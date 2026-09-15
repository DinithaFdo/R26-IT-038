"""Final research detector integration: real 3-branch fusion + config guards.

Covers, in one place:

* a real end-to-end run of CNN-V2 + AASIST-Light V2 + SSL (ASVspoof5) through
  the actual production `FusionEngine`, proving the fused score is exactly
  the arithmetic mean of the three real branch scores and that Glottal never
  enters the calculation (Phase 5, item 12 of the final-detector integration);
* configuration regressions that would silently break the frozen research
  contract: threshold drift, a wrong primary branch set, Glottal leaking into
  the primary decision, or losing the explicit `simple_average` method
  (Phase 5, item 13).

None of this needs labelled audio -- these are mechanical/arithmetic and
configuration-loading checks, not accuracy claims. See
`scripts/validate_colab_parity.py` for the (currently unrunnable, no local
audio) accuracy-parity tooling.
"""

from __future__ import annotations

import math

import pytest

from app.config.settings import Settings
from app.models.factory import ModelFactory
from app.schemas.common import BranchStatus
from app.utils.fusion import (
    FROZEN_RESEARCH_THRESHOLD,
    FusionContractError,
    FusionEngine,
)
from tests.real_model_helpers import (
    processed_audio,
    real_model_settings,
    requires_aasist_checkpoint,
    requires_cnn_checkpoint,
    requires_ssl_checkpoint,
    requires_torch,
    requires_transformers,
)

pytestmark = requires_torch

PRIMARY_BRANCHES = ("lfcc_cnn_tcn", "aasist", "ssl_sequence")


# ---------------------------------------------------------------------------
# 1. Real 3-branch fusion (item 12)
# ---------------------------------------------------------------------------


@requires_cnn_checkpoint
@requires_aasist_checkpoint
@requires_ssl_checkpoint
@requires_transformers
@pytest.mark.integration
@pytest.mark.network
def test_real_three_branch_fusion_matches_simple_average_of_real_branches() -> None:
    """The single most important end-to-end check for this integration.

    Loads the three real, currently-deployed checkpoints (CNN-V2, AASIST-Light
    V2, SSL/ASVspoof5), runs one real prediction through each, and asserts:
    every branch probability is finite and in [0, 1]; the fused score is
    exactly the arithmetic mean of the three; the threshold is exactly the
    frozen value; and Glottal (disabled, no checkpoint) never contributes.

    Marked `integration`/`network` because SSL fetches/uses the XLS-R
    backbone -- matching the existing convention for real SSL e2e tests.
    """

    settings = real_model_settings(ssl_model_mode="real")
    factory = ModelFactory(settings)
    audio = processed_audio()

    predictions = [factory.create(branch).predict_safe(audio) for branch in PRIMARY_BRANCHES]
    for branch, prediction in zip(PRIMARY_BRANCHES, predictions, strict=True):
        assert prediction.status == BranchStatus.success, (branch, prediction.error)
        assert prediction.probabilities is not None
        assert math.isfinite(prediction.probabilities.spoof)
        assert 0.0 <= prediction.probabilities.spoof <= 1.0
        assert math.isfinite(prediction.probabilities.bonafide)
        assert 0.0 <= prediction.probabilities.bonafide <= 1.0

    engine = FusionEngine.from_settings(settings)
    result = engine.fuse(predictions)

    assert result.status == BranchStatus.success
    assert result.method == "simple_average"
    assert result.decision_threshold == pytest.approx(FROZEN_RESEARCH_THRESHOLD, abs=1e-12)
    assert engine.spoof_threshold == pytest.approx(FROZEN_RESEARCH_THRESHOLD, abs=1e-12)

    manual_mean = sum(p.probabilities.spoof for p in predictions) / 3
    assert result.probabilities is not None
    assert result.probabilities.spoof == pytest.approx(manual_mean, abs=1e-9)

    # Glottal never contributes: it is not in contributing_branches, and its
    # weight/score never entered the mean above (proven by the exact-match
    # assertion, not just by absence from this list).
    contributing_canonical = set(result.contributing_branches)
    assert "glottal_features" not in contributing_canonical
    assert "glottal" not in contributing_canonical

    prediction_verdict = "spoof" if manual_mean >= FROZEN_RESEARCH_THRESHOLD else "bonafide"
    assert result.prediction is not None
    assert result.prediction.value == prediction_verdict


@requires_cnn_checkpoint
@requires_aasist_checkpoint
def test_two_of_three_primary_branches_is_degraded_not_research_eligible() -> None:
    """SSL failing (disabled here) must still produce an operational fused
    score -- never a silent claim that a 2-branch result equals the 3-branch
    research detector."""

    settings = real_model_settings(ssl_model_mode="disabled")
    factory = ModelFactory(settings)
    audio = processed_audio()

    predictions = [factory.create(branch).predict_safe(audio) for branch in ("lfcc_cnn_tcn", "aasist")]
    engine = FusionEngine.from_settings(settings)
    result = engine.fuse(predictions)

    assert result.status == BranchStatus.success
    assert result.system_stage == "partial"
    assert result.eligible_for_research_evaluation is False
    assert "incomplete_branch_set" in result.research_blockers
    assert result.warning is not None


# ---------------------------------------------------------------------------
# 2. Configuration regressions (item 13)
# ---------------------------------------------------------------------------


def test_threshold_cannot_silently_fall_back_to_point_five() -> None:
    """An explicit FUSION_DECISION_THRESHOLD override that disagrees with the
    frozen contract must fail startup, never silently win or silently lose."""

    with pytest.raises(Exception) as excinfo:
        Settings(_env_file=None, fusion_decision_threshold=0.5)
    assert "FUSION_DECISION_THRESHOLD" in str(excinfo.value)
    assert "0.5519237850482265" in str(excinfo.value) or "conflicts" in str(excinfo.value)


def test_threshold_resolved_from_settings_is_exactly_the_frozen_value() -> None:
    settings = Settings(_env_file=None)
    contract = settings.frozen_fusion_contract
    assert contract is not None
    assert contract.decision_threshold == FROZEN_RESEARCH_THRESHOLD
    engine = FusionEngine.from_settings(settings)
    assert engine.spoof_threshold == FROZEN_RESEARCH_THRESHOLD


def test_primary_research_branches_are_exactly_cnn_aasist_ssl() -> None:
    from app.utils.fusion import FULL_SYSTEM_BRANCHES

    assert set(FULL_SYSTEM_BRANCHES) == {"lfcc_cnn_tcn", "aasist", "ssl_sequence"}
    assert "glottal" not in FULL_SYSTEM_BRANCHES

    settings = Settings(_env_file=None)
    contract = settings.frozen_fusion_contract
    assert contract is not None
    assert set(contract.primary_branches) == {"lfcc_cnn_tcn", "aasist", "ssl_sequence"}


def test_simple_average_is_the_resolved_fusion_method() -> None:
    settings = Settings(_env_file=None)
    engine = FusionEngine.from_settings(settings)
    assert engine.method == "simple_average"


def test_fusion_method_override_conflicting_with_contract_fails_startup() -> None:
    with pytest.raises(Exception) as excinfo:
        Settings(_env_file=None, fusion_method="weighted_average")
    assert "FUSION_METHOD" in str(excinfo.value)


def test_glottal_is_excluded_from_primary_fusion_even_if_forced_successful() -> None:
    """Belt-and-braces guard: even a synthetic, artificially-successful
    Glottal `BranchPrediction` must never enter the primary fusion sum when
    the engine was built from the frozen contract."""

    from app.schemas.common import ModelMode, PredictionLabel
    from app.schemas.prediction import BranchPrediction, ProbabilityScores

    settings = Settings(_env_file=None)
    engine = FusionEngine.from_settings(settings)
    assert engine.contract_version is not None

    cnn = BranchPrediction(
        model_name="cnn_acoustic",
        display_name="CNN",
        status=BranchStatus.success,
        mode=ModelMode.dummy,
        prediction=PredictionLabel.bonafide,
        confidence=0.9,
        probabilities=ProbabilityScores(bonafide=0.9, spoof=0.1),
        processing_time_ms=1.0,
    )
    aasist = cnn.model_copy(update={"model_name": "aasist", "display_name": "AASIST"})
    ssl = cnn.model_copy(update={"model_name": "ssl_wavlm_xlsr", "display_name": "SSL"})
    glottal_forced_high_spoof = BranchPrediction(
        model_name="glottal_features",
        display_name="Glottal",
        status=BranchStatus.success,
        mode=ModelMode.dummy,
        prediction=PredictionLabel.spoof,
        confidence=1.0,
        probabilities=ProbabilityScores(bonafide=0.0, spoof=1.0),
        processing_time_ms=1.0,
    )

    result = engine.fuse([cnn, aasist, ssl, glottal_forced_high_spoof])

    assert result.status == BranchStatus.success
    assert result.probabilities is not None
    # If Glottal had leaked in, a spoof=1.0 fourth branch would pull the mean
    # up from 0.1 toward 0.325; it must instead be exactly the 3-way mean.
    assert result.probabilities.spoof == pytest.approx(0.1, abs=1e-9)
    assert "glottal_features" not in result.contributing_branches


def test_missing_contract_file_falls_back_to_plain_settings_fusion() -> None:
    """FUSION_CONTRACT_PATH="" must disable contract loading entirely, not error."""

    settings = Settings(_env_file=None, fusion_contract_path="")
    assert settings.frozen_fusion_contract is None
    engine = FusionEngine.from_settings(settings)
    assert engine.contract_version is None
    assert engine.method == settings.fusion_method
    assert engine.spoof_threshold == settings.fusion_decision_threshold


def test_load_fusion_contract_rejects_a_tampered_threshold(tmp_path) -> None:
    import json

    from app.utils.fusion import load_fusion_contract

    tampered = tmp_path / "final_detector_v1.json"
    tampered.write_text(
        json.dumps(
            {
                "contract_version": "tampered",
                "fusion_method": "simple_average",
                "decision_threshold": 0.5,
                "primary_branches": ["lfcc_cnn_tcn", "aasist", "ssl_sequence"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FusionContractError):
        load_fusion_contract(tampered)


def test_load_fusion_contract_rejects_glottal_in_primary_branches(tmp_path) -> None:
    import json

    from app.utils.fusion import load_fusion_contract

    tampered = tmp_path / "final_detector_v1.json"
    tampered.write_text(
        json.dumps(
            {
                "contract_version": "tampered",
                "fusion_method": "simple_average",
                "decision_threshold": FROZEN_RESEARCH_THRESHOLD,
                "primary_branches": ["lfcc_cnn_tcn", "aasist", "ssl_sequence", "glottal"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FusionContractError):
        load_fusion_contract(tampered)
