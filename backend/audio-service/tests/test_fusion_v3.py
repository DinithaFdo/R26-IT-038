"""Fusion V3 (constrained 4-branch convex weighted fusion): artifact
validation, exact fusion math, branch-requirement policy, the legacy
explicit-fallback contract, response metadata, mathematical parity against
the research eval CSV, and one real end-to-end VoiceService run.

See `app/utils/fusion.py` (`ConvexFusionContract`, `load_convex_fusion_contract`,
`ConstrainedFourBranchFusion`) and `app/services/voice_service.py`
(`VoiceService._fuse`) for the implementation this covers.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import pytest

from app.config.settings import Settings
from app.models.base import BaseVoiceModel
from app.models.registry import ModelRegistry
from app.schemas.common import BranchStatus, ModelMode, PredictionLabel
from app.schemas.prediction import BranchPrediction, ProbabilityScores
from app.services.voice_service import VoiceService
from app.utils.fusion import (
    CONVEX_V3_BRANCH_ORDER,
    CONVEX_V3_THRESHOLD,
    CONVEX_V3_VERSION,
    FROZEN_RESEARCH_THRESHOLD,
    ConstrainedFourBranchFusion,
    ConvexFusionContractError,
    FusionEngine,
    load_convex_fusion_contract,
)
from tests.real_model_helpers import (
    processed_audio,
    requires_aasist_checkpoint,
    requires_cnn_checkpoint,
    requires_ssl_checkpoint,
    requires_torch,
    requires_transformers,
)
from tests.test_glottal_real_branch import (
    requires_glottal_artifacts,
    requires_glottal_dependencies,
)

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_ROOT.parent
_MODEL_ARTIFACTS_ROOT = _REPO_ROOT / "model_artifacts" / "fusion"
_V3_JSON_PATH = _MODEL_ARTIFACTS_ROOT / "fusion_convex_4branch_v3.json"
_V3_JOBLIB_PATH = _MODEL_ARTIFACTS_ROOT / "fusion_convex_4branch_v3.joblib"
_EVAL_CSV_PATH = (
    _REPO_ROOT
    / "research-artifacts"
    / "fusion"
    / "fusion_convex_4branch_v3_eval_predictions.csv"
)

_EXPECTED_WEIGHTS = {
    "lfcc_cnn_tcn": 0.3488902015351922,
    "aasist": 0.289776249915524,
    "ssl_sequence": 0.22524436863750652,
    "glottal": 0.13608917991177738,
}

requires_v3_artifacts = pytest.mark.skipif(
    not (_V3_JSON_PATH.is_file() and _V3_JOBLIB_PATH.is_file()),
    reason=f"Fusion V3 artifacts not found under {_MODEL_ARTIFACTS_ROOT}.",
)
requires_eval_csv = pytest.mark.skipif(
    not _EVAL_CSV_PATH.is_file(),
    reason=f"Fusion V3 eval predictions CSV not found at {_EVAL_CSV_PATH}.",
)


def _branch(
    model_name: str,
    spoof_probability: float,
    *,
    mode: ModelMode = ModelMode.real,
    status: BranchStatus = BranchStatus.success,
    research_result: bool = True,
) -> BranchPrediction:
    if status != BranchStatus.success:
        return BranchPrediction(
            model_name=model_name,
            display_name=model_name,
            status=status,
            mode=mode,
            processing_time_ms=1.0,
            error="branch unavailable",
            metadata={"error_code": "branch_failed", "research_result": False},
        )
    bonafide_probability = 1.0 - spoof_probability
    # Matches `_validate_prediction_confidence`'s own tie-break (dict-order
    # `max`, bonafide first) exactly: a strict `>`, not `>=`, at the 0.5 tie.
    prediction = (
        PredictionLabel.spoof
        if spoof_probability > bonafide_probability
        else PredictionLabel.bonafide
    )
    return BranchPrediction(
        model_name=model_name,
        display_name=model_name,
        status=status,
        mode=mode,
        prediction=prediction,
        confidence=max(spoof_probability, bonafide_probability),
        probabilities=ProbabilityScores(bonafide=bonafide_probability, spoof=spoof_probability),
        processing_time_ms=1.0,
        metadata={"research_result": research_result},
    )


def _four_real_branches(
    *,
    cnn: float = 0.2,
    aasist: float = 0.4,
    ssl: float = 0.6,
    glottal: float = 0.9,
) -> list[BranchPrediction]:
    return [
        _branch("cnn_acoustic", cnn),
        _branch("aasist", aasist),
        _branch("ssl_wavlm_xlsr", ssl),
        _branch("glottal_features", glottal),
    ]


def _v3_engine() -> ConstrainedFourBranchFusion:
    contract = load_convex_fusion_contract(_V3_JSON_PATH, _V3_JOBLIB_PATH)
    assert contract is not None
    return ConstrainedFourBranchFusion(contract=contract)


# ---------------------------------------------------------------------------
# 1-11: Artifact loading and validation
# ---------------------------------------------------------------------------


@requires_v3_artifacts
def test_v3_config_loads() -> None:
    contract = load_convex_fusion_contract(_V3_JSON_PATH, _V3_JOBLIB_PATH)

    assert contract is not None
    assert contract.version == CONVEX_V3_VERSION


@requires_v3_artifacts
def test_v3_feature_order_validation() -> None:
    contract = load_convex_fusion_contract(_V3_JSON_PATH, _V3_JOBLIB_PATH)

    assert contract is not None
    assert contract.feature_order == (
        "cnn_spoof_probability",
        "aasist_spoof_probability",
        "ssl_spoof_probability",
        "glottal_spoof_probability",
    )


@requires_v3_artifacts
def test_v3_four_weights_loaded_all_non_negative_sum_to_one() -> None:
    contract = load_convex_fusion_contract(_V3_JSON_PATH, _V3_JOBLIB_PATH)

    assert contract is not None
    assert len(contract.weights) == 4
    assert all(weight >= 0 for weight in contract.weights.values())
    assert sum(contract.weights.values()) == pytest.approx(1.0, abs=1e-9)


@requires_v3_artifacts
def test_v3_threshold_is_exactly_half() -> None:
    contract = load_convex_fusion_contract(_V3_JSON_PATH, _V3_JOBLIB_PATH)

    assert contract is not None
    assert contract.threshold == pytest.approx(CONVEX_V3_THRESHOLD, abs=1e-12)


@requires_v3_artifacts
@pytest.mark.parametrize(
    ("branch", "expected"),
    list(_EXPECTED_WEIGHTS.items()),
)
def test_v3_each_branch_weight_is_exact(branch: str, expected: float) -> None:
    contract = load_convex_fusion_contract(_V3_JSON_PATH, _V3_JOBLIB_PATH)

    assert contract is not None
    assert contract.weights[branch] == pytest.approx(expected, abs=1e-15)


def test_v3_version_validation_rejects_wrong_version(tmp_path) -> None:
    import json

    bad = tmp_path / "fusion_convex_4branch_v3.json"
    bad.write_text(
        json.dumps(
            {
                "version": "v2",
                "architecture": "constrained_convex_weighted_fusion",
                "feature_order": [
                    "cnn_spoof_probability",
                    "aasist_spoof_probability",
                    "ssl_spoof_probability",
                    "glottal_spoof_probability",
                ],
                "weights": {**{f"{k}_spoof_probability": v for k, v in {
                    "cnn": 0.35, "aasist": 0.29, "ssl": 0.22, "glottal": 0.14,
                }.items()}},
                "constraints": {
                    "non_negative_weights": True,
                    "weights_sum_to_one": True,
                    "intercept": False,
                    "post_sigmoid_calibration": False,
                },
                "decision": {"threshold": 0.5},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConvexFusionContractError, match="version"):
        load_convex_fusion_contract(bad)


def test_v3_feature_order_validation_rejects_wrong_order(tmp_path) -> None:
    import json

    bad = tmp_path / "fusion_convex_4branch_v3.json"
    bad.write_text(
        json.dumps(
            {
                "version": "v3",
                "architecture": "constrained_convex_weighted_fusion",
                "feature_order": [
                    "aasist_spoof_probability",
                    "cnn_spoof_probability",
                    "ssl_spoof_probability",
                    "glottal_spoof_probability",
                ],
                "weights": {
                    "cnn_spoof_probability": 0.35,
                    "aasist_spoof_probability": 0.29,
                    "ssl_spoof_probability": 0.22,
                    "glottal_spoof_probability": 0.14,
                },
                "constraints": {
                    "non_negative_weights": True,
                    "weights_sum_to_one": True,
                    "intercept": False,
                    "post_sigmoid_calibration": False,
                },
                "decision": {"threshold": 0.5},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConvexFusionContractError, match="feature_order"):
        load_convex_fusion_contract(bad)


@pytest.mark.parametrize(
    "weights",
    [
        # Negative weight.
        {
            "cnn_spoof_probability": -0.05,
            "aasist_spoof_probability": 0.35,
            "ssl_spoof_probability": 0.35,
            "glottal_spoof_probability": 0.35,
        },
        # Does not sum to 1.
        {
            "cnn_spoof_probability": 0.5,
            "aasist_spoof_probability": 0.5,
            "ssl_spoof_probability": 0.5,
            "glottal_spoof_probability": 0.5,
        },
    ],
)
def test_v3_invalid_weights_rejected(tmp_path, weights: dict[str, float]) -> None:
    import json

    bad = tmp_path / "fusion_convex_4branch_v3.json"
    bad.write_text(
        json.dumps(
            {
                "version": "v3",
                "architecture": "constrained_convex_weighted_fusion",
                "feature_order": [
                    "cnn_spoof_probability",
                    "aasist_spoof_probability",
                    "ssl_spoof_probability",
                    "glottal_spoof_probability",
                ],
                "weights": weights,
                "constraints": {
                    "non_negative_weights": True,
                    "weights_sum_to_one": True,
                    "intercept": False,
                    "post_sigmoid_calibration": False,
                },
                "decision": {"threshold": 0.5},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConvexFusionContractError):
        load_convex_fusion_contract(bad)


def test_v3_tampered_threshold_rejected(tmp_path) -> None:
    import json

    bad = tmp_path / "fusion_convex_4branch_v3.json"
    bad.write_text(
        json.dumps(
            {
                "version": "v3",
                "architecture": "constrained_convex_weighted_fusion",
                "feature_order": [
                    "cnn_spoof_probability",
                    "aasist_spoof_probability",
                    "ssl_spoof_probability",
                    "glottal_spoof_probability",
                ],
                "weights": {
                    "cnn_spoof_probability": 0.35,
                    "aasist_spoof_probability": 0.29,
                    "ssl_spoof_probability": 0.22,
                    "glottal_spoof_probability": 0.14,
                },
                "constraints": {
                    "non_negative_weights": True,
                    "weights_sum_to_one": True,
                    "intercept": False,
                    "post_sigmoid_calibration": False,
                },
                "decision": {"threshold": 0.6},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConvexFusionContractError, match="threshold"):
        load_convex_fusion_contract(bad)


def test_missing_v3_config_file_returns_none(tmp_path) -> None:
    assert load_convex_fusion_contract(tmp_path / "does_not_exist.json") is None


# ---------------------------------------------------------------------------
# 12-15: Exact fusion math (no sigmoid, no intercept)
# ---------------------------------------------------------------------------


@requires_v3_artifacts
def test_v3_exact_dot_product_math() -> None:
    engine = _v3_engine()
    branches = _four_real_branches(cnn=0.2, aasist=0.4, ssl=0.6, glottal=0.9)

    result = engine.fuse(branches)

    expected = np.dot(
        [0.2, 0.4, 0.6, 0.9],
        [_EXPECTED_WEIGHTS[b] for b in CONVEX_V3_BRANCH_ORDER],
    )
    assert result.status == BranchStatus.success
    assert result.probabilities is not None
    assert result.probabilities.spoof == pytest.approx(float(expected), abs=1e-12)


@requires_v3_artifacts
def test_v3_no_sigmoid_applied() -> None:
    """A pure weighted sum is linear in each input; a sigmoid would not be."""

    engine = _v3_engine()
    low = engine.fuse(_four_real_branches(cnn=0.1, aasist=0.1, ssl=0.1, glottal=0.1))
    high = engine.fuse(_four_real_branches(cnn=0.9, aasist=0.9, ssl=0.9, glottal=0.9))

    assert low.probabilities is not None
    assert high.probabilities is not None
    # Every branch weighted by 0.1 vs 0.9 over weights summing to 1 -> exactly
    # 0.1 and 0.9 output. A sigmoid/calibration layer would compress these
    # away from the raw linear values.
    assert low.probabilities.spoof == pytest.approx(0.1, abs=1e-12)
    assert high.probabilities.spoof == pytest.approx(0.9, abs=1e-12)


@requires_v3_artifacts
def test_v3_no_intercept_applied() -> None:
    engine = _v3_engine()

    result = engine.fuse(_four_real_branches(cnn=0.0, aasist=0.0, ssl=0.0, glottal=0.0))

    assert result.probabilities is not None
    assert result.probabilities.spoof == pytest.approx(0.0, abs=1e-12)


@requires_v3_artifacts
def test_v3_glottal_materially_changes_fusion_result() -> None:
    engine = _v3_engine()

    low_glottal = engine.fuse(_four_real_branches(glottal=0.0))
    high_glottal = engine.fuse(_four_real_branches(glottal=1.0))

    assert low_glottal.probabilities is not None
    assert high_glottal.probabilities is not None
    delta = high_glottal.probabilities.spoof - low_glottal.probabilities.spoof
    assert delta == pytest.approx(_EXPECTED_WEIGHTS["glottal"], abs=1e-12)


# ---------------------------------------------------------------------------
# 16-20: Branch requirement policy
# ---------------------------------------------------------------------------


@requires_v3_artifacts
def test_all_four_successful_branches_uses_v3() -> None:
    engine = _v3_engine()

    result = engine.fuse(_four_real_branches())

    assert result.status == BranchStatus.success
    assert result.fusion_version == CONVEX_V3_VERSION
    assert result.fallback_used is False


@requires_v3_artifacts
def test_glottal_included_in_v3_contributing_branches() -> None:
    engine = _v3_engine()

    result = engine.fuse(_four_real_branches())

    assert "glottal_features" in result.contributing_branches


@requires_v3_artifacts
def test_one_branch_failed_v3_not_fabricated() -> None:
    engine = _v3_engine()
    branches = _four_real_branches()
    branches[3] = _branch("glottal_features", 0.0, status=BranchStatus.failed)

    result = engine.fuse(branches)

    assert result.status == BranchStatus.failed
    assert result.probabilities is None
    assert result.prediction is None
    assert result.contributing_branches == []


@requires_v3_artifacts
def test_no_automatic_three_weight_renormalization_when_glottal_fails() -> None:
    """The 3 remaining branches must never be renormalised and scored as a
    substitute V3 result -- the failed result must carry no probabilities at
    all, not a 3-way-weighted approximation."""

    engine = _v3_engine()
    branches = _four_real_branches()
    branches[3] = _branch("glottal_features", 0.99, status=BranchStatus.failed)

    result = engine.fuse(branches)

    assert result.status == BranchStatus.failed
    assert result.branch_weights == {}
    assert result.excluded_branches.get("glottal_features") is not None


@requires_v3_artifacts
def test_dummy_mode_branch_makes_v3_unavailable() -> None:
    """A dummy-mode placeholder standing in for a required real branch must
    not let the frozen, research-calibrated V3 contract run over it."""

    engine = _v3_engine()
    branches = _four_real_branches()
    branches[3] = _branch("glottal_features", 0.9, mode=ModelMode.dummy)

    result = engine.fuse(branches)

    assert result.status == BranchStatus.failed
    assert result.excluded_branches.get("glottal_features") == "not_real_mode"


# ---------------------------------------------------------------------------
# 21-22: Legacy fallback contract
# ---------------------------------------------------------------------------


def test_legacy_fallback_uses_only_cnn_aasist_ssl(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        glottal_model_path="glottal/glottal_logreg_selected20_v1.joblib",
    )
    engine = FusionEngine.from_settings(settings)
    assert engine.contract_version is not None  # frozen legacy contract active

    result = engine.fuse(_four_real_branches(cnn=0.2, aasist=0.4, ssl=0.6, glottal=0.999))

    assert result.status == BranchStatus.success
    assert result.probabilities is not None
    assert result.probabilities.spoof == pytest.approx((0.2 + 0.4 + 0.6) / 3, abs=1e-9)
    assert "glottal_features" not in result.contributing_branches


def test_legacy_fallback_threshold_is_the_frozen_value() -> None:
    settings = Settings(
        _env_file=None,
        glottal_model_path="glottal/glottal_logreg_selected20_v1.joblib",
    )
    engine = FusionEngine.from_settings(settings)

    result = engine.fuse(_four_real_branches())

    assert result.decision_threshold == pytest.approx(FROZEN_RESEARCH_THRESHOLD, abs=1e-12)
    assert engine.fusion_version == "legacy-3branch-frozen-v1"


# ---------------------------------------------------------------------------
# VoiceService selection logic: try V3 first, explicit-named fallback
# ---------------------------------------------------------------------------


class _FixedRealModel(BaseVoiceModel):
    def __init__(self, model_name: str, spoof_probability: float) -> None:
        super().__init__()
        self._name = model_name
        self._spoof_probability = spoof_probability

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._name

    @property
    def mode(self) -> ModelMode:
        return ModelMode.real

    @property
    def is_loaded(self) -> bool:
        return True

    def predict(self, processed_audio) -> BranchPrediction:
        return _branch(self._name, self._spoof_probability, mode=ModelMode.real)


def _voice_service_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "cnn_model_mode": "real",
        "aasist_model_mode": "real",
        "ssl_model_mode": "real",
        "glottal_model_mode": "real",
        "glottal_model_path": "glottal/glottal_logreg_selected20_v1.joblib",
        "glottal_selected_features_path": "glottal/glottal_selected_features_v1.json",
        "model_root_dir": "../model_artifacts",
        "model_device_policy": "cpu",
        "model_default_device": "cpu",
        "model_load_strategy": "lazy",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _voice_service_with_fixed_branches(settings: Settings, scores: dict[str, float]) -> VoiceService:
    registry = ModelRegistry(
        models=[
            _FixedRealModel("cnn_acoustic", scores["cnn_acoustic"]),
            _FixedRealModel("aasist", scores["aasist"]),
            _FixedRealModel("ssl_wavlm_xlsr", scores["ssl_wavlm_xlsr"]),
            _FixedRealModel("glottal_features", scores["glottal_features"]),
        ],
        app_settings=settings,
    )
    return VoiceService(
        model_registry=registry,
        preprocess_fn=lambda _upload: processed_audio(),
        app_settings=settings,
    )


@requires_v3_artifacts
def test_response_metadata_states_which_fusion_version_ran(tmp_path) -> None:
    settings = _voice_service_settings()
    service = _voice_service_with_fixed_branches(
        settings,
        {"cnn_acoustic": 0.2, "aasist": 0.4, "ssl_wavlm_xlsr": 0.6, "glottal_features": 0.9},
    )

    response = service.predict_from_validated_upload(_upload_metadata(tmp_path), cleanup_upload=False)

    assert response.fusion.status == BranchStatus.success
    assert response.fusion.fusion_version == CONVEX_V3_VERSION
    assert response.fusion.fusion_mode == "learned_constrained"
    assert response.fusion.fallback_used is False
    assert response.fusion.decision_threshold == pytest.approx(0.5, abs=1e-12)
    assert response.fusion.branch_weights["glottal_features"] == pytest.approx(
        _EXPECTED_WEIGHTS["glottal"], abs=1e-12
    )
    assert response.provenance is not None
    assert response.provenance.fusion.fusion_version == CONVEX_V3_VERSION
    assert response.provenance.fusion.fallback_used is False


@requires_v3_artifacts
def test_v3_unavailable_falls_back_to_legacy_with_fallback_flag(tmp_path) -> None:
    """SSL failing makes V3 unavailable (only 3/4 required branches real+
    successful); VoiceService must fall back to the legacy detector and mark
    the response `fallback_used=True`, never call it "Fusion V3"."""

    settings = _voice_service_settings()
    registry = ModelRegistry(
        models=[
            _FixedRealModel("cnn_acoustic", 0.2),
            _FixedRealModel("aasist", 0.4),
            _FailingModel("ssl_wavlm_xlsr"),
            _FixedRealModel("glottal_features", 0.9),
        ],
        app_settings=settings,
    )
    service = VoiceService(
        model_registry=registry,
        preprocess_fn=lambda _upload: processed_audio(),
        app_settings=settings,
    )

    response = service.predict_from_validated_upload(_upload_metadata(tmp_path), cleanup_upload=False)

    assert response.fusion.status == BranchStatus.success
    assert response.fusion.fusion_version == "legacy-3branch-frozen-v1"
    assert response.fusion.fallback_used is True
    assert response.fusion.decision_threshold == pytest.approx(FROZEN_RESEARCH_THRESHOLD, abs=1e-12)
    assert "glottal_features" not in response.fusion.contributing_branches


class _FailingModel(BaseVoiceModel):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        self._name = model_name

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._name

    @property
    def mode(self) -> ModelMode:
        return ModelMode.real

    @property
    def is_loaded(self) -> bool:
        return True

    def predict(self, processed_audio) -> BranchPrediction:
        return _branch(self._name, 0.0, status=BranchStatus.failed)


def test_legacy_mode_setting_disables_v3_without_being_a_fallback(tmp_path) -> None:
    settings = _voice_service_settings(fusion_mode="legacy_3branch")
    service = _voice_service_with_fixed_branches(
        settings,
        {"cnn_acoustic": 0.2, "aasist": 0.4, "ssl_wavlm_xlsr": 0.6, "glottal_features": 0.9},
    )

    response = service.predict_from_validated_upload(_upload_metadata(tmp_path), cleanup_upload=False)

    assert response.fusion.fusion_version == "legacy-3branch-frozen-v1"
    # Deliberately configured primary, not a fallback from a failed V3 attempt.
    assert response.fusion.fallback_used is False


# ---------------------------------------------------------------------------
# 23-24: Threshold boundary / label mapping
# ---------------------------------------------------------------------------


@requires_v3_artifacts
def test_v3_threshold_boundary_around_half() -> None:
    engine = _v3_engine()

    # An exact 0.5/0.5 tie: `FusionResult`'s own validator independently
    # derives the winning label as argmax(bonafide, spoof), which favours
    # bonafide on a tie -- the fusion decision must agree with it exactly
    # here, or constructing the result would raise (see the comment on the
    # `>` in `ConstrainedFourBranchFusion.fuse`).
    tie = engine.fuse(_four_real_branches(cnn=0.5, aasist=0.5, ssl=0.5, glottal=0.5))
    assert tie.probabilities.spoof == pytest.approx(0.5, abs=1e-12)
    assert tie.prediction == PredictionLabel.bonafide

    just_above = engine.fuse(_four_real_branches(cnn=0.51, aasist=0.51, ssl=0.51, glottal=0.51))
    assert just_above.prediction == PredictionLabel.spoof


@requires_v3_artifacts
def test_v3_bonafide_spoof_mapping_correct() -> None:
    engine = _v3_engine()

    spoof_leaning = engine.fuse(_four_real_branches(cnn=0.9, aasist=0.9, ssl=0.9, glottal=0.9))
    bonafide_leaning = engine.fuse(_four_real_branches(cnn=0.1, aasist=0.1, ssl=0.1, glottal=0.1))

    assert spoof_leaning.prediction == PredictionLabel.spoof
    assert bonafide_leaning.prediction == PredictionLabel.bonafide


# ---------------------------------------------------------------------------
# 11: Mathematical parity against the research eval CSV
# ---------------------------------------------------------------------------


@requires_v3_artifacts
@requires_eval_csv
def test_v3_matches_research_eval_csv_within_machine_precision() -> None:
    engine = _v3_engine()

    with _EVAL_CSV_PATH.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows

    bonafide_row = next(row for row in rows if row["target"] == "0")
    spoof_row = next(row for row in rows if row["target"] == "1")
    rng = random.Random(0)
    sample_rows = [bonafide_row, spoof_row, *rng.sample(rows, k=min(80, len(rows)))]

    diffs: list[float] = []
    for row in sample_rows:
        branches = [
            _branch("cnn_acoustic", float(row["cnn_spoof_probability"])),
            _branch("aasist", float(row["aasist_spoof_probability"])),
            _branch("ssl_wavlm_xlsr", float(row["ssl_spoof_probability"])),
            _branch("glottal_features", float(row["glottal_spoof_probability"])),
        ]
        result = engine.fuse(branches)
        assert result.status == BranchStatus.success
        assert result.probabilities is not None

        expected = float(row["fusion_v3_spoof_probability"])
        diffs.append(abs(result.probabilities.spoof - expected))

        expected_prediction = int(row["fusion_v3_prediction"])
        actual_prediction = 1 if result.prediction == PredictionLabel.spoof else 0
        assert actual_prediction == expected_prediction, row["audio_id"]

    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    assert max_diff <= 1e-9, f"max abs diff {max_diff} exceeds tolerance (mean={mean_diff})"


# ---------------------------------------------------------------------------
# Real end-to-end: VoiceService -> CNN -> AASIST -> SSL -> Glottal -> V3
# ---------------------------------------------------------------------------


@requires_v3_artifacts
@requires_torch
@requires_transformers
@requires_cnn_checkpoint
@requires_aasist_checkpoint
@requires_ssl_checkpoint
@requires_glottal_artifacts
@requires_glottal_dependencies
@pytest.mark.integration
@pytest.mark.network
def test_real_four_branch_request_reaches_v3_fusion(tmp_path) -> None:
    """The one mandatory proof point: a real audio input driven through the
    real CNN, AASIST, SSL, and Glottal branches reaches Fusion V3 -- not a
    synthetic BranchPrediction stand-in for any of them."""

    settings = _voice_service_settings(
        # Generous: a cold SSL (XLS-R backbone) load plus DisVoice/IAIF
        # extraction for Glottal can each take several seconds on CPU.
        model_branch_timeout_seconds=120,
    )
    service = VoiceService(app_settings=settings, preprocess_fn=lambda _upload: processed_audio())

    response = service.predict_from_validated_upload(_upload_metadata(tmp_path), cleanup_upload=False)

    for branch in response.branches:
        assert branch.status == BranchStatus.success, (branch.model_name, branch.error)
        assert branch.mode == ModelMode.real
        assert branch.probabilities is not None

    assert response.fusion.status == BranchStatus.success
    assert response.fusion.fusion_version == CONVEX_V3_VERSION
    assert response.fusion.fallback_used is False
    assert response.fusion.decision_threshold == pytest.approx(0.5, abs=1e-12)
    for branch_name, weight in _EXPECTED_WEIGHTS.items():
        assert response.fusion.branch_weights[
            next(b.model_name for b in response.branches if _canonical(b.model_name) == branch_name)
        ] == pytest.approx(weight, abs=1e-12)

    scores = {_canonical(b.model_name): b.probabilities.spoof for b in response.branches}
    expected = sum(_EXPECTED_WEIGHTS[name] * scores[name] for name in _EXPECTED_WEIGHTS)
    assert response.fusion.probabilities.spoof == pytest.approx(expected, abs=1e-9)
    expected_prediction = (
        PredictionLabel.spoof if expected >= 0.5 else PredictionLabel.bonafide
    )
    assert response.fusion.prediction == expected_prediction

    print(
        "\nFusion V3 real end-to-end result:\n"
        f"  branch scores: {scores}\n"
        f"  fusion_spoof_probability: {response.fusion.probabilities.spoof}\n"
        f"  prediction: {response.fusion.prediction}\n"
        f"  fusion_version: {response.fusion.fusion_version}\n"
        f"  fallback_used: {response.fusion.fallback_used}\n"
    )


def _canonical(model_name: str) -> str:
    from app.models.runtime import canonical_branch_name

    return canonical_branch_name(model_name)


def _upload_metadata(tmp_path: Path):
    from app.ingestion.audio import AudioInspectionResult, AudioUploadMetadata

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not-real-audio-bytes")
    return AudioUploadMetadata(
        original_filename="sample.wav",
        sanitized_filename="sample.wav",
        saved_filename="sample.wav",
        saved_path=audio_path,
        content_type="audio/wav",
        file_size_bytes=audio_path.stat().st_size,
        duration_seconds=3.0,
        sample_rate=16000,
        channels=1,
        original_extension="wav",
        detected_container="wav",
        detected_codec="pcm_s16le",
        inspection=AudioInspectionResult(
            duration_seconds=3.0,
            sample_rate=16000,
            channels=1,
            detected_container="wav",
            detected_codec="pcm_s16le",
            audio_stream_index=0,
        ),
    )
