"""CNN verification foundation + real-model guard-repair tests.

Covers, in one place:

* checkpoint identity resolution proving test-path resolution == production
  path resolution (independently cross-checked against a raw parse of
  `backend/.env`, not just against the same resolver under test);
* pinned AASIST-Light V2 checkpoint identity (sha256/architecture/epoch/
  parameter count) so a silently swapped checkpoint fails loudly;
* the repaired `ProcessedAudio` test helper actually carrying
  `unnormalised_waveform`, and real AASIST V2 running through it;
* mechanical (not evidentiary) contract tests for CNN class-order selection;
* a regression-characterization test documenting the known gap where AASIST's
  runtime spoof index is *not* cross-checked against the checkpoint's own
  attested `spoof_class_index` (finding B5 in the integration audit) -- not
  fixed here, deliberately, per this phase's scope.

None of this changes production model code, fusion weights, thresholds, or
`CNN_CLASS_ORDER`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

import pytest

from app.config.settings import Settings
from app.models.factory import ModelFactory
from app.schemas.common import BranchStatus, ModelMode
from tests.real_model_helpers import (
    _BACKEND_ROOT,
    _ENV_FILE,
    AASIST_CHECKPOINT,
    AASIST_CHECKPOINT_IDENTITY,
    AASIST_LIGHT_V1_CHECKPOINT,
    CNN_CHECKPOINT,
    CNN_CHECKPOINT_IDENTITY,
    deployment_settings,
    processed_audio,
    real_model_settings,
    requires_aasist_checkpoint,
    requires_cnn_checkpoint,
    requires_torch,
)

pytestmark = requires_torch

#: Pinned by the CNN verification phase audit against the actual deployed
#: file (`shasum -a 256 model_artifacts/aasist/aasist_light_v2_best.pt`).
EXPECTED_AASIST_V2_SHA256 = (
    "c944c69f05a0135ad9dfdaf86fb8a31815339f862d780c5f2c5d834f181e64ec"
)
EXPECTED_AASIST_V2_ARCHITECTURE = "aasist-light-v2-finalized-baseline"
EXPECTED_AASIST_V2_EPOCH = 13
EXPECTED_AASIST_V2_PARAMETER_COUNT = 641_795


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_env_value(env_text: str, key: str) -> str | None:
    """Deliberately dumb, regex-only .env line parser.

    Independent of `Settings`/`pydantic-settings` on purpose: this exists to
    cross-check the resolver under test, not to reuse it.
    """

    match = re.search(rf"^{re.escape(key)}=(.*)$", env_text, re.MULTILINE)
    if match is None:
        return None
    value = match.group(1).strip()
    if value.startswith(("'", '"')) and value.endswith(value[0]) and len(value) >= 2:
        value = value[1:-1]
    return value or None


# ---------------------------------------------------------------------------
# 1. Checkpoint resolution: test path resolution == production path resolution
# ---------------------------------------------------------------------------


def test_cnn_checkpoint_resolves_through_runtime_settings_not_a_guessed_path() -> None:
    """CNN_CHECKPOINT must come from the same Settings-driven resolver
    ModelFactory uses in production, not a hardcoded test-only assumption."""

    settings = deployment_settings()
    production_identity = ModelFactory(settings).branch_config("lfcc_cnn_tcn").checkpoint

    assert CNN_CHECKPOINT_IDENTITY.valid == production_identity.valid
    assert CNN_CHECKPOINT_IDENTITY.safe_path == production_identity.safe_path
    assert CNN_CHECKPOINT_IDENTITY.sha256 == production_identity.sha256


def test_aasist_checkpoint_resolves_through_runtime_settings_not_a_guessed_path() -> None:
    settings = deployment_settings()
    production_identity = ModelFactory(settings).branch_config("aasist").checkpoint

    assert AASIST_CHECKPOINT_IDENTITY.valid == production_identity.valid
    assert AASIST_CHECKPOINT_IDENTITY.safe_path == production_identity.safe_path
    assert AASIST_CHECKPOINT_IDENTITY.sha256 == production_identity.sha256


@requires_cnn_checkpoint
def test_cnn_checkpoint_path_matches_an_independent_raw_dotenv_parse() -> None:
    """Cross-checks resolution against a hand-rolled parse of `backend/.env`
    -- not against the same code path under test -- so a future regression to
    a hardcoded/guessed checkpoint path (the original B2 bug) fails loudly
    here instead of silently skipping every real-model test again."""

    if not _ENV_FILE.is_file():
        pytest.skip("backend/.env is not present on this machine.")
    text = _ENV_FILE.read_text(encoding="utf-8")
    configured = _parse_env_value(text, "CNN_MODEL_PATH")
    root = _parse_env_value(text, "MODEL_ROOT_DIR") or "../model_artifacts"
    if not configured:
        pytest.skip("CNN_MODEL_PATH is not set in backend/.env.")

    expected = (_BACKEND_ROOT / root / configured).resolve()
    assert CNN_CHECKPOINT_IDENTITY.safe_path == expected


@requires_aasist_checkpoint
def test_aasist_checkpoint_path_matches_an_independent_raw_dotenv_parse() -> None:
    if not _ENV_FILE.is_file():
        pytest.skip("backend/.env is not present on this machine.")
    text = _ENV_FILE.read_text(encoding="utf-8")
    configured = _parse_env_value(text, "AASIST_MODEL_PATH") or "aasist/aasist_light_v2_best.pt"
    root = _parse_env_value(text, "MODEL_ROOT_DIR") or "../model_artifacts"

    expected = (_BACKEND_ROOT / root / configured).resolve()
    assert AASIST_CHECKPOINT_IDENTITY.safe_path == expected


def test_the_real_model_guard_suite_is_not_silently_skipped_on_this_machine() -> None:
    """Guards against B2 regressing to "5 skipped, 0 executed".

    If both checkpoints resolve as valid here, every `@requires_cnn_checkpoint`
    / `@requires_aasist_checkpoint` gated test in this suite MUST execute, not
    skip. This does not re-derive the same resolver under test -- it is
    cross-checked independently by the two `..._matches_an_independent_raw_dotenv_parse`
    tests above; this test only states the consequence plainly so a reader
    scanning results immediately sees whether coverage is live.
    """

    if not (_BACKEND_ROOT.parent / "models").exists():
        pytest.skip("Local model artifact directories are not present on this machine.")
    assert CNN_CHECKPOINT_IDENTITY.valid, (
        "CNN checkpoint did not resolve -- real-model tests will (correctly) "
        f"skip. Reason: {CNN_CHECKPOINT_IDENTITY.error_code}"
    )
    assert AASIST_CHECKPOINT_IDENTITY.valid, (
        "AASIST checkpoint did not resolve -- real-model tests will "
        f"(correctly) skip. Reason: {AASIST_CHECKPOINT_IDENTITY.error_code}"
    )


# ---------------------------------------------------------------------------
# 2. Pinned checkpoint identity
# ---------------------------------------------------------------------------


@requires_aasist_checkpoint
def test_aasist_v2_checkpoint_identity_is_pinned() -> None:
    """A silently swapped AASIST checkpoint file must fail this test."""

    assert AASIST_CHECKPOINT is not None
    assert AASIST_CHECKPOINT.name == "aasist_light_v2_best.pt"
    assert AASIST_CHECKPOINT_IDENTITY.sha256 == EXPECTED_AASIST_V2_SHA256
    assert _sha256_of(AASIST_CHECKPOINT) == EXPECTED_AASIST_V2_SHA256

    settings = real_model_settings()
    model = ModelFactory(settings).create("aasist")
    model.load()
    runtime_model = model._runtime_model

    assert runtime_model.architecture_version == EXPECTED_AASIST_V2_ARCHITECTURE
    assert runtime_model.parameter_count == EXPECTED_AASIST_V2_PARAMETER_COUNT
    assert runtime_model.checkpoint_metadata.get("epoch") == EXPECTED_AASIST_V2_EPOCH
    assert runtime_model.checkpoint_metadata.get("class_mapping") == {
        "bonafide": 0,
        "spoof": 1,
    }
    # spoof_index=1 is currently sourced from AASIST_CLASS_ORDER, not directly
    # from checkpoint_metadata['class_mapping'] -- see the characterization
    # test below. It happens to agree here.
    assert runtime_model.spoof_index == 1
    assert runtime_model.bonafide_index == 0


@requires_aasist_checkpoint
def test_aasist_checkpoint_resolution_is_not_the_superseded_v1_file() -> None:
    """Directly proves AASIST_CHECKPOINT != the old V1 artifact.

    Regression guard for the exact bug this phase fixed: `real_model_helpers`
    previously pointed `AASIST_CHECKPOINT` at
    `model_artifacts/best_aasist_light_full_weighted.pth` (a V1-era filename
    that never existed at that location), while production had already moved
    to AASIST-Light V2.
    """

    assert AASIST_CHECKPOINT.name != "best_aasist_light_full_weighted.pth"
    assert AASIST_CHECKPOINT_IDENTITY.sha256 == EXPECTED_AASIST_V2_SHA256
    if AASIST_LIGHT_V1_CHECKPOINT.is_file():
        assert AASIST_CHECKPOINT_IDENTITY.sha256 != _sha256_of(AASIST_LIGHT_V1_CHECKPOINT)


#: Pinned by this phase's CNN-V2 integration against the actual deployed
#: file (`shasum -a 256 model_artifacts/cnn/cnn_v2_lfcc_delta_aug_asvspoof2019_inference_best.pt`).
EXPECTED_CNN_V2_SHA256 = (
    "2d8b7d2dc57ba657101e5e38f377f3e60b3f0b7395d3affdfeab3d7dd48dc1f2"
)
EXPECTED_CNN_V2_ARCHITECTURE = "cnn-v2-lfcc-delta-vgg-4block-v1"
EXPECTED_CNN_V2_PARAMETER_COUNT = 616_162


@requires_cnn_checkpoint
def test_cnn_v2_checkpoint_identity_is_recorded_through_the_provenance_mechanism() -> None:
    """CNN-V2, unlike the superseded checkpoint this test previously covered,
    IS self-describing: its own config/feature_pipeline/label_map are read
    and validated at load time (`load_cnn_v2_checkpoint_strict`), then
    recorded through the same `CheckpointIdentity`/provenance mechanism
    AASIST uses."""

    assert CNN_CHECKPOINT is not None
    assert CNN_CHECKPOINT_IDENTITY.sha256 == _sha256_of(CNN_CHECKPOINT)
    assert CNN_CHECKPOINT_IDENTITY.size_bytes == CNN_CHECKPOINT.stat().st_size
    assert CNN_CHECKPOINT_IDENTITY.sha256 == EXPECTED_CNN_V2_SHA256

    settings = real_model_settings()
    model = ModelFactory(settings).create("lfcc_cnn_tcn")
    model.load()
    runtime_model = model._runtime_model

    assert runtime_model.architecture_version == EXPECTED_CNN_V2_ARCHITECTURE
    assert runtime_model.parameter_count == EXPECTED_CNN_V2_PARAMETER_COUNT
    assert runtime_model.checkpoint_metadata.get("label_map") == ["bonafide", "spoof"]
    assert runtime_model.checkpoint_metadata.get("positive_class") == "spoof"
    assert runtime_model.checkpoint_metadata.get("class_mapping") == {
        "bonafide": 0,
        "spoof": 1,
    }
    assert runtime_model.spoof_index == 1
    assert runtime_model.bonafide_index == 0

    prediction = model.predict_safe(processed_audio())
    assert prediction.metadata["checkpoint_metadata"]["label_map"] == ["bonafide", "spoof"]


# ---------------------------------------------------------------------------
# 3. Repaired ProcessedAudio helper
# ---------------------------------------------------------------------------


def test_processed_audio_helper_includes_unnormalised_waveform() -> None:
    import numpy as np

    audio = processed_audio()

    assert audio.unnormalised_waveform is not None
    assert isinstance(audio.unnormalised_waveform, np.ndarray)
    assert audio.unnormalised_waveform.shape == audio.waveform.shape
    assert np.array_equal(audio.unnormalised_waveform, audio.waveform)


@requires_aasist_checkpoint
def test_real_aasist_v2_can_run_through_the_repaired_helper() -> None:
    """Regression test for B3: previously raised `model_input_invalid`
    ("aasist requires the unnormalised decoded waveform") because
    `processed_audio()` did not populate `unnormalised_waveform` at all."""

    settings = real_model_settings()
    model = ModelFactory(settings).create("aasist")

    prediction = model.predict_safe(processed_audio())

    assert prediction.status == BranchStatus.success
    assert prediction.mode == ModelMode.real
    assert prediction.metadata["architecture"] == EXPECTED_AASIST_V2_ARCHITECTURE
    assert prediction.probabilities is not None


# ---------------------------------------------------------------------------
# 4. Mechanical label-mapping contract tests (not evidentiary)
# ---------------------------------------------------------------------------


def test_configured_spoof_index_selects_the_matching_logit_column() -> None:
    """Contract test only: proves the configured class order picks the logit
    column it claims to, using controlled logits and a stub runtime -- no
    checkpoint required. Does NOT prove which column is actually spoof for
    any real checkpoint; that requires labelled evidence (see
    scripts/validate_cnn_class_order.py and the CNN Label Mapping Status
    section of the integration audit).
    """

    import torch

    from app.models.real.inference import LoadedTorchBranch, _spoof_probability_from_logits

    settings = real_model_settings()
    config = ModelFactory(settings).branch_config("lfcc_cnn_tcn")

    def runtime_for(spoof_index: int) -> LoadedTorchBranch:
        return LoadedTorchBranch(
            module=None,
            device="cpu",
            front_end=None,
            spoof_index=spoof_index,
            architecture_version="contract-test-stub",
            preprocessing={},
            verification={
                "preprocessing_verified": False,
                "class_mapping_verified": False,
                "verified": False,
            },
            class_order="bonafide_spoof" if spoof_index == 1 else "spoof_bonafide",
            parameter_count=0,
            load_time_ms=0.0,
            state_dict_keys=0,
        )

    high_index0_low_index1 = torch.tensor([[8.0, -8.0]])
    low_index0_high_index1 = torch.tensor([[-8.0, 8.0]])

    bonafide_spoof = runtime_for(spoof_index=1)
    spoof_bonafide = runtime_for(spoof_index=0)

    # Under bonafide=0/spoof=1: a high index-1 logit must read as very spoof.
    assert _spoof_probability_from_logits(
        torch, low_index0_high_index1, bonafide_spoof, config
    ) == pytest.approx(1.0, abs=1e-3)
    # Under spoof=0/bonafide=1: the SAME logits must read as very NOT spoof.
    assert _spoof_probability_from_logits(
        torch, low_index0_high_index1, spoof_bonafide, config
    ) == pytest.approx(0.0, abs=1e-3)

    # Mirror case.
    assert _spoof_probability_from_logits(
        torch, high_index0_low_index1, bonafide_spoof, config
    ) == pytest.approx(0.0, abs=1e-3)
    assert _spoof_probability_from_logits(
        torch, high_index0_low_index1, spoof_bonafide, config
    ) == pytest.approx(1.0, abs=1e-3)


@requires_cnn_checkpoint
def test_cnn_class_order_setting_mechanically_inverts_the_real_branchs_output() -> None:
    """Same contract, exercised through the real loaded CNN checkpoint this
    time: flipping CNN_CLASS_ORDER must invert every score. This restates the
    existing `test_class_order_selects_the_opposite_logit`
    (test_real_model_adapters.py) deliberately -- pinning the mechanical
    contract here keeps it next to the rest of this phase's verification
    tests."""

    forward = ModelFactory(
        real_model_settings(cnn_class_order="bonafide_spoof")
    ).create("lfcc_cnn_tcn")
    reversed_order = ModelFactory(
        real_model_settings(cnn_class_order="spoof_bonafide")
    ).create("lfcc_cnn_tcn")
    audio = processed_audio()

    forward_spoof = forward.predict_safe(audio).probabilities.spoof
    reversed_spoof = reversed_order.predict_safe(audio).probabilities.spoof

    assert forward_spoof == pytest.approx(1.0 - reversed_spoof, abs=1e-6)


# ---------------------------------------------------------------------------
# 5. Characterization test: AASIST spoof index is not cross-checked against
#    the checkpoint's own attestation (finding B5). Documented, not fixed.
# ---------------------------------------------------------------------------


@requires_aasist_checkpoint
def test_characterization_aasist_class_order_env_var_is_not_cross_checked_against_checkpoint() -> None:
    """CHARACTERIZATION TEST -- documents a known gap, does not fix it.

    `app/models/real/inference.py::_load_branch` sets
    ``spoof_index = 1 if class_order == "bonafide_spoof" else 0`` directly
    from the ``AASIST_CLASS_ORDER`` setting. It never compares that against
    the checkpoint's own attested ``spoof_class_index``/``class_mapping``,
    even though the loader already read and validated those fields moments
    earlier (`load_aasist_light_v2_checkpoint_strict`). Concretely: setting
    ``AASIST_CLASS_ORDER=spoof_bonafide`` in `backend/.env` would silently
    invert every AASIST prediction with no load error, even though the
    checkpoint itself says ``bonafide_class_index: 0, spoof_class_index: 1``.

    This test PASSING means that gap still exists (the misconfigured branch
    loads without error). If a future change adds the cross-check described
    in the integration audit's recommended fix (derive `spoof_index` from
    `checkpoint_metadata['class_mapping']`, raising
    `CheckpointCompatibilityError` on disagreement), this test's expectation
    inverts: it should then assert the settings override is REJECTED, and
    this docstring/test should be rewritten, not just deleted.

    Deliberately not fixed in this phase -- CNN Verification Foundation is
    explicitly scoped to test/diagnostic infrastructure, not AASIST
    production code changes.
    """

    misconfigured = real_model_settings(aasist_class_order="spoof_bonafide")
    model = ModelFactory(misconfigured).create("aasist")

    # No exception, no rejection: the loader accepts a class order that
    # contradicts the checkpoint's own attested class_mapping.
    model.load()
    runtime_model = model._runtime_model

    assert runtime_model.checkpoint_metadata.get("class_mapping") == {
        "bonafide": 0,
        "spoof": 1,
    }
    # The checkpoint says spoof=1; the misconfigured runtime now reads spoof=0.
    assert runtime_model.spoof_index == 0
    assert runtime_model.spoof_index != runtime_model.checkpoint_metadata.get(
        "class_mapping", {}
    ).get("spoof")
