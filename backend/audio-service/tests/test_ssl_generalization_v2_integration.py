"""Generalization V2 SSL checkpoint integration guard.

Complements ``test_ssl_sequence_integration.py`` (architecture/preprocessing/
artifact-structure) with checks specific to swapping in the new
trainable-state-only ``ssl_xlsr_mamba_generalization_v2_best.pt`` artifact:
checkpoint identity pinning, the self-describing/trainable-state-only
loader fallback, label-mapping mechanics, and the deterministic center-crop
policy. Hermetic tests run by default; the two that load the real ~1.2 GB
XLS-R backbone are marked ``integration`` + ``network``, matching the
existing e2e test in ``test_ssl_sequence_integration.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.exceptions import ModelLoadError
from app.models.factory import ModelFactory
from tests.real_model_helpers import (
    deployment_settings,
    processed_audio,
    real_model_settings,
    requires_ssl_checkpoint,
    requires_ssl_generalization_v2_checkpoint,
    requires_torch,
    requires_transformers,
    SSL_CHECKPOINT_IDENTITY,
    SSL_GENERALIZATION_V2_CHECKPOINT,
)

pytestmark = requires_torch

#: ssl_asvspoof5_best.pt -- the current default (see Settings.ssl_model_path).
EXPECTED_SSL_V2_SHA256 = (
    "b684111dc07643e1307f493352f4051664f1edb1b0e3f7f1db76f81f410b0c8e"
)
EXPECTED_SUPERSEDED_V1_FILENAME = "xlsr_mamba_asvspoof2019_best.pt"
#: The immediately-prior default, now superseded by ssl_asvspoof5_best.pt.
EXPECTED_SUPERSEDED_GENERALIZATION_V2_FILENAME = "ssl_xlsr_mamba_generalization_v2_best.pt"
EXPECTED_TRAINABLE_PARAMETER_COUNT = 1_173_634


# ---------------------------------------------------------------------------
# 1. Checkpoint resolution -- proves the deployed artifact is the new one,
#    resolved through the same Settings path production uses.
# ---------------------------------------------------------------------------


def test_ssl_checkpoint_path_resolves_through_runtime_settings_not_a_guessed_path() -> None:
    settings = deployment_settings()
    resolved = ModelFactory(settings).branch_config("ssl_sequence").checkpoint
    assert resolved.safe_path == SSL_CHECKPOINT_IDENTITY.safe_path


def test_default_ssl_model_path_points_at_asvspoof5_not_a_superseded_artifact() -> None:
    """Settings' own Python-level default (no .env involved) must select the
    finalized ASVspoof5 checkpoint's filename, matching AASIST_MODEL_PATH's
    existing "declare a real relative default" convention."""

    from app.config.settings import Settings

    settings = Settings(_env_file=None)
    assert settings.ssl_model_path.endswith("ssl_asvspoof5_best.pt")
    assert EXPECTED_SUPERSEDED_V1_FILENAME not in settings.ssl_model_path
    assert EXPECTED_SUPERSEDED_GENERALIZATION_V2_FILENAME not in settings.ssl_model_path


@requires_ssl_checkpoint
def test_ssl_checkpoint_identity_is_pinned() -> None:
    """Regression guard against a silent checkpoint swap: SHA-256 must match
    exactly what was verified during this integration."""

    assert SSL_CHECKPOINT_IDENTITY.sha256 == EXPECTED_SSL_V2_SHA256


@requires_ssl_checkpoint
def test_ssl_checkpoint_resolution_is_not_the_superseded_v1_file() -> None:
    assert SSL_CHECKPOINT_IDENTITY.safe_path is not None
    assert SSL_CHECKPOINT_IDENTITY.safe_path.name != EXPECTED_SUPERSEDED_V1_FILENAME


# ---------------------------------------------------------------------------
# 2. Loader fallback: a trainable-state-only artifact (no architecture /
#    xlsr_model_name / xlsr_hidden_size / mamba_dim / num_classes /
#    label_mapping keys) must load using the DECLARED_* constants, and a
#    self-describing artifact (the superseded convention) must still use its
#    own embedded values in preference to the declared defaults.
# ---------------------------------------------------------------------------


def test_loader_fills_in_declared_defaults_for_a_trainable_state_only_artifact(tmp_path) -> None:
    import torch

    from app.models.architectures.xlsr_mamba import build_xlsr_mamba_net
    from app.models.real.ssl_sequence_inference import _load_artifact_package

    class _FakeXlsr:
        def parameters(self):
            return iter(())

        def eval(self):
            return self

    module = build_xlsr_mamba_net(xlsr_backbone=_FakeXlsr())
    artifact = tmp_path / "trainable_state_only.pt"
    torch.save(
        {
            "epoch": 7,
            "stage": "unit_test_stage",
            "train_loss": 0.123,
            "trainable_model_state": module.state_dict(),
        },
        artifact,
    )

    settings = real_model_settings(
        ssl_model_mode="real",
        model_root_dir=str(tmp_path),
        ssl_model_path=artifact.name,
    )

    class _FakeConfig:
        branch_name = "ssl_sequence"

    package = _load_artifact_package(torch, artifact, _FakeConfig(), settings)
    assert package["architecture"] == "XLSRMambaClassifier"
    assert package["xlsr_model_name"] == settings.ssl_xlsr_model_name
    assert package["xlsr_hidden_size"] == 1024
    assert package["mamba_dim"] == 256
    assert package["num_classes"] == 2
    assert package["label_mapping"] == {"bonafide": 0, "spoof": 1}
    assert package["epoch"] == 7
    assert package["stage"] == "unit_test_stage"
    assert package["train_loss"] == pytest.approx(0.123)


def test_loader_prefers_embedded_values_over_declared_defaults_for_a_self_describing_artifact(
    tmp_path,
) -> None:
    """The superseded artifact format's own embedded fields must still win --
    the declared fallback must never silently override an explicit value."""

    import torch

    from app.models.real.ssl_sequence_inference import _load_artifact_package

    artifact = tmp_path / "self_describing.pt"
    torch.save(
        {
            "architecture": "XLSRMambaClassifier",
            "xlsr_model_name": "some/other-backbone",
            "xlsr_hidden_size": 999,
            "mamba_dim": 111,
            "num_classes": 2,
            "label_mapping": {"bonafide": 1, "spoof": 0},
            "trainable_model_state": {"x": torch.zeros(1)},
        },
        artifact,
    )

    settings = real_model_settings(
        ssl_model_mode="real",
        model_root_dir=str(tmp_path),
        ssl_model_path=artifact.name,
    )

    class _FakeConfig:
        branch_name = "ssl_sequence"

    package = _load_artifact_package(torch, artifact, _FakeConfig(), settings)
    assert package["xlsr_model_name"] == "some/other-backbone"
    assert package["xlsr_hidden_size"] == 999
    assert package["mamba_dim"] == 111
    assert package["label_mapping"] == {"bonafide": 1, "spoof": 0}


def test_loader_still_rejects_a_declared_architecture_mismatch_for_a_self_describing_artifact(
    tmp_path,
) -> None:
    import torch

    from app.models.real.ssl_sequence_inference import _load_artifact_package

    artifact = tmp_path / "wrong_arch.pt"
    torch.save(
        {
            "architecture": "SomeOtherModel",
            "trainable_model_state": {"x": torch.zeros(1)},
        },
        artifact,
    )
    settings = real_model_settings(
        ssl_model_mode="real", model_root_dir=str(tmp_path), ssl_model_path=artifact.name
    )

    class _FakeConfig:
        branch_name = "ssl_sequence"

    with pytest.raises(ModelLoadError) as excinfo:
        _load_artifact_package(torch, artifact, _FakeConfig(), settings)
    assert excinfo.value.error_code == "checkpoint_incompatible"


@requires_ssl_checkpoint
@requires_ssl_checkpoint
def test_real_asvspoof5_artifact_has_embedded_self_description() -> None:
    """Confirms the deployed ASVspoof5 artifact self-describes its backbone
    name, label map, sample rate, and max duration -- unlike the Generalization
    V2 artifact it superseded (see the companion test below), so the loader's
    embedded-values-win-over-declared-defaults path (not the fallback path) is
    what real deployments exercise today."""

    import torch

    from app.models.torch_support import numpy_scalar_safe_globals
    from tests.real_model_helpers import SSL_CHECKPOINT

    with torch.serialization.safe_globals(numpy_scalar_safe_globals()):
        package = torch.load(str(SSL_CHECKPOINT), map_location="cpu", weights_only=True)
    for key in ("xlsr_model_name", "label_mapping", "sample_rate", "max_duration_seconds"):
        assert key in package, (
            f"Deployed SSL artifact no longer embeds {key!r} -- update this "
            "test's expectations if that is intentional."
        )
    assert "trainable_model_state" in package
    assert package["xlsr_model_name"] == "facebook/wav2vec2-large-xlsr-53"
    assert package["label_mapping"] == {"bonafide": 0, "spoof": 1}


@requires_ssl_generalization_v2_checkpoint
def test_superseded_generalization_v2_artifact_has_no_embedded_self_description() -> None:
    """Regression documentation: the immediately-prior default (still on disk,
    never deleted) really is the trainable-state-only format the loader's
    DECLARED_* fallback exists for -- distinct from the current ASVspoof5
    default, which self-describes (see the test above)."""

    import torch

    from tests.real_model_helpers import SSL_GENERALIZATION_V2_CHECKPOINT

    package = torch.load(
        str(SSL_GENERALIZATION_V2_CHECKPOINT), map_location="cpu", weights_only=True
    )
    for key in ("architecture", "xlsr_model_name", "xlsr_hidden_size", "mamba_dim", "label_mapping"):
        assert key not in package
    assert "trainable_model_state" in package


# ---------------------------------------------------------------------------
# 3. Label-mapping mechanics (bonafide=0, spoof=1): a contract test only --
#    proves the configured mapping is applied as claimed using controlled
#    logits, not that it is correct for the real checkpoint.
# ---------------------------------------------------------------------------


def test_configured_label_mapping_selects_the_matching_logit_column_mechanically() -> None:
    import torch

    from app.models.real.ssl_sequence_inference import (
        LoadedSSLBranch,
        _spoof_probability_from_logits,
    )

    settings = real_model_settings()
    config = ModelFactory(settings).branch_config("ssl_sequence")

    def runtime_for(spoof_index: int) -> LoadedSSLBranch:
        return LoadedSSLBranch(
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
            trainable_parameter_count=0,
            load_time_ms=0.0,
            xlsr_model_name="stub",
            mamba_dim=256,
            label_mapping={"bonafide": 0, "spoof": 1},
        )

    high_index0_low_index1 = torch.tensor([[8.0, -8.0]])
    low_index0_high_index1 = torch.tensor([[-8.0, 8.0]])

    bonafide_spoof = runtime_for(spoof_index=1)  # index 0 = bonafide, index 1 = spoof
    spoof_bonafide = runtime_for(spoof_index=0)  # index 0 = spoof, index 1 = bonafide

    # bonafide_spoof convention: high logit at index 0 -> low spoof probability.
    assert _spoof_probability_from_logits(
        torch, high_index0_low_index1, bonafide_spoof, config
    ) < 0.01
    # ... high logit at index 1 -> high spoof probability.
    assert _spoof_probability_from_logits(
        torch, low_index0_high_index1, bonafide_spoof, config
    ) > 0.99
    # spoof_bonafide convention inverts both.
    assert _spoof_probability_from_logits(
        torch, high_index0_low_index1, spoof_bonafide, config
    ) > 0.99
    assert _spoof_probability_from_logits(
        torch, low_index0_high_index1, spoof_bonafide, config
    ) < 0.01


def test_ssl_class_order_setting_matches_the_declared_bonafide_zero_spoof_one_mapping() -> None:
    from app.config.settings import Settings

    settings = Settings(_env_file=None)
    assert settings.branch_class_order("ssl_sequence") == "bonafide_spoof"


# ---------------------------------------------------------------------------
# 4. Preprocessing: deterministic center crop for long audio (content check,
#    not just shape -- a trailing/leading crop would also pass a shape-only
#    assertion).
# ---------------------------------------------------------------------------


def test_center_crop_keeps_the_middle_of_long_audio_not_the_start_or_end() -> None:
    """A single spike at the source signal's exact midpoint uniquely pins
    down which window was kept, without the affine-invariance ambiguity a
    plain ramp would have under zero-mean/peak normalisation (any window of
    a straight line normalises to ~the same shape, which would pass a
    leading-crop OR a center-crop assertion)."""

    import torch

    from app.models.preprocessing.ssl_waveform import SSLWaveformConfig, SSLWaveformFrontEnd

    config = SSLWaveformConfig(sample_rate=16000, target_samples=96000)
    front_end = SSLWaveformFrontEnd(config)

    total_samples = 160_000  # 10s, longer than the 6s window
    spike_index = total_samples // 2  # 80000
    waveform = torch.zeros(total_samples, dtype=torch.float32)
    waveform[spike_index] = 1.0

    expected_start = (total_samples - config.target_samples) // 2  # 32000
    expected_spike_position_in_output = spike_index - expected_start  # 48000, i.e. dead center

    input_values, attention_mask = front_end.prepare(waveform)

    assert input_values.shape == (1, config.target_samples)
    assert int(attention_mask.sum()) == config.target_samples  # fully real, nothing padded

    observed_spike_position = int(torch.argmax(input_values[0]).item())
    assert observed_spike_position == expected_spike_position_in_output

    # A leading crop (old behaviour, keeping samples [0:96000)) would have
    # placed the spike at its original index (80000), not the output's
    # center (48000) -- explicitly confirm that is NOT what happened.
    assert observed_spike_position != spike_index


# ---------------------------------------------------------------------------
# 5. Real end-to-end: checkpoint metadata (epoch/stage/train_loss) surfaces
#    in prediction metadata; HF backbone-unavailable degrades to a load
#    error rather than a silent dummy fallback. Both need the real ~1.2 GB
#    backbone, so both are integration+network like the existing e2e test.
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.network
@requires_ssl_checkpoint
@requires_transformers
def test_checkpoint_metadata_and_threshold_are_exposed_in_prediction_metadata() -> None:
    settings = real_model_settings(ssl_model_mode="real")
    model = ModelFactory(settings).create("ssl_sequence")

    waveform = np.random.RandomState(1).randn(48000).astype(np.float32) * 0.05
    prediction = model.predict(processed_audio(waveform))

    assert prediction.metadata["checkpoint_epoch"] == 2
    assert prediction.metadata["checkpoint_stage"] == "Generalization_V2"
    assert prediction.metadata["checkpoint_train_loss"] == pytest.approx(0.2633038581501354)
    assert prediction.metadata["threshold"] == 0.5
    assert prediction.metadata["label_mapping"] == {"bonafide": 0, "spoof": 1}


@pytest.mark.integration
@pytest.mark.network
def test_unavailable_hf_backbone_name_fails_loudly_rather_than_falling_back_silently() -> None:
    """SSL must fail closed (a real ModelLoadError) rather than silently
    substituting a dummy prediction when the configured backbone cannot be
    fetched -- e.g. a typo'd SSL_XLSR_MODEL_NAME."""

    settings = real_model_settings(
        ssl_model_mode="real",
        ssl_xlsr_model_name="facebook/this-repo-does-not-exist-multiscope-guard",
    )
    model = ModelFactory(settings).create("ssl_sequence")

    with pytest.raises(ModelLoadError) as excinfo:
        model.load()
    assert excinfo.value.error_code == "checkpoint_missing"
