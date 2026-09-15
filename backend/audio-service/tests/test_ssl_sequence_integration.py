"""XLS-R + Mamba SSL branch (``ssl_sequence``) integration tests.

Split by cost/network dependency, mirroring the CNN/AASIST test philosophy in
``real_model_helpers.py``:

* architecture/preprocessing/artifact-structure tests are hermetic (no
  network, no real XLS-R backbone) and run by default;
* the one true end-to-end inference test needs the real ~1.2GB Hugging Face
  backbone and is marked ``integration`` + ``network`` (excluded from the
  default ``pytest`` run, same as the existing Cloudinary/MongoDB tests).
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config.settings import Settings
from app.core.exceptions import ModelLoadError
from app.models.factory import ModelFactory
from app.schemas.common import BranchStatus, ModelMode, PredictionLabel
from app.schemas.prediction import BranchPrediction, ProbabilityScores
from app.utils.fusion import FusionEngine
from tests.real_model_helpers import (
    processed_audio,
    real_model_settings,
    requires_ssl_checkpoint,
    requires_torch,
    requires_transformers,
)

pytestmark = requires_torch


# --------------------------------------------------------------------------
# Architecture reconstruction (no checkpoint, no network -- a fake backbone
# stub is enough to validate the trainable-head module graph).
# --------------------------------------------------------------------------


class _FakeXlsrBackbone:
    """Stands in for the frozen Hugging Face backbone in shape-only tests."""

    def parameters(self):
        return iter(())

    def eval(self):
        return self


def test_mamba_dt_rank_matches_mamba_ssm_auto_formula() -> None:
    from app.models.architectures.xlsr_mamba import mamba_dt_rank

    # mamba_ssm's dt_rank="auto" default: ceil(d_model / 16).
    assert mamba_dt_rank(256) == 16
    assert mamba_dt_rank(16) == 1
    assert mamba_dt_rank(17) == 2


@requires_ssl_checkpoint
def test_reconstructed_architecture_state_dict_matches_real_artifact_exactly() -> None:
    """The single most important structural check: every key/shape in the
    real deployment artifact's trainable_model_state must map onto this
    reconstruction with strict compatibility. No network, no real backbone --
    just confirms the reverse-engineered module graph is correct.

    The deployed artifact may be either the self-describing legacy format or
    the trainable-state-only Generalization V2 format (see
    ``app/models/real/ssl_sequence_inference.py``'s DECLARED_* constants) --
    this test falls back to the same declared values the real loader uses so
    it exercises whichever artifact is actually configured via
    ``SSL_MODEL_PATH`` rather than assuming the legacy shape.
    """

    import torch

    from app.models.architectures.xlsr_mamba import build_xlsr_mamba_net
    from app.models.real.ssl_sequence_inference import (
        DECLARED_MAMBA_DIM,
        DECLARED_NUM_CLASSES,
        DECLARED_XLSR_HIDDEN_SIZE,
    )
    from app.models.torch_support import numpy_scalar_safe_globals
    from tests.real_model_helpers import SSL_CHECKPOINT

    with torch.serialization.safe_globals(numpy_scalar_safe_globals()):
        package = torch.load(str(SSL_CHECKPOINT), map_location="cpu", weights_only=True)
    trainable_state = package["trainable_model_state"]

    module = build_xlsr_mamba_net(
        xlsr_backbone=_FakeXlsrBackbone(),
        xlsr_hidden_size=package.get("xlsr_hidden_size", DECLARED_XLSR_HIDDEN_SIZE),
        mamba_dim=package.get("mamba_dim", DECLARED_MAMBA_DIM),
        num_classes=package.get("num_classes", DECLARED_NUM_CLASSES),
    )
    own_state = module.state_dict()

    missing_in_module = [key for key in trainable_state if key not in own_state]
    assert missing_in_module == []

    shape_mismatches = [
        key
        for key in trainable_state
        if tuple(own_state[key].shape) != tuple(trainable_state[key].shape)
    ]
    assert shape_mismatches == []

    load_result = module.load_state_dict(trainable_state, strict=False)
    assert load_result.unexpected_keys == []
    # Everything left "missing" by the partial load must be a frozen backbone
    # parameter (xlsr.*) -- never part of the trainable head.
    assert all(key.startswith("xlsr.") for key in load_result.missing_keys)


def test_ssl_backbone_import_error_reports_broken_scientific_runtime(monkeypatch) -> None:
    """A broken SciPy/scikit-learn import should not be misreported as
    'transformers is not installed'. It is usually the real root cause of the
    SSL loader failing at import time on Windows deployments."""

    import builtins
    from types import SimpleNamespace

    import app.models.real.ssl_sequence_inference as ssl_sequence_inference

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("DLL load failed while importing _ufuncs: The specified module could not be found.")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModelLoadError, match="SciPy|scikit-learn|required scientific dependency"):
        ssl_sequence_inference._load_frozen_xlsr_backbone(
            "facebook/wav2vec2-large-xlsr-53",
            SimpleNamespace(branch_name="ssl_sequence"),
        )


def test_forward_pass_shape_with_a_fake_backbone() -> None:
    """A fake backbone (correct output shape, zero compute) exercises the
    full projection -> mamba1 -> mamba2 -> norm -> pool -> classifier graph
    without downloading anything."""

    import torch

    from app.models.architectures.xlsr_mamba import build_xlsr_mamba_net

    hidden_size, mamba_dim, num_classes, seq_len = 1024, 256, 2, 12

    class FakeBackboneWithOutput(_FakeXlsrBackbone):
        def __call__(self, input_values, attention_mask=None):
            batch = input_values.shape[0]
            return type(
                "Output", (), {"last_hidden_state": torch.randn(batch, seq_len, hidden_size)}
            )()

        def _get_feature_vector_attention_mask(self, feature_length, attention_mask):
            return torch.ones(attention_mask.shape[0], feature_length, dtype=torch.long)

    module = build_xlsr_mamba_net(
        xlsr_backbone=FakeBackboneWithOutput(),
        xlsr_hidden_size=hidden_size,
        mamba_dim=mamba_dim,
        num_classes=num_classes,
    )
    module.eval()

    input_values = torch.randn(1, 96000)
    attention_mask = torch.ones(1, 96000, dtype=torch.long)
    with torch.inference_mode():
        logits = module(input_values, attention_mask)

    assert logits.shape == (1, num_classes)
    assert bool(torch.isfinite(logits).all())


# --------------------------------------------------------------------------
# Preprocessing parity (16kHz mono, 6s/96000-sample window).
# --------------------------------------------------------------------------


def test_ssl_waveform_front_end_pads_short_audio_and_marks_valid_length() -> None:
    import torch

    from app.models.preprocessing.ssl_waveform import SSLWaveformConfig, SSLWaveformFrontEnd

    front_end = SSLWaveformFrontEnd(SSLWaveformConfig(sample_rate=16000, target_samples=96000))
    waveform = torch.ones(48000, dtype=torch.float32) * 0.1  # 3s, half the target window

    input_values, attention_mask = front_end.prepare(waveform)

    assert input_values.shape == (1, 96000)
    assert attention_mask.shape == (1, 96000)
    assert int(attention_mask.sum()) == 48000
    assert bool((attention_mask[0, 48000:] == 0).all())
    # Padding stays exactly zero -- normalisation must not touch it.
    assert bool((input_values[0, 48000:] == 0).all())


def test_ssl_waveform_front_end_crops_long_audio_to_exactly_the_target_window() -> None:
    import torch

    from app.models.preprocessing.ssl_waveform import SSLWaveformConfig, SSLWaveformFrontEnd

    front_end = SSLWaveformFrontEnd(SSLWaveformConfig(sample_rate=16000, target_samples=96000))
    waveform = torch.arange(160000, dtype=torch.float32)  # 10s, longer than the 6s window

    input_values, attention_mask = front_end.prepare(waveform)

    assert input_values.shape == (1, 96000)
    assert int(attention_mask.sum()) == 96000  # fully "real" -- nothing padded


def test_ssl_waveform_front_end_normalizes_zero_mean_peak_over_real_samples_only() -> None:
    import torch

    from app.models.preprocessing.ssl_waveform import SSLWaveformConfig, SSLWaveformFrontEnd

    front_end = SSLWaveformFrontEnd(SSLWaveformConfig(sample_rate=16000, target_samples=96000))
    torch.manual_seed(0)
    waveform = torch.randn(96000, dtype=torch.float32) * 5.0 + 3.0  # exact-length, non-trivial stats

    input_values, _ = front_end.prepare(waveform)

    assert float(input_values.mean()) == pytest.approx(0.0, abs=1e-4)
    assert float(input_values.abs().max()) == pytest.approx(1.0, abs=1e-5)


def test_ssl_waveform_front_end_handles_empty_waveform_without_crashing() -> None:
    import torch

    from app.models.preprocessing.ssl_waveform import SSLWaveformConfig, SSLWaveformFrontEnd

    front_end = SSLWaveformFrontEnd(SSLWaveformConfig(sample_rate=16000, target_samples=96000))
    input_values, attention_mask = front_end.prepare(torch.zeros(0, dtype=torch.float32))

    assert input_values.shape == (1, 96000)
    assert int(attention_mask.sum()) == 0


def test_original_forward_attention_is_reduced_to_a_compact_timeline() -> None:
    import torch

    from app.models.real.ssl_sequence_inference import (
        _reduce_original_forward_attention,
    )

    attention = torch.full((1, 2, 3, 3), 1.0 / 3.0)
    evidence = _reduce_original_forward_attention(
        torch,
        attentions=(attention,),
        frame_mask=torch.tensor([[True, True, True]]),
        attention_mask=torch.ones((1, 16_000), dtype=torch.long),
        sample_rate=16_000,
        runtime_model=None,
        spoof_probability=0.8,
    )

    assert len(evidence) == 1
    window = evidence[0]
    assert window.start_seconds == 0.0
    assert window.end_seconds == pytest.approx(1.0)
    assert window.token_times_seconds.shape == (3,)
    assert window.attention_density.shape == (3,)
    assert np.isfinite(window.attention_density).all()
    assert float(window.attention_density.mean()) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Artifact / loader failure isolation (hermetic -- constructs bad artifacts
# in-memory rather than touching the real checkpoint file).
# --------------------------------------------------------------------------


def test_loader_rejects_artifact_missing_required_keys(tmp_path, monkeypatch) -> None:
    import torch

    from app.models.factory import ModelFactory

    bad_artifact = tmp_path / "bad_ssl.pt"
    torch.save({"architecture": "XLSRMambaClassifier"}, bad_artifact)  # missing everything else

    settings = real_model_settings(
        ssl_model_mode="real",
        model_root_dir=str(tmp_path),
        ssl_model_path=bad_artifact.name,
    )
    model = ModelFactory(settings).create("ssl_sequence")

    with pytest.raises(ModelLoadError) as excinfo:
        model.load()
    assert excinfo.value.error_code == "checkpoint_incompatible"


def test_loader_rejects_artifact_with_wrong_architecture_name(tmp_path) -> None:
    import torch

    wrong_artifact = tmp_path / "wrong_arch.pt"
    torch.save(
        {
            "architecture": "SomeOtherModel",
            "xlsr_model_name": "facebook/wav2vec2-large-xlsr-53",
            "xlsr_hidden_size": 1024,
            "mamba_dim": 256,
            "num_classes": 2,
            "label_mapping": {"bonafide": 0, "spoof": 1},
            "trainable_model_state": {"x": torch.zeros(1)},
        },
        wrong_artifact,
    )

    settings = real_model_settings(
        ssl_model_mode="real",
        model_root_dir=str(tmp_path),
        ssl_model_path=wrong_artifact.name,
    )
    model = ModelFactory(settings).create("ssl_sequence")

    with pytest.raises(ModelLoadError) as excinfo:
        model.load()
    assert excinfo.value.error_code == "checkpoint_incompatible"


def test_loader_reports_checkpoint_missing_when_ssl_model_path_is_unset() -> None:
    settings = real_model_settings(ssl_model_mode="real", ssl_model_path="")
    model = ModelFactory(settings).create("ssl_sequence")

    with pytest.raises(ModelLoadError) as excinfo:
        model.load()
    assert excinfo.value.error_code == "checkpoint_missing"


def test_disabled_ssl_branch_is_unaffected_by_the_real_adapter_wiring() -> None:
    """The current default (.env ships ssl_model_mode=real, but a deployment
    without the extra dependencies/artifact must still degrade safely)."""

    settings = real_model_settings(ssl_model_mode="disabled")
    model = ModelFactory(settings).create("ssl_sequence")
    prediction = model.predict_safe(processed_audio())

    assert prediction.status == BranchStatus.skipped
    assert prediction.mode == ModelMode.disabled


# --------------------------------------------------------------------------
# Fusion contract: SSL contributes like any other branch, and a failed SSL
# branch must not take the rest of fusion down with it.
# --------------------------------------------------------------------------


def _successful_branch(model_name: str, spoof: float) -> BranchPrediction:
    return BranchPrediction(
        model_name=model_name,
        display_name=model_name,
        status=BranchStatus.success,
        mode=ModelMode.real,
        prediction=PredictionLabel.spoof if spoof >= 0.5 else PredictionLabel.bonafide,
        confidence=max(spoof, 1 - spoof),
        probabilities=ProbabilityScores(bonafide=1 - spoof, spoof=spoof),
        processing_time_ms=5.0,
        metadata={"research_result": False},
    )


def _failed_branch(model_name: str) -> BranchPrediction:
    return BranchPrediction(
        model_name=model_name,
        display_name=model_name,
        status=BranchStatus.failed,
        mode=ModelMode.real,
        error="model_inference_failed",
        processing_time_ms=0.0,
        metadata={},
    )


def test_fusion_includes_a_successful_ssl_branch_alongside_cnn_and_aasist() -> None:
    settings = real_model_settings()  # fusion_weight_ssl_sequence defaults to 0.25
    engine = FusionEngine.from_settings(settings)

    branches = [
        _successful_branch("cnn_acoustic", 0.7),
        _successful_branch("aasist", 0.6),
        _successful_branch("ssl_wavlm_xlsr", 0.8),
    ]
    result = engine.fuse(branches)

    assert result.status == BranchStatus.success
    assert "ssl_wavlm_xlsr" in result.contributing_branches
    assert "ssl_sequence" in result.branch_weights or "ssl_wavlm_xlsr" in result.branch_weights


def test_fusion_survives_a_failed_ssl_branch_using_only_cnn_and_aasist() -> None:
    """Adding SSL must never make the whole prediction fail when SSL alone fails."""

    settings = real_model_settings()
    engine = FusionEngine.from_settings(settings)

    branches = [
        _successful_branch("cnn_acoustic", 0.3),
        _successful_branch("aasist", 0.4),
        _failed_branch("ssl_wavlm_xlsr"),
    ]
    result = engine.fuse(branches)

    assert result.status == BranchStatus.success
    assert "ssl_wavlm_xlsr" not in result.contributing_branches
    assert result.excluded_branches.get("ssl_wavlm_xlsr") is not None


# --------------------------------------------------------------------------
# Readiness / configuration wiring.
# --------------------------------------------------------------------------


def test_settings_expose_ssl_branch_verification_and_class_order() -> None:
    settings = Settings(
        _env_file=None,
        ssl_preprocessing_verified=True,
        ssl_class_mapping_verified=False,
    )
    verification = settings.branch_verification("ssl_sequence")
    assert verification == {
        "preprocessing_verified": True,
        "class_mapping_verified": False,
        "verified": False,
    }
    assert settings.branch_class_order("ssl_sequence") == "bonafide_spoof"


def test_real_ssl_mode_in_production_requires_verification() -> None:
    with pytest.raises(Exception):
        Settings(
            _env_file=None,
            app_env="production",
            ssl_model_mode="real",
            ssl_preprocessing_verified=False,
            ssl_class_mapping_verified=False,
            mongodb_uri="mongodb://example.invalid/test",
            mongodb_database="test",
            clerk_issuer="https://example.invalid",
            clerk_jwks_url="https://example.invalid/.well-known/jwks.json",
            clerk_audience="aud",
            clerk_authorized_parties="party",
            cloudinary_cloud_name="x",
            cloudinary_api_key="x",
            cloudinary_api_secret="x",
            api_key_hash_secret="a" * 32,
        )


# --------------------------------------------------------------------------
# True end-to-end inference: real artifact + real downloaded XLS-R backbone.
# Slow (backbone download + CPU selective-scan) and network-dependent, so it
# is excluded from the default test run, same as the Mongo/Cloudinary tests.
# --------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.network
@requires_ssl_checkpoint
@requires_transformers
def test_real_end_to_end_inference_produces_valid_probabilities() -> None:
    settings = real_model_settings(ssl_model_mode="real")
    model = ModelFactory(settings).create("ssl_sequence")

    waveform = np.random.RandomState(0).randn(48000).astype(np.float32) * 0.05
    prediction = model.predict(processed_audio(waveform))

    assert prediction.status == BranchStatus.success
    assert prediction.mode == ModelMode.real
    assert prediction.prediction in (PredictionLabel.bonafide, PredictionLabel.spoof)
    assert 0.0 <= prediction.probabilities.bonafide <= 1.0
    assert 0.0 <= prediction.probabilities.spoof <= 1.0
    assert prediction.probabilities.bonafide + prediction.probabilities.spoof == pytest.approx(
        1.0, abs=1e-3
    )
    # Honest by construction: the pure-PyTorch scan has not been re-verified
    # against the documented evaluation on this runtime yet.
    assert prediction.metadata["research_result"] is False
    assert "warning" in prediction.metadata
