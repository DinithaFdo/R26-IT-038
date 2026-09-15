from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.config.settings import Settings
from app.ingestion.audio import ProcessedAudio
from app.models.factory import ModelFactory
from app.models.runtime import checkpoint_identity_for_path
from app.schemas.common import BranchStatus
from tests.real_model_helpers import CHECKPOINT_ROOT, requires_torch

pytestmark = requires_torch

AASIST_V2_CHECKPOINT = CHECKPOINT_ROOT / "aasist" / "aasist_light_v2_best.pt"
AASIST_V2_SUMMARY = CHECKPOINT_ROOT / "aasist" / "aasist_light_v2_final_summary.json"


requires_aasist_v2_checkpoint = pytest.mark.skipif(
    not AASIST_V2_CHECKPOINT.is_file(),
    reason=f"AASIST-Light V2 checkpoint not present at {AASIST_V2_CHECKPOINT}.",
)


def _aasist_v2_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "model_root_dir": str(CHECKPOINT_ROOT),
        "aasist_model_mode": "real",
        "cnn_model_mode": "disabled",
        "ssl_model_mode": "disabled",
        "glottal_model_mode": "disabled",
        "required_model_branches": "aasist",
        "fusion_min_successful_branches": 1,
        "model_device_policy": "cpu",
        "model_default_device": "cpu",
        "model_load_strategy": "lazy",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _valid_processed_audio() -> ProcessedAudio:
    rng = np.random.RandomState(13)
    raw = rng.normal(0.0, 0.05, 70_000).astype(np.float32)
    return ProcessedAudio(
        waveform=raw * 0.5,
        unnormalised_waveform=raw,
        sample_rate=16_000,
        original_sample_rate=16_000,
        original_channels=1,
        duration_seconds=raw.size / 16_000,
        was_resampled=False,
        was_converted_to_mono=False,
        normalisation_applied=True,
        peak_amplitude=float(np.abs(raw * 0.5).max()),
        rms_energy=float(np.sqrt(np.mean((raw * 0.5) ** 2))),
    )


@requires_aasist_v2_checkpoint
def test_canonical_aasist_real_branch_loads_aasist_light_v2() -> None:
    from app.models.preprocessing.aasist_light_v2 import AasistLightV2WaveformFrontEnd

    settings = _aasist_v2_settings()
    model = ModelFactory(settings).create("aasist")

    model.load()
    runtime_model = model._runtime_model

    assert model.branch_name == "aasist"
    assert model.model_name == "aasist"
    assert runtime_model.architecture_version == "aasist-light-v2-finalized-baseline"
    assert isinstance(runtime_model.front_end, AasistLightV2WaveformFrontEnd)
    assert runtime_model.preprocessing["target_samples"] == 64_600
    assert runtime_model.preprocessing["normalization"] == "per_waveform_zscore"
    assert runtime_model.parameter_count == 641_795
    assert runtime_model.state_dict_keys == 45
    assert runtime_model.spoof_index == 1
    assert runtime_model.bonafide_index == 0


@requires_aasist_v2_checkpoint
def test_aasist_v2_checkpoint_resolves_inside_model_root_with_sha256() -> None:
    identity = checkpoint_identity_for_path(
        "aasist/aasist_light_v2_best.pt",
        model_root_dir=CHECKPOINT_ROOT,
    )

    assert identity.valid is True
    assert identity.safe_path == AASIST_V2_CHECKPOINT.resolve()
    assert identity.filename == "aasist_light_v2_best.pt"
    assert identity.extension == ".pt"
    assert identity.sha256 is not None
    assert identity.sha256_short == identity.sha256[:12]


def test_aasist_v2_missing_checkpoint_is_a_controlled_branch_failure(tmp_path: Path) -> None:
    model = ModelFactory(
        _aasist_v2_settings(
            model_root_dir=str(tmp_path),
            aasist_model_path="aasist/missing.pt",
        )
    ).create("aasist")

    prediction = model.predict_safe(_valid_processed_audio())

    assert prediction.status == BranchStatus.failed
    assert prediction.mode.value == "real"
    assert prediction.probabilities is None
    assert prediction.metadata["error_code"] == "checkpoint_not_found"


def test_aasist_v2_invalid_checkpoint_payload_fails_strictly(tmp_path: Path) -> None:
    import torch

    from app.models.architectures.aasist_light_v2 import build_aasist_light_v2_net
    from app.models.real.inference import load_aasist_light_v2_checkpoint_strict
    from app.models.torch_support import CheckpointCompatibilityError

    checkpoint = tmp_path / "invalid.pt"
    torch.save({"epoch": 13, "class_mapping": {"bonafide": 0, "spoof": 1}}, checkpoint)

    with pytest.raises(CheckpointCompatibilityError):
        load_aasist_light_v2_checkpoint_strict(
            build_aasist_light_v2_net(),
            checkpoint,
        )


def test_aasist_v2_missing_state_dict_keys_fail_strictly(tmp_path: Path) -> None:
    import torch

    from app.models.architectures.aasist_light_v2 import build_aasist_light_v2_net
    from app.models.real.inference import load_aasist_light_v2_checkpoint_strict
    from app.models.torch_support import CheckpointCompatibilityError

    module = build_aasist_light_v2_net()
    state_dict = module.state_dict()
    state_dict.pop("classifier.3.bias")
    checkpoint = tmp_path / "missing.pt"
    torch.save(_checkpoint_payload(state_dict), checkpoint)

    with pytest.raises(CheckpointCompatibilityError):
        load_aasist_light_v2_checkpoint_strict(build_aasist_light_v2_net(), checkpoint)


def test_aasist_v2_unexpected_state_dict_keys_fail_strictly(tmp_path: Path) -> None:
    import torch

    from app.models.architectures.aasist_light_v2 import build_aasist_light_v2_net
    from app.models.real.inference import load_aasist_light_v2_checkpoint_strict
    from app.models.torch_support import CheckpointCompatibilityError

    module = build_aasist_light_v2_net()
    state_dict = module.state_dict()
    state_dict["unexpected.weight"] = torch.ones(1)
    checkpoint = tmp_path / "unexpected.pt"
    torch.save(_checkpoint_payload(state_dict), checkpoint)

    with pytest.raises(CheckpointCompatibilityError):
        load_aasist_light_v2_checkpoint_strict(build_aasist_light_v2_net(), checkpoint)


def test_aasist_v2_reversed_class_mapping_is_rejected(tmp_path: Path) -> None:
    import torch

    from app.models.architectures.aasist_light_v2 import build_aasist_light_v2_net
    from app.models.real.inference import load_aasist_light_v2_checkpoint_strict
    from app.models.torch_support import CheckpointCompatibilityError

    checkpoint = tmp_path / "reversed.pt"
    torch.save(
        _checkpoint_payload(
            build_aasist_light_v2_net().state_dict(),
            class_mapping={"spoof": 0, "bonafide": 1},
        ),
        checkpoint,
    )

    with pytest.raises(CheckpointCompatibilityError):
        load_aasist_light_v2_checkpoint_strict(build_aasist_light_v2_net(), checkpoint)


@requires_aasist_v2_checkpoint
def test_aasist_v2_model_is_eval_on_selected_device_and_reused() -> None:
    model = ModelFactory(_aasist_v2_settings()).create("aasist")

    first = model.predict_safe(_valid_processed_audio())
    first_runtime = model._runtime_model
    second = model.predict_safe(_valid_processed_audio())

    assert first.status == BranchStatus.success
    assert second.status == BranchStatus.success
    assert model._runtime_model is first_runtime
    assert first_runtime.device == "cpu"
    assert next(first_runtime.module.parameters()).device.type == "cpu"
    assert first_runtime.module.training is False


@requires_aasist_v2_checkpoint
def test_aasist_v2_optional_logits_smoke_test() -> None:
    import torch

    model = ModelFactory(_aasist_v2_settings()).create("aasist")
    model.load()
    runtime_model = model._runtime_model
    prepared = runtime_model.front_end.prepare(torch.randn(64_600))

    with torch.inference_mode():
        logits = runtime_model.module(prepared.to(runtime_model.device))
    probabilities = torch.softmax(logits.to(dtype=torch.float32), dim=-1)

    assert tuple(logits.shape) == (1, 2)
    assert bool(torch.isfinite(probabilities).all())
    assert float(probabilities.sum()) == pytest.approx(1.0, abs=1e-6)


@requires_aasist_v2_checkpoint
def test_aasist_v2_model_health_reports_ready_after_load() -> None:
    model = ModelFactory(_aasist_v2_settings()).create("aasist")

    before = model.health()
    model.load()
    after = model.health()

    assert before["branch_name"] == "aasist"
    assert before["checkpoint_valid"] is True
    assert before["ready"] is False
    assert after["ready"] is True
    assert after["is_loaded"] is True
    assert after["architecture"] == "aasist-light-v2-finalized-baseline"
    assert after["checkpoint_hash_short"]


def test_phase2_fusion_configuration_defaults_are_unchanged() -> None:
    settings = Settings(_env_file=None)

    assert settings.fusion_method == "weighted_average"
    assert settings.fusion_decision_threshold == 0.5
    assert settings.fusion_weight_lfcc_cnn_tcn == 0.25
    assert settings.fusion_weight_aasist == 0.25
    assert settings.fusion_weight_ssl_sequence == 0.25
    assert settings.fusion_weight_glottal == 0.25


def test_phase2_non_aasist_branch_configuration_is_unchanged() -> None:
    settings = Settings(_env_file=None)
    factory = ModelFactory(settings)

    assert factory.branch_config("lfcc_cnn_tcn").model_name == "cnn_acoustic"
    assert factory.branch_config("ssl_sequence").model_name == "ssl_wavlm_xlsr"
    assert factory.branch_config("glottal").model_name == "glottal_features"


@requires_aasist_v2_checkpoint
def test_aasist_v2_final_summary_metadata_is_loaded_internally() -> None:
    model = ModelFactory(_aasist_v2_settings()).create("aasist")

    model.load()
    summary = model._runtime_model.summary_metadata

    assert AASIST_V2_SUMMARY.is_file()
    assert summary["summary_available"] is True
    assert summary["model_name"] == "AASIST-Light V2"
    assert summary["status"] == "FINALIZED_BASELINE"
    assert summary["best_epoch"] == 13
    assert summary["label_mapping"] == {"bonafide": 0, "spoof": 1}
    assert "Cross-domain EER remains high on ASVspoof2021 DF" in summary[
        "known_weaknesses"
    ]


def _checkpoint_payload(
    state_dict: dict,
    *,
    class_mapping: dict[str, int] | None = None,
) -> dict:
    return {
        "model_state_dict": state_dict,
        "epoch": 13,
        "best_eer": 0.039245587694380565,
        "class_mapping": class_mapping or {"bonafide": 0, "spoof": 1},
        "eer_threshold": 0.5,
    }
