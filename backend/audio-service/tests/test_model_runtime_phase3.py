from __future__ import annotations

from collections import Counter
from pathlib import Path
import threading
import time

import numpy as np
import pytest

from app.config.settings import Settings
from app.ingestion.audio import AudioUploadMetadata, ProcessedAudio
from app.models.factory import ModelFactory
from app.models.real.base import real_prediction_from_spoof_probability
from app.models.registry import ModelRegistry
from app.models.runtime import (
    StaticDeviceCapabilityProvider,
    checkpoint_identity_for_path,
)
from app.schemas.common import BranchStatus
from app.services.voice_service import VoiceService


def test_checkpoint_validation_hashes_safe_relative_file_without_path_leak(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "lfcc.pt"
    checkpoint.write_bytes(b"fake checkpoint bytes")

    identity = checkpoint_identity_for_path(
        "lfcc.pt",
        model_root_dir=tmp_path,
    )

    assert identity.valid is True
    assert identity.filename == "lfcc.pt"
    assert identity.sha256 is not None
    assert identity.sha256_short == identity.sha256[:12]
    assert str(tmp_path) not in str(identity.public_dict())


@pytest.mark.parametrize(
    ("path_name", "expected_error"),
    [
        ("missing.pt", "checkpoint_not_found"),
        ("directory.pt", "checkpoint_not_file"),
        ("weights.txt", "checkpoint_unsupported_extension"),
        ("../escape.pt", "checkpoint_path_escape"),
    ],
)
def test_checkpoint_validation_rejects_unsafe_paths(
    tmp_path,
    path_name,
    expected_error,
) -> None:
    (tmp_path / "directory.pt").mkdir()
    (tmp_path / "weights.txt").write_text("not a checkpoint", encoding="utf-8")
    (tmp_path.parent / "escape.pt").write_bytes(b"escape")

    identity = checkpoint_identity_for_path(path_name, model_root_dir=tmp_path)

    assert identity.valid is False
    assert identity.error_code == expected_error
    assert str(tmp_path) not in str(identity.public_dict())


def test_factory_creates_dummy_and_real_adapters_from_settings(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path, "lfcc.pt")
    factory = ModelFactory(
        Settings(
            _env_file=None,
            model_root_dir=str(tmp_path),
            cnn_model_mode="real",
            cnn_model_path=checkpoint.name,
        ),
        real_loaders={"lfcc_cnn_tcn": lambda _config: object()},
        real_predictors={"lfcc_cnn_tcn": _fake_real_predictor(0.8)},
    )

    models = factory.create_all()

    assert [model.branch_name for model in models] == [
        "lfcc_cnn_tcn",
        "aasist",
        "ssl_sequence",
        "glottal",
    ]
    assert [model.mode.value for model in models] == [
        "real",
        "dummy",
        "dummy",
        "dummy",
    ]
    assert models[0].health()["checkpoint_hash_short"]
    assert models[0].health()["adapter_type"] == "real"
    assert models[1].health()["adapter_type"] == "dummy"


def test_device_policy_reports_cuda_fallback_and_unavailable(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path, "lfcc.pt")
    fallback_factory = ModelFactory(
        Settings(
            _env_file=None,
            model_root_dir=str(tmp_path),
            cnn_model_mode="real",
            cnn_model_path=checkpoint.name,
            model_device_policy="cuda",
            model_default_device="cuda",
            model_allow_cpu_fallback=True,
        ),
        device_capabilities=StaticDeviceCapabilityProvider(cuda=False),
    )
    strict_factory = ModelFactory(
        Settings(
            _env_file=None,
            model_root_dir=str(tmp_path),
            cnn_model_mode="real",
            cnn_model_path=checkpoint.name,
            model_device_policy="cuda",
            model_default_device="cuda",
            model_allow_cpu_fallback=False,
        ),
        device_capabilities=StaticDeviceCapabilityProvider(cuda=False),
    )

    fallback_health = fallback_factory.create("lfcc_cnn_tcn").health()
    strict_health = strict_factory.create("lfcc_cnn_tcn").health()

    assert fallback_health["resolved_device"] == "cpu"
    assert fallback_health["fallback_used"] is True
    assert strict_health["resolved_device"] == "unavailable"
    assert strict_health["last_error_code"] is None


def test_real_adapter_loads_once_for_concurrent_callers(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path, "lfcc.pt")
    load_counter = Counter()
    model = ModelFactory(
        Settings(
            _env_file=None,
            model_root_dir=str(tmp_path),
            cnn_model_mode="real",
            cnn_model_path=checkpoint.name,
        ),
        real_loaders={
            "lfcc_cnn_tcn": lambda _config: load_counter.update(["load"]) or object()
        },
        real_predictors={"lfcc_cnn_tcn": _fake_real_predictor(0.7)},
    ).create("lfcc_cnn_tcn")

    threads = [threading.Thread(target=model.load) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert load_counter["load"] == 1
    assert model.health()["lifecycle_state"] == "ready"

    model.unload()
    model.unload()
    assert model.health()["lifecycle_state"] == "unloaded"


def test_real_adapter_retry_policy_can_recover_from_transient_load_failure(
    tmp_path,
) -> None:
    checkpoint = _checkpoint(tmp_path, "lfcc.pt")
    load_counter = Counter()

    def flaky_loader(_config):
        load_counter.update(["load"])
        if load_counter["load"] == 1:
            raise RuntimeError("transient framework startup")
        return object()

    model = ModelFactory(
        Settings(
            _env_file=None,
            model_root_dir=str(tmp_path),
            cnn_model_mode="real",
            cnn_model_path=checkpoint.name,
            model_load_retry_count=1,
        ),
        real_loaders={"lfcc_cnn_tcn": flaky_loader},
        real_predictors={"lfcc_cnn_tcn": _fake_real_predictor(0.7)},
    ).create("lfcc_cnn_tcn")

    model.load()

    assert load_counter["load"] == 2
    assert model.health()["lifecycle_state"] == "ready"


def test_real_mode_without_checkpoint_fails_honestly_without_dummy_prediction(
    tmp_path,
) -> None:
    model = ModelFactory(
        Settings(
            _env_file=None,
            model_root_dir=str(tmp_path),
            cnn_model_mode="real",
            # CNN-V2's own default path is no longer blank (see
            # Settings.cnn_model_path) -- explicitly clear it here so this
            # test still exercises "no checkpoint configured at all", not
            # "configured but not found".
            cnn_model_path="",
        )
    ).create("lfcc_cnn_tcn")

    prediction = model.predict_safe(_processed_audio())

    assert prediction.status == BranchStatus.failed
    assert prediction.mode.value == "real"
    assert prediction.probabilities is None
    assert prediction.metadata["error_code"] == "checkpoint_missing"
    assert model.health()["checkpoint_configured"] is False


def test_per_branch_timeout_isolated_from_remaining_dummy_branches(tmp_path) -> None:
    checkpoint = _checkpoint(tmp_path, "lfcc.pt")
    settings = Settings(
        _env_file=None,
        model_root_dir=str(tmp_path),
        cnn_model_mode="real",
        cnn_model_path=checkpoint.name,
        lfcc_cnn_tcn_timeout_seconds=0.01,
    )
    registry = ModelRegistry(
        models=ModelFactory(
            settings,
            real_loaders={"lfcc_cnn_tcn": lambda _config: object()},
            real_predictors={"lfcc_cnn_tcn": _slow_real_predictor},
        ).create_all(),
        app_settings=settings,
    )
    service = VoiceService(
        model_registry=registry,
        preprocess_fn=lambda _upload: _processed_audio(),
        app_settings=settings,
    )

    response = service.predict_from_validated_upload(
        _upload_metadata(tmp_path),
        cleanup_upload=False,
    )

    assert response.branches[0].status == BranchStatus.failed
    assert response.branches[0].metadata["error_code"] == "model_timeout"
    assert [branch.model_name for branch in response.branches] == [
        "cnn_acoustic",
        "aasist",
        "ssl_wavlm_xlsr",
        "glottal_features",
    ]
    assert sum(branch.status == BranchStatus.success for branch in response.branches) == 3
    assert response.fusion.status == BranchStatus.success


@pytest.mark.parametrize(
    ("real_branches", "research_eligible"),
    [
        (set(), False),
        ({"lfcc_cnn_tcn"}, False),
        ({"lfcc_cnn_tcn", "aasist"}, False),
        ({"lfcc_cnn_tcn", "aasist", "ssl_sequence"}, False),
        ({"lfcc_cnn_tcn", "aasist", "ssl_sequence", "glottal"}, True),
    ],
)
def test_mixed_dummy_fake_real_modes_preserve_order_and_research_policy(
    tmp_path,
    real_branches,
    research_eligible,
) -> None:
    settings_kwargs = {"model_root_dir": str(tmp_path)}
    for branch in real_branches:
        path = _checkpoint(tmp_path, f"{branch}.pt")
        if branch == "lfcc_cnn_tcn":
            settings_kwargs.update(cnn_model_mode="real", cnn_model_path=path.name)
        elif branch == "aasist":
            settings_kwargs.update(aasist_model_mode="real", aasist_model_path=path.name)
        elif branch == "ssl_sequence":
            settings_kwargs.update(ssl_model_mode="real", ssl_model_path=path.name)
        elif branch == "glottal":
            settings_kwargs.update(glottal_model_mode="real", glottal_model_path=path.name)
    settings = Settings(_env_file=None, **settings_kwargs)
    registry = ModelRegistry(
        models=ModelFactory(
            settings,
            real_loaders={branch: lambda _config: object() for branch in real_branches},
            real_predictors={
                branch: _fake_real_predictor(0.65) for branch in real_branches
            },
        ).create_all(),
        app_settings=settings,
    )
    service = VoiceService(
        model_registry=registry,
        preprocess_fn=lambda _upload: _processed_audio(),
        app_settings=settings,
    )

    response = service.predict_from_validated_upload(
        _upload_metadata(tmp_path),
        cleanup_upload=False,
    )

    assert [branch.model_name for branch in response.branches] == [
        "cnn_acoustic",
        "aasist",
        "ssl_wavlm_xlsr",
        "glottal_features",
    ]
    assert response.fusion.eligible_for_research_evaluation is research_eligible
    assert response.fusion.contains_dummy_branches is (not research_eligible)


def _checkpoint(tmp_path: Path, filename: str) -> Path:
    path = tmp_path / filename
    path.write_bytes(f"fake checkpoint {filename}".encode("utf-8"))
    return path


def _fake_real_predictor(spoof_probability: float):
    def predictor(processed_audio, _runtime_model, config):
        assert processed_audio.sample_rate == 16000
        return real_prediction_from_spoof_probability(
            config=config,
            spoof_probability=spoof_probability,
        )

    return predictor


def _slow_real_predictor(processed_audio, _runtime_model, config):
    time.sleep(0.2)
    return real_prediction_from_spoof_probability(
        config=config,
        spoof_probability=0.9,
    )


def _processed_audio() -> ProcessedAudio:
    sample_rate = 16000
    waveform = np.sin(
        2 * np.pi * 440 * np.arange(sample_rate, dtype=np.float32) / sample_rate
    ).astype(np.float32)
    return ProcessedAudio(
        waveform=waveform,
        sample_rate=sample_rate,
        original_sample_rate=sample_rate,
        original_channels=1,
        duration_seconds=1.0,
        was_resampled=False,
        was_converted_to_mono=False,
        normalisation_applied=False,
        peak_amplitude=float(np.max(np.abs(waveform))),
        rms_energy=float(np.sqrt(np.mean(np.square(waveform, dtype=np.float32)))),
    )


def _upload_metadata(tmp_path: Path) -> AudioUploadMetadata:
    audio_path = tmp_path / "upload.wav"
    audio_path.write_bytes(b"validated")
    return AudioUploadMetadata(
        original_filename="upload.wav",
        sanitized_filename="upload.wav",
        saved_filename="upload.wav",
        saved_path=audio_path,
        content_type="audio/wav",
        file_size_bytes=audio_path.stat().st_size,
        duration_seconds=1.0,
        sample_rate=16000,
        channels=1,
        original_extension="wav",
        detected_container="wav",
        detected_codec="pcm_s16le",
    )
