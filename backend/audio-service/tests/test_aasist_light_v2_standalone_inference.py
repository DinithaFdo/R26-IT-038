from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np
import pytest

from app.core.exceptions import AudioUploadError, ModelInferenceError
from app.ingestion.audio import preprocess_waveform
from app.models.factory import ModelFactory
from app.models.runtime import REAL_ARCHITECTURES
from app.schemas.common import BranchStatus
from scripts.validate_aasist_light_v2_inference import (
    classify_processed_audio,
    validate_audio_file,
    validation_settings,
)
from tests.real_model_helpers import CHECKPOINT_ROOT, requires_torch

pytestmark = requires_torch

AASIST_V2_CHECKPOINT = CHECKPOINT_ROOT / "aasist" / "aasist_light_v2_best.pt"
requires_aasist_v2_checkpoint = pytest.mark.skipif(
    not AASIST_V2_CHECKPOINT.is_file(),
    reason=f"AASIST-Light V2 checkpoint not present at {AASIST_V2_CHECKPOINT}.",
)


@pytest.fixture()
def aasist_settings(tmp_path: Path):
    return validation_settings(
        model_root_dir=str(CHECKPOINT_ROOT),
        upload_dir=str(tmp_path / "uploads"),
    )


@pytest.fixture()
def valid_audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "phase3-valid.wav"
    _write_wav(path, _synthetic_voice_like_waveform(seconds=1.25))
    return path


@pytest.fixture()
def validation_result(valid_audio_file: Path, aasist_settings):
    return validate_audio_file(valid_audio_file, app_settings=aasist_settings)


@requires_aasist_v2_checkpoint
def test_standalone_inference_harness_uses_canonical_aasist(validation_result) -> None:
    assert validation_result["branch"] == "aasist"
    assert validation_result["model"] == "aasist-light-v2-finalized-baseline"


@requires_aasist_v2_checkpoint
def test_actual_real_adapter_produces_two_logits(validation_result) -> None:
    assert validation_result["logits_shape"] == [1, 2]
    assert len(validation_result["logits"][0]) == 2


@requires_aasist_v2_checkpoint
def test_softmax_index_zero_is_bonafide(validation_result) -> None:
    import torch

    logits = torch.tensor(validation_result["logits"], dtype=torch.float32)
    probabilities = torch.softmax(logits, dim=-1)[0]

    assert validation_result["checkpoint"]["class_mapping"]["bonafide"] == 0
    assert validation_result["bonafide_probability"] == pytest.approx(
        float(probabilities[0])
    )


@requires_aasist_v2_checkpoint
def test_softmax_index_one_is_spoof(validation_result) -> None:
    import torch

    logits = torch.tensor(validation_result["logits"], dtype=torch.float32)
    probabilities = torch.softmax(logits, dim=-1)[0]

    assert validation_result["checkpoint"]["class_mapping"]["spoof"] == 1
    assert validation_result["spoof_probability"] == pytest.approx(
        float(probabilities[1])
    )


@requires_aasist_v2_checkpoint
def test_probabilities_are_finite(validation_result) -> None:
    assert math.isfinite(validation_result["bonafide_probability"])
    assert math.isfinite(validation_result["spoof_probability"])


@requires_aasist_v2_checkpoint
def test_probabilities_sum_to_one(validation_result) -> None:
    assert validation_result["probability_sum"] == pytest.approx(1.0, abs=1e-6)


@requires_aasist_v2_checkpoint
def test_v2_model_input_is_exactly_64600_samples(validation_result) -> None:
    assert validation_result["preprocessing"]["model_input_samples"] == 64_600
    assert validation_result["preprocessing"]["model_input_shape"] == [1, 1, 64_600]


@requires_aasist_v2_checkpoint
def test_v2_model_input_mean_is_approximately_zero(validation_result) -> None:
    assert abs(validation_result["preprocessing"]["model_input_mean"]) < 1e-5


@requires_aasist_v2_checkpoint
def test_v2_model_input_std_is_approximately_one(validation_result) -> None:
    assert validation_result["preprocessing"]["model_input_std"] == pytest.approx(
        1.0,
        abs=1e-4,
    )


@requires_aasist_v2_checkpoint
def test_shared_peak_normalized_waveform_is_not_used(
    valid_audio_file: Path,
    aasist_settings,
) -> None:
    result = validate_audio_file(valid_audio_file, app_settings=aasist_settings)

    assert result["ingestion"]["shared_normalisation_applied"] is True
    assert result["preprocessing"]["used_unnormalised_waveform"] is True
    assert result["preprocessing"]["source_peak"] != pytest.approx(
        result["preprocessing"]["shared_waveform_peak"]
    )


@requires_aasist_v2_checkpoint
def test_known_valid_audio_runs_without_error(validation_result) -> None:
    assert validation_result["prediction"] in {"bonafide", "spoof"}
    assert validation_result["model_training"] is False


def test_corrupted_audio_fails_through_controlled_ingestion_exception(
    tmp_path: Path,
    aasist_settings,
) -> None:
    bad_audio = tmp_path / "corrupted.wav"
    bad_audio.write_bytes(b"not a valid wav file")

    with pytest.raises(AudioUploadError):
        validate_audio_file(bad_audio, app_settings=aasist_settings)


@requires_aasist_v2_checkpoint
def test_near_silent_audio_fails_through_v2_preprocessing(aasist_settings) -> None:
    time = np.arange(64_600, dtype=np.float32) / 16_000
    near_silent = (5e-8 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    processed = preprocess_waveform(
        near_silent,
        sample_rate=16_000,
        target_sample_rate=16_000,
        app_settings=aasist_settings,
    )

    with pytest.raises(ModelInferenceError) as error:
        classify_processed_audio(processed, app_settings=aasist_settings)

    assert error.value.error_code == "model_input_invalid"


@requires_aasist_v2_checkpoint
def test_repeated_inference_on_same_file_is_deterministic(
    valid_audio_file: Path,
    aasist_settings,
) -> None:
    model = ModelFactory(aasist_settings).create("aasist")

    first = validate_audio_file(valid_audio_file, app_settings=aasist_settings, model=model)
    second = validate_audio_file(valid_audio_file, app_settings=aasist_settings, model=model)

    assert abs(first["spoof_probability"] - second["spoof_probability"]) < 1e-8
    assert first["logits"][0] == pytest.approx(second["logits"][0])


@requires_aasist_v2_checkpoint
def test_model_is_reused_rather_than_reloaded_per_call(
    valid_audio_file: Path,
    aasist_settings,
) -> None:
    model = ModelFactory(aasist_settings).create("aasist")

    validate_audio_file(valid_audio_file, app_settings=aasist_settings, model=model)
    runtime_model_id = id(model._runtime_model)
    validate_audio_file(valid_audio_file, app_settings=aasist_settings, model=model)

    assert id(model._runtime_model) == runtime_model_id


def test_public_branch_remains_aasist(aasist_settings) -> None:
    model = ModelFactory(aasist_settings).create("aasist")

    assert model.branch_name == "aasist"


def test_cnn_is_now_v2() -> None:
    """Was `test_cnn_remains_unchanged`. CNN intentionally moved to CNN-V2 in
    the final-detector integration -- pins the deliberate new value instead."""

    assert REAL_ARCHITECTURES["lfcc_cnn_tcn"] == "cnn-v2-lfcc-delta-vgg-4block-v1"


def test_ssl_remains_unchanged() -> None:
    assert (
        REAL_ARCHITECTURES["ssl_sequence"]
        == "xlsr-mamba-sequence-frozen-backbone-pure-pytorch-scan-v1"
    )


def test_glottal_is_now_real() -> None:
    """Was `test_glottal_remains_unchanged`. Glottal intentionally moved from
    its placeholder contract string to a trained real pipeline (logistic
    regression over 20 selected DisVoice/parselmouth/spectral features) in
    the Glottal branch integration -- pins the deliberate new value instead.
    """

    assert REAL_ARCHITECTURES["glottal"] == "glottal-logreg-selected20-v1"


def test_fusion_remains_unchanged(aasist_settings) -> None:
    assert aasist_settings.fusion_method == "weighted_average"
    assert aasist_settings.fusion_decision_threshold == 0.5
    assert aasist_settings.fusion_weight_lfcc_cnn_tcn == 0.25
    assert aasist_settings.fusion_weight_aasist == 0.25
    assert aasist_settings.fusion_weight_ssl_sequence == 0.25
    assert aasist_settings.fusion_weight_glottal == 0.25


def test_api_schemas_remain_unchanged(aasist_settings) -> None:
    from app.main import create_app

    app = create_app(app_settings=aasist_settings)
    schema = app.openapi()

    assert "/api/v1/voice/models/health" in schema["paths"]
    assert not any("validate_aasist" in path for path in schema["paths"])
    assert not any("aasist-light-v2" in path for path in schema["paths"])


def _synthetic_voice_like_waveform(
    *,
    seconds: float,
    sample_rate: int = 16_000,
) -> np.ndarray:
    time = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    signal = (
        0.45 * np.sin(2 * np.pi * 140 * time)
        + 0.25 * np.sin(2 * np.pi * 280 * time)
        + 0.05 * np.sin(2 * np.pi * 800 * time)
    )
    return signal.astype(np.float32)


def _write_wav(path: Path, waveform: np.ndarray, sample_rate: int = 16_000) -> None:
    clipped = np.clip(waveform, -1.0, 1.0)
    pcm = (clipped * np.iinfo(np.int16).max).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
