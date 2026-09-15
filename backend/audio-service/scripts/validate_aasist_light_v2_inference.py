"""Standalone AASIST-Light V2 inference verification.

This developer-only harness sends one local audio file through the backend's
real ingestion and canonical ``aasist`` model path, then prints deterministic
diagnostics for backend-vs-Colab parity checks.

Usage:
    python scripts/validate_aasist_light_v2_inference.py /path/to/audio.wav
    python scripts/validate_aasist_light_v2_inference.py /path/to/audio.wav --json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from time import perf_counter
from typing import Any
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings(
    "ignore",
    message='Field "model_.*" in Settings has conflict with protected namespace',
)

from app.config.settings import Settings
from app.core.exceptions import ModelInferenceError
from app.ingestion.audio import (
    AudioInspectionResult,
    ProcessedAudio,
    inspect_audio_file,
    preprocess_audio_file,
)
from app.models.factory import ModelFactory
from app.models.real.inference import (
    PROBABILITY_SUM_TOLERANCE,
    _prepare_input,
    _spoof_probability_from_logits,
)
from app.models.runtime import REAL_ARCHITECTURES
from app.models.torch_support import require_torch

EXPECTED_AASIST_VERSION = "aasist-light-v2-finalized-baseline"
EXPECTED_AASIST_SAMPLES = 64_600
EXPECTED_CLASS_MAPPING = {"bonafide": 0, "spoof": 1}


def validation_settings(**overrides: object) -> Settings:
    """Return settings for isolated canonical AASIST real-branch validation."""

    values: dict[str, object] = {
        "model_root_dir": "../model_artifacts",
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


def validate_audio_file(
    audio_path: Path,
    *,
    app_settings: Settings | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    """Run a local file through production ingestion and AASIST V2 inference."""

    settings = app_settings or validation_settings()
    path = audio_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    extension = path.suffix.lower().lstrip(".")

    total_start = perf_counter()
    preprocessing_start = perf_counter()
    inspection = inspect_audio_file(path, extension, settings)
    processed_audio = preprocess_audio_file(path, extension, settings, inspection)
    preprocessing_time_ms = _elapsed_ms(preprocessing_start)

    result = classify_processed_audio(
        processed_audio,
        app_settings=settings,
        model=model,
        audio_path=path,
        inspection=inspection,
        preprocessing_time_ms=preprocessing_time_ms,
    )
    result["performance"]["total_time_ms"] = _elapsed_ms(total_start)
    return result


def classify_processed_audio(
    processed_audio: ProcessedAudio,
    *,
    app_settings: Settings,
    model: Any | None = None,
    audio_path: Path | None = None,
    inspection: AudioInspectionResult | None = None,
    preprocessing_time_ms: float = 0.0,
) -> dict[str, Any]:
    """Classify already-ingested audio using the backend's real AASIST adapter."""

    torch = require_torch()
    aasist_model = model or ModelFactory(app_settings).create("aasist")
    if aasist_model.branch_name != "aasist":
        raise ModelInferenceError(
            "AASIST validation harness did not create the canonical aasist branch.",
            error_code="model_output_invalid",
        )

    load_start = perf_counter()
    aasist_model.load()
    load_time_ms = _elapsed_ms(load_start)

    config = aasist_model._config
    runtime_model = aasist_model._runtime_model
    _validate_runtime_contract(runtime_model)

    source_waveform = _model_source_waveform(processed_audio, runtime_model)
    inference_start = perf_counter()
    model_input = _prepare_input(torch, processed_audio, runtime_model, config)
    model_input_cpu = model_input.detach().cpu().reshape(-1)
    with torch.inference_mode():
        logits = runtime_model.module(model_input)
    inference_time_ms = _elapsed_ms(inference_start)

    _validate_logits(torch, logits)
    probabilities = torch.softmax(logits.to(dtype=torch.float32), dim=-1)[0]
    bonafide_probability = float(probabilities[0])
    spoof_probability = _spoof_probability_from_logits(
        torch,
        logits,
        runtime_model,
        config,
    )
    _validate_probabilities(bonafide_probability, spoof_probability)

    input_samples = int(model_input_cpu.numel())
    result = {
        "file": {
            "basename": audio_path.name if audio_path is not None else None,
            "path": str(audio_path) if audio_path is not None else None,
        },
        "branch": aasist_model.branch_name,
        "model": runtime_model.architecture_version,
        "model_version": config.model_version,
        "checkpoint": {
            "filename": config.checkpoint.filename,
            "sha256": config.checkpoint.sha256,
            "epoch": runtime_model.checkpoint_metadata.get("epoch"),
            "best_eer": runtime_model.checkpoint_metadata.get("best_eer"),
            "class_mapping": runtime_model.checkpoint_metadata.get("class_mapping"),
        },
        "device": runtime_model.device,
        "model_training": bool(runtime_model.module.training),
        "ingestion": {
            "decoded_sample_rate": processed_audio.sample_rate,
            "decoded_sample_count": int(
                processed_audio.unnormalised_waveform.shape[0]
                if processed_audio.unnormalised_waveform is not None
                else processed_audio.waveform.shape[0]
            ),
            "original_sample_rate": processed_audio.original_sample_rate,
            "original_channels": processed_audio.original_channels,
            "ffprobe_sample_rate": inspection.sample_rate if inspection else None,
            "ffprobe_channels": inspection.channels if inspection else None,
            "shared_normalisation_applied": processed_audio.normalisation_applied,
        },
        "preprocessing": {
            "used_unnormalised_waveform": bool(
                getattr(runtime_model.front_end, "requires_unnormalized_waveform", False)
            ),
            "source_samples": int(source_waveform.size),
            "source_peak": float(np.max(np.abs(source_waveform))),
            "shared_waveform_peak": float(np.max(np.abs(processed_audio.waveform))),
            "model_input_shape": list(model_input.shape),
            "model_input_samples": input_samples,
            "model_input_mean": float(model_input_cpu.mean().item()),
            "model_input_std": float(model_input_cpu.std(unbiased=False).item()),
            "model_input_finite": bool(torch.isfinite(model_input_cpu).all()),
            "crop_occurred": int(source_waveform.size) > EXPECTED_AASIST_SAMPLES,
            "padding_occurred": int(source_waveform.size) < EXPECTED_AASIST_SAMPLES,
            "normalization": runtime_model.preprocessing.get("normalization"),
            "length_policy": runtime_model.preprocessing.get("length_policy"),
        },
        "logits": logits.detach().cpu().tolist(),
        "logits_shape": list(logits.shape),
        "bonafide_probability": bonafide_probability,
        "spoof_probability": spoof_probability,
        "probability_sum": bonafide_probability + spoof_probability,
        "prediction": (
            "spoof" if spoof_probability >= bonafide_probability else "bonafide"
        ),
        "performance": {
            "model_load_time_ms": load_time_ms,
            "preprocessing_time_ms": preprocessing_time_ms,
            "inference_time_ms": inference_time_ms,
            "total_time_ms": preprocessing_time_ms + load_time_ms + inference_time_ms,
        },
    }
    _validate_result(result)
    return result


def _validate_runtime_contract(runtime_model: Any) -> None:
    if runtime_model.architecture_version != EXPECTED_AASIST_VERSION:
        raise ModelInferenceError(
            "AASIST runtime architecture version is not the finalized V2 baseline.",
            error_code="model_output_invalid",
        )
    if runtime_model.spoof_index != 1 or runtime_model.bonafide_index != 0:
        raise ModelInferenceError(
            "AASIST runtime class mapping is not bonafide=0, spoof=1.",
            error_code="model_output_invalid",
        )
    if runtime_model.checkpoint_metadata.get("class_mapping") != EXPECTED_CLASS_MAPPING:
        raise ModelInferenceError(
            "AASIST checkpoint class mapping is not bonafide=0, spoof=1.",
            error_code="model_output_invalid",
        )


def _model_source_waveform(
    processed_audio: ProcessedAudio,
    runtime_model: Any,
) -> np.ndarray:
    if getattr(runtime_model.front_end, "requires_unnormalized_waveform", False):
        if processed_audio.unnormalised_waveform is None:
            raise ModelInferenceError(
                "AASIST-Light V2 requires the unnormalised decoded waveform.",
                public_message="Audio could not be analysed.",
                error_code="model_input_invalid",
            )
        return np.asarray(processed_audio.unnormalised_waveform, dtype=np.float32)
    return np.asarray(processed_audio.waveform, dtype=np.float32)


def _validate_logits(torch: Any, logits: Any) -> None:
    if list(logits.shape) != [1, 2]:
        raise ModelInferenceError(
            f"AASIST-Light V2 logits shape {list(logits.shape)} != [1, 2].",
            error_code="model_output_invalid",
        )
    if not bool(torch.isfinite(logits).all()):
        raise ModelInferenceError(
            "AASIST-Light V2 logits are non-finite.",
            error_code="model_output_invalid",
        )


def _validate_probabilities(
    bonafide_probability: float,
    spoof_probability: float,
) -> None:
    values = (bonafide_probability, spoof_probability)
    if not all(np.isfinite(value) for value in values):
        raise ModelInferenceError(
            "AASIST-Light V2 probabilities are non-finite.",
            error_code="model_output_invalid",
        )
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ModelInferenceError(
            "AASIST-Light V2 probabilities are out of range.",
            error_code="model_output_invalid",
        )
    if abs(sum(values) - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ModelInferenceError(
            "AASIST-Light V2 probabilities do not sum to 1.",
            error_code="model_output_invalid",
        )


def _validate_result(result: dict[str, Any]) -> None:
    if result["branch"] != "aasist":
        raise ModelInferenceError("AASIST validation used a non-canonical branch.")
    if result["model"] != REAL_ARCHITECTURES["aasist"]:
        raise ModelInferenceError("AASIST validation used an unexpected model.")
    preprocessing = result["preprocessing"]
    if preprocessing["model_input_samples"] != EXPECTED_AASIST_SAMPLES:
        raise ModelInferenceError("AASIST V2 input sample count is incorrect.")
    if preprocessing["normalization"] != "per_waveform_zscore":
        raise ModelInferenceError("AASIST V2 z-score preprocessing was not used.")
    if preprocessing["length_policy"] != "first_crop_right_zero_pad":
        raise ModelInferenceError("AASIST V2 length policy was not used.")
    if result["model_training"]:
        raise ModelInferenceError("AASIST V2 model is still in training mode.")


def _elapsed_ms(start: float) -> float:
    return max((perf_counter() - start) * 1000.0, 0.0)


def _print_human(result: dict[str, Any]) -> None:
    preprocessing = result["preprocessing"]
    ingestion = result["ingestion"]
    performance = result["performance"]
    checkpoint = result["checkpoint"]
    print(f"Input file: {result['file']['path']}")
    print(f"Checkpoint: {checkpoint['filename']}")
    print(f"SHA-256: {checkpoint['sha256']}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Device: {result['device']}")
    print(f"Decoded sample rate: {ingestion['decoded_sample_rate']}")
    print(f"Decoded sample count: {ingestion['decoded_sample_count']}")
    print(f"V2 samples: {preprocessing['model_input_samples']}")
    print(f"V2 mean: {preprocessing['model_input_mean']:.12f}")
    print(f"V2 std: {preprocessing['model_input_std']:.12f}")
    print(f"Logits: {result['logits']}")
    print(f"P(bonafide): {result['bonafide_probability']:.12f}")
    print(f"P(spoof): {result['spoof_probability']:.12f}")
    print(f"Probability sum: {result['probability_sum']:.12f}")
    print(f"Prediction: {result['prediction']}")
    print(f"Preprocessing time ms: {performance['preprocessing_time_ms']:.3f}")
    print(f"Inference time ms: {performance['inference_time_ms']:.3f}")
    print(f"Total time ms: {performance['total_time_ms']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output file for Colab parity comparison.",
    )
    args = parser.parse_args()

    if args.json:
        logging.disable(logging.CRITICAL)

    result = validate_audio_file(args.audio_path)
    if args.output is not None:
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_human(result)


if __name__ == "__main__":
    main()
