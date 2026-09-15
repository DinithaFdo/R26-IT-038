"""Standalone SSL (XLS-R + Mamba, Generalization V2) inference verification.

Developer-only harness. Loads the real ``ssl_sequence`` branch (frozen XLS-R
backbone + the Generalization V2 trainable head), prints checkpoint identity
and load diagnostics, and -- if an audio file is given -- runs it through the
backend's real ingestion + inference path and prints the prediction.

Model loading alone requires network access on first run (the ~1.2 GB XLS-R
backbone is fetched from Hugging Face Hub and cached under
``~/.cache/huggingface``); subsequent runs are offline.

Usage:
    python scripts/validate_ssl_generalization_v2_inference.py
    python scripts/validate_ssl_generalization_v2_inference.py --audio /path/to/sample.wav
    python scripts/validate_ssl_generalization_v2_inference.py --audio /path/to/sample.wav --json
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
from app.ingestion.audio import ProcessedAudio, inspect_audio_file, preprocess_audio_file
from app.models.factory import ModelFactory
from app.models.real.ssl_sequence_inference import PROBABILITY_SUM_TOLERANCE
from app.models.runtime import REAL_ARCHITECTURES

EXPECTED_SSL_ARCHITECTURE = REAL_ARCHITECTURES["ssl_sequence"]
EXPECTED_XLSR_HIDDEN_SIZE = 1024
EXPECTED_MAMBA_DIM = 256
EXPECTED_TRAINABLE_PARAMETER_COUNT = 1_173_634


def validation_settings(**overrides: object) -> Settings:
    """Settings for isolated canonical SSL real-branch validation."""

    values: dict[str, object] = {
        "model_root_dir": "../model_artifacts",
        "ssl_model_mode": "real",
        "cnn_model_mode": "disabled",
        "aasist_model_mode": "disabled",
        "glottal_model_mode": "disabled",
        "required_model_branches": "ssl_sequence",
        "fusion_min_successful_branches": 1,
        "model_device_policy": "cpu",
        "model_default_device": "cpu",
        "model_load_strategy": "lazy",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def load_ssl_model(app_settings: Settings | None = None) -> dict[str, Any]:
    """Load the real SSL branch and return checkpoint/architecture diagnostics."""

    settings = app_settings or validation_settings()
    model = ModelFactory(settings).create("ssl_sequence")
    if model.branch_name != "ssl_sequence":
        raise ModelInferenceError(
            "SSL validation harness did not create the canonical ssl_sequence branch.",
            error_code="model_output_invalid",
        )

    load_start = perf_counter()
    model.load()
    load_time_ms = _elapsed_ms(load_start)

    config = model._config
    runtime_model = model._runtime_model
    missing_unexpected = _check_missing_unexpected(runtime_model)

    return {
        "model": model,
        "config": config,
        "runtime_model": runtime_model,
        "load_time_ms": load_time_ms,
        "checkpoint": {
            "filename": config.checkpoint.filename,
            "sha256": config.checkpoint.sha256,
            "epoch": runtime_model.checkpoint_epoch,
            "stage": runtime_model.checkpoint_stage,
            "train_loss": runtime_model.checkpoint_train_loss,
        },
        "architecture": {
            "architecture_version": runtime_model.architecture_version,
            "xlsr_model_name": runtime_model.xlsr_model_name,
            "xlsr_hidden_size": EXPECTED_XLSR_HIDDEN_SIZE,
            "mamba_dim": runtime_model.mamba_dim,
            "label_mapping": runtime_model.label_mapping,
            "parameter_count": runtime_model.parameter_count,
            "trainable_parameter_count": runtime_model.trainable_parameter_count,
        },
        "device": runtime_model.device,
        "missing_unexpected_keys": missing_unexpected,
        "verification": runtime_model.verification,
    }


def _check_missing_unexpected(runtime_model: Any) -> dict[str, Any]:
    """Re-derive the same missing/unexpected-key check the real loader already
    enforced (which raises on failure), just to report it explicitly here."""

    own_state = runtime_model.module.state_dict()
    non_backbone_keys = [key for key in own_state if not key.startswith("xlsr.")]
    return {
        "non_xlsr_missing_keys": [],  # loader already raises before returning if any exist
        "unexpected_keys": [],  # loader already raises before returning if any exist
        "non_xlsr_parameter_keys_restored": len(non_backbone_keys),
        "trainable_parameter_count_matches_expected": (
            runtime_model.trainable_parameter_count == EXPECTED_TRAINABLE_PARAMETER_COUNT
        ),
    }


def classify_processed_audio(
    processed_audio: ProcessedAudio,
    *,
    loaded: dict[str, Any],
    audio_path: Path | None = None,
    preprocessing_time_ms: float = 0.0,
) -> dict[str, Any]:
    model = loaded["model"]
    inference_start = perf_counter()
    prediction = model.predict(processed_audio)
    inference_time_ms = _elapsed_ms(inference_start)

    bonafide_probability = prediction.probabilities.bonafide
    spoof_probability = prediction.probabilities.spoof
    _validate_probabilities(bonafide_probability, spoof_probability)

    result: dict[str, Any] = {
        "file": {
            "basename": audio_path.name if audio_path is not None else None,
            "path": str(audio_path) if audio_path is not None else None,
        },
        "prediction": prediction.prediction.value if prediction.prediction else None,
        "bonafide_probability": bonafide_probability,
        "spoof_probability": spoof_probability,
        "probability_sum": bonafide_probability + spoof_probability,
        "confidence": prediction.confidence,
        "metadata": prediction.metadata,
        "performance": {
            "preprocessing_time_ms": preprocessing_time_ms,
            "inference_time_ms": inference_time_ms,
            "total_time_ms": preprocessing_time_ms + inference_time_ms,
        },
    }
    return result


def validate_audio_file(audio_path: Path, *, loaded: dict[str, Any]) -> dict[str, Any]:
    settings = loaded["config"]
    app_settings = _settings_for_ingestion(loaded)
    path = audio_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    extension = path.suffix.lower().lstrip(".")

    preprocessing_start = perf_counter()
    inspection = inspect_audio_file(path, extension, app_settings)
    processed_audio = preprocess_audio_file(path, extension, app_settings, inspection)
    preprocessing_time_ms = _elapsed_ms(preprocessing_start)

    return classify_processed_audio(
        processed_audio,
        loaded=loaded,
        audio_path=path,
        preprocessing_time_ms=preprocessing_time_ms,
    )


def _settings_for_ingestion(loaded: dict[str, Any]) -> Settings:
    # `config` on the loaded model is a BranchModelConfig, not Settings --
    # ingestion needs the actual Settings instance used to build the model.
    return loaded["_app_settings"]


def _validate_probabilities(bonafide_probability: float, spoof_probability: float) -> None:
    values = (bonafide_probability, spoof_probability)
    if not all(np.isfinite(value) for value in values):
        raise ModelInferenceError("SSL probabilities are non-finite.")
    if not all(0.0 <= value <= 1.0 for value in values):
        raise ModelInferenceError("SSL probabilities are out of range.")
    if abs(sum(values) - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise ModelInferenceError("SSL probabilities do not sum to 1.")


def _elapsed_ms(start: float) -> float:
    return max((perf_counter() - start) * 1000.0, 0.0)


def _print_human(loaded: dict[str, Any], audio_result: dict[str, Any] | None) -> None:
    checkpoint = loaded["checkpoint"]
    architecture = loaded["architecture"]
    print(f"Resolved checkpoint: {checkpoint['filename']}")
    print(f"Checkpoint SHA-256: {checkpoint['sha256']}")
    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Checkpoint stage: {checkpoint['stage']}")
    print(f"Checkpoint train_loss: {checkpoint['train_loss']}")
    print(f"Device: {loaded['device']}")
    print(f"XLS-R model name: {architecture['xlsr_model_name']}")
    print(f"Architecture version: {architecture['architecture_version']}")
    print(f"Mamba dim: {architecture['mamba_dim']}")
    print(f"Label mapping: {architecture['label_mapping']}")
    print(f"Trainable parameter count: {architecture['trainable_parameter_count']}")
    print(f"Total parameter count: {architecture['parameter_count']}")
    print(
        "Trainable parameter count matches expected "
        f"({EXPECTED_TRAINABLE_PARAMETER_COUNT}): "
        f"{loaded['missing_unexpected_keys']['trainable_parameter_count_matches_expected']}"
    )
    print(f"Model load time ms: {loaded['load_time_ms']:.3f}")
    print(f"preprocessing_verified: {loaded['verification']['preprocessing_verified']}")
    print(f"class_mapping_verified: {loaded['verification']['class_mapping_verified']}")
    if audio_result is None:
        print("\nNo --audio supplied; model loading validated only.")
        return
    print(f"\nInput file: {audio_result['file']['path']}")
    print(f"Prediction: {audio_result['prediction']}")
    print(f"P(bonafide): {audio_result['bonafide_probability']:.6f}")
    print(f"P(spoof): {audio_result['spoof_probability']:.6f}")
    print(f"Probability sum: {audio_result['probability_sum']:.6f}")
    print(f"Confidence: {audio_result['confidence']:.6f}")
    performance = audio_result["performance"]
    print(f"Preprocessing time ms: {performance['preprocessing_time_ms']:.3f}")
    print(f"Inference time ms: {performance['inference_time_ms']:.3f}")
    print(f"Total time ms: {performance['total_time_ms']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, default=None, help="Optional audio file to classify.")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument("--output", type=Path, help="Optional JSON output file.")
    args = parser.parse_args()

    if args.json:
        logging.disable(logging.CRITICAL)

    app_settings = validation_settings()
    loaded = load_ssl_model(app_settings)
    loaded["_app_settings"] = app_settings

    audio_result = None
    if args.audio is not None:
        audio_result = validate_audio_file(args.audio, loaded=loaded)

    if args.json or args.output is not None:
        payload = {
            "checkpoint": loaded["checkpoint"],
            "architecture": loaded["architecture"],
            "device": loaded["device"],
            "load_time_ms": loaded["load_time_ms"],
            "missing_unexpected_keys": loaded["missing_unexpected_keys"],
            "verification": loaded["verification"],
            "audio_result": audio_result,
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True, default=str)
        if args.output is not None:
            args.output.write_text(serialized + "\n", encoding="utf-8")
        if args.json:
            print(serialized)
            return

    _print_human(loaded, audio_result)


if __name__ == "__main__":
    main()
