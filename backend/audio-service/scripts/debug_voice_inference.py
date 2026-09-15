"""Standalone MULTI-SCOPE inference diagnostic.

Runs one audio file through the exact same decode -> shared-preprocessing ->
branch-front-end -> model.forward() -> softmax path the API uses, but prints
every intermediate value instead of only the final API response. Read-only:
loads checkpoints for inference exactly as the app does, never trains or
writes anything.

Usage:
    backend/.venv/bin/python scripts/debug_voice_inference.py /path/to/audio.wav

Run from the backend/ directory so app.config.settings picks up backend/.env.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from app.config.settings import settings
from app.ingestion.audio import inspect_audio_file, preprocess_audio_file
from app.models.factory import ModelFactory
from app.models.real.inference import (
    _prepare_input,
    _spoof_probability_from_logits,
    build_aasist_loader,
    build_cnn_loader,
)
from app.models.real.ssl_sequence_inference import (
    _prepare_ssl_input,
    build_ssl_loader,
)
from app.models.real.ssl_sequence_inference import (
    _spoof_probability_from_logits as ssl_spoof_probability_from_logits,
)

SEP = "=" * 60


def section(title: str) -> None:
    print(f"\n{SEP}\n{title}\n{SEP}")


def describe_audio(path: Path) -> "tuple[object, object]":
    section("AUDIO")
    extension = path.suffix.lower().lstrip(".")
    inspection = inspect_audio_file(path, extension, settings)
    processed = preprocess_audio_file(path, extension, settings, inspection)
    print(f"path:                  {path}")
    print(f"extension:             {extension}")
    print(f"codec (ffprobe):       {getattr(inspection, 'codec_name', 'n/a')}")
    print(f"original_sample_rate:  {processed.original_sample_rate}")
    print(f"canonical_sample_rate: {processed.sample_rate}")
    print(f"original_channels:     {processed.original_channels}")
    print(f"was_resampled:         {processed.was_resampled}")
    print(f"was_converted_to_mono: {processed.was_converted_to_mono}")
    print(f"normalisation_applied: {processed.normalisation_applied}")
    print(f"duration_seconds:      {processed.duration_seconds:.4f}")
    print(f"samples:               {processed.waveform.shape[0]}")
    print(f"peak_amplitude:        {processed.peak_amplitude:.6f}")
    print(f"rms_energy:            {processed.rms_energy:.6f}")
    print(f"preprocessing_version: {processed.preprocessing_version}")
    return inspection, processed


def run_cnn_or_aasist(branch_name: str, processed_audio, factory: ModelFactory) -> None:
    section(f"{branch_name.upper()} DEBUG")
    config = factory.branch_config(branch_name)
    print(f"mode:                   {config.mode}")
    print(f"checkpoint_configured:  {config.checkpoint.configured}")
    print(f"checkpoint_valid:       {config.checkpoint.valid}")
    print(f"checkpoint_filename:    {config.checkpoint.filename}")
    print(f"checkpoint_size_bytes:  {config.checkpoint.size_bytes}")
    print(f"checkpoint_sha256:      {config.checkpoint.sha256}")
    print(f"checkpoint_error_code:  {config.checkpoint.error_code}")
    print(f"resolved_device:        {config.resolved_device}")
    print(f"class_order (config):   {settings.branch_class_order(branch_name)}")
    print(f"preprocessing_verified: {config.preprocessing_verified}")
    print(f"class_mapping_verified: {config.class_mapping_verified}")

    if config.mode.value != "real" or not config.checkpoint.valid:
        print("-> skipped: branch is not real+valid, no inference to run.")
        return

    loader = build_cnn_loader(settings) if branch_name == "lfcc_cnn_tcn" else build_aasist_loader(settings)
    runtime_model = loader(config)
    print(f"architecture_version:   {runtime_model.architecture_version}")
    print(f"parameter_count:        {runtime_model.parameter_count}")
    print(f"state_dict_keys:        {runtime_model.state_dict_keys}")
    print(f"module.training:        {runtime_model.module.training}  (must be False)")
    print(f"spoof_index:            {runtime_model.spoof_index}")
    print(f"bonafide_index:         {runtime_model.bonafide_index}")
    print(f"front_end preprocessing: {runtime_model.preprocessing}")

    model_input = _prepare_input(torch, processed_audio, runtime_model, config)
    print(f"model_input shape:      {tuple(model_input.shape)}")
    with torch.inference_mode():
        logits = runtime_model.module(model_input)
    print(f"raw_logits:             {logits.tolist()}")
    probs = torch.softmax(logits.to(dtype=torch.float32), dim=-1)[0]
    print(f"softmax:                {probs.tolist()}")
    spoof = _spoof_probability_from_logits(torch, logits, runtime_model, config)
    bonafide = 1.0 - spoof
    print(f"mapped spoof=%.4f%%  bonafide=%.4f%%" % (spoof * 100, bonafide * 100))
    print(f"final prediction:       {'spoof' if spoof >= bonafide else 'bonafide'}")


def run_ssl(processed_audio, factory: ModelFactory) -> None:
    section("SSL_SEQUENCE DEBUG")
    config = factory.branch_config("ssl_sequence")
    print(f"mode:                   {config.mode}")
    print(f"checkpoint_configured:  {config.checkpoint.configured}")
    print(f"checkpoint_valid:       {config.checkpoint.valid}")
    print(f"checkpoint_filename:    {config.checkpoint.filename}")
    print(f"checkpoint_size_bytes:  {config.checkpoint.size_bytes}")
    print(f"checkpoint_sha256:      {config.checkpoint.sha256}")
    print(f"checkpoint_error_code:  {config.checkpoint.error_code}")
    print(f"resolved_device:        {config.resolved_device}")
    print(f"class_order (config):   {settings.branch_class_order('ssl_sequence')}")
    print(f"preprocessing_verified: {config.preprocessing_verified}")
    print(f"class_mapping_verified: {config.class_mapping_verified}")

    if config.mode.value != "real" or not config.checkpoint.valid:
        print("-> skipped: branch is not real+valid, no inference to run.")
        return

    loader = build_ssl_loader(settings)
    runtime_model = loader(config)
    print(f"architecture_version:   {runtime_model.architecture_version}")
    print(f"xlsr_model_name:        {runtime_model.xlsr_model_name}")
    print(f"mamba_dim:              {runtime_model.mamba_dim}")
    print(f"parameter_count:        {runtime_model.parameter_count}")
    print(f"trainable_parameters:   {runtime_model.trainable_parameter_count}")
    print(f"module.training:        {runtime_model.module.training}  (must be False)")
    print(f"artifact label_mapping: {runtime_model.label_mapping}")
    print(f"configured spoof_index: {runtime_model.spoof_index}")
    print(f"configured bonafide_ix: {runtime_model.bonafide_index}")
    artifact_spoof_ix = runtime_model.label_mapping.get("spoof")
    if artifact_spoof_ix is not None and artifact_spoof_ix != runtime_model.spoof_index:
        print(
            "  !! MISMATCH: artifact's own label_mapping disagrees with the "
            "configured SSL_CLASS_ORDER."
        )
    print(f"front_end preprocessing: {runtime_model.preprocessing}")

    input_values, attention_mask = _prepare_ssl_input(torch, processed_audio, runtime_model, config)
    print(f"input_values shape:     {tuple(input_values.shape)}")
    print(f"attention_mask sum:     {int(attention_mask.sum())} / {attention_mask.shape[-1]}")
    with torch.inference_mode():
        logits = runtime_model.module(input_values, attention_mask)
    print(f"raw_logits:             {logits.tolist()}")
    probs = torch.softmax(logits.to(dtype=torch.float32), dim=-1)[0]
    print(f"softmax:                {probs.tolist()}")
    spoof = ssl_spoof_probability_from_logits(torch, logits, runtime_model, config)
    bonafide = 1.0 - spoof
    print(f"mapped spoof=%.4f%%  bonafide=%.4f%%" % (spoof * 100, bonafide * 100))
    print(f"final prediction:       {'spoof' if spoof >= bonafide else 'bonafide'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_path", type=Path)
    parser.add_argument(
        "--branches",
        default="lfcc_cnn_tcn,aasist,ssl_sequence",
        help="Comma-separated branches to run (default: all real branches).",
    )
    args = parser.parse_args()

    if not args.audio_path.exists():
        raise SystemExit(f"Audio file not found: {args.audio_path}")

    print(f"{SEP}\nMULTI-SCOPE INFERENCE DEBUG\n{SEP}")
    _, processed = describe_audio(args.audio_path)

    factory = ModelFactory(settings)
    requested = {b.strip() for b in args.branches.split(",") if b.strip()}

    if "lfcc_cnn_tcn" in requested:
        run_cnn_or_aasist("lfcc_cnn_tcn", processed, factory)
    if "aasist" in requested:
        run_cnn_or_aasist("aasist", processed, factory)
    if "ssl_sequence" in requested:
        run_ssl(processed, factory)

    print(f"\n{SEP}\nDONE\n{SEP}")


if __name__ == "__main__":
    main()
