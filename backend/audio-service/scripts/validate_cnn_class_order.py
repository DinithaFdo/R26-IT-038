"""Developer-only CNN label-order sanity check and saturation diagnostic.

The `lfcc_cnn_tcn` (public name `cnn_acoustic`) checkpoint is a bare
``OrderedDict`` state dict with no class mapping, no label map, and no
preprocessing metadata (see `app/models/architectures/cnn.py` and
`app/models/preprocessing/spectral.py`). `CNN_CLASS_ORDER=bonafide_spoof` in
`backend/.env` is therefore an UNBACKED assertion, not something the
checkpoint itself proves -- unlike AASIST-Light V2, whose checkpoint carries
its own attested `class_mapping`/`spoof_class_index` and is validated at load
time (`app/models/real/inference.py::load_aasist_light_v2_checkpoint_strict`).

This script:

  1. Runs the deployed CNN checkpoint through the REAL production ingestion
     and feature-extraction path exactly once per file (same
     `preprocess_audio_file` / `SpectralFeatureExtractor` / model forward pass
     production inference uses -- see `app/models/real/inference.py`), so
     there is no risk of a hand-rolled reimplementation drifting from what the
     server actually does.
  2. Reads the raw 2-logit output and interprets it BOTH ways (bonafide=0/
     spoof=1, and spoof=0/bonafide=1) from that SAME single forward pass --
     never loads two differently-configured models.
  3. If a labelled manifest is supplied, scores both interpretations against
     the true labels and reports per-mapping accuracy/balanced-accuracy/
     recall/confusion-matrix, explicitly captioned as a SMALL LABELLED SANITY
     VALIDATION, never as a benchmark.
  4. Reports CNN saturation (how often P >= 0.99 or P <= 0.01) across whatever
     inputs were scored -- labelled files when given, otherwise a small
     deterministic battery of synthetic probes reused from
     `tests/test_real_model_discrimination.py`, captioned as diagnostics only,
     never as accuracy evidence.
  5. Prints ONE explicit recommendation line at the end:
       RECOMMEND KEEP bonafide_spoof
       RECOMMEND INVESTIGATE spoof_bonafide
       INSUFFICIENT EVIDENCE
     and never writes `CNN_CLASS_ORDER` or any other configuration anywhere.
     Changing the deployed setting is a separate, explicit task.

Manifest format (CSV, header required):

    audio_path,true_label
    samples/human_01.wav,bonafide
    samples/elevenlabs_01.wav,spoof

``audio_path`` may be absolute or relative to the manifest file's own
directory. ``true_label`` must be exactly ``bonafide`` or ``spoof``.

Usage:
    python scripts/validate_cnn_class_order.py --manifest path/to/manifest.csv
    python scripts/validate_cnn_class_order.py --manifest path/to/manifest.csv --json
    python scripts/validate_cnn_class_order.py --no-synthetic-probes --manifest ...
    python scripts/validate_cnn_class_order.py   # synthetic saturation probes only
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import sys
import warnings

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings(
    "ignore",
    message='Field "model_.*" in Settings has conflict with protected namespace',
)

from app.config.settings import Settings
from app.ingestion.audio import ProcessedAudio, inspect_audio_file, preprocess_audio_file
from app.models.factory import ModelFactory
from app.models.real.inference import _prepare_input
from app.models.torch_support import require_torch

BACKEND_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = BACKEND_ROOT / ".env"

VALID_LABELS = {"bonafide", "spoof"}
DEFAULT_DECISION_MARGIN = 0.15
DEFAULT_MIN_SAMPLES_PER_CLASS = 3


@dataclass
class ManifestRow:
    audio_path: Path
    true_label: str


@dataclass
class SampleResult:
    name: str
    true_label: str | None
    logits: list[float]
    p0: float
    p1: float
    max_probability: float
    entropy: float
    saturated: bool
    mapping_a_prediction: str
    mapping_a_spoof_probability: float
    mapping_a_correct: bool | None
    mapping_b_prediction: str
    mapping_b_spoof_probability: float
    mapping_b_correct: bool | None


@dataclass
class MappingMetrics:
    name: str
    label_convention: str
    correct: int = 0
    total: int = 0
    confusion: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            "bonafide": {"bonafide": 0, "spoof": 0},
            "spoof": {"bonafide": 0, "spoof": 0},
        }
    )

    @property
    def accuracy(self) -> float | None:
        if self.total == 0:
            return None
        return self.correct / self.total

    @property
    def bonafide_recall(self) -> float | None:
        row = self.confusion["bonafide"]
        denom = row["bonafide"] + row["spoof"]
        return row["bonafide"] / denom if denom else None

    @property
    def spoof_recall(self) -> float | None:
        row = self.confusion["spoof"]
        denom = row["bonafide"] + row["spoof"]
        return row["spoof"] / denom if denom else None

    @property
    def balanced_accuracy(self) -> float | None:
        recalls = [r for r in (self.bonafide_recall, self.spoof_recall) if r is not None]
        return sum(recalls) / len(recalls) if recalls else None

    def record(self, true_label: str, predicted_label: str) -> None:
        self.total += 1
        if predicted_label == true_label:
            self.correct += 1
        self.confusion[true_label][predicted_label] += 1


def deployment_settings() -> Settings:
    """The same Settings a locally started server would build from
    `backend/.env`. Read-only: this script never writes configuration."""

    return Settings(_env_file=str(ENV_FILE) if ENV_FILE.is_file() else None)


def load_manifest(manifest_path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) < {
            "audio_path",
            "true_label",
        }:
            raise ValueError(
                "Manifest must be a CSV with header 'audio_path,true_label'."
            )
        for line_number, record in enumerate(reader, start=2):
            raw_path = (record.get("audio_path") or "").strip()
            label = (record.get("true_label") or "").strip().lower()
            if not raw_path:
                continue
            if label not in VALID_LABELS:
                raise ValueError(
                    f"Manifest line {line_number}: true_label must be "
                    f"'bonafide' or 'spoof', got {label!r}."
                )
            candidate = Path(raw_path).expanduser()
            resolved = (
                candidate if candidate.is_absolute() else manifest_path.parent / candidate
            )
            rows.append(ManifestRow(audio_path=resolved, true_label=label))
    return rows


def synthetic_probes() -> dict[str, np.ndarray]:
    """The same acoustically-diverse, unlabelled probes
    `tests/test_real_model_discrimination.py` uses. Saturation diagnostics
    only -- never scored against a true label, never treated as accuracy
    evidence."""

    sample_rate = 16000
    time = np.arange(sample_rate * 3) / sample_rate
    generator = np.random.RandomState(0)
    return {
        "synthetic_harmonic": (
            0.30 * np.sin(2 * np.pi * 140 * time)
            + 0.15 * np.sin(2 * np.pi * 280 * time)
            + 0.02 * generator.randn(time.size)
        ).astype(np.float32),
        "synthetic_broadband_noise": (0.20 * generator.randn(time.size)).astype(
            np.float32
        ),
        "synthetic_pure_tone": (0.50 * np.sin(2 * np.pi * 440 * time)).astype(
            np.float32
        ),
        "synthetic_chirp": (
            0.40 * np.sin(2 * np.pi * (100 + 300 * time / 3) * time)
        ).astype(np.float32),
    }


def _entropy(p0: float, p1: float) -> float:
    values = [p for p in (p0, p1) if p > 0.0]
    return -sum(p * math.log(p, 2) for p in values)


def score_processed_audio(
    processed_audio: ProcessedAudio,
    *,
    model,
    name: str,
    true_label: str | None,
) -> SampleResult:
    """One forward pass; both class-order interpretations read from it."""

    torch = require_torch()
    runtime_model = model._runtime_model
    config = model._config

    model_input = _prepare_input(torch, processed_audio, runtime_model, config)
    with torch.inference_mode():
        logits = runtime_model.module(model_input)
    if list(logits.shape) != [1, 2]:
        raise ValueError(f"CNN logits shape {list(logits.shape)} != [1, 2] for {name}.")
    probabilities = torch.softmax(logits.to(dtype=torch.float32), dim=-1)[0]
    p0 = float(probabilities[0])
    p1 = float(probabilities[1])
    max_probability = max(p0, p1)

    # Interpretation A: bonafide=0, spoof=1 -- the configured CNN_CLASS_ORDER.
    spoof_a = p1
    prediction_a = "spoof" if spoof_a >= 0.5 else "bonafide"
    # Interpretation B: spoof=0, bonafide=1 -- the opposite convention.
    spoof_b = p0
    prediction_b = "spoof" if spoof_b >= 0.5 else "bonafide"

    return SampleResult(
        name=name,
        true_label=true_label,
        logits=[float(x) for x in logits.detach().cpu().reshape(-1).tolist()],
        p0=p0,
        p1=p1,
        max_probability=max_probability,
        entropy=_entropy(p0, p1),
        # P0 + P1 == 1, so max(P0, P1) >= 0.99 already implies the other
        # class's probability is <= 0.01 -- one condition covers both.
        saturated=max_probability >= 0.99,
        mapping_a_prediction=prediction_a,
        mapping_a_spoof_probability=spoof_a,
        mapping_a_correct=(prediction_a == true_label) if true_label else None,
        mapping_b_prediction=prediction_b,
        mapping_b_spoof_probability=spoof_b,
        mapping_b_correct=(prediction_b == true_label) if true_label else None,
    )


def run(
    *,
    manifest_rows: list[ManifestRow],
    include_synthetic_probes: bool,
) -> dict:
    settings = deployment_settings()
    config = ModelFactory(settings).branch_config("lfcc_cnn_tcn")
    model = ModelFactory(settings).create("lfcc_cnn_tcn")
    model.load()

    checkpoint_report = {
        "configured_path": settings.cnn_model_path,
        "resolved_path": (
            str(config.checkpoint.safe_path) if config.checkpoint.safe_path else None
        ),
        "filename": config.checkpoint.filename,
        "sha256": config.checkpoint.sha256,
        "size_bytes": config.checkpoint.size_bytes,
        "modified_at": (
            config.checkpoint.modified_at.isoformat()
            if config.checkpoint.modified_at
            else None
        ),
        "configured_class_order": settings.cnn_class_order,
        "note": (
            "The checkpoint itself contains no class mapping metadata -- "
            "configured_class_order is an operator assertion, not something "
            "the checkpoint proves. Contrast with AASIST-Light V2, whose "
            "checkpoint attests its own class_mapping and is rejected at "
            "load time if it disagrees."
        ),
    }

    results: list[SampleResult] = []
    errors: list[dict[str, str]] = []

    for row in manifest_rows:
        try:
            extension = row.audio_path.suffix.lower().lstrip(".")
            inspection = inspect_audio_file(row.audio_path, extension, settings)
            processed = preprocess_audio_file(
                row.audio_path, extension, settings, inspection
            )
            results.append(
                score_processed_audio(
                    processed,
                    model=model,
                    name=row.audio_path.name,
                    true_label=row.true_label,
                )
            )
        except Exception as error:  # noqa: BLE001 - reported, not raised
            errors.append({"file": str(row.audio_path), "error": str(error)})

    probe_results: list[SampleResult] = []
    if include_synthetic_probes:
        for probe_name, waveform in synthetic_probes().items():
            processed = ProcessedAudio(
                waveform=waveform,
                sample_rate=16000,
                original_sample_rate=16000,
                original_channels=1,
                duration_seconds=float(waveform.size / 16000),
                was_resampled=False,
                was_converted_to_mono=False,
                normalisation_applied=True,
                peak_amplitude=float(np.abs(waveform).max()),
                rms_energy=float(np.sqrt((waveform**2).mean())),
                unnormalised_waveform=np.ascontiguousarray(waveform.copy()),
            )
            probe_results.append(
                score_processed_audio(
                    processed, model=model, name=probe_name, true_label=None
                )
            )

    mapping_a = MappingMetrics(name="Mapping A", label_convention="bonafide=0, spoof=1")
    mapping_b = MappingMetrics(name="Mapping B", label_convention="spoof=0, bonafide=1")
    for sample in results:
        assert sample.true_label is not None
        mapping_a.record(sample.true_label, sample.mapping_a_prediction)
        mapping_b.record(sample.true_label, sample.mapping_b_prediction)

    all_scored = results + probe_results
    saturated_count = sum(1 for sample in all_scored if sample.saturated)

    return {
        "checkpoint": checkpoint_report,
        "labelled_samples": [_sample_dict(s) for s in results],
        "synthetic_probe_samples": [_sample_dict(s) for s in probe_results],
        "errors": errors,
        "mapping_a": _metrics_dict(mapping_a),
        "mapping_b": _metrics_dict(mapping_b),
        "saturation": {
            "scored_count": len(all_scored),
            "saturated_count": saturated_count,
            "saturated_fraction": (
                saturated_count / len(all_scored) if all_scored else None
            ),
            "threshold_note": "saturated = max(P0, P1) >= 0.99",
        },
        "recommendation": _recommendation(
            mapping_a,
            mapping_b,
            decision_margin=DEFAULT_DECISION_MARGIN,
            min_samples_per_class=DEFAULT_MIN_SAMPLES_PER_CLASS,
        ),
    }


def _sample_dict(sample: SampleResult) -> dict:
    return {
        "filename": sample.name,
        "true_label": sample.true_label,
        "logits": sample.logits,
        "softmax_index_0": sample.p0,
        "softmax_index_1": sample.p1,
        "max_probability": sample.max_probability,
        "entropy_bits": sample.entropy,
        "saturated": sample.saturated,
        "mapping_a_prediction": sample.mapping_a_prediction,
        "mapping_a_spoof_probability": sample.mapping_a_spoof_probability,
        "mapping_a_correct": sample.mapping_a_correct,
        "mapping_b_prediction": sample.mapping_b_prediction,
        "mapping_b_spoof_probability": sample.mapping_b_spoof_probability,
        "mapping_b_correct": sample.mapping_b_correct,
    }


def _metrics_dict(metrics: MappingMetrics) -> dict:
    return {
        "name": metrics.name,
        "label_convention": metrics.label_convention,
        "total_samples": metrics.total,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "bonafide_recall": metrics.bonafide_recall,
        "spoof_recall": metrics.spoof_recall,
        "confusion_matrix": metrics.confusion,
    }


def _recommendation(
    mapping_a: MappingMetrics,
    mapping_b: MappingMetrics,
    *,
    decision_margin: float,
    min_samples_per_class: int,
) -> dict:
    bonafide_count = mapping_a.confusion["bonafide"]["bonafide"] + mapping_a.confusion[
        "bonafide"
    ]["spoof"]
    spoof_count = mapping_a.confusion["spoof"]["bonafide"] + mapping_a.confusion["spoof"][
        "spoof"
    ]

    if (
        bonafide_count < min_samples_per_class
        or spoof_count < min_samples_per_class
        or mapping_a.accuracy is None
        or mapping_b.accuracy is None
    ):
        return {
            "verdict": "INSUFFICIENT EVIDENCE",
            "reason": (
                f"Need at least {min_samples_per_class} bonafide and "
                f"{min_samples_per_class} spoof labelled samples to compare "
                f"mappings; got bonafide={bonafide_count}, spoof={spoof_count}."
            ),
        }

    gap = mapping_a.accuracy - mapping_b.accuracy
    if gap >= decision_margin:
        return {
            "verdict": "RECOMMEND KEEP bonafide_spoof",
            "reason": (
                f"Mapping A (bonafide=0, spoof=1 -- current CNN_CLASS_ORDER) "
                f"accuracy {mapping_a.accuracy:.3f} exceeds Mapping B "
                f"{mapping_b.accuracy:.3f} by {gap:.3f} (>= margin "
                f"{decision_margin})."
            ),
        }
    if -gap >= decision_margin:
        return {
            "verdict": "RECOMMEND INVESTIGATE spoof_bonafide",
            "reason": (
                f"Mapping B (spoof=0, bonafide=1) accuracy {mapping_b.accuracy:.3f} "
                f"exceeds Mapping A (current CNN_CLASS_ORDER) "
                f"{mapping_a.accuracy:.3f} by {-gap:.3f} (>= margin "
                f"{decision_margin}). This does NOT change CNN_CLASS_ORDER "
                "automatically -- that is a separate, explicit task."
            ),
        }
    return {
        "verdict": "INSUFFICIENT EVIDENCE",
        "reason": (
            f"Mapping A accuracy {mapping_a.accuracy:.3f} vs Mapping B "
            f"{mapping_b.accuracy:.3f}: gap {abs(gap):.3f} is below the "
            f"{decision_margin} decision margin. Collect more labelled "
            "samples before drawing a conclusion."
        ),
    }


def _print_human(report: dict) -> None:
    checkpoint = report["checkpoint"]
    print("=" * 78)
    print("CNN checkpoint identity (production-equivalent resolution)")
    print("=" * 78)
    for key in (
        "configured_path",
        "resolved_path",
        "filename",
        "sha256",
        "size_bytes",
        "modified_at",
        "configured_class_order",
    ):
        print(f"  {key}: {checkpoint[key]}")
    print(f"  NOTE: {checkpoint['note']}")

    if report["errors"]:
        print()
        print("Errors while scoring manifest rows:")
        for error in report["errors"]:
            print(f"  {error['file']}: {error['error']}")

    if report["labelled_samples"]:
        print()
        print("=" * 78)
        print(
            "SMALL LABELLED SANITY VALIDATION -- not a benchmark evaluation "
            f"({len(report['labelled_samples'])} sample(s))"
        )
        print("=" * 78)
        header = (
            f"{'file':30} {'true':9} {'P0':>8} {'P1':>8} "
            f"{'A_pred':9} {'A_ok':5} {'B_pred':9} {'B_ok':5}"
        )
        print(header)
        for sample in report["labelled_samples"]:
            print(
                f"{sample['filename']:30} {str(sample['true_label']):9} "
                f"{sample['softmax_index_0']:8.4f} {sample['softmax_index_1']:8.4f} "
                f"{sample['mapping_a_prediction']:9} "
                f"{str(sample['mapping_a_correct']):5} "
                f"{sample['mapping_b_prediction']:9} "
                f"{str(sample['mapping_b_correct']):5}"
            )

        for key in ("mapping_a", "mapping_b"):
            metrics = report[key]
            print()
            print(f"{metrics['name']} ({metrics['label_convention']}):")
            print(f"  samples: {metrics['total_samples']}")
            print(f"  accuracy: {_fmt(metrics['accuracy'])}")
            print(f"  balanced_accuracy: {_fmt(metrics['balanced_accuracy'])}")
            print(f"  bonafide_recall: {_fmt(metrics['bonafide_recall'])}")
            print(f"  spoof_recall: {_fmt(metrics['spoof_recall'])}")
            print(f"  confusion_matrix (rows=true, cols=predicted): "
                  f"{metrics['confusion_matrix']}")
    else:
        print()
        print(
            "No labelled manifest supplied (or it scored zero rows) -- "
            "mapping accuracy cannot be computed. Pass --manifest to enable "
            "the label-order comparison."
        )

    if report["synthetic_probe_samples"]:
        print()
        print("=" * 78)
        print(
            "SATURATION DIAGNOSTIC (synthetic probes) -- NOT accuracy evidence, "
            "unlabelled signals only"
        )
        print("=" * 78)
        for sample in report["synthetic_probe_samples"]:
            print(
                f"{sample['filename']:26} P0={sample['softmax_index_0']:.6f} "
                f"P1={sample['softmax_index_1']:.6f} "
                f"max={sample['max_probability']:.6f} "
                f"entropy_bits={sample['entropy_bits']:.4f} "
                f"saturated={sample['saturated']}"
            )

    saturation = report["saturation"]
    print()
    print("=" * 78)
    print("Saturation summary (across all scored inputs, real + synthetic)")
    print("=" * 78)
    print(f"  scored: {saturation['scored_count']}")
    print(f"  saturated (max prob >= 0.99): {saturation['saturated_count']}")
    print(f"  saturated_fraction: {_fmt(saturation['saturated_fraction'])}")

    recommendation = report["recommendation"]
    print()
    print("=" * 78)
    print(f"RECOMMENDATION: {recommendation['verdict']}")
    print("=" * 78)
    print(recommendation["reason"])
    print(
        "\nThis script never writes CNN_CLASS_ORDER or any other "
        "configuration. Acting on a recommendation is a separate, explicit "
        "task."
    )


def _fmt(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV with header 'audio_path,true_label' (bonafide|spoof).",
    )
    parser.add_argument(
        "--no-synthetic-probes",
        dest="synthetic_probes",
        action="store_false",
        help="Skip the synthetic saturation-only probe battery.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument("--output", type=Path, help="Optional JSON output file.")
    parser.set_defaults(synthetic_probes=True)
    args = parser.parse_args()

    manifest_rows: list[ManifestRow] = []
    if args.manifest is not None:
        manifest_rows = load_manifest(args.manifest)
        if not manifest_rows:
            print(f"Manifest {args.manifest} contained no usable rows.", file=sys.stderr)

    report = run(
        manifest_rows=manifest_rows,
        include_synthetic_probes=args.synthetic_probes,
    )

    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2, default=str))

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        _print_human(report)


if __name__ == "__main__":
    main()
