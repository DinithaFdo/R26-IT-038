"""Colab-vs-backend parity validation against ``xai_handoff_eval_predictions_v1.csv``.

The reference CSV (``research-artifacts/xai/xai_handoff_eval_predictions_v1.csv``)
records, per ``audio_id``, the Colab research pipeline's CNN/AASIST/SSL/Glottal
spoof probabilities, the primary fusion probability, the primary prediction,
and the frozen threshold (0.5519237850482265). It does NOT ship any audio
files -- this repository has none anywhere (confirmed by a repo-wide search
during this integration). This script is therefore validation TOOLING, not a
validation RESULT: it does nothing useful until someone supplies audio files
named ``<audio_id>.<ext>`` for at least a few CSV rows.

Usage:
    python scripts/validate_colab_parity.py --audio-dir /path/to/wavs
    python scripts/validate_colab_parity.py --audio-dir /path/to/wavs --limit 20
    python scripts/validate_colab_parity.py --audio-dir /path/to/wavs --json

Exit status is informational only (this is a developer harness, not a CI
gate): 0 if every locally-available row matched within tolerance, 1 if any
matched row exceeded tolerance, 2 if no matching audio files were found at
all (parity could not be evaluated -- this is the expected result today).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings(
    "ignore",
    message='Field "model_.*" in Settings has conflict with protected namespace',
)

from app.config.settings import Settings
from app.ingestion.audio import inspect_audio_file, preprocess_audio_file
from app.models.factory import ModelFactory
from app.utils.fusion import FusionEngine

DEFAULT_CSV_PATH = (
    Path(__file__).resolve().parent.parent
    .parent
    / "research-artifacts"
    / "xai"
    / "xai_handoff_eval_predictions_v1.csv"
)
DEFAULT_TOLERANCE = 1e-4
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".opus", ".ogg", ".webm")
PRIMARY_BRANCHES = ("lfcc_cnn_tcn", "aasist", "ssl_sequence")
CSV_PROBABILITY_COLUMNS = {
    "lfcc_cnn_tcn": "cnn_spoof_probability",
    "aasist": "aasist_spoof_probability",
    "ssl_sequence": "ssl_spoof_probability",
}


@dataclass
class ReferenceRow:
    audio_id: str
    cnn_spoof_probability: float
    aasist_spoof_probability: float
    ssl_spoof_probability: float
    glottal_spoof_probability: float | None
    target: int
    primary_fusion_probability: float
    primary_prediction: str
    primary_threshold: float
    glottal_role: str


@dataclass
class RowComparison:
    audio_id: str
    audio_path: str
    branch_diffs: dict[str, float] = field(default_factory=dict)
    fusion_diff: float | None = None
    prediction_matches: bool | None = None
    within_tolerance: bool = False
    error: str | None = None


def load_reference_csv(path: Path) -> list[ReferenceRow]:
    rows: list[ReferenceRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            rows.append(
                ReferenceRow(
                    audio_id=record["audio_id"],
                    cnn_spoof_probability=float(record["cnn_spoof_probability"]),
                    aasist_spoof_probability=float(record["aasist_spoof_probability"]),
                    ssl_spoof_probability=float(record["ssl_spoof_probability"]),
                    glottal_spoof_probability=(
                        float(record["glottal_spoof_probability"])
                        if record.get("glottal_spoof_probability")
                        else None
                    ),
                    target=int(record["target"]),
                    primary_fusion_probability=float(record["primary_fusion_probability"]),
                    primary_prediction=record["primary_prediction"],
                    primary_threshold=float(record["primary_threshold"]),
                    glottal_role=record.get("glottal_role", ""),
                )
            )
    return rows


def find_local_audio_file(audio_dir: Path, audio_id: str) -> Path | None:
    for extension in AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{audio_id}{extension}"
        if candidate.is_file():
            return candidate
    return None


def parity_settings() -> Settings:
    """Real CNN + AASIST + SSL, matching the live primary-detector deployment."""

    return Settings(
        _env_file=None,
        model_root_dir="../model_artifacts",
        cnn_model_mode="real",
        aasist_model_mode="real",
        ssl_model_mode="real",
        glottal_model_mode="disabled",
        required_model_branches="lfcc_cnn_tcn,aasist",
        model_device_policy="cpu",
        model_default_device="cpu",
        model_load_strategy="lazy",
    )


def compare_row(
    row: ReferenceRow,
    *,
    audio_path: Path,
    factory: ModelFactory,
    engine: FusionEngine,
    app_settings: Settings,
    tolerance: float,
) -> RowComparison:
    comparison = RowComparison(audio_id=row.audio_id, audio_path=str(audio_path))
    try:
        extension = audio_path.suffix.lower().lstrip(".")
        inspection = inspect_audio_file(audio_path, extension, app_settings)
        processed_audio = preprocess_audio_file(audio_path, extension, app_settings, inspection)

        predictions = [factory.create(branch).predict_safe(processed_audio) for branch in PRIMARY_BRANCHES]
        reference_by_branch = {
            "lfcc_cnn_tcn": row.cnn_spoof_probability,
            "aasist": row.aasist_spoof_probability,
            "ssl_sequence": row.ssl_spoof_probability,
        }
        for branch_name, prediction in zip(PRIMARY_BRANCHES, predictions, strict=True):
            actual = prediction.probabilities.spoof if prediction.probabilities else None
            expected = reference_by_branch[branch_name]
            comparison.branch_diffs[branch_name] = (
                abs(actual - expected) if actual is not None else float("inf")
            )

        fusion = engine.fuse(predictions)
        actual_fusion = fusion.probabilities.spoof if fusion.probabilities else None
        comparison.fusion_diff = (
            abs(actual_fusion - row.primary_fusion_probability)
            if actual_fusion is not None
            else float("inf")
        )
        comparison.prediction_matches = (
            fusion.prediction is not None and fusion.prediction.value == row.primary_prediction
        )
        comparison.within_tolerance = (
            all(diff <= tolerance for diff in comparison.branch_diffs.values())
            and comparison.fusion_diff is not None
            and comparison.fusion_diff <= tolerance
            and bool(comparison.prediction_matches)
        )
    except Exception as error:  # noqa: BLE001 - reported, not raised, per-row
        comparison.error = f"{type(error).__name__}: {error}"
    return comparison


def run(
    *,
    csv_path: Path,
    audio_dir: Path,
    tolerance: float,
    limit: int | None,
) -> dict[str, Any]:
    reference_rows = load_reference_csv(csv_path)
    app_settings = parity_settings()
    factory = ModelFactory(app_settings)
    engine = FusionEngine.from_settings(app_settings)

    matched: list[RowComparison] = []
    unmatched_count = 0
    for row in reference_rows:
        if limit is not None and len(matched) >= limit:
            break
        audio_path = find_local_audio_file(audio_dir, row.audio_id)
        if audio_path is None:
            unmatched_count += 1
            continue
        matched.append(
            compare_row(
                row,
                audio_path=audio_path,
                factory=factory,
                engine=engine,
                app_settings=app_settings,
                tolerance=tolerance,
            )
        )

    passed = [comparison for comparison in matched if comparison.within_tolerance]
    failed = [comparison for comparison in matched if not comparison.within_tolerance]
    return {
        "csv_path": str(csv_path),
        "audio_dir": str(audio_dir),
        "reference_rows": len(reference_rows),
        "rows_with_local_audio": len(matched),
        "rows_without_local_audio": unmatched_count,
        "tolerance": tolerance,
        "passed": len(passed),
        "failed": len(failed),
        "results": [comparison.__dict__ for comparison in matched],
        "parity_evaluated": bool(matched),
    }


def _print_human(summary: dict[str, Any]) -> None:
    print(f"Reference CSV: {summary['csv_path']} ({summary['reference_rows']} rows)")
    print(f"Audio directory: {summary['audio_dir']}")
    print(f"Rows with a local audio file: {summary['rows_with_local_audio']}")
    print(f"Rows without a local audio file: {summary['rows_without_local_audio']}")
    if not summary["parity_evaluated"]:
        print()
        print(
            "NO LOCAL AUDIO FILES MATCHED ANY audio_id IN THE REFERENCE CSV. "
            "Colab parity has NOT been evaluated -- this is expected until "
            "audio files named <audio_id>.<ext> are supplied via --audio-dir. "
            "This tool does not claim parity in the absence of matching audio."
        )
        return
    print(f"Tolerance: {summary['tolerance']}")
    print(f"Passed: {summary['passed']}  Failed: {summary['failed']}")
    for result in summary["results"]:
        status = "PASS" if result["within_tolerance"] else "FAIL"
        if result["error"]:
            print(f"  [{status}] {result['audio_id']}: ERROR {result['error']}")
            continue
        diffs = ", ".join(f"{branch}={diff:.6g}" for branch, diff in result["branch_diffs"].items())
        print(
            f"  [{status}] {result['audio_id']}: {diffs}, "
            f"fusion_diff={result['fusion_diff']:.6g}, "
            f"prediction_matches={result['prediction_matches']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Reference CSV path.")
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing <audio_id>.<ext> files to compare against the CSV.",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--limit", type=int, default=None, help="Max matched rows to evaluate.")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    args = parser.parse_args()

    summary = run(
        csv_path=args.csv,
        audio_dir=args.audio_dir,
        tolerance=args.tolerance,
        limit=args.limit,
    )

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    else:
        _print_human(summary)

    if not summary["parity_evaluated"]:
        sys.exit(2)
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
