"""Colab-vs-backend parity validation for the Glottal branch.

Reference files this script expects (NOT present in this repository as of
this integration -- confirmed by a repo-wide search):

    features/glottal_dev_balanced_2000_selected20.csv
    features/glottal_eval_balanced_4000_selected20.csv
    fusion_exports/glottal_dev_predictions.csv
    fusion_exports/glottal_eval_predictions.csv

Also not present: any audio files. This script is therefore validation
TOOLING, not a validation RESULT -- it does nothing useful until someone
supplies (a) the reference CSVs above and (b) audio files named
``<audio_id>.<ext>`` for at least a few of their rows. It refuses to print a
pass/fail parity verdict when it cannot actually compare against real
reference data; see ``--json`` output's ``parity_evaluated`` field.

For each row present in BOTH the features CSV and the predictions CSV, keyed
by ``audio_id`` (column name configurable via --id-column for CSVs that use a
different key), with a matching local audio file, this script:

    A. extracts the same 20 selected features the backend Glottal branch
       computes (app/models/preprocessing/glottal_v1.py) and diffs each
       against the features CSV's columns for that feature name;
    B. runs the same real Glottal loader/predictor the backend uses
       (app/models/real/glottal_inference.py) and diffs its
       glottal_spoof_probability against the predictions CSV.

The predictions CSV's probability column name is not known in advance (no
reference file to inspect) -- pass --probability-column if it is not one of
the common candidates this script tries by default
(glottal_spoof_probability, spoof_probability, probability).

Usage:
    python scripts/validate_glottal_colab_parity.py \\
        --features-csv /path/to/glottal_dev_balanced_2000_selected20.csv \\
        --predictions-csv /path/to/glottal_dev_predictions.csv \\
        --audio-dir /path/to/wavs

Exit status is informational only (developer harness, not a CI gate): 0 if
every locally-available row matched within tolerance, 1 if any matched row
exceeded tolerance, 2 if reference files and/or matching local audio were not
found (parity could not be evaluated -- the expected result today).
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
from app.models.factory import ModelFactory
from app.models.preprocessing import glottal_v1

DEFAULT_TOLERANCE = 1e-4
#: Feature-level tolerance is looser than the probability tolerance above:
#: DisVoice/parselmouth/librosa involve FFT, filtering, and Praat algorithms
#: whose bit-exact output can differ slightly across library/platform
#: versions even when the pipeline is identical. Report the observed
#: differences explicitly rather than asserting exact equality.
DEFAULT_FEATURE_TOLERANCE = 1e-3
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".opus", ".ogg", ".webm")
CANDIDATE_PROBABILITY_COLUMNS = (
    "glottal_spoof_probability",
    "spoof_probability",
    "probability",
)


@dataclass
class RowComparison:
    audio_id: str
    audio_path: str
    feature_diffs: dict[str, float] = field(default_factory=dict)
    probability_diff: float | None = None
    max_feature_diff: float | None = None
    within_tolerance: bool = False
    error: str | None = None


def load_csv_by_id(path: Path, *, id_column: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            audio_id = record.get(id_column)
            if audio_id:
                rows[audio_id] = record
    return rows


def find_local_audio_file(audio_dir: Path, audio_id: str) -> Path | None:
    for extension in AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{audio_id}{extension}"
        if candidate.is_file():
            return candidate
    return None


def resolve_probability_column(
    predictions_row: dict[str, str], explicit_column: str | None
) -> str | None:
    if explicit_column is not None:
        return explicit_column if explicit_column in predictions_row else None
    for candidate in CANDIDATE_PROBABILITY_COLUMNS:
        if candidate in predictions_row:
            return candidate
    return None


def parity_settings() -> Settings:
    return Settings(
        _env_file=None,
        model_root_dir="../model_artifacts",
        glottal_model_mode="real",
        required_model_branches="",
        model_device_policy="cpu",
        model_default_device="cpu",
        model_load_strategy="lazy",
    )


def compare_row(
    *,
    audio_id: str,
    audio_path: Path,
    features_row: dict[str, str],
    predictions_row: dict[str, str],
    selected_features: list[str],
    model: Any,
    probability_column: str | None,
    feature_tolerance: float,
    probability_tolerance: float,
) -> RowComparison:
    comparison = RowComparison(audio_id=audio_id, audio_path=str(audio_path))
    try:
        cleaned = glottal_v1.load_and_clean_glottal_waveform(audio_path)
        all_features = glottal_v1.extract_all_glottal_features(cleaned)
        vector = glottal_v1.build_selected_feature_vector(all_features, selected_features)

        for name in selected_features:
            if name not in features_row:
                continue
            try:
                expected = float(features_row[name])
            except ValueError:
                continue
            actual = all_features[name]
            comparison.feature_diffs[name] = abs(actual - expected)
        comparison.max_feature_diff = (
            max(comparison.feature_diffs.values()) if comparison.feature_diffs else None
        )

        probabilities = model.predict_proba(vector)[0]
        spoof_index = list(model.classes_).index(1)
        actual_probability = float(probabilities[spoof_index])

        resolved_column = resolve_probability_column(predictions_row, probability_column)
        if resolved_column is not None:
            expected_probability = float(predictions_row[resolved_column])
            comparison.probability_diff = abs(actual_probability - expected_probability)

        feature_ok = comparison.max_feature_diff is None or (
            comparison.max_feature_diff <= feature_tolerance
        )
        probability_ok = comparison.probability_diff is None or (
            comparison.probability_diff <= probability_tolerance
        )
        comparison.within_tolerance = feature_ok and probability_ok and (
            bool(comparison.feature_diffs) or comparison.probability_diff is not None
        )
    except Exception as error:  # noqa: BLE001 - reported, not raised, per-row
        comparison.error = f"{type(error).__name__}: {error}"
    return comparison


def run(
    *,
    features_csv: Path,
    predictions_csv: Path,
    audio_dir: Path,
    id_column: str,
    probability_column: str | None,
    feature_tolerance: float,
    probability_tolerance: float,
    limit: int | None,
) -> dict[str, Any]:
    missing_files = [
        str(path) for path in (features_csv, predictions_csv) if not path.is_file()
    ]
    if missing_files:
        return {
            "features_csv": str(features_csv),
            "predictions_csv": str(predictions_csv),
            "audio_dir": str(audio_dir),
            "missing_reference_files": missing_files,
            "parity_evaluated": False,
        }

    features_by_id = load_csv_by_id(features_csv, id_column=id_column)
    predictions_by_id = load_csv_by_id(predictions_csv, id_column=id_column)
    common_ids = sorted(set(features_by_id) & set(predictions_by_id))

    app_settings = parity_settings()
    factory = ModelFactory(app_settings)
    glottal_model = factory.create("glottal")
    glottal_model.load()
    from app.models.real.glottal_inference import LoadedGlottalBranch

    runtime_model: LoadedGlottalBranch = glottal_model._runtime_model
    selected_features = list(runtime_model.selected_features)
    model = runtime_model.pipeline

    matched: list[RowComparison] = []
    unmatched_count = 0
    for audio_id in common_ids:
        if limit is not None and len(matched) >= limit:
            break
        audio_path = find_local_audio_file(audio_dir, audio_id)
        if audio_path is None:
            unmatched_count += 1
            continue
        matched.append(
            compare_row(
                audio_id=audio_id,
                audio_path=audio_path,
                features_row=features_by_id[audio_id],
                predictions_row=predictions_by_id[audio_id],
                selected_features=selected_features,
                model=model,
                probability_column=probability_column,
                feature_tolerance=feature_tolerance,
                probability_tolerance=probability_tolerance,
            )
        )

    passed = [comparison for comparison in matched if comparison.within_tolerance]
    failed = [comparison for comparison in matched if not comparison.within_tolerance]
    return {
        "features_csv": str(features_csv),
        "predictions_csv": str(predictions_csv),
        "audio_dir": str(audio_dir),
        "reference_rows_in_both_csvs": len(common_ids),
        "rows_with_local_audio": len(matched),
        "rows_without_local_audio": unmatched_count,
        "feature_tolerance": feature_tolerance,
        "probability_tolerance": probability_tolerance,
        "passed": len(passed),
        "failed": len(failed),
        "results": [comparison.__dict__ for comparison in matched],
        "parity_evaluated": bool(matched),
    }


def _print_human(summary: dict[str, Any]) -> None:
    if "missing_reference_files" in summary:
        print("Reference file(s) not found:")
        for path in summary["missing_reference_files"]:
            print(f"  - {path}")
        print()
        print(
            "Colab parity has NOT been evaluated -- this tool does not claim "
            "parity in the absence of the reference CSVs."
        )
        return

    print(f"Features CSV: {summary['features_csv']}")
    print(f"Predictions CSV: {summary['predictions_csv']}")
    print(f"Audio directory: {summary['audio_dir']}")
    print(f"Rows present in both reference CSVs: {summary['reference_rows_in_both_csvs']}")
    print(f"Rows with a local audio file: {summary['rows_with_local_audio']}")
    print(f"Rows without a local audio file: {summary['rows_without_local_audio']}")
    if not summary["parity_evaluated"]:
        print()
        print(
            "NO LOCAL AUDIO FILES MATCHED ANY audio_id COMMON TO BOTH "
            "REFERENCE CSVS. Colab parity has NOT been evaluated -- this is "
            "expected until audio files named <audio_id>.<ext> are supplied "
            "via --audio-dir. This tool does not claim parity in the absence "
            "of matching audio."
        )
        return
    print(
        f"Feature tolerance: {summary['feature_tolerance']}  "
        f"Probability tolerance: {summary['probability_tolerance']}"
    )
    print(f"Passed: {summary['passed']}  Failed: {summary['failed']}")
    for result in summary["results"]:
        status = "PASS" if result["within_tolerance"] else "FAIL"
        if result["error"]:
            print(f"  [{status}] {result['audio_id']}: ERROR {result['error']}")
            continue
        max_diff = result["max_feature_diff"]
        max_diff_str = f"{max_diff:.6g}" if max_diff is not None else "n/a"
        prob_diff = result["probability_diff"]
        prob_diff_str = f"{prob_diff:.6g}" if prob_diff is not None else "n/a"
        print(
            f"  [{status}] {result['audio_id']}: max_feature_diff={max_diff_str}, "
            f"probability_diff={prob_diff_str} "
            f"({len(result['feature_diffs'])} of 20 features compared)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-csv", type=Path, required=True, help=(
        "e.g. features/glottal_dev_balanced_2000_selected20.csv"
    ))
    parser.add_argument("--predictions-csv", type=Path, required=True, help=(
        "e.g. fusion_exports/glottal_dev_predictions.csv"
    ))
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Directory containing <audio_id>.<ext> files to compare against the CSVs.",
    )
    parser.add_argument("--id-column", type=str, default="audio_id")
    parser.add_argument(
        "--probability-column",
        type=str,
        default=None,
        help=(
            "Column in --predictions-csv holding glottal_spoof_probability. "
            f"Auto-detected from {CANDIDATE_PROBABILITY_COLUMNS} if omitted."
        ),
    )
    parser.add_argument("--feature-tolerance", type=float, default=DEFAULT_FEATURE_TOLERANCE)
    parser.add_argument("--probability-tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--limit", type=int, default=None, help="Max matched rows to evaluate.")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    args = parser.parse_args()

    summary = run(
        features_csv=args.features_csv,
        predictions_csv=args.predictions_csv,
        audio_dir=args.audio_dir,
        id_column=args.id_column,
        probability_column=args.probability_column,
        feature_tolerance=args.feature_tolerance,
        probability_tolerance=args.probability_tolerance,
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
