"""Validate trusted model exports and atomically activate an immutable bundle.

Stage the exported files, verify their contracts and prediction/SHAP output,
and publish the version before updating the active model reference.

Run from audio-service:
python -m app.voice_xai.semantic.import_bundle --source PATH_TO_MODEL_EXPORTS
"""
import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from app.voice_xai.semantic.artifact_loader import (
    BUNDLE_VERSION, OPTIONAL_FILES, REQUIRED_FILES,
    SemanticArtifactCompatibilityError, file_hash,
    load_production_semantic_artifacts, read_json, validate_training_contract,
)
from app.voice_xai.semantic.schema import FEATURE_EXTRACTION_VERSION, FEATURE_SCHEMA_VERSION


def import_bundle(source: Path, destination: Path, *, imputer_name="imputer.joblib") -> str:
    """Only import user-trusted joblib exports; never accept public HTTP uploads."""
    source, destination = source.resolve(), destination.resolve()
    if Path(imputer_name).name != imputer_name:
        raise SemanticArtifactCompatibilityError("Imputer must be a filename inside the source folder.")
    for name in REQUIRED_FILES:
        original = imputer_name if name == "imputer.joblib" else name
        if not (source / original).is_file():
            raise SemanticArtifactCompatibilityError(f"Missing required model export: {original}.")
    # Validate the source declaration before creating or activating any bundle.
    # A new manifest must never relabel exports trained with an older extractor.
    validate_training_contract(read_json(source / "training_metadata.json"))
    destination.mkdir(parents=True, exist_ok=True)
    versions = destination / "bundles"
    versions.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".import-", dir=destination) as staging:
        stage = Path(staging)
        names = set(REQUIRED_FILES)
        for name in REQUIRED_FILES:
            shutil.copyfile(source / (imputer_name if name == "imputer.joblib" else name), stage / name)
        for name in OPTIONAL_FILES:
            path = source / name
            if name == "teacher_assets.json" and not path.exists():
                candidates = list(source.glob("teacher_assets_*.json"))
                if len(candidates) == 1:
                    path = candidates[0]
            if path.is_file():
                shutil.copyfile(path, stage / name)
                names.add(name)
        hashes = {name: file_hash(stage / name) for name in sorted(names)}
        identity = sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
        metadata = read_json(stage / "training_metadata.json")
        if not isinstance(metadata, dict):
            raise SemanticArtifactCompatibilityError("Training metadata must be a JSON object.")
        experiment = re.sub(r"[^A-Za-z0-9_-]", "_", str(metadata.get("experiment_name", "surrogate")))[:80]
        version = f"{experiment}-{identity}"
        manifest = {
            "bundle_version": BUNDLE_VERSION,
            "model_version": version,
            "extractor_version": FEATURE_EXTRACTION_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "prediction_transform": "clip_raw",
            "files": hashes,
        }
        (stage / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # Model, imputer, provenance, SHAP and median-vector inference must
        # succeed before active.json is touched. R² is never a runtime gate.
        artifacts = load_production_semantic_artifacts(stage)
        from app.voice_xai.semantic.features import FeatureExtractionResult
        from app.voice_xai.semantic.schema import FEATURE_NAMES
        from app.voice_xai.semantic.service import XgboostShapEvidenceProvider
        XgboostShapEvidenceProvider(artifacts).extract_features(FeatureExtractionResult(
            FEATURE_NAMES, artifacts.imputer.statistics_.copy(), 16000, 6.0,
            FEATURE_EXTRACTION_VERSION, {},
        ))
        target = versions / identity
        if target.exists():
            if file_hash(target / "bundle_manifest.json") != file_hash(stage / "bundle_manifest.json"):
                raise SemanticArtifactCompatibilityError("Existing bundle identity collision.")
            load_production_semantic_artifacts(target)
        else:
            # Rename within the same filesystem publishes a complete immutable bundle.
            stage.rename(target)
        pointer = {"bundle": identity, "manifest_sha256": file_hash(target / "bundle_manifest.json")}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination,
                                         prefix=".active-", suffix=".json", delete=False) as handle:
            json.dump(pointer, handle)
            temporary = Path(handle.name)
        try:
            os.replace(temporary, destination / "active.json")
        finally:
            temporary.unlink(missing_ok=True)
    return version


def main():
    parser = argparse.ArgumentParser(description="Validate and activate a semantic surrogate model bundle.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--imputer", default="imputer.joblib")
    args = parser.parse_args()
    if args.destination is None:
        from app.config.settings import settings
        args.destination = settings.resolve_xai_artifact_path(settings.xai_semantic_bundle_dir)
    try:
        version = import_bundle(args.source, args.destination, imputer_name=args.imputer)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Import failed; active model was not changed: {error}\n")
    print(f"Activated {version}. Restart the backend to load this model.")


if __name__ == "__main__":
    main()
