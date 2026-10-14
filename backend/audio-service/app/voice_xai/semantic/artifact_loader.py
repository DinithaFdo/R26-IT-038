"""Validate model bundles and load the active semantic inference dependencies.

Verify file hashes, feature order, regression objective, and model provenance
before loading the fitted median imputer, XGBoost regressor, and Tree SHAP explainer.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
from threading import RLock
from typing import Any
import numpy as np
from app.voice_xai.semantic.schema import (
    FEATURE_NAMES, FEATURE_EXTRACTION_VERSION, FEATURE_SCHEMA_VERSION,
    PREPROCESSING_CONTRACT, SELECTED20,
)

BUNDLE_VERSION = "semantic-surrogate-bundle-v2"
REQUIRED_FILES = {"surrogate_xgboost.json", "feature_columns.json", "imputer.joblib", "training_metadata.json"}
OPTIONAL_FILES = {"probability_calibrator.joblib", "teacher_fusion.json", "teacher_assets.json", "selected_glottal_features_20.json"}
_EXPLAINER_LOCK = RLock()

class SemanticArtifactCompatibilityError(ValueError):
    pass

@dataclass(frozen=True)
class LoadedSemanticArtifacts:
    model_version: str
    booster: Any
    shap_explainer: Any
    imputer: Any
    metadata: dict
    threshold: float
    directory: Path

    def fidelity(self):
        """Read the current model's metric without gating runtime on its value."""
        metric = self.metadata.get("primary_test_fidelity", {})
        if not isinstance(metric, dict):
            return None
        value, count = metric.get("r2"), metric.get("n")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(value) or value > 1 or isinstance(count, bool) or not isinstance(count, int) or count < 1:
            return None
        if self.metadata.get("test_rows", count) != count:
            return None
        return float(value), count

def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()

def read_json(path: Path):
    try:
        # Serialized metadata can contain NaN for unavailable optional metrics.
        # Validate inference fields below; fidelity() treats such metrics as
        # unavailable rather than preventing an otherwise valid model load.
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise SemanticArtifactCompatibilityError(f"Cannot read {path.name}.") from error

def resolve_bundle(root: Path) -> Path:
    root = root.resolve()
    if (root / "bundle_manifest.json").is_file():
        return root
    pointer = read_json(root / "active.json")
    if not isinstance(pointer, dict):
        raise SemanticArtifactCompatibilityError("Invalid active bundle reference.")
    name = pointer.get("bundle")
    if not isinstance(name, str) or not name or Path(name).name != name or name in {".", ".."}:
        raise SemanticArtifactCompatibilityError("Invalid active bundle reference.")
    directory = (root / "bundles" / name).resolve()
    if not directory.is_relative_to(root / "bundles"):
        raise SemanticArtifactCompatibilityError("Active bundle escaped its root.")
    if file_hash(directory / "bundle_manifest.json") != pointer.get("manifest_sha256"):
        raise SemanticArtifactCompatibilityError("Active bundle manifest hash mismatch.")
    return directory

def validate_training_contract(metadata):
    """Require explicit evidence of training with the current extraction contract."""
    if (not isinstance(metadata, dict)
            or metadata.get("extractor_version") != FEATURE_EXTRACTION_VERSION
            or metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
            or metadata.get("preprocessing") != PREPROCESSING_CONTRACT):
        raise SemanticArtifactCompatibilityError(
            "Re-extract training features and retrain the surrogate with the shared "
            "full-clip preprocessing contract before importing this model. "
            "Training metadata must declare the matching extractor_version, "
            "feature_schema_version, and preprocessing settings."
        )


def validate_bundle(directory: Path):
    manifest = read_json(directory / "bundle_manifest.json")
    if (not isinstance(manifest, dict) or manifest.get("bundle_version") != BUNDLE_VERSION
            or manifest.get("extractor_version") != FEATURE_EXTRACTION_VERSION
            or manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
            or manifest.get("prediction_transform") != "clip_raw"
            or not isinstance(manifest.get("model_version"), str) or not manifest["model_version"]):
        raise SemanticArtifactCompatibilityError(
            "Unsupported semantic model bundle contract. Full-clip extraction requires "
            "a retrained v2 bundle; import it and restart the backend."
        )
    files = manifest.get("files", {})
    if not isinstance(files, dict) or not REQUIRED_FILES <= files.keys() or not files.keys() <= REQUIRED_FILES | OPTIONAL_FILES:
        raise SemanticArtifactCompatibilityError("Incomplete or unknown bundle files.")
    for name, expected in files.items():
        path = directory / name
        if not path.is_file() or file_hash(path) != expected:
            raise SemanticArtifactCompatibilityError(f"Artifact hash mismatch: {name}.")
    columns = read_json(directory / "feature_columns.json")
    if not isinstance(columns, list) or tuple(columns) != FEATURE_NAMES:
        raise SemanticArtifactCompatibilityError("Expected the exact semantic model 165-feature order.")
    metadata = read_json(directory / "training_metadata.json")
    validate_training_contract(metadata)
    if (not isinstance(metadata, dict) or metadata.get("model_type") != "XGBRegressor"
            or metadata.get("target") != "teacher_fused_spoof_probability"
            or metadata.get("ground_truth_used_as_training_target") is not False
            or metadata.get("feature_count") != len(FEATURE_NAMES)):
        raise SemanticArtifactCompatibilityError("Metadata must describe teacher-probability regression over 165 features.")
    threshold = metadata.get("decision_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise SemanticArtifactCompatibilityError("Metadata requires a valid decision threshold.")
    model = read_json(directory / "surrogate_xgboost.json")
    try:
        learner = model["learner"]
        valid = learner["objective"]["name"] == "reg:squarederror" and int(learner["learner_model_param"]["num_feature"]) == len(FEATURE_NAMES)
    except (KeyError, TypeError, ValueError) as error:
        raise SemanticArtifactCompatibilityError("Invalid XGBoost export.") from error
    if not valid:
        raise SemanticArtifactCompatibilityError("Expected a 165-feature XGBRegressor with the reg:squarederror objective.")
    if "selected_glottal_features_20.json" in files:
        selected = read_json(directory / "selected_glottal_features_20.json")
        if tuple(selected.get("selected_features", ())) != SELECTED20:
            raise SemanticArtifactCompatibilityError("Selected-20 feature contract mismatch.")
    if "teacher_assets.json" in files:
        assets = read_json(directory / "teacher_assets.json")
        hashes = assets.get("hashes", {})
        fingerprint = sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
        if (assets.get("teacher_fingerprint") != fingerprint
                or tuple(hashes.get("feature_schema", ())) != FEATURE_NAMES
                or any(values != [fingerprint] for values in metadata.get("teacher_fingerprints", {}).values())):
            raise SemanticArtifactCompatibilityError("Teacher provenance mismatch.")
        if "teacher_fusion.json" in files and hashes.get("fusion") != read_json(directory / "teacher_fusion.json"):
            raise SemanticArtifactCompatibilityError("Teacher fusion provenance mismatch.")
    return manifest, metadata, float(threshold)

def _build_tree_explainer(shap, booster):
    # Avoid SHAP's unrelated optional Transformers/TensorFlow probe.
    module = shap.explainers._explainer
    with _EXPLAINER_LOCK:
        original = module.is_transformers_lm
        module.is_transformers_lm = lambda _model: False
        try:
            return shap.TreeExplainer(booster, model_output="raw", feature_perturbation="tree_path_dependent")
        finally:
            module.is_transformers_lm = original

def load_production_semantic_artifacts(bundle_dir: Path) -> LoadedSemanticArtifacts:
    directory = resolve_bundle(bundle_dir)
    manifest, metadata, threshold = validate_bundle(directory)
    try:
        import joblib
        import shap
        import xgboost
        from sklearn.impute import SimpleImputer
        imputer = joblib.load(directory / "imputer.joblib")
        statistics = np.asarray(getattr(imputer, "statistics_", ()))
        if (not isinstance(imputer, SimpleImputer) or imputer.strategy != "median"
                or statistics.shape != (len(FEATURE_NAMES),) or not np.isfinite(statistics).all()
                or int(getattr(imputer, "n_features_in_", 0)) != len(FEATURE_NAMES)):
            raise SemanticArtifactCompatibilityError("Expected a fitted 165-feature median imputer.")
        if hasattr(imputer, "feature_names_in_") and tuple(imputer.feature_names_in_) != FEATURE_NAMES:
            raise SemanticArtifactCompatibilityError("Imputer feature order mismatch.")
        booster = xgboost.Booster(model_file=bytearray((directory / "surrogate_xgboost.json").read_bytes()))
        explainer = _build_tree_explainer(shap, booster)
    except SemanticArtifactCompatibilityError:
        raise
    except Exception as error:
        raise SemanticArtifactCompatibilityError("Semantic model dependencies or artifacts could not be loaded.") from error
    return LoadedSemanticArtifacts(manifest["model_version"], booster, explainer, imputer, metadata, threshold, directory)
