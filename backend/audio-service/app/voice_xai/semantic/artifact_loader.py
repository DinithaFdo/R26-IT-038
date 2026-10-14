"""Validate and load semantic model exports from one fixed directory.

Check feature order, preprocessing, regression objective, and optional teacher
provenance before loading the imputer, regressor, and Tree SHAP explainer.
"""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re
from threading import RLock
from typing import Any

import numpy as np

from app.voice_xai.semantic.schema import (
    FEATURE_NAMES, FEATURE_EXTRACTION_VERSION, FEATURE_SCHEMA_VERSION,
    PREPROCESSING_CONTRACT, SELECTED20,
)

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
    fingerprint: str

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


def _read_model_files(directory: Path) -> dict[str, bytes]:
    """Snapshot the required exports and any known optional companions."""
    files = {}
    try:
        for name in sorted(REQUIRED_FILES | OPTIONAL_FILES):
            path = directory / name
            if path.is_file():
                files[name] = path.read_bytes()
            elif name in REQUIRED_FILES:
                raise SemanticArtifactCompatibilityError(f"Missing required model file: {name}.")
        if "teacher_assets.json" not in files:
            candidates = sorted(directory.glob("teacher_assets_*.json"))
            if len(candidates) > 1:
                raise SemanticArtifactCompatibilityError("Keep only the matching teacher_assets file in the model directory.")
            if candidates:
                files["teacher_assets.json"] = candidates[0].read_bytes()
    except OSError as error:
        raise SemanticArtifactCompatibilityError("Semantic model files could not be read.") from error
    return files


def _files_fingerprint(files: dict[str, bytes]) -> str:
    hashes = {name: sha256(data).hexdigest() for name, data in sorted(files.items())}
    return sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def model_fingerprint(directory: Path) -> str:
    """Identify model contents automatically without a manifest or pointer file."""
    return _files_fingerprint(_read_model_files(directory.resolve()))


def _read_json(files: dict[str, bytes], name: str):
    try:
        # Optional evaluation metrics may serialize as NaN; fidelity() handles
        # those as unavailable. Inference fields are validated separately.
        return json.loads(files[name])
    except (KeyError, ValueError) as error:
        raise SemanticArtifactCompatibilityError(f"Cannot read {name}.") from error


def _validate_training_contract(metadata):
    if (not isinstance(metadata, dict)
            or metadata.get("extractor_version") != FEATURE_EXTRACTION_VERSION
            or metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION
            or metadata.get("preprocessing") != PREPROCESSING_CONTRACT):
        raise SemanticArtifactCompatibilityError(
            "Re-extract training features and retrain the surrogate with the shared "
            "full-clip preprocessing contract. Training metadata must declare the "
            "matching extractor_version, feature_schema_version, and preprocessing settings."
        )


def _validate_model_files(files: dict[str, bytes]):
    columns = _read_json(files, "feature_columns.json")
    if not isinstance(columns, list) or tuple(columns) != FEATURE_NAMES:
        raise SemanticArtifactCompatibilityError("Expected the exact semantic model 165-feature order.")
    metadata = _read_json(files, "training_metadata.json")
    _validate_training_contract(metadata)
    if (metadata.get("model_type") != "XGBRegressor"
            or metadata.get("target") != "teacher_fused_spoof_probability"
            or metadata.get("ground_truth_used_as_training_target") is not False
            or metadata.get("feature_count") != len(FEATURE_NAMES)):
        raise SemanticArtifactCompatibilityError("Metadata must describe teacher-probability regression over 165 features.")
    threshold = metadata.get("decision_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise SemanticArtifactCompatibilityError("Metadata requires a valid decision threshold.")
    model = _read_json(files, "surrogate_xgboost.json")
    try:
        learner = model["learner"]
        valid = learner["objective"]["name"] == "reg:squarederror" and int(learner["learner_model_param"]["num_feature"]) == len(FEATURE_NAMES)
    except (KeyError, TypeError, ValueError) as error:
        raise SemanticArtifactCompatibilityError("Invalid XGBoost export.") from error
    if not valid:
        raise SemanticArtifactCompatibilityError("Expected a 165-feature XGBRegressor with the reg:squarederror objective.")
    if "selected_glottal_features_20.json" in files:
        selected = _read_json(files, "selected_glottal_features_20.json")
        if not isinstance(selected, dict) or selected.get("selected_features") != list(SELECTED20):
            raise SemanticArtifactCompatibilityError("Selected-20 feature contract mismatch.")
    if "teacher_assets.json" in files:
        assets = _read_json(files, "teacher_assets.json")
        hashes = assets.get("hashes") if isinstance(assets, dict) else None
        if not isinstance(hashes, dict):
            raise SemanticArtifactCompatibilityError("Teacher provenance mismatch.")
        fingerprint = sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
        fingerprints = metadata.get("teacher_fingerprints", {})
        if (assets.get("teacher_fingerprint") != fingerprint
                or hashes.get("feature_schema") != list(FEATURE_NAMES)
                or not isinstance(fingerprints, dict)
                or any(values != [fingerprint] for values in fingerprints.values())):
            raise SemanticArtifactCompatibilityError("Teacher provenance mismatch.")
        if "teacher_fusion.json" in files and hashes.get("fusion") != _read_json(files, "teacher_fusion.json"):
            raise SemanticArtifactCompatibilityError("Teacher fusion provenance mismatch.")
    return metadata, float(threshold)


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


def load_production_semantic_artifacts(model_dir: Path) -> LoadedSemanticArtifacts:
    directory = model_dir.resolve()
    files = _read_model_files(directory)
    metadata, threshold = _validate_model_files(files)
    try:
        import joblib
        import shap
        import xgboost
        from sklearn.impute import SimpleImputer

        imputer = joblib.load(BytesIO(files["imputer.joblib"]))
        statistics = np.asarray(getattr(imputer, "statistics_", ()))
        if (not isinstance(imputer, SimpleImputer) or imputer.strategy != "median"
                or statistics.shape != (len(FEATURE_NAMES),) or not np.isfinite(statistics).all()
                or int(getattr(imputer, "n_features_in_", 0)) != len(FEATURE_NAMES)):
            raise SemanticArtifactCompatibilityError("Expected a fitted 165-feature median imputer.")
        if hasattr(imputer, "feature_names_in_") and tuple(imputer.feature_names_in_) != FEATURE_NAMES:
            raise SemanticArtifactCompatibilityError("Imputer feature order mismatch.")
        booster = xgboost.Booster(model_file=bytearray(files["surrogate_xgboost.json"]))
        explainer = _build_tree_explainer(shap, booster)
    except SemanticArtifactCompatibilityError:
        raise
    except Exception as error:
        raise SemanticArtifactCompatibilityError("Semantic model dependencies or artifacts could not be loaded.") from error

    fingerprint = _files_fingerprint(files)
    experiment = re.sub(r"[^A-Za-z0-9_-]", "_", str(metadata.get("experiment_name", "surrogate")))[:80]
    artifacts = LoadedSemanticArtifacts(
        f"{experiment}-{fingerprint[:16]}", booster, explainer, imputer,
        metadata, threshold, directory, fingerprint,
    )
    # Validate one prediction/SHAP reconstruction before startup completes.
    from app.voice_xai.semantic.features import FeatureExtractionResult
    from app.voice_xai.semantic.service import XgboostShapEvidenceProvider
    XgboostShapEvidenceProvider(artifacts).extract_features(FeatureExtractionResult(
        FEATURE_NAMES, statistics.copy(), 16000, 1.0, FEATURE_EXTRACTION_VERSION, {},
    ))
    return artifacts
