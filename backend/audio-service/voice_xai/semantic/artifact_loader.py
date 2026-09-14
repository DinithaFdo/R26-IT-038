"""Validation and one-time loading for versioned semantic-XAI artifacts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.voice_xai.semantic.feature_labels import label_for_feature
from app.voice_xai.semantic.features import FEATURE_EXTRACTION_VERSION, FEATURE_NAMES


SEMANTIC_ARTIFACT_MANIFEST_VERSION = "voice-xai-semantic-manifest-v2"
_SHAP_TREE_EXPLAINER_INIT_LOCK = RLock()


class SemanticArtifactCompatibilityError(ValueError):
    """Raised when an artifact cannot explain the runtime feature contract."""


@dataclass(frozen=True)
class SemanticArtifactManifest:
    model_version: str
    feature_schema_version: str
    extractor_version: str
    output_space: str
    feature_count: int
    model_sha256: str
    feature_columns_sha256: str
    decision_threshold_sha256: str
    training_metadata_sha256: str
    imputer_sha256: str
    decision_threshold: float
    training_dataset_version: str
    target_type: str = "independent_acoustic_evidence_model"
    manifest_version: str = SEMANTIC_ARTIFACT_MANIFEST_VERSION
    shap_background_required: bool = False
    feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.manifest_version != SEMANTIC_ARTIFACT_MANIFEST_VERSION:
            raise SemanticArtifactCompatibilityError("Unsupported semantic manifest version.")
        if not self.model_version.strip() or not self.extractor_version.strip():
            raise SemanticArtifactCompatibilityError("model_version and extractor_version are required.")
        if not self.feature_schema_version.strip() or not self.training_dataset_version.strip():
            raise SemanticArtifactCompatibilityError("Semantic schema and training dataset versions are required.")
        if self.extractor_version != FEATURE_EXTRACTION_VERSION:
            raise SemanticArtifactCompatibilityError("Artifact extractor version does not match the runtime extractor.")
        if self.feature_count != len(FEATURE_NAMES):
            raise SemanticArtifactCompatibilityError("Artifact feature count does not match the runtime extractor.")
        if self.feature_names and self.feature_names != FEATURE_NAMES:
            raise SemanticArtifactCompatibilityError("Artifact feature names do not match the runtime order.")
        if not 0.0 < self.decision_threshold < 1.0:
            raise SemanticArtifactCompatibilityError("Decision threshold must be strictly between zero and one.")
        if self.output_space not in {"raw_margin", "log_odds", "probability"}:
            raise SemanticArtifactCompatibilityError("Unsupported SHAP output space.")
        if self.target_type not in {"classifier_surrogate", "independent_acoustic_evidence_model"}:
            raise SemanticArtifactCompatibilityError("Unsupported semantic target type.")
        for digest in (self.model_sha256, self.feature_columns_sha256,
                       self.decision_threshold_sha256, self.training_metadata_sha256,
                       self.imputer_sha256):
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise SemanticArtifactCompatibilityError("Artifact hashes must be lowercase SHA-256 values.")

    @classmethod
    def from_file(cls, path: Path) -> "SemanticArtifactManifest":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise SemanticArtifactCompatibilityError("Semantic manifest does not satisfy the required contract.") from error


@dataclass(frozen=True)
class LoadedSemanticArtifacts:
    manifest: SemanticArtifactManifest
    booster: Any
    shap_explainer: Any
    imputer: Any


def load_production_semantic_artifacts(*, manifest_path: Path, model_path: Path,
                                      feature_columns_path: Path, threshold_path: Path,
                                      training_metadata_path: Path,
                                      imputer_path: Path,
                                      shap_background_path: Path | None) -> LoadedSemanticArtifacts:
    """Validate every published v4 artifact before loading XGBoost and SHAP once."""
    manifest = SemanticArtifactManifest.from_file(manifest_path)
    feature_names = _load_feature_columns(feature_columns_path, manifest)
    manifest = replace(manifest, feature_names=feature_names)
    _validate_threshold_file(threshold_path, manifest)
    _validate_training_metadata(training_metadata_path, manifest)
    _validate_model_file(model_path, manifest)
    imputer = _load_imputer(imputer_path, manifest)
    background = _load_shap_background(shap_background_path, manifest)
    try:
        import shap
        import xgboost
    except ImportError as error:
        raise SemanticArtifactCompatibilityError("Production semantic inference requires optional xai dependencies.") from error
    try:
        booster = xgboost.Booster(model_file=str(model_path))
        explainer = _build_xgboost_tree_explainer(shap, booster, background)
    except Exception as error:
        raise SemanticArtifactCompatibilityError("The configured XGBoost or SHAP artifact could not be loaded.") from error
    return LoadedSemanticArtifacts(manifest=manifest, booster=booster, shap_explainer=explainer, imputer=imputer)


def _build_xgboost_tree_explainer(shap: Any, booster: Any, background: NDArray[np.float64] | None) -> Any:
    """Build a SHAP TreeExplainer without probing unrelated Transformers backends.

    SHAP's base explainer checks whether every model is a transformer language
    model.  When the SSL classifier has already imported ``transformers``, that
    check can trigger its optional TensorFlow/Keras import and fail on a Keras
    3 installation.  The semantic model is a validated XGBoost ``Booster``, so
    it cannot be a transformer; suppressing that one check during construction
    is both safe and keeps TensorFlow out of this independent XAI path.
    """
    explainer_module = shap.explainers._explainer
    original_is_transformers_lm = explainer_module.is_transformers_lm
    with _SHAP_TREE_EXPLAINER_INIT_LOCK:
        explainer_module.is_transformers_lm = lambda _model: False
        try:
            return shap.TreeExplainer(booster, data=background, model_output="raw")
        finally:
            explainer_module.is_transformers_lm = original_is_transformers_lm


def _load_feature_columns(path: Path, manifest: SemanticArtifactManifest) -> tuple[str, ...]:
    _validate_hash(path, manifest.feature_columns_sha256, "feature-columns")
    try:
        names = tuple(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SemanticArtifactCompatibilityError("Feature columns must be a JSON array of names.") from error
    if names != FEATURE_NAMES or len(set(names)) != manifest.feature_count:
        raise SemanticArtifactCompatibilityError("Feature columns do not exactly match the runtime extractor order.")
    return names


def _validate_threshold_file(path: Path, manifest: SemanticArtifactManifest) -> None:
    _validate_hash(path, manifest.decision_threshold_sha256, "threshold")
    try:
        threshold = float(json.loads(path.read_text(encoding="utf-8"))["threshold"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SemanticArtifactCompatibilityError("Decision threshold artifact is invalid.") from error
    if not np.isclose(threshold, manifest.decision_threshold, rtol=0.0, atol=1e-15):
        raise SemanticArtifactCompatibilityError("Decision threshold does not match the manifest.")


def _validate_training_metadata(path: Path, manifest: SemanticArtifactManifest) -> None:
    _validate_hash(path, manifest.training_metadata_sha256, "training metadata")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        valid = (int(metadata["feature_count"]) == manifest.feature_count
                 and int(metadata["target_sr"]) == 16_000
                 and np.isclose(float(metadata["max_audio_seconds"]), 6.0, rtol=0.0, atol=1e-12)
                 and np.isclose(float(metadata["decision_threshold"]), manifest.decision_threshold, rtol=0.0, atol=1e-15)
                 and metadata["label_convention"] == {"bonafide": 0, "spoof": 1})
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SemanticArtifactCompatibilityError("Training metadata artifact is invalid.") from error
    if not valid:
        raise SemanticArtifactCompatibilityError("Training metadata does not match the semantic model contract.")


def _validate_model_file(path: Path, manifest: SemanticArtifactManifest) -> None:
    _validate_hash(path, manifest.model_sha256, "XGBoost model")
    try:
        learner = json.loads(path.read_text(encoding="utf-8"))["learner"]
        objective = learner["objective"]["name"]
        feature_count = int(learner["learner_model_param"]["num_feature"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise SemanticArtifactCompatibilityError("XGBoost model must be a readable JSON model.") from error
    if objective != "binary:logistic" or feature_count != manifest.feature_count:
        raise SemanticArtifactCompatibilityError("XGBoost model does not match the semantic feature contract.")


def _load_imputer(path: Path, manifest: SemanticArtifactManifest) -> Any:
    """Load only the hash-pinned training imputer after static artifact checks."""
    _validate_hash(path, manifest.imputer_sha256, "imputer")
    try:
        import joblib
        from sklearn.impute import SimpleImputer
    except ImportError as error:
        raise SemanticArtifactCompatibilityError(
            "Production semantic inference requires joblib and scikit-learn for the training imputer."
        ) from error
    try:
        imputer = joblib.load(path)
    except Exception as error:
        raise SemanticArtifactCompatibilityError("The configured training imputer could not be loaded.") from error
    statistics = np.asarray(getattr(imputer, "statistics_", ()), dtype=np.float64)
    if (
        not isinstance(imputer, SimpleImputer)
        or imputer.strategy != "median"
        or int(getattr(imputer, "n_features_in_", 0)) != manifest.feature_count
        or statistics.shape != (manifest.feature_count,)
        or not np.isfinite(statistics).all()
    ):
        raise SemanticArtifactCompatibilityError(
            "Training imputer must be a fitted finite median SimpleImputer for the v4 feature schema."
        )
    _restore_simple_imputer_runtime_state(imputer)
    return imputer


def _restore_simple_imputer_runtime_state(imputer: Any) -> None:
    """Supply state introduced by newer scikit-learn releases for a pinned v4 artifact.

    The checked-in imputer was trained with scikit-learn 1.6.1.  Newer releases
    use ``_fill_dtype`` during ``transform`` but do not populate it while
    unpickling that older, hash-validated artifact.  Its fitted input dtype is
    the correct fill dtype for median statistics, so restoring it preserves the
    original estimator behaviour without changing the serialized artifact.
    """
    if hasattr(imputer, "_fill_dtype"):
        return
    fit_dtype = getattr(imputer, "_fit_dtype", None)
    if fit_dtype is None:
        raise SemanticArtifactCompatibilityError(
            "Training imputer is missing the fitted dtype required by this scikit-learn version."
        )
    imputer._fill_dtype = np.dtype(fit_dtype)


def _load_shap_background(path: Path | None, manifest: SemanticArtifactManifest) -> NDArray[np.float64] | None:
    if path is None:
        if manifest.shap_background_required:
            raise SemanticArtifactCompatibilityError("The manifest requires a SHAP background array.")
        return None
    _require_file(path, "SHAP background")
    try:
        values = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise SemanticArtifactCompatibilityError("SHAP background must be a readable NumPy array without pickles.") from error
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != manifest.feature_count or not np.isfinite(values).all():
        raise SemanticArtifactCompatibilityError("SHAP background has an invalid shape or non-finite values.")
    return np.asarray(values, dtype=np.float64)


def reference_summary_for(_manifest: SemanticArtifactManifest, feature_name: str) -> str:
    """The supplied v4 artifacts contain no reference distribution summary."""
    return f"Reference distribution is unavailable for {label_for_feature(feature_name).display_name}."


def expected_units() -> dict[str, str | None]:
    return {name: label_for_feature(name).unit for name in FEATURE_NAMES}


def _validate_hash(path: Path, expected: str, label: str) -> None:
    _require_file(path, label)
    if _sha256(path) != expected:
        raise SemanticArtifactCompatibilityError(f"{label.capitalize()} SHA-256 does not match manifest.")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SemanticArtifactCompatibilityError(f"{label.capitalize()} is not a readable file.")


def _sha256(path: Path) -> str:
    """Hash binary artifacts exactly and JSON artifacts independent of checkout EOLs.

    Semantic JSON artifacts are hash-pinned from their LF-authored training
    exports. Git's Windows ``core.autocrlf`` setting can rewrite only those
    textual files to CRLF without changing their parsed JSON content. Normalize
    that transport-only newline difference before hashing; joblib and every
    other binary artifact remain strictly byte-for-byte verified.
    """
    digest = sha256()
    if path.suffix.lower() == ".json":
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    else:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
