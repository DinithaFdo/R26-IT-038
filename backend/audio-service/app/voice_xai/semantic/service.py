"""Run clip-level surrogate inference and produce additive Tree SHAP evidence.

Extract or accept acoustic features, apply the active model's median imputer,
predict the raw regression score, and verify that SHAP reconstructs that score.
"""
from pathlib import Path
import numpy as np
from app.voice_xai.semantic.artifact_loader import SemanticArtifactCompatibilityError, load_production_semantic_artifacts, model_fingerprint
from app.voice_xai.semantic.contracts import SemanticAnalysisResult, SemanticEvidence
from app.voice_xai.semantic.features import AcousticFeatureExtractor
from app.voice_xai.semantic.schema import FEATURE_NAMES, FEATURE_EXTRACTION_VERSION
from app.voice_xai.semantic.explainer import build_semantic_explanation

class XgboostShapEvidenceProvider:
    def __init__(self, artifacts):
        self.artifacts = artifacts

    def extract_features(self, features):
        import pandas as pd
        import xgboost
        if features.feature_names != FEATURE_NAMES or features.extractor_version != FEATURE_EXTRACTION_VERSION:
            raise SemanticArtifactCompatibilityError("Feature extraction does not match the semantic model contract.")
        raw = np.asarray(features.values, dtype=np.float64)
        if raw.shape != (len(FEATURE_NAMES),):
            raise SemanticArtifactCompatibilityError("Expected 165 feature values.")
        raw = np.where(np.isfinite(raw), raw, np.nan)
        values = np.asarray(self.artifacts.imputer.transform(pd.DataFrame([raw], columns=FEATURE_NAMES)), dtype=np.float64)
        if values.shape != (1, len(FEATURE_NAMES)) or not np.isfinite(values).all():
            raise SemanticArtifactCompatibilityError("Imputation did not produce 165 finite values.")
        matrix = xgboost.DMatrix(values, feature_names=list(FEATURE_NAMES))
        score = float(self.artifacts.booster.predict(matrix, output_margin=True)[0])
        shap = np.asarray(self.artifacts.shap_explainer.shap_values(values, check_additivity=True), dtype=np.float64).reshape(-1)
        expected = np.asarray(self.artifacts.shap_explainer.expected_value).reshape(-1)
        if expected.size != 1 or shap.shape != (len(FEATURE_NAMES),):
            raise SemanticArtifactCompatibilityError("Unexpected Tree SHAP output shape.")
        base = float(expected[0])
        evidence = SemanticEvidence(values[0], shap, base, score, int(np.isnan(raw).sum()))
        if abs(base + float(shap.sum()) - score) > 1e-4:
            raise SemanticArtifactCompatibilityError("Tree SHAP does not reconstruct the raw regressor score.")
        return evidence

class ProductionSemanticExplanationService:
    """Keep one validated model in memory until the backend restarts."""
    def __init__(self, artifacts, *, feature_extractor=None):
        self.artifacts = artifacts
        self._provider = XgboostShapEvidenceProvider(artifacts)
        self._feature_extractor = feature_extractor or AcousticFeatureExtractor()

    @classmethod
    def load_once(cls, *, model_dir: Path):
        return cls(load_production_semantic_artifacts(model_dir))

    def analyze(self, inference):
        try:
            current = model_fingerprint(self.artifacts.directory)
        except SemanticArtifactCompatibilityError as error:
            raise SemanticArtifactCompatibilityError("Model files changed or are incomplete. Restart after replacing the complete model export.") from error
        if current != self.artifacts.fingerprint:
            raise SemanticArtifactCompatibilityError("Model files changed. Restart the backend before creating explanations.")
        features = inference.acoustic_features
        if features is None:
            if inference.processed_audio is None:
                raise SemanticArtifactCompatibilityError("Semantic inference requires original audio or extracted features.")
            features = self._feature_extractor.extract_processed_audio(inference.processed_audio)
        evidence = self._provider.extract_features(features)
        return SemanticAnalysisResult(build_semantic_explanation(evidence, self.artifacts))
