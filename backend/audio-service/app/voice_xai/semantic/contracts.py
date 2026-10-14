"""Internal evidence and result types for a clip-level semantic analysis.

Validate the 165 imputed feature values, signed SHAP contributions, and raw
regression score, and carry the serialized explanation to the workflow.
"""
from dataclasses import dataclass
import numpy as np
from app.schemas.xai import SemanticExplanation
from app.voice_xai.semantic.schema import FEATURE_NAMES

@dataclass(frozen=True)
class SemanticEvidence:
    feature_values: np.ndarray
    shap_values: np.ndarray
    base_value: float
    predicted_value: float
    imputed_feature_count: int = 0

    def __post_init__(self):
        for values in (self.feature_values, self.shap_values):
            if np.asarray(values).shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
                raise ValueError("Semantic evidence requires 165 finite values per feature and SHAP vector.")
        if not np.isfinite([self.base_value, self.predicted_value]).all():
            raise ValueError("Semantic base and prediction scores must be finite.")
        if not 0 <= self.imputed_feature_count <= len(FEATURE_NAMES):
            raise ValueError("Invalid imputed feature count.")

@dataclass(frozen=True)
class SemanticAnalysisResult:
    explanation: SemanticExplanation
    development_placeholder: bool = False
    research_eligible: bool = False
    warnings: tuple[str, ...] = ()
