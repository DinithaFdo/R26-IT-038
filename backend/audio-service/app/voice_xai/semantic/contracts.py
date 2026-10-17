"""Internal evidence and result types for a clip-level semantic analysis.

Validate the 165 imputed feature values, signed SHAP contributions, and raw
regression score, and carry the serialized explanation to the workflow.
"""
from dataclasses import dataclass
import numpy as np
from app.schemas.xai import SemanticExplanation
from app.voice_xai.semantic.schema import FEATURE_NAMES

WINDOW_EXTRACTION_VERSION = 'shared-clean-window-audio-v1'
WINDOW_LOCALIZATION_VERSION = 'window-abs-shap-duration-percentile-v1'
WINDOW_PREPROCESSING_CONTRACT = {
    'version': WINDOW_EXTRACTION_VERSION,
    'preprocessing_scope': 'shared_full_cleaned_clip',
    'window_trim': False,
    'window_normalization': False,
    'padding': False,
    'tail_policy': 'end_anchored_short_unpadded',
    'time_reference': 'source_audio',
}


@dataclass(frozen=True)
class SemanticWindowOptions:
    enabled: bool = True
    window_seconds: float = 2.0
    hop_seconds: float = 1.0
    max_windows: int = 32
    budget_seconds: float = 120.0
    activity_percentile: float = 90.0

    # Reject invalid or impractically small windows before allocating DSP work.
    def __post_init__(self):
        if (not np.isfinite([self.window_seconds, self.hop_seconds, self.budget_seconds,
                             self.activity_percentile]).all()
                or self.window_seconds < 0.2 or not 0 < self.hop_seconds <= self.window_seconds
                or self.hop_seconds < 1 / 16000 or isinstance(self.max_windows, bool)
                or not isinstance(self.max_windows, int) or not 1 <= self.max_windows <= 256
                or self.budget_seconds <= 0 or not 0 <= self.activity_percentile <= 100):
            raise ValueError('Invalid semantic window configuration.')

    # Bind offline validation metadata to exactly the runtime measurement recipe.
    def model_contract(self):
        return {**WINDOW_PREPROCESSING_CONTRACT, 'window_seconds': self.window_seconds,
                'hop_seconds': self.hop_seconds}

    # Keep application settings and direct/offline service construction consistent.
    @classmethod
    def from_settings(cls, settings):
        return cls(enabled=settings.xai_semantic_windows_enabled,
                   window_seconds=settings.xai_semantic_window_seconds,
                   hop_seconds=settings.xai_semantic_hop_seconds,
                   max_windows=settings.xai_semantic_max_windows,
                   budget_seconds=settings.xai_semantic_window_budget_seconds,
                   activity_percentile=settings.xai_semantic_activity_percentile)

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

    def __post_init__(self):
        if (self.development_placeholder != self.explanation.development_placeholder
                or self.research_eligible != self.explanation.research_eligible):
            raise ValueError('Semantic result and explanation eligibility flags differ.')
