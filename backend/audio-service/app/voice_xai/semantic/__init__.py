"""Feature-schema and SHAP-based semantic explanation services."""

from app.voice_xai.semantic.features import (
    FEATURE_NAMES,
    AcousticFeatureExtractor,
    FeatureExtractionConfig,
    FeatureExtractionResult,
)

from app.voice_xai.semantic.service import (
    MockSemanticExplanationService,
    ProductionSemanticExplanationService,
)

__all__ = [
    "FEATURE_NAMES",
    "AcousticFeatureExtractor",
    "FeatureExtractionConfig",
    "FeatureExtractionResult",
    "MockSemanticExplanationService",
    "ProductionSemanticExplanationService",
]
