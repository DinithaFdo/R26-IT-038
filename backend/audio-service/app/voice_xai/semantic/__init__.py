"""Public entry points for acoustic extraction and semantic explanations.

Expose the ordered feature contract, audio extractor, extraction result, and
XGBoost explanation service used by the application.
"""
from app.voice_xai.semantic.features import AcousticFeatureExtractor, FeatureExtractionResult
from app.voice_xai.semantic.schema import FEATURE_NAMES
from app.voice_xai.semantic.service import ProductionSemanticExplanationService
__all__ = ["FEATURE_NAMES", "AcousticFeatureExtractor", "FeatureExtractionResult", "ProductionSemanticExplanationService"]
