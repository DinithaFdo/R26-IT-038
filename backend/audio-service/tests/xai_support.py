"""Static semantic model evidence for workflow tests."""
from app.schemas.xai import SemanticExplanation, SemanticFeatureContribution, ShapDirection, ComponentStatus
from app.voice_xai.semantic.contracts import SemanticAnalysisResult
from app.voice_xai.semantic.schema import FEATURE_EXTRACTION_VERSION, FEATURE_SCHEMA_VERSION

class StaticSemanticService:
    def analyze(self, inference):
        return SemanticAnalysisResult(
            explanation=SemanticExplanation(status=ComponentStatus.completed,
                target_type="classifier_surrogate", model_version="test-model-v1",
                feature_schema_version=FEATURE_SCHEMA_VERSION, extractor_version=FEATURE_EXTRACTION_VERSION,
                output_space="raw_regression", development_placeholder=True,
                base_value=0.3, predicted_value=0.6,
                feature_importance=[SemanticFeatureContribution(rank=rank,
                    feature_name=name, display_name=name, value=0.1, shap_value=0.1,
                    direction=ShapDirection.toward_spoof)
                    for rank,name in enumerate(("jitter","shimmer","hnr"),1)]),
            development_placeholder=True,
        )
