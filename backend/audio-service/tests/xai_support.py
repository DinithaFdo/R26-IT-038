"""Static semantic responses for orchestration tests; never loaded by the app."""

from app.schemas.xai import SemanticFeatureContribution, ShapDirection
from app.voice_xai.semantic.contracts import (
    SemanticAnalysisResult,
    SemanticExplanationConfig,
    SemanticWindowAnalysisResult,
)
from app.voice_xai.semantic.explainer import build_windowed_semantic_explanation
from app.voice_xai.semantic.features import FEATURE_EXTRACTION_VERSION
from app.voice_xai.semantic.windows import SemanticWindow


class StaticSemanticService:
    def analyze(self, inference):
        result = self._result(inference.prediction.audio.duration_seconds)
        return SemanticAnalysisResult(
            explanation=result.explanation,
            development_placeholder=True,
        )

    def analyze_windows(self, audio, *, request_id):
        return self._result(audio.duration_seconds)

    @staticmethod
    def _result(duration):
        window = SemanticWindow(
            start_seconds=0.0,
            end_seconds=duration,
            contributions=tuple(SemanticFeatureContribution(
                rank=rank,
                feature_name=name,
                display_name="Test feature",
                value=0.1,
                shap_value=0.2 / rank,
                direction=ShapDirection.toward_spoof,
                start_seconds=0.0,
                end_seconds=duration,
            ) for rank, name in enumerate(("jitter", "shimmer", "hnr"), start=1)),
            spoof_probability=0.8,
        )
        explanation = build_windowed_semantic_explanation(
            (window,),
            SemanticExplanationConfig(model_version="test-model-v1"),
            extractor_version=FEATURE_EXTRACTION_VERSION,
        ).model_copy(update={"development_placeholder": True, "research_eligible": False})
        return SemanticWindowAnalysisResult(
            explanation=explanation,
            windows=(window,),
            development_placeholder=True,
        )
