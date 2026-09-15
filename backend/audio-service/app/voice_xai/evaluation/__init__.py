"""Metrics and fail-closed research-readiness policies for Voice XAI."""

from app.voice_xai.evaluation.consistency import quality_for_analysis
from app.voice_xai.evaluation.eligibility import evaluate_research_eligibility
from app.voice_xai.evaluation.offline_metrics import load_surrogate_fidelity_artifact

__all__ = [
    "evaluate_research_eligibility",
    "load_surrogate_fidelity_artifact",
    "quality_for_analysis",
]
