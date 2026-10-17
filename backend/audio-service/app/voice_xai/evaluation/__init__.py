"""Metrics and fail-closed research-readiness policies for Voice XAI."""

from app.voice_xai.evaluation.consistency import quality_for_analysis
from app.voice_xai.evaluation.eligibility import evaluate_research_eligibility

__all__ = [
    "evaluate_research_eligibility",
    "quality_for_analysis",
]
