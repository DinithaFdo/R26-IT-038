"""Per-analysis Voice XAI quality calculations with explicit availability."""

from __future__ import annotations

from app.schemas.xai import (
    ExplanationQuality,
    MetricEvidence,
    MetricStatus,
    SemanticExplanation,
    TemporalExplanation,
)
from app.voice_xai.evaluation.offline_metrics import surrogate_fidelity_for_model

TOP_SHAP_FEATURE_COUNT = 3


def quality_for_analysis(
    temporal: TemporalExplanation | None,
    semantic: SemanticExplanation | None,
) -> ExplanationQuality:
    """Compare attention regions with exactly the top-three SHAP windows.

    The sliding semantic-window grid is intentionally not used here: it often
    spans nearly the whole clip and would make overlap a coverage statistic,
    rather than agreement between the strongest semantic evidence and the
    attention rollout.
    """

    surrogate_fidelity = surrogate_fidelity_for_model(
        semantic.model_version if semantic is not None else None
    )
    if temporal is None or semantic is None:
        return ExplanationQuality(
            surrogate_fidelity_r2=surrogate_fidelity,
            temporal_semantic_iou=MetricEvidence(
                status=MetricStatus.not_computed,
                scope="per_analysis",
                reason="Temporal or semantic explanation is unavailable.",
            )
        )
    if not temporal.high_attention_regions:
        return ExplanationQuality(
            surrogate_fidelity_r2=surrogate_fidelity,
            temporal_semantic_iou=MetricEvidence(
                status=MetricStatus.not_computed,
                scope="per_analysis",
                reason="No temporal high-attention regions were selected.",
            )
        )
    top_shap = sorted(
        semantic.feature_importance,
        key=lambda contribution: contribution.rank,
    )[:TOP_SHAP_FEATURE_COUNT]
    if len(top_shap) != TOP_SHAP_FEATURE_COUNT:
        return ExplanationQuality(
            surrogate_fidelity_r2=surrogate_fidelity,
            temporal_semantic_iou=MetricEvidence(
                status=MetricStatus.not_computed,
                scope="per_analysis",
                reason="Three timestamped top-ranked SHAP contributions are unavailable.",
            )
        )
    if any(
        contribution.start_seconds is None or contribution.end_seconds is None
        for contribution in top_shap
    ):
        return ExplanationQuality(
            surrogate_fidelity_r2=surrogate_fidelity,
            temporal_semantic_iou=MetricEvidence(
                status=MetricStatus.not_computed,
                scope="per_analysis",
                reason="A top-three SHAP contribution has no timestamped time window.",
            )
        )
    temporal_intervals = [
        (region.start_seconds, region.end_seconds)
        for region in temporal.high_attention_regions
    ]
    semantic_intervals = [
        (float(contribution.start_seconds), float(contribution.end_seconds))
        for contribution in top_shap
    ]
    intersection = _measure(_intersections(temporal_intervals, semantic_intervals))
    union = _measure(temporal_intervals) + _measure(semantic_intervals) - intersection
    return ExplanationQuality(
        surrogate_fidelity_r2=surrogate_fidelity,
        temporal_semantic_iou=MetricEvidence(
            status=MetricStatus.available,
            value=intersection / union if union else 0.0,
            scope="per_analysis",
        )
    )


def _intersections(left: list[tuple[float, float]], right: list[tuple[float, float]]):
    return [
        (max(a_start, b_start), min(a_end, b_end))
        for a_start, a_end in left
        for b_start, b_end in right
        if max(a_start, b_start) < min(a_end, b_end)
    ]


def _measure(intervals: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)
