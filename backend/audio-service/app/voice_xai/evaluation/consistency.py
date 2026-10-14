"""Expose model fidelity and metric applicability for clip-level explanations."""
from app.schemas.xai import ExplanationQuality, MetricEvidence, MetricStatus

def quality_for_analysis(temporal, semantic):
    fidelity = semantic.fidelity_r2 if semantic is not None else None
    if fidelity is None:
        fidelity = MetricEvidence(status=MetricStatus.not_computed, scope="offline_validation", reason="Current model fidelity metadata is unavailable.")
    return ExplanationQuality(
        surrogate_fidelity_r2=fidelity,
        temporal_semantic_iou=MetricEvidence(status=MetricStatus.not_applicable, scope="per_analysis",
            reason="Clip-level SHAP has no temporal intervals for IoU."),
    )
