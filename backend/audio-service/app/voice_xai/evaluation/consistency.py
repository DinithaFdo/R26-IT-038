"""Measure descriptive temporal/semantic agreement only on common valid coverage."""
from app.schemas.xai import ExplanationQuality, MetricEvidence, MetricStatus
from app.voice_xai.evaluation.intervals import intersect_spans, union_spans, span_duration


# Compute interval IoU without counting missing evidence, failed windows or padding.
def temporal_semantic_iou(temporal, semantic, temporal_spans=None):
    localization = getattr(semantic, 'localization', None)
    if localization is None:
        return MetricEvidence(status=MetricStatus.not_applicable, scope='per_analysis',
                              reason='Clip-level SHAP has no temporal intervals for IoU.')
    if temporal is None:
        return MetricEvidence(status=MetricStatus.not_computed, scope='per_analysis', reason='Temporal evidence is unavailable.')
    if temporal_spans is None:
        temporal_spans = [(span.start_seconds, span.end_seconds) for span in localization.analyzed_spans]
    common = intersect_spans(temporal_spans, [(span.start_seconds, span.end_seconds) for span in localization.valid_spans])
    duration = span_duration(common)
    if not duration:
        return MetricEvidence(status=MetricStatus.not_computed, scope='per_analysis', reason='No common valid temporal/semantic coverage.')
    left = intersect_spans([(region.start_seconds, region.end_seconds) for region in temporal.high_attention_regions], common)
    right = intersect_spans([(region.start_seconds, region.end_seconds) for region in localization.regions], common)
    denominator = span_duration(union_spans((*left, *right)))
    if not denominator:
        return MetricEvidence(status=MetricStatus.not_computed, scope='per_analysis', reason='Neither method selects a region within common valid coverage; IoU is undefined.')
    value = span_duration(intersect_spans(left, right)) / denominator
    return MetricEvidence(status=MetricStatus.available, scope='per_analysis', value=min(1.0, max(0.0, value)),
                          reason=f'Descriptive overlap agreement over {duration:.6f}s of common valid coverage; '
                                 f'window model is {localization.model_validation_status}. This is not localization accuracy.')


# Preserve full-clip fidelity while exposing independently scoped interval agreement.
def quality_for_analysis(temporal, semantic, *, temporal_spans=None):
    fidelity = semantic.fidelity_r2 if semantic is not None else None
    if fidelity is None:
        fidelity = MetricEvidence(status=MetricStatus.not_computed, scope="offline_validation", reason="Current model fidelity metadata is unavailable.")
    return ExplanationQuality(
        surrogate_fidelity_r2=fidelity,
        temporal_semantic_iou=temporal_semantic_iou(temporal, semantic, temporal_spans),
    )
