"""Source-audio interval operations and descriptive temporal/semantic agreement."""
import math

from app.schemas.xai import ExplanationQuality, MetricEvidence, MetricStatus


# Merge overlapping/touching half-open spans without filling unknown gaps.
def union_spans(spans):
    ordered = sorted((float(start), float(end)) for start, end in spans)
    result = []
    for start, end in ordered:
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError('Expected finite, positive source-audio intervals.')
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return tuple(result)


# Intersect interval unions so evidence is compared only on common coverage.
def intersect_spans(left, right):
    left, right = union_spans(left), union_spans(right)
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        start, end = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if end > start:
            result.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return union_spans(result)


# Remove failed or unprocessed window supports, including overlapping portions.
def subtract_spans(spans, excluded):
    result = []
    excluded = union_spans(excluded)
    for start, end in union_spans(spans):
        cursor = start
        for cut_start, cut_end in excluded:
            if cut_end <= cursor:
                continue
            if cut_start >= end:
                break
            if cut_start > cursor:
                result.append((cursor, min(cut_start, end)))
            cursor = max(cursor, cut_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))
    return tuple(result)


# Measure union duration once, avoiding double-counted overlapping windows.
def span_duration(spans):
    return sum(end - start for start, end in union_spans(spans))


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
