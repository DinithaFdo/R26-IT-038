"""Build the semantic API response from raw regression and Tree SHAP evidence.

Rank all 165 signed feature contributions, clip the score for the probability
summary, format display labels, and attach the model's threshold and fidelity metadata.
"""
import numpy as np
from app.schemas.common import PredictionLabel
from app.schemas.xai import (
    ComponentStatus, MetricEvidence, MetricStatus, SemanticClipSummary,
    SemanticExplanation, SemanticFeatureContribution, SemanticTargetType, ShapDirection,
    SemanticWindowExplanation, SemanticLocalization, SemanticActivityBin, SemanticRegion,
)
from app.voice_xai.semantic.schema import FEATURE_NAMES, FEATURE_EXTRACTION_VERSION, FEATURE_SCHEMA_VERSION
from app.voice_xai.semantic.contracts import WINDOW_EXTRACTION_VERSION, WINDOW_LOCALIZATION_VERSION
from app.voice_xai.evaluation.intervals import union_spans, intersect_spans, subtract_spans, span_duration


def label_for_feature(name: str) -> str:
    if name not in FEATURE_NAMES:
        raise ValueError(f"Unknown semantic feature: {name}")
    return name.replace("_", " ").capitalize()


# Rank all signed feature contributions with the same stable order for clips and windows.
def feature_contributions(evidence):
    contributions = []
    for rank, index in enumerate(np.argsort(-np.abs(evidence.shap_values), kind="stable"), 1):
        name = FEATURE_NAMES[index]
        value = float(evidence.shap_values[index])
        contributions.append(SemanticFeatureContribution(
            rank=rank, feature_name=name, display_name=label_for_feature(name),
            value=float(evidence.feature_values[index]), shap_value=value,
            direction=ShapDirection.toward_spoof if value > 0 else ShapDirection.toward_bonafide if value < 0 else ShapDirection.neutral,
        ))
    return contributions


# Preserve the existing full-clip explanation and its raw-score interpretation.
def build_semantic_explanation(evidence, artifacts):
    probability = float(np.clip(evidence.predicted_value, 0, 1))
    contributions = feature_contributions(evidence)
    fidelity = artifacts.fidelity()
    metric = MetricEvidence(status=MetricStatus.available, value=fidelity[0], scope="offline_validation", dataset_version=f"Current model test export (n={fidelity[1]})") if fidelity else MetricEvidence(status=MetricStatus.not_computed, scope="offline_validation", reason="This model export has no valid test fidelity metadata.")
    warning = "Tree SHAP explains the surrogate raw score before clipping. It approximates the teacher and provides no time localisation."
    if evidence.imputed_feature_count:
        warning += f" {evidence.imputed_feature_count} missing features used training medians."
    return SemanticExplanation(
        status=ComponentStatus.completed, target_type=SemanticTargetType.classifier_surrogate,
        model_version=artifacts.model_version, feature_schema_version=FEATURE_SCHEMA_VERSION,
        extractor_version=FEATURE_EXTRACTION_VERSION, output_space="raw_regression",
        development_placeholder=False, research_eligible=False,
        base_value=evidence.base_value, predicted_value=evidence.predicted_value,
        clip_summary=SemanticClipSummary(spoof_probability=probability, bonafide_probability=1-probability,
            decision_threshold=artifacts.threshold, predicted_label=PredictionLabel.spoof if probability >= artifacts.threshold else PredictionLabel.bonafide),
        feature_importance=contributions, fidelity_r2=metric, warning=warning,
    )


# Record one window's source boundaries and its complete additive SHAP evidence.
def build_window_explanation(window, evidence, *, window_id, sample_rate, processing_seconds):
    return SemanticWindowExplanation(
        window_id=window_id, start_sample=window.start_sample,
        end_sample_exclusive=window.end_sample,
        start_seconds=window.start_sample / sample_rate, end_seconds=window.end_sample / sample_rate,
        status='completed', predicted_value=evidence.predicted_value, base_value=evidence.base_value,
        shap_activity=float(np.abs(evidence.shap_values).sum()),
        imputed_feature_count=evidence.imputed_feature_count, processing_seconds=processing_seconds,
        feature_importance=feature_contributions(evidence),
    )


# Aggregate only available window supports on exact boundaries, excluding failed overlaps.
def aggregate_window_activity(windows, analyzed_spans):
    completed = [window for window in windows if window.status == 'completed']
    successful = [(window.start_seconds, window.end_seconds) for window in completed]
    failed = [(window.start_seconds, window.end_seconds) for window in windows if window.status != 'completed']
    valid = subtract_spans(intersect_spans(successful, analyzed_spans), failed)
    timeline = []
    for start, end in valid:
        bounds = sorted({start, end} | {
            bound for window in completed for bound in (window.start_seconds, window.end_seconds)
            if start < bound < end
        })
        for left, right in zip(bounds, bounds[1:]):
            active = [window.shap_activity for window in completed
                      if window.start_seconds <= left and window.end_seconds >= right]
            if active:
                timeline.append(SemanticActivityBin(start_seconds=left, end_seconds=right,
                                                    activity=float(np.mean(active)), supporting_windows=len(active)))
    return timeline, valid


# Select a percentile by duration, rather than by an arbitrary plotting resolution.
def activity_percentile(timeline, percentile):
    ordered = sorted(timeline, key=lambda item: item.activity)
    target = sum(item.end_seconds - item.start_seconds for item in ordered) * percentile / 100
    duration = 0.0
    for item in ordered:
        duration += item.end_seconds - item.start_seconds
        if duration >= target:
            return item.activity
    return ordered[-1].activity


# Merge adjacent high-activity bins without filling evidence gaps or failed windows.
def semantic_regions(timeline, threshold):
    groups = []
    for item in timeline:
        if item.activity <= 0 or item.activity < threshold:
            continue
        if groups and abs(groups[-1][-1].end_seconds - item.start_seconds) < 1e-9:
            groups[-1].append(item)
        else:
            groups.append([item])
    return [SemanticRegion(
        region_id=index, start_seconds=group[0].start_seconds, end_seconds=group[-1].end_seconds,
        peak_activity=max(item.activity for item in group),
        mean_activity=sum(item.activity * (item.end_seconds - item.start_seconds) for item in group)
                      / sum(item.end_seconds - item.start_seconds for item in group),
    ) for index, group in enumerate(groups, 1)]


# Describe coverage and experimental validity separately from measured agreement.
def build_semantic_localization(windows, analyzed_spans, *, options, artifacts,
                                sample_rate=16000, trim_start_sample=0, reason=None, processing_seconds=0.0):
    analyzed_spans = union_spans(analyzed_spans)
    timeline, valid = aggregate_window_activity(windows, analyzed_spans)
    threshold = activity_percentile(timeline, options.activity_percentile) if timeline else None
    validated, validation_reason, validation = artifacts.window_validation(options)
    fidelity = (MetricEvidence(status=MetricStatus.available, value=validation['r2'], scope='offline_validation',
                              dataset_version=f"Held-out windows (n={validation['n_windows']}, groups={validation['n_recordings']})")
                if validated else MetricEvidence(status=MetricStatus.not_computed, scope='offline_validation', reason=validation_reason))
    available = bool(timeline)
    requested_duration = span_duration(analyzed_spans)
    valid_duration = span_duration(valid)
    incomplete_coverage = valid_duration < requested_duration - 1e-9
    partial = any(window.status != 'completed' for window in windows) or reason is not None or incomplete_coverage
    if reason is None and available and incomplete_coverage:
        reason = 'Some temporal evidence has no valid semantic window support.'
    return SemanticLocalization(
        status=('partial' if partial else 'completed') if available else 'not_available',
        reason=reason or (None if available else 'No valid semantic window coverage is available.'),
        method_version=WINDOW_LOCALIZATION_VERSION, extractor_version=WINDOW_EXTRACTION_VERSION,
        model_version=artifacts.model_version, model_validation_status='validated' if validated else 'experimental',
        model_validation_reason=validation_reason, window_fidelity_r2=fidelity,
        window_seconds=options.window_seconds, hop_seconds=options.hop_seconds,
        sample_rate_hz=sample_rate, trim_start_sample=trim_start_sample,
        analyzed_spans=[{'start_seconds': start, 'end_seconds': end} for start, end in analyzed_spans],
        valid_spans=[{'start_seconds': start, 'end_seconds': end} for start, end in valid],
        coverage_percentage=min(100.0, 100 * valid_duration / requested_duration) if requested_duration else 0.0,
        threshold_percentile=options.activity_percentile, activity_threshold=threshold,
        flat_activity_map=bool(len({item.activity for item in timeline}) == 1) if timeline else None,
        windows=windows, timeline=timeline, regions=semantic_regions(timeline, threshold) if timeline else [],
        processing_seconds=processing_seconds,
    )
