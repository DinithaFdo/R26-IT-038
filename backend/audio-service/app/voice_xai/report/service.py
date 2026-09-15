"""Controlled report composition for independent Voice XAI evidence."""

from __future__ import annotations

from app.schemas.common import PredictionLabel
from app.schemas.xai import (
    ClassifierSnapshot,
    CombinedExplanationReport,
    ComponentStatus,
    ExplanationQuality,
    MetricStatus,
    ReportDisposition,
    SemanticExplanation,
    TemporalExplanation,
)


class DeterministicReportComposer:
    """Create a repeatable human-review report without generative inference."""

    _DISCLAIMER = (
        "This explanation is decision-support evidence only. It does not prove "
        "that audio is spoofed or bonafide and requires qualified human review."
    )

    def compose(
        self,
        *,
        classifier: ClassifierSnapshot,
        temporal: TemporalExplanation | None,
        semantic: SemanticExplanation | None,
        quality: ExplanationQuality | None = None,
    ) -> CombinedExplanationReport:
        """Return the same report for the same validated inputs."""

        disposition = _disposition(classifier)
        finding = _finding(classifier, temporal, semantic)
        return CombinedExplanationReport(
            status=ComponentStatus.completed,
            disposition=disposition,
            requires_human_review=True,
            finding=finding,
            primary_evidence=_primary_evidence(classifier, temporal, semantic),
            quality_checks=_quality_checks(quality),
            limitation=_limitations(classifier, temporal, semantic),
            recommendation=(
                "Review the original audio, classifier snapshot, evidence intervals, "
                "and provenance before making a decision or recording sign-off."
            ),
            disclaimer=self._DISCLAIMER,
        )


def _disposition(classifier: ClassifierSnapshot) -> ReportDisposition:
    if classifier.verdict == PredictionLabel.spoof:
        return ReportDisposition.spoof_suspected
    if classifier.verdict == PredictionLabel.bonafide:
        return ReportDisposition.bonafide_suspected
    return ReportDisposition.inconclusive


def _finding(
    classifier: ClassifierSnapshot,
    temporal: TemporalExplanation | None,
    semantic: SemanticExplanation | None,
) -> str:
    if classifier.verdict is None or classifier.spoof_probability is None:
        decision = "The classifier verdict and score are unavailable"
    else:
        decision = (
            f"The classifier returned {classifier.verdict.value} with a spoof score "
            f"of {classifier.spoof_probability:.2f}."
        )
    return " ".join((decision, _temporal_sentence(temporal), _semantic_sentence(semantic)))


def _primary_evidence(
    classifier: ClassifierSnapshot,
    temporal: TemporalExplanation | None,
    semantic: SemanticExplanation | None,
) -> str:
    parts = [
        "Classifier decision threshold " f"was {classifier.decision_threshold:.2f}."
    ]
    if temporal is not None and temporal.status == ComponentStatus.completed:
        parts.append(_temporal_sentence(temporal))
    if semantic is not None and semantic.status == ComponentStatus.completed:
        parts.append(_semantic_sentence(semantic))
    if len(parts) == 1:
        parts.append("No completed temporal or semantic evidence is available.")
    return " ".join(parts)


def _temporal_sentence(temporal: TemporalExplanation | None) -> str:
    if temporal is None or temporal.status != ComponentStatus.completed:
        return "Temporal high-attention evidence is unavailable."
    if not temporal.high_attention_regions:
        return "No high-attention regions were selected by the temporal configuration."
    regions = ", ".join(
        f"{region.start_seconds:.2f}-{region.end_seconds:.2f} seconds"
        for region in temporal.high_attention_regions
    )
    return f"High-attention activity occurred at {regions}."


def _semantic_sentence(semantic: SemanticExplanation | None) -> str:
    if semantic is None or semantic.status != ComponentStatus.completed:
        return "Semantic acoustic evidence is unavailable."
    if not semantic.feature_importance:
        return "No ranked semantic feature contributions are available."
    features = ", ".join(
        contribution.display_name for contribution in semantic.feature_importance[:2]
    )
    return f"The largest semantic {semantic.target_type.value} contributions were {features}."


def _quality_checks(quality: ExplanationQuality | None) -> str:
    if quality is None:
        return "No per-analysis quality metrics were supplied."
    available = []
    unavailable = []
    for name, metric in quality:
        if metric is None:
            continue
        if metric.status == MetricStatus.available and metric.value is not None:
            # Keep the full-precision value in ``quality`` itself. The report is
            # a human-facing summary, so render a stable, readable precision
            # together with the metric scope.
            available.append(f"{name}={metric.value:.4f} ({metric.scope})")
            continue
        reason = f" ({metric.reason})" if metric.reason else ""
        unavailable.append(f"{name}{reason}")
    parts = []
    if available:
        parts.append("Available quality metrics: " + ", ".join(available) + ".")
    if unavailable:
        parts.append("Unavailable quality metrics: " + ", ".join(unavailable) + ".")
    return " ".join(parts) or "No per-analysis quality metrics were supplied."


def _limitations(
    classifier: ClassifierSnapshot,
    temporal: TemporalExplanation | None,
    semantic: SemanticExplanation | None,
) -> str:
    limitations = []
    if classifier.contains_dummy_branches:
        limitations.append(
            "The classifier contains development/dummy branches and is not research eligible."
        )
    for component in (temporal, semantic):
        if component is not None and component.warning:
            limitations.append(component.warning)
    if not limitations:
        limitations.append(
            "Temporal attention and semantic contributions are correlational decision-support evidence."
        )
    return " ".join(limitations)
