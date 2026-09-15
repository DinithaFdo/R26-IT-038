from app.schemas.common import BranchStatus, ModelMode, PredictionLabel
from app.schemas.xai import (
    CanonicalXaiBranch,
    ClassifierBranchSnapshot,
    ClassifierSnapshot,
    ComponentStatus,
    ExplanationQuality,
    MetricEvidence,
    MetricStatus,
    SemanticExplanation,
    SemanticFeatureContribution,
    SemanticTargetType,
    ShapDirection,
    TemporalExplanation,
    TemporalRegion,
)
from app.voice_xai.report.service import DeterministicReportComposer


def test_report_composer_is_deterministic_and_uses_controlled_evidence_text() -> None:
    composer = DeterministicReportComposer()
    classifier = ClassifierSnapshot(
        verdict=PredictionLabel.spoof,
        spoof_probability=0.8,
        bonafide_probability=0.2,
        confidence=0.8,
        decision_threshold=0.5,
        contains_dummy_branches=True,
        research_eligible=False,
        branches=[
            ClassifierBranchSnapshot(
                branch_name=CanonicalXaiBranch.lfcc_cnn_tcn,
                model_name="cnn_acoustic",
                status=BranchStatus.success,
                mode=ModelMode.dummy,
                spoof_probability=0.8,
            )
        ],
    )
    temporal = TemporalExplanation(
        status=ComponentStatus.completed,
        method_version="fixture-v1",
        attention_score_peak=0.9,
        threshold_percentile=80,
        regions=[
            TemporalRegion(
                region_id=1,
                start_seconds=1.82,
                end_seconds=2.30,
                attention_score=0.9,
            )
        ],
        combined_region_duration_seconds=0.48,
    )
    semantic = SemanticExplanation(
        status=ComponentStatus.completed,
        target_type=SemanticTargetType.classifier_surrogate,
        model_version="fixture-v1",
        feature_schema_version="esvas-acoustic-28-v1",
        extractor_version="fixture-v1",
        output_space="raw_margin",
        feature_importance=[
            SemanticFeatureContribution(
                rank=1,
                feature_name="jitter",
                display_name="Jitter",
                value=0.01,
                shap_value=0.2,
                direction=ShapDirection.toward_spoof,
            )
        ],
    )

    first = composer.compose(classifier=classifier, temporal=temporal, semantic=semantic)
    second = composer.compose(classifier=classifier, temporal=temporal, semantic=semantic)

    assert first == second
    assert "spoof score of 0.80" in first.finding
    assert "1.82-2.30 seconds" in first.finding
    assert "requires qualified human review" in first.disclaimer


def test_report_includes_available_iou_value_and_scope() -> None:
    composer = DeterministicReportComposer()
    classifier = ClassifierSnapshot(
        verdict=PredictionLabel.spoof,
        spoof_probability=0.8,
        bonafide_probability=0.2,
        confidence=0.8,
        decision_threshold=0.5,
        contains_dummy_branches=False,
        research_eligible=True,
        branches=[
            ClassifierBranchSnapshot(
                branch_name=CanonicalXaiBranch.lfcc_cnn_tcn,
                model_name="cnn_acoustic",
                status=BranchStatus.success,
                mode=ModelMode.real,
                spoof_probability=0.8,
            )
        ],
    )
    quality = ExplanationQuality(
        temporal_semantic_iou=MetricEvidence(
            status=MetricStatus.available,
            value=1 / 3,
            scope="per_analysis",
        )
    )

    report = composer.compose(
        classifier=classifier,
        temporal=None,
        semantic=None,
        quality=quality,
    )

    assert report.quality_checks == (
        "Available quality metrics: "
        "temporal_semantic_iou=0.3333 (per_analysis)."
    )
