from app.schemas.common import BranchStatus, ModelMode, PredictionLabel
from app.schemas.xai import (
    CanonicalXaiBranch,
    ClassifierBranchSnapshot,
    ClassifierSnapshot,
    ComponentStatus,
    SemanticExplanation,
    SemanticTargetType,
    TemporalExplanation,
)
from app.voice_xai.evaluation.eligibility import (
    RESEARCH_ELIGIBILITY_POLICY_VERSION,
    evaluate_research_eligibility,
)


def _classifier(*, dummy: bool, eligible: bool) -> ClassifierSnapshot:
    return ClassifierSnapshot(
        verdict=PredictionLabel.spoof,
        spoof_probability=0.8,
        bonafide_probability=0.2,
        confidence=0.8,
        decision_threshold=0.5,
        contains_dummy_branches=dummy,
        research_eligible=eligible,
        branches=[
            ClassifierBranchSnapshot(
                branch_name=CanonicalXaiBranch.lfcc_cnn_tcn,
                model_name="cnn_acoustic",
                status=BranchStatus.success,
                mode=ModelMode.dummy if dummy else ModelMode.real,
                spoof_probability=0.8,
            )
        ],
    )


def _temporal(*, eligible: bool) -> TemporalExplanation:
    return TemporalExplanation(
        status=ComponentStatus.completed,
        method_version="v1",
        development_placeholder=not eligible,
        research_eligible=eligible,
        attention_score_peak=0.9,
        threshold_percentile=80.0,
        regions=[],
        combined_region_duration_seconds=0.0,
    )


def _semantic(*, eligible: bool) -> SemanticExplanation:
    return SemanticExplanation(
        status=ComponentStatus.completed,
        target_type=SemanticTargetType.independent_acoustic_evidence_model,
        model_version="v1",
        feature_schema_version="esvas-acoustic-28-v1",
        extractor_version="v1",
        output_space="probability",
        development_placeholder=not eligible,
        research_eligible=eligible,
        base_value=0.0,
        predicted_value=0.5,
    )


def test_default_dummy_mock_run_is_never_research_eligible() -> None:
    decision = evaluate_research_eligibility(
        classifier=_classifier(dummy=True, eligible=False),
        temporal=_temporal(eligible=False),
        semantic=_semantic(eligible=False),
    )

    assert decision.eligible is False
    assert decision.policy_version == RESEARCH_ELIGIBILITY_POLICY_VERSION
    assert "classifier_contains_dummy_branches" in decision.unmet_requirements
    assert "classifier_not_research_eligible" in decision.unmet_requirements
    assert "temporal_not_research_eligible" in decision.unmet_requirements
    assert "semantic_not_research_eligible" in decision.unmet_requirements


def test_missing_components_are_reported_as_unmet_not_silently_ignored() -> None:
    decision = evaluate_research_eligibility(
        classifier=_classifier(dummy=False, eligible=True),
        temporal=None,
        semantic=None,
    )

    assert decision.eligible is False
    assert "temporal_missing" in decision.unmet_requirements
    assert "semantic_missing" in decision.unmet_requirements


def test_every_requirement_satisfied_is_eligible() -> None:
    decision = evaluate_research_eligibility(
        classifier=_classifier(dummy=False, eligible=True),
        temporal=_temporal(eligible=True),
        semantic=_semantic(eligible=True),
    )

    assert decision.eligible is True
    assert decision.unmet_requirements == ()


def test_one_unmet_requirement_fails_the_whole_decision_closed() -> None:
    decision = evaluate_research_eligibility(
        classifier=_classifier(dummy=False, eligible=True),
        temporal=_temporal(eligible=True),
        semantic=_semantic(eligible=False),
    )

    assert decision.eligible is False
    assert decision.unmet_requirements == ("semantic_not_research_eligible",)
