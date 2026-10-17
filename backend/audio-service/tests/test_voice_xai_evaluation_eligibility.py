from dataclasses import asdict

import pytest

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
    OFFLINE_SEMANTIC_AOPC,
    OFFLINE_TEMPORAL_AOPC,
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
        attention_threshold=0.8,
        high_attention_regions=[],
        high_attention_combined_duration_seconds=0.0,
    )


def _semantic(*, eligible: bool) -> SemanticExplanation:
    return SemanticExplanation(
        status=ComponentStatus.completed,
        target_type=SemanticTargetType.classifier_surrogate,
        model_version="v1",
        feature_schema_version="fixture-features-v1",
        extractor_version="v1",
        output_space="raw_regression",
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


def test_selected_offline_metrics_preserve_source_and_negative_temporal_comparison():
    decision = evaluate_research_eligibility(
        classifier=_classifier(dummy=False, eligible=True),
        temporal=_temporal(eligible=True), semantic=_semantic(eligible=True),
    )
    metrics = {metric['name']: metric for metric in asdict(decision)['offline_metrics']}
    assert metrics['semantic_aopc']['value'] == pytest.approx(0.517231721941027)
    assert metrics['temporal_aopc']['value'] == pytest.approx(0.1838843630005916)
    assert all(metric['scope'] == 'offline_validation' and metric['n_audio'] == 1
               and metric['fixed_reference'] and metric['audio_id'] == 'LA_D_2302632.wav'
               for metric in metrics.values())
    assert OFFLINE_SEMANTIC_AOPC.advantage_over_random > 0
    assert OFFLINE_TEMPORAL_AOPC.advantage_over_random == pytest.approx(-0.0869459662780476)
    assert decision.eligible is True  # Eligibility is separate from a ranking pass/fail test.


@pytest.mark.parametrize('component', ['temporal', 'semantic'])
@pytest.mark.parametrize('status', [ComponentStatus.running, ComponentStatus.failed])
def test_fixed_offline_values_cannot_make_incomplete_components_eligible(component, status):
    components = {'temporal': _temporal(eligible=True), 'semantic': _semantic(eligible=True)}
    components[component] = components[component].model_copy(update={'status': status})
    decision = evaluate_research_eligibility(classifier=_classifier(dummy=False, eligible=True), **components)
    assert decision.eligible is False
    assert f'{component}_not_research_eligible' in decision.unmet_requirements


def test_placeholder_semantic_cannot_be_enabled_by_a_true_flag():
    semantic = _semantic(eligible=True).model_copy(update={'development_placeholder': True})
    decision = evaluate_research_eligibility(
        classifier=_classifier(dummy=False, eligible=True), temporal=_temporal(eligible=True), semantic=semantic,
    )
    assert decision.eligible is False
