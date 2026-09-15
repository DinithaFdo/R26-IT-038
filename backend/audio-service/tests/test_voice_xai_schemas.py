from datetime import UTC, datetime
import math

import pytest
from pydantic import ValidationError

from app.schemas.common import BranchStatus, ModelMode, PredictionLabel
from app.schemas.xai import (
    XAI_SCHEMA_VERSION,
    CanonicalXaiBranch,
    ClassifierBranchSnapshot,
    ClassifierSnapshot,
    CombinedExplanationReport,
    ComponentStatus,
    ExplanationComponentStatuses,
    ExplanationComponentErrors,
    ExplanationComponent,
    ExplanationError,
    ExplanationQuality,
    ExplanationProvenance,
    ExplanationStatus,
    MetricEvidence,
    MetricStatus,
    ReportDisposition,
    SemanticFeatureContribution,
    SemanticClipSummary,
    ShapDirection,
    TemporalRegion,
    XaiExplanationResponse,
)


def test_xai_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        TemporalRegion(
            region_id=1,
            start_seconds=0.1,
            end_seconds=0.2,
            attention_score=0.8,
            unsupported=True,
        )


def test_classifier_snapshot_rejects_dummy_research_result() -> None:
    with pytest.raises(ValidationError, match="not research eligible"):
        _classifier_snapshot(research_eligible=True)


def test_classifier_snapshot_rejects_inconsistent_dummy_summary() -> None:
    payload = _classifier_snapshot(research_eligible=False).model_dump()
    payload["contains_dummy_branches"] = False

    with pytest.raises(ValidationError, match="dummy-branch summary"):
        ClassifierSnapshot.model_validate(payload)


def test_metric_requires_value_or_unavailability_reason() -> None:
    with pytest.raises(ValidationError, match="require a value"):
        MetricEvidence(
            status=MetricStatus.available,
            value=None,
            scope="offline_validation",
        )

    with pytest.raises(ValidationError, match="require a reason"):
        MetricEvidence(
            status=MetricStatus.not_computed,
            scope="per_analysis",
        )


def test_quality_reads_legacy_null_metrics_but_does_not_serialize_them() -> None:
    quality = ExplanationQuality.model_validate(
        {
            "surrogate_fidelity_r2": None,
            "temporal_localization_iou": None,
            "temporal_semantic_iou": None,
            "robustness_stability": None,
        }
    )

    assert quality.model_dump() == {
        "surrogate_fidelity_r2": None,
        "temporal_semantic_iou": None,
    }


def test_temporal_and_semantic_values_reject_invalid_numbers_and_windows() -> None:
    with pytest.raises(ValidationError, match="after its start"):
        TemporalRegion(
            region_id=1,
            start_seconds=0.4,
            end_seconds=0.3,
            attention_score=0.8,
        )

    with pytest.raises(ValidationError):
        SemanticFeatureContribution(
            rank=1,
            feature_name="jitter",
            display_name="Pitch jitter",
            value=math.nan,
            shap_value=0.3,
            direction=ShapDirection.toward_spoof,
        )

    with pytest.raises(ValidationError, match="predicted_label"):
        SemanticClipSummary(
            spoof_probability=0.2,
            bonafide_probability=0.8,
            decision_threshold=0.5,
            predicted_label=PredictionLabel.spoof,
            window_count=1,
        )


def test_complete_xai_response_has_versioned_independent_status() -> None:
    now = datetime.now(UTC)
    report = CombinedExplanationReport(
        status=ComponentStatus.completed,
        disposition=ReportDisposition.inconclusive,
        finding="Classifier output requires review.",
        primary_evidence="No validated real explanation is available.",
        quality_checks="Research quality metrics are not yet available.",
        limitation="Classifier branches are development placeholders.",
        recommendation="Review the source audio and provenance.",
        disclaimer="Explanation output is decision-support evidence only.",
    )

    response = XaiExplanationResponse(
        explanation_id="explanation-1",
        prediction_id="prediction-1",
        request_id="request-1",
        status=ExplanationStatus.completed,
        component_statuses=ExplanationComponentStatuses(
            temporal=ComponentStatus.not_available,
            semantic=ComponentStatus.not_available,
            report=ComponentStatus.completed,
        ),
        component_errors=ExplanationComponentErrors(
            temporal=ExplanationError(
                component=ExplanationComponent.temporal,
                code="attention_unavailable",
                message="Temporal attention is unavailable.",
            ),
            semantic=ExplanationError(
                component=ExplanationComponent.semantic,
                code="semantic_model_unavailable",
                message="Semantic explanation is unavailable.",
            ),
        ),
        classifier_snapshot=_classifier_snapshot(research_eligible=False),
        quality=ExplanationQuality(
            surrogate_fidelity_r2=MetricEvidence(
                status=MetricStatus.available,
                value=0.6342774342328266,
                scope="offline_validation",
                dataset_version="ASVspoof2019_LA",
            ),
            temporal_semantic_iou=MetricEvidence(
                status=MetricStatus.not_computed,
                scope="per_analysis",
                reason="Temporal or semantic explanation is unavailable.",
            ),
        ),
        combined_report=report,
        provenance=ExplanationProvenance(
            pipeline_version="voice-xai-pipeline-v1",
            classifier_contract_version="classifier-xai-v1",
        ),
        created_at=now,
        updated_at=now,
        completed_at=now,
    )

    payload = response.model_dump(mode="json")
    assert payload["schema_version"] == XAI_SCHEMA_VERSION
    assert payload["status"] == "completed"
    assert payload["classifier_snapshot"]["research_eligible"] is False
    assert payload["quality"] == {
        "surrogate_fidelity_r2": {
            "status": "available",
            "value": pytest.approx(0.6342774342328266),
            "scope": "offline_validation",
            "reason": None,
            "dataset_version": "ASVspoof2019_LA",
        },
        "temporal_semantic_iou": {
            "status": "not_computed",
            "value": None,
            "scope": "per_analysis",
            "reason": "Temporal or semantic explanation is unavailable.",
            "dataset_version": None,
        },
    }


def test_xai_response_rejects_inconsistent_terminal_state() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="require completed_at"):
        XaiExplanationResponse(
            explanation_id="explanation-1",
            prediction_id="prediction-1",
            request_id="request-1",
            status=ExplanationStatus.completed,
            component_statuses=ExplanationComponentStatuses(),
            classifier_snapshot=_classifier_snapshot(research_eligible=False),
            provenance=ExplanationProvenance(
                pipeline_version="voice-xai-pipeline-v1",
                classifier_contract_version="classifier-xai-v1",
            ),
            created_at=now,
            updated_at=now,
        )


def test_xai_response_rejects_component_status_without_safe_error() -> None:
    now = datetime.now(UTC)
    with pytest.raises(ValidationError, match="requires a safe component error"):
        XaiExplanationResponse(
            explanation_id="explanation-1",
            prediction_id="prediction-1",
            request_id="request-1",
            status=ExplanationStatus.partial,
            component_statuses=ExplanationComponentStatuses(
                temporal=ComponentStatus.not_available,
            ),
            classifier_snapshot=_classifier_snapshot(research_eligible=False),
            provenance=ExplanationProvenance(
                pipeline_version="voice-xai-pipeline-v1",
                classifier_contract_version="classifier-xai-v1",
            ),
            created_at=now,
            updated_at=now,
        )


def test_xai_can_be_research_ineligible_when_classifier_is_ready() -> None:
    now = datetime.now(UTC)
    classifier_payload = _classifier_snapshot(research_eligible=False).model_dump()
    classifier_payload["contains_dummy_branches"] = False
    classifier_payload["research_eligible"] = True
    classifier_payload["branches"][0]["mode"] = ModelMode.real
    response = XaiExplanationResponse(
        explanation_id="explanation-1",
        prediction_id="prediction-1",
        request_id="request-1",
        development_placeholder=True,
        research_eligible=False,
        status=ExplanationStatus.queued,
        component_statuses=ExplanationComponentStatuses(),
        classifier_snapshot=ClassifierSnapshot.model_validate(classifier_payload),
        provenance=ExplanationProvenance(
            pipeline_version="voice-xai-pipeline-v1",
            classifier_contract_version="classifier-xai-v1",
        ),
        created_at=now,
        updated_at=now,
    )
    assert response.research_eligible is False


def _classifier_snapshot(*, research_eligible: bool) -> ClassifierSnapshot:
    return ClassifierSnapshot(
        verdict=PredictionLabel.spoof,
        spoof_probability=0.8,
        bonafide_probability=0.2,
        confidence=0.8,
        decision_threshold=0.5,
        contains_dummy_branches=True,
        research_eligible=research_eligible,
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
