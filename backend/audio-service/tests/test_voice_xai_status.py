import pytest

from app.schemas.common import BranchStatus, ModelMode, PredictionLabel, SourceType
from app.schemas.xai import (
    CanonicalXaiBranch,
    ClassifierBranchSnapshot,
    ClassifierSnapshot,
    CombinedExplanationReport,
    ComponentStatus,
    ExplanationComponent,
    ExplanationComponentStatuses,
    ExplanationError,
    ExplanationStatus,
    ReportDisposition,
    TemporalExplanation,
    TemporalRegion,
)
from app.voice_xai.persistence.mongodb import MongoXaiExplanationRepository
from app.voice_xai.status import (
    XaiExplanationNotFoundError,
    XaiStatusLifecycleError,
    XaiStatusService,
    derive_explanation_status,
)
from tests.test_mongodb import FakeDatabase


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        (ExplanationComponentStatuses(), ExplanationStatus.queued),
        (
            ExplanationComponentStatuses(temporal=ComponentStatus.running),
            ExplanationStatus.running,
        ),
        (
            ExplanationComponentStatuses(
                temporal=ComponentStatus.completed,
                semantic=ComponentStatus.running,
            ),
            ExplanationStatus.partial,
        ),
        (
            ExplanationComponentStatuses(temporal=ComponentStatus.blocked),
            ExplanationStatus.blocked,
        ),
        (
            ExplanationComponentStatuses(
                temporal=ComponentStatus.completed,
                semantic=ComponentStatus.not_available,
                report=ComponentStatus.completed,
            ),
            ExplanationStatus.completed,
        ),
        (
            ExplanationComponentStatuses(report=ComponentStatus.failed),
            ExplanationStatus.failed,
        ),
    ],
)
def test_derive_explanation_status(
    statuses: ExplanationComponentStatuses,
    expected: ExplanationStatus,
) -> None:
    assert derive_explanation_status(statuses) == expected


@pytest.mark.anyio
async def test_status_service_completes_report_with_partial_input_evidence() -> None:
    database = FakeDatabase("xai_status_complete_test")
    repository = MongoXaiExplanationRepository(database)
    service = XaiStatusService(repository)
    explanation_id = await _create_explanation(repository)

    await service.complete_component(
        explanation_id=explanation_id,
        owner_user_id="user-123",
        component=ExplanationComponent.temporal,
        result=_temporal_result(),
    )
    document = await _get(repository, explanation_id)
    assert document["status"] == "partial"

    semantic_error = ExplanationError(
        component=ExplanationComponent.semantic,
        code="semantic_model_unavailable",
        message="The semantic model is unavailable.",
    )
    await service.mark_component_not_available(
        explanation_id=explanation_id,
        owner_user_id="user-123",
        component=ExplanationComponent.semantic,
        error=semantic_error,
    )
    await service.complete_component(
        explanation_id=explanation_id,
        owner_user_id="user-123",
        component=ExplanationComponent.report,
        result=_report_result(),
    )
    await service.mark_component_not_available(
        explanation_id=explanation_id,
        owner_user_id="user-123",
        component=ExplanationComponent.narrative,
        error=ExplanationError(
            component=ExplanationComponent.narrative,
            code="narrative_disabled",
            message="AI narrative generation is disabled.",
        ),
    )

    document = await _get(repository, explanation_id)
    assert document["status"] == "completed"
    assert document["component_statuses"] == {
        "temporal": "completed",
        "semantic": "not_available",
        "report": "completed",
        "narrative": "not_available",
    }
    assert document["component_errors"]["semantic"] == semantic_error.model_dump(
        mode="python"
    )
    assert document["error"] is None
    assert document["completed_at"].tzinfo is not None
    assert "prediction_status" not in document


@pytest.mark.anyio
async def test_report_failure_is_the_only_component_failure_that_fails_run() -> None:
    database = FakeDatabase("xai_status_failure_test")
    repository = MongoXaiExplanationRepository(database)
    service = XaiStatusService(repository)
    explanation_id = await _create_explanation(repository)

    await service.fail_component(
        explanation_id=explanation_id,
        owner_user_id="user-123",
        component=ExplanationComponent.temporal,
        error=ExplanationError(
            component=ExplanationComponent.temporal,
            code="attention_unavailable",
            message="Attention tensors are unavailable.",
        ),
    )
    document = await _get(repository, explanation_id)
    assert document["status"] == "partial"
    assert document["completed_at"] is None

    report_error = ExplanationError(
        component=ExplanationComponent.report,
        code="report_generation_failed",
        message="The combined report could not be generated.",
    )
    await service.fail_component(
        explanation_id=explanation_id,
        owner_user_id="user-123",
        component=ExplanationComponent.report,
        error=report_error,
    )
    document = await _get(repository, explanation_id)
    assert document["status"] == "failed"
    assert document["error"] == report_error.model_dump(mode="python")
    assert document["completed_at"].tzinfo is not None


@pytest.mark.anyio
async def test_blocked_component_must_be_explicitly_requeued() -> None:
    database = FakeDatabase("xai_status_blocked_test")
    repository = MongoXaiExplanationRepository(database)
    service = XaiStatusService(repository)
    explanation_id = await _create_explanation(repository)

    await service.block_component(
        explanation_id=explanation_id,
        owner_user_id="user-123",
        component=ExplanationComponent.temporal,
        error=ExplanationError(
            component=ExplanationComponent.temporal,
            code="waiting_for_real_model",
            message="A real SSL model is required.",
        ),
    )
    document = await _get(repository, explanation_id)
    assert document["status"] == "blocked"

    with pytest.raises(XaiStatusLifecycleError, match="must requeue"):
        await service.start_component(
            explanation_id=explanation_id,
            owner_user_id="user-123",
            component=ExplanationComponent.semantic,
        )

    await service.requeue_component(
        explanation_id=explanation_id,
        owner_user_id="user-123",
        component=ExplanationComponent.temporal,
    )
    document = await _get(repository, explanation_id)
    assert document["status"] == "queued"
    assert document["component_statuses"]["temporal"] == "queued"
    assert document["component_errors"]["temporal"] is None

    await service.start_component(
        explanation_id=explanation_id,
        owner_user_id="user-123",
        component=ExplanationComponent.temporal,
    )
    document = await _get(repository, explanation_id)
    assert document["status"] == "running"


@pytest.mark.anyio
async def test_report_cannot_start_before_input_components_are_terminal() -> None:
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_report_gate_test"))
    service = XaiStatusService(repository)
    explanation_id = await _create_explanation(repository)

    with pytest.raises(XaiStatusLifecycleError, match="requires terminal"):
        await service.start_component(
            explanation_id=explanation_id,
            owner_user_id="user-123",
            component=ExplanationComponent.report,
        )


@pytest.mark.anyio
async def test_status_service_preserves_owner_isolation() -> None:
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_owner_test"))
    service = XaiStatusService(repository)
    explanation_id = await _create_explanation(repository)

    with pytest.raises(XaiExplanationNotFoundError):
        await service.start_run(
            explanation_id=explanation_id,
            owner_user_id="different-user",
        )


async def _create_explanation(repository: MongoXaiExplanationRepository) -> str:
    return await repository.create_explanation(
        prediction_id="prediction-123",
        request_id="request-123",
        owner_user_id="user-123",
        source_type=SourceType.dashboard_upload,
        classifier_snapshot=_classifier_snapshot(),
        pipeline_version="voice-xai-pipeline-v1",
    )


async def _get(
    repository: MongoXaiExplanationRepository,
    explanation_id: str,
):
    return await repository.get_explanation_for_owner(
        explanation_id=explanation_id,
        owner_user_id="user-123",
    )


def _classifier_snapshot() -> ClassifierSnapshot:
    return ClassifierSnapshot(
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


def _temporal_result() -> TemporalExplanation:
    return TemporalExplanation(
        status=ComponentStatus.completed,
        method_version="attention-rollout-v1",
        attention_score_peak=0.9,
        threshold_percentile=80,
        regions=[
            TemporalRegion(
                region_id=1,
                start_seconds=0.2,
                end_seconds=0.5,
                attention_score=0.9,
            )
        ],
        combined_region_duration_seconds=0.3,
    )


def _report_result() -> CombinedExplanationReport:
    return CombinedExplanationReport(
        status=ComponentStatus.completed,
        disposition=ReportDisposition.inconclusive,
        finding="Classifier output requires qualified review.",
        primary_evidence="Temporal evidence is available.",
        quality_checks="Semantic evidence is unavailable.",
        limitation="The semantic model was unavailable.",
        recommendation="Review the original audio and provenance.",
        disclaimer="Explanation output is decision-support evidence only.",
    )
