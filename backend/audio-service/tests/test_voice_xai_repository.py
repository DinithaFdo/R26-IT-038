from datetime import UTC, datetime

import pytest

from app.schemas.common import BranchStatus, ModelMode, PredictionLabel, SourceType
from app.schemas.xai import (
    CanonicalXaiBranch,
    ClassifierBranchSnapshot,
    ClassifierSnapshot,
    ComponentStatus,
    ExplanationComponent,
    ExplanationError,
    ExplanationStatus,
    TemporalExplanation,
    TemporalRegion,
)
from app.voice_xai.persistence.mongodb import (
    MongoXaiExplanationRepository,
    XaiPersistenceConsistencyError,
)
from tests.test_mongodb import FakeDatabase


def test_xai_repository_accepts_a_database_that_rejects_boolean_coercion() -> None:
    class AsyncDatabaseLike:
        def __bool__(self) -> bool:
            raise NotImplementedError("AsyncDatabase objects do not support bool().")

        def __getitem__(self, name: str):
            return f"collection:{name}"

    repository = MongoXaiExplanationRepository(AsyncDatabaseLike())

    assert repository.collection == "collection:xai_explanations"


@pytest.mark.anyio
async def test_xai_repository_creates_versioned_owner_scoped_record() -> None:
    database = FakeDatabase("xai_repository_test")
    repository = MongoXaiExplanationRepository(database)

    explanation_id = await _create_explanation(repository)
    document = await repository.get_explanation_for_owner(
        explanation_id=explanation_id,
        owner_user_id="user-123",
    )

    assert document is not None
    assert document["id"] == explanation_id
    assert document["schema_version"] == "voice-xai-api-v1"
    assert document["pipeline_version"] == "voice-xai-pipeline-v1"
    assert document["status"] == "queued"
    assert document["component_statuses"] == {
        "temporal": "queued",
        "semantic": "queued",
        "report": "queued",
        "narrative": "queued",
    }
    assert document["classifier_snapshot"]["branches"][0]["branch_name"] == (
        "lfcc_cnn_tcn"
    )
    assert document["provenance"]["classifier_contract_version"] == (
        "classifier-xai-v1"
    )
    assert document["created_at"].tzinfo is not None
    assert document["updated_at"].tzinfo is not None

    assert (
        await repository.get_explanation_for_owner(
            explanation_id=explanation_id,
            owner_user_id="another-user",
        )
        is None
    )


@pytest.mark.anyio
async def test_xai_repository_persists_component_and_status_history() -> None:
    database = FakeDatabase("xai_component_test")
    repository = MongoXaiExplanationRepository(database)
    explanation_id = await _create_explanation(repository)

    await repository.transition_status(explanation_id, ExplanationStatus.running)
    await repository.update_component_status(
        explanation_id,
        ExplanationComponent.temporal,
        ComponentStatus.running,
    )
    await repository.save_component_result(
        explanation_id,
        ExplanationComponent.temporal,
        TemporalExplanation(
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
        ),
    )
    await repository.transition_status(explanation_id, ExplanationStatus.partial)

    document = await repository.get_explanation_for_owner(
        explanation_id=explanation_id,
        owner_user_id="user-123",
    )
    assert document["component_statuses"]["temporal"] == "completed"
    assert document["temporal"]["score_kind"] == "high_attention"
    assert [entry["status"] for entry in document["status_history"]] == [
        "queued",
        "running",
        "partial",
    ]


@pytest.mark.anyio
async def test_xai_repository_rejects_invalid_or_unsafe_terminal_updates() -> None:
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_terminal_test"))
    explanation_id = await _create_explanation(repository)

    with pytest.raises(XaiPersistenceConsistencyError, match="Invalid XAI"):
        await repository.transition_status(
            explanation_id,
            ExplanationStatus.completed,
        )

    await repository.transition_status(explanation_id, ExplanationStatus.running)
    with pytest.raises(ValueError, match="require a safe error"):
        await repository.transition_status(explanation_id, ExplanationStatus.failed)

    await repository.transition_status(
        explanation_id,
        ExplanationStatus.failed,
        error=ExplanationError(
            component=ExplanationComponent.temporal,
            code="attention_unavailable",
            message="Temporal attention is unavailable.",
        ),
    )

    with pytest.raises(XaiPersistenceConsistencyError, match="terminal state"):
        await repository.update_component_status(
            explanation_id,
            ExplanationComponent.semantic,
            ComponentStatus.running,
        )


@pytest.mark.anyio
async def test_xai_repository_returns_latest_owner_prediction_run() -> None:
    database = FakeDatabase("xai_latest_test")
    repository = MongoXaiExplanationRepository(database)
    first_id = await _create_explanation(repository)
    await repository.transition_status(first_id, ExplanationStatus.blocked)
    database["xai_explanations"].documents[0]["created_at"] = datetime(
        2020,
        1,
        1,
        tzinfo=UTC,
    )
    second_id = await _create_explanation(repository)

    latest = await repository.get_latest_explanation_for_prediction(
        prediction_id="prediction-123",
        owner_user_id="user-123",
    )

    assert latest is not None
    assert latest["id"] == second_id
    assert latest["id"] != first_id


@pytest.mark.anyio
async def test_xai_repository_reuses_active_prediction_run_idempotently() -> None:
    database = FakeDatabase("xai_active_idempotency")
    repository = MongoXaiExplanationRepository(database)

    first_id = await _create_explanation(repository)
    second_id = await _create_explanation(repository)

    documents = database["xai_explanations"].documents
    assert second_id == first_id
    assert len(documents) == 1


@pytest.mark.anyio
async def test_xai_repository_allows_new_run_after_active_run_is_blocked() -> None:
    database = FakeDatabase("xai_active_blocked_retry")
    repository = MongoXaiExplanationRepository(database)

    first_id = await _create_explanation(repository)
    await repository.transition_status(first_id, ExplanationStatus.blocked)
    second_id = await _create_explanation(repository)

    assert second_id != first_id
    assert len(database["xai_explanations"].documents) == 2


@pytest.mark.anyio
async def test_xai_repository_deletes_only_owner_scoped_prediction_runs() -> None:
    database = FakeDatabase("xai_prediction_delete_test")
    repository = MongoXaiExplanationRepository(database)
    matching_id = await _create_explanation(repository)
    other_prediction_id = await repository.create_explanation(
        prediction_id="other-prediction",
        request_id="request-other-prediction",
        owner_user_id="user-123",
        source_type=SourceType.dashboard_upload,
        classifier_snapshot=_classifier_snapshot(),
        pipeline_version="voice-xai-pipeline-v1",
    )
    other_owner_id = await repository.create_explanation(
        prediction_id="prediction-123",
        request_id="request-other-owner",
        owner_user_id="other-user",
        source_type=SourceType.dashboard_upload,
        classifier_snapshot=_classifier_snapshot(),
        pipeline_version="voice-xai-pipeline-v1",
    )

    matches = await repository.list_explanations_for_prediction_for_owner(
        prediction_id="prediction-123",
        owner_user_id="user-123",
    )
    deleted = await repository.delete_explanations_for_prediction_for_owner(
        prediction_id="prediction-123",
        owner_user_id="user-123",
    )

    assert [document["id"] for document in matches] == [matching_id]
    assert deleted == 1
    assert await repository.get_explanation_for_owner(
        explanation_id=matching_id,
        owner_user_id="user-123",
    ) is None
    assert await repository.get_explanation_for_owner(
        explanation_id=other_prediction_id,
        owner_user_id="user-123",
    ) is not None
    assert await repository.get_explanation_for_owner(
        explanation_id=other_owner_id,
        owner_user_id="other-user",
    ) is not None


async def _create_explanation(repository: MongoXaiExplanationRepository) -> str:
    return await repository.create_explanation(
        prediction_id="prediction-123",
        request_id="request-123",
        owner_user_id="user-123",
        source_type=SourceType.dashboard_upload,
        classifier_snapshot=_classifier_snapshot(),
        pipeline_version="voice-xai-pipeline-v1",
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
