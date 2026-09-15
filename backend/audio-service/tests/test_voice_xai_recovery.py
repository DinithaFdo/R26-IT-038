from datetime import UTC, datetime, timedelta

import pytest

from app.schemas.common import SourceType
from app.schemas.xai import ComponentStatus, ExplanationStatus
from app.voice_xai.orchestrator import classifier_snapshot_from_prediction
from app.voice_xai.persistence.mongodb import (
    MongoXaiExplanationRepository,
    XaiPersistenceConsistencyError,
)
from app.voice_xai.recovery import (
    STALE_RUN_ERROR_CODE,
    recover_stale_explanations,
)
from tests.test_mongodb import FakeDatabase
from tests.test_voice_xai_orchestrator import _prediction


async def _create_explanation(
    repository,
    *,
    status: str,
    age_seconds: float,
    prediction_id: str = "prediction-123",
) -> str:
    prediction = _prediction()
    explanation_id = await repository.create_explanation(
        prediction_id=prediction_id,
        request_id=prediction.request_id,
        owner_user_id="user-123",
        source_type=SourceType.dashboard_upload,
        classifier_snapshot=classifier_snapshot_from_prediction(prediction),
        pipeline_version="voice-xai-pipeline-v1",
    )
    stored_at = datetime.now(UTC) - timedelta(seconds=age_seconds)
    for document in repository.collection.documents:
        if document["id"] == explanation_id:
            document["status"] = status
            document["updated_at"] = stored_at
    return explanation_id


@pytest.mark.anyio
async def test_stale_running_run_is_recovered_to_failed() -> None:
    repository = MongoXaiExplanationRepository(FakeDatabase("recovery_stale_running"))
    explanation_id = await _create_explanation(
        repository, status="running", age_seconds=600
    )

    result = await recover_stale_explanations(repository, stale_after_seconds=300)

    assert result.scanned == 1
    assert result.recovered == 1
    document = await repository.get_explanation_for_owner(
        explanation_id=explanation_id, owner_user_id="user-123"
    )
    assert document["status"] == ExplanationStatus.failed.value
    assert document["error"]["code"] == STALE_RUN_ERROR_CODE
    assert document["component_statuses"] == {
        "temporal": ComponentStatus.failed.value,
        "semantic": ComponentStatus.failed.value,
        "report": ComponentStatus.failed.value,
        "narrative": ComponentStatus.failed.value,
    }
    assert all(
        error["code"] == STALE_RUN_ERROR_CODE
        for error in document["component_errors"].values()
    )


@pytest.mark.anyio
async def test_legacy_narrative_component_status_does_not_break_recovery() -> None:
    repository = MongoXaiExplanationRepository(FakeDatabase("recovery_legacy_narrative"))
    explanation_id = await _create_explanation(
        repository, status="running", age_seconds=600
    )
    for document in repository.collection.documents:
        if document["id"] == explanation_id:
            document["component_statuses"]["narrative"] = "queued"

    result = await recover_stale_explanations(repository, stale_after_seconds=300)

    assert result.scanned == 1
    assert result.recovered == 1
    document = await repository.get_explanation_for_owner(
        explanation_id=explanation_id, owner_user_id="user-123"
    )
    assert document["status"] == ExplanationStatus.failed.value
    assert document["component_statuses"] == {
        "temporal": ComponentStatus.failed.value,
        "semantic": ComponentStatus.failed.value,
        "report": ComponentStatus.failed.value,
        "narrative": "queued",
    }


@pytest.mark.anyio
async def test_stale_queued_and_blocked_runs_are_recovered() -> None:
    repository = MongoXaiExplanationRepository(FakeDatabase("recovery_stale_various"))
    queued_id = await _create_explanation(repository, status="queued", age_seconds=600)
    blocked_id = await _create_explanation(
        repository,
        status="blocked",
        age_seconds=600,
        prediction_id="prediction-456",
    )

    result = await recover_stale_explanations(repository, stale_after_seconds=300)

    assert result.recovered == 2
    for explanation_id in (queued_id, blocked_id):
        document = await repository.get_explanation_for_owner(
            explanation_id=explanation_id, owner_user_id="user-123"
        )
        assert document["status"] == ExplanationStatus.failed.value


@pytest.mark.anyio
async def test_fresh_active_run_is_not_touched() -> None:
    repository = MongoXaiExplanationRepository(FakeDatabase("recovery_fresh_running"))
    explanation_id = await _create_explanation(
        repository, status="running", age_seconds=1
    )

    result = await recover_stale_explanations(repository, stale_after_seconds=300)

    assert result.scanned == 0
    assert result.recovered == 0
    document = await repository.get_explanation_for_owner(
        explanation_id=explanation_id, owner_user_id="user-123"
    )
    assert document["status"] == "running"


@pytest.mark.anyio
async def test_already_terminal_run_is_never_selected() -> None:
    repository = MongoXaiExplanationRepository(FakeDatabase("recovery_terminal"))
    explanation_id = await _create_explanation(
        repository, status="completed", age_seconds=600
    )

    result = await recover_stale_explanations(repository, stale_after_seconds=300)

    assert result.scanned == 0
    document = await repository.get_explanation_for_owner(
        explanation_id=explanation_id, owner_user_id="user-123"
    )
    assert document["status"] == "completed"


@pytest.mark.anyio
async def test_concurrent_status_change_during_recovery_is_skipped_not_crashed() -> None:
    repository = MongoXaiExplanationRepository(FakeDatabase("recovery_race"))
    await _create_explanation(repository, status="running", age_seconds=600)

    class RacyRepository:
        def __init__(self, inner) -> None:
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        async def transition_status(self, *args, **kwargs):
            raise XaiPersistenceConsistencyError("changed concurrently")

    result = await recover_stale_explanations(
        RacyRepository(repository), stale_after_seconds=300
    )

    assert result.scanned == 1
    assert result.recovered == 0
    assert result.skipped == 1
