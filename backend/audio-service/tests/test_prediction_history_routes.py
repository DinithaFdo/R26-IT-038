import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_prediction_history_service
from app.api.dependencies import get_prediction_rerun_service
from app.auth.clerk_auth import require_clerk_user
from app.auth.schemas import AuthPrincipal
from app.config.settings import Settings
from app.core.exceptions import CorruptedAudioError
from app.ingestion.audio import AudioUploadMetadata
from app.main import app
from app.schemas.common import BranchStatus, ModelMode, PredictionLabel
from app.schemas.common import PredictionStatus
from app.schemas.common import is_valid_prediction_status_transition
from app.schemas.prediction import AudioMetadata, AudioStorageMetadata
from app.schemas.prediction import BranchPrediction, FusionResult
from app.schemas.prediction import ProbabilityScores, VoicePredictionResponse
from app.services.prediction_history_service import PredictionHistoryService
from app.services.prediction_job_runner import InlinePredictionJobRunner
from app.services.prediction_persistence_service import PredictionPersistenceService
from app.services.prediction_rerun_service import PredictionRerunService


def test_own_record_detail_access_does_not_expose_cloudinary_public_id() -> None:
    repository = FakeHistoryRepository([prediction_document()])
    with history_client(repository, user_id="user_123") as client:
        response = client.get("/api/v1/me/predictions/prediction_1")

    assert response.status_code == 200
    body = response.json()
    assert body["prediction_id"] == "prediction_1"
    assert body["audio"]["detected_codec"] == "pcm_s16le"
    assert "public_id" not in str(body)
    assert body["research_eligible"] is False
    assert body["warnings"]


def test_cross_user_prediction_access_returns_404() -> None:
    repository = FakeHistoryRepository([prediction_document()])
    with history_client(repository, user_id="user_456") as client:
        response = client.get("/api/v1/me/predictions/prediction_1")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


def test_list_predictions_uses_newest_first_pagination_and_ignores_owner_query() -> None:
    now = datetime.now(UTC)
    repository = FakeHistoryRepository(
        [
            prediction_document(
                prediction_id="oldest",
                owner_user_id="user_123",
                created_at=now - timedelta(days=2),
            ),
            prediction_document(
                prediction_id="middle",
                owner_user_id="user_123",
                created_at=now - timedelta(days=1),
            ),
            prediction_document(
                prediction_id="newest",
                owner_user_id="user_123",
                created_at=now,
            ),
            prediction_document(
                prediction_id="other-user",
                owner_user_id="user_456",
                created_at=now + timedelta(days=1),
            ),
        ]
    )
    with history_client(repository, user_id="user_123") as client:
        response = client.get(
            "/api/v1/me/predictions",
            params={"limit": 2, "page": 1, "owner_user_id": "user_456"},
        )

    assert response.status_code == 200
    body = response.json()
    assert [item["prediction_id"] for item in body["items"]] == [
        "newest",
        "middle",
    ]
    assert body["has_next"] is True


def test_list_predictions_filters_status_source_label_and_date_range() -> None:
    now = datetime.now(UTC)
    repository = FakeHistoryRepository(
        [
            prediction_document(
                prediction_id="match",
                status="completed",
                source_type="dashboard_upload",
                final_prediction="spoof",
                created_at=now,
            ),
            prediction_document(
                prediction_id="wrong-label",
                status="completed",
                source_type="dashboard_upload",
                final_prediction="bonafide",
                created_at=now,
            ),
            prediction_document(
                prediction_id="wrong-source",
                status="completed",
                source_type="public_api",
                final_prediction="spoof",
                created_at=now,
            ),
            prediction_document(
                prediction_id="too-old",
                status="completed",
                source_type="dashboard_upload",
                final_prediction="spoof",
                created_at=now - timedelta(days=10),
            ),
        ]
    )
    with history_client(repository, user_id="user_123") as client:
        response = client.get(
            "/api/v1/me/predictions",
            params={
                "status": "completed",
                "source_type": "dashboard_upload",
                "prediction_label": "spoof",
                "created_from": (now - timedelta(days=1)).isoformat(),
                "created_to": (now + timedelta(days=1)).isoformat(),
            },
        )

    assert response.status_code == 200
    assert [item["prediction_id"] for item in response.json()["items"]] == ["match"]


def test_deleted_records_are_hidden_from_read_endpoints() -> None:
    repository = FakeHistoryRepository(
        [prediction_document(status=PredictionStatus.deleted.value)]
    )
    with history_client(repository, user_id="user_123") as client:
        list_response = client.get("/api/v1/me/predictions")
        detail_response = client.get("/api/v1/me/predictions/prediction_1")
        audio_response = client.get("/api/v1/me/predictions/prediction_1/audio")

    assert list_response.status_code == 200
    assert list_response.json()["items"] == []
    assert detail_response.status_code == 404
    assert audio_response.status_code == 404


def test_audio_endpoint_returns_signed_playback_url_without_public_id() -> None:
    storage = FakeHistoryStorage()
    repository = FakeHistoryRepository([prediction_document()])
    with history_client(repository, storage=storage, user_id="user_123") as client:
        response = client.get("/api/v1/me/predictions/prediction_1/audio")

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "playback_url": "https://signed.example/audio",
        "expires_in_seconds": 300,
    }
    assert storage.signed_requests == [
        ("multiscope/audio/user_123/audio_123", "user_123", 300)
    ]
    assert "multiscope/audio" not in str(body)


def test_repeated_delete_is_idempotent_and_deletes_cloudinary_asset_once() -> None:
    storage = FakeHistoryStorage()
    repository = FakeHistoryRepository([prediction_document()])
    with history_client(repository, storage=storage, user_id="user_123") as client:
        first_response = client.delete("/api/v1/me/predictions/prediction_1")
        second_response = client.delete("/api/v1/me/predictions/prediction_1")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert first_response.json() == {
        "prediction_id": "prediction_1",
        "status": "deleted",
    }
    assert second_response.json() == first_response.json()
    assert storage.deleted_public_ids == ["multiscope/audio/user_123/audio_123"]
    assert repository.documents[0]["cloudinary_asset"] is None


def test_delete_removes_only_the_owners_explanations_and_artifacts() -> None:
    storage = FakeHistoryStorage()
    repository = FakeHistoryRepository([prediction_document()])
    xai_repository = FakeHistoryXaiRepository(
        [
            {"id": "explanation-owner", "prediction_id": "prediction_1", "owner_user_id": "user_123"},
            {"id": "explanation-other-owner", "prediction_id": "prediction_1", "owner_user_id": "user_456"},
            {"id": "explanation-other-prediction", "prediction_id": "prediction_2", "owner_user_id": "user_123"},
        ]
    )
    artifact_store = FakeHistoryArtifactStore()

    with history_client(
        repository,
        storage=storage,
        xai_repository=xai_repository,
        xai_artifact_store=artifact_store,
        user_id="user_123",
    ) as client:
        first_response = client.delete("/api/v1/me/predictions/prediction_1")
        second_response = client.delete("/api/v1/me/predictions/prediction_1")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert artifact_store.deleted_explanations == ["explanation-owner"]
    assert [document["id"] for document in xai_repository.documents] == [
        "explanation-other-owner",
        "explanation-other-prediction",
    ]


def test_rerun_cross_user_prediction_returns_404() -> None:
    storage = FakeHistoryStorage()
    repository = FakeHistoryRepository([prediction_document()])

    with history_client(repository, storage=storage, user_id="user_456") as client:
        response = client.post("/api/v1/me/predictions/prediction_1/rerun")

    assert response.status_code == 404
    assert storage.download_requests == []
    assert len(repository.documents) == 1


def test_rerun_missing_source_audio_returns_409() -> None:
    document = prediction_document()
    document["cloudinary_asset"] = None
    repository = FakeHistoryRepository([document])

    with history_client(repository, user_id="user_123") as client:
        response = client.post("/api/v1/me/predictions/prediction_1/rerun")

    assert response.status_code == 409
    assert response.json()["error"]["code"] == "http_error"
    assert len(repository.documents) == 1


def test_successful_rerun_creates_new_prediction_record(monkeypatch, tmp_path) -> None:
    repository = FakeHistoryRepository([prediction_document()])
    storage = FakeHistoryStorage()
    voice = FakeRerunVoiceService()
    validated_path = tmp_path / "validated.wav"
    validated_path.write_bytes(b"validated")

    def validated_local_audio(source_path, **kwargs):
        assert source_path.exists()
        assert kwargs["original_filename"] == "sample.wav"
        return rerun_upload_metadata(validated_path)

    monkeypatch.setattr(
        "app.services.prediction_rerun_service.save_validated_local_audio_file",
        validated_local_audio,
    )

    with history_client(
        repository,
        storage=storage,
        voice=voice,
        user_id="user_123",
    ) as client:
        response = client.post(
            "/api/v1/me/predictions/prediction_1/rerun",
            data={"rerun_reason": "model refresh"},
            headers={"X-Request-ID": "rerun-request"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction_id"] == "prediction_2"
    UUID(body["request_id"])
    assert body["request_id"] != "rerun-request"
    assert response.headers["X-Client-Correlation-ID"] == "rerun-request"
    assert repository.documents[0]["status"] == "completed"
    rerun_document = repository.documents[1]
    assert rerun_document["parent_prediction_id"] == "prediction_1"
    assert rerun_document["rerun_reason"] == "model refresh"
    assert rerun_document["preprocessing_version"]
    assert rerun_document["model_versions"]["cnn_acoustic"]["mode"] == "dummy"
    assert rerun_document["status"] == "completed"
    assert storage.download_requests == [
        ("multiscope/audio/user_123/audio_123", "user_123")
    ]
    assert storage.upload_count == 1
    assert voice.call_count == 1


def test_completed_rerun_automatically_enqueues_xai(monkeypatch, tmp_path) -> None:
    class RecordingXaiOrchestrator:
        def __init__(self) -> None:
            self.bundle = None

        async def enqueue(self, bundle) -> None:
            self.bundle = bundle

    repository = FakeHistoryRepository([prediction_document()])
    storage = FakeHistoryStorage()
    voice = FakeRerunVoiceService()
    xai = RecordingXaiOrchestrator()
    validated_path = tmp_path / "validated.wav"
    validated_path.write_bytes(b"validated")

    monkeypatch.setattr(
        "app.services.prediction_rerun_service.save_validated_local_audio_file",
        lambda _source_path, **_kwargs: rerun_upload_metadata(validated_path),
    )

    with history_client(
        repository,
        storage=storage,
        voice=voice,
        xai_orchestrator=xai,
        app_settings=Settings(_env_file=None, xai_enabled=True, xai_mode="mock"),
        user_id="user_123",
    ) as client:
        response = client.post("/api/v1/me/predictions/prediction_1/rerun")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert xai.bundle is not None
    assert xai.bundle.prediction_id == "prediction_2"
    assert xai.bundle.owner_user_id == "user_123"
    assert xai.bundle.source_type.value == "dashboard_upload"
    assert xai.bundle.prediction.request_id == xai.bundle.request_id


def test_rerun_xai_enqueue_failure_does_not_change_completed_prediction(
    monkeypatch,
    tmp_path,
) -> None:
    class FailingXaiOrchestrator:
        async def enqueue(self, _bundle) -> None:
            raise RuntimeError("queue unavailable")

    repository = FakeHistoryRepository([prediction_document()])
    validated_path = tmp_path / "validated.wav"
    validated_path.write_bytes(b"validated")
    monkeypatch.setattr(
        "app.services.prediction_rerun_service.save_validated_local_audio_file",
        lambda _source_path, **_kwargs: rerun_upload_metadata(validated_path),
    )

    with history_client(
        repository,
        xai_orchestrator=FailingXaiOrchestrator(),
        app_settings=Settings(_env_file=None, xai_enabled=True, xai_mode="mock"),
        user_id="user_123",
    ) as client:
        response = client.post("/api/v1/me/predictions/prediction_1/rerun")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert repository.documents[1]["status"] == "completed"


def test_failed_rerun_preserves_failed_child_record(monkeypatch) -> None:
    repository = FakeHistoryRepository([prediction_document()])
    storage = FakeHistoryStorage()
    voice = FakeRerunVoiceService()

    def invalid_local_audio(*args, **kwargs):
        raise CorruptedAudioError("Downloaded source is corrupted.")

    monkeypatch.setattr(
        "app.services.prediction_rerun_service.save_validated_local_audio_file",
        invalid_local_audio,
    )

    with history_client(
        repository,
        storage=storage,
        voice=voice,
        user_id="user_123",
    ) as client:
        response = client.post("/api/v1/me/predictions/prediction_1/rerun")

    assert response.status_code == 400
    assert len(repository.documents) == 2
    rerun_document = repository.documents[1]
    assert rerun_document["parent_prediction_id"] == "prediction_1"
    assert rerun_document["status"] == "failed"
    assert rerun_document["error_summary"][-1]["code"] == "corrupted_audio"
    assert storage.upload_count == 0
    assert voice.call_count == 0


@pytest.mark.anyio
async def test_cancellation_during_rerun_download_removes_temporary_file(
    monkeypatch,
    tmp_path,
) -> None:
    download_path = tmp_path / "download.wav"
    repository = FakeHistoryRepository([prediction_document()])
    storage = CancelingDownloadStorage()
    service = PredictionRerunService(
        repository=repository,
        persistence=PredictionPersistenceService(repository),
        storage=storage,
        voice_service=FakeRerunVoiceService(),
        job_runner=InlinePredictionJobRunner(),
    )

    monkeypatch.setattr(
        "app.services.prediction_rerun_service._safe_download_path",
        lambda: download_path,
    )

    with pytest.raises(asyncio.CancelledError):
        await service.rerun_prediction(
            principal=AuthPrincipal(
                subject="user:user_123",
                principal_type="clerk_user",
                user_id="user_123",
            ),
            prediction_id="prediction_1",
            request_id="server-rerun",
        )

    assert download_path.exists() is False
    assert len(repository.documents) == 1


def test_cloudinary_delete_success_but_mongodb_delete_failure_is_not_marked_deleted() -> None:
    storage = FakeHistoryStorage()
    repository = DeletePersistenceFailureRepository([prediction_document()])

    with history_client(repository, storage=storage, user_id="user_123") as client:
        response = client.delete("/api/v1/me/predictions/prediction_1")

    assert response.status_code == 500
    assert storage.deleted_public_ids == ["multiscope/audio/user_123/audio_123"]
    assert repository.documents[0]["status"] == "deleting"
    assert repository.documents[0]["deleted_at"] is None
    assert repository.documents[0]["cloudinary_asset"] is not None


@contextmanager
def history_client(
    repository,
    *,
    storage=None,
    voice=None,
    xai_orchestrator=None,
    app_settings=None,
    xai_repository=None,
    xai_artifact_store=None,
    user_id: str,
) -> Generator[TestClient, None, None]:
    storage = storage or FakeHistoryStorage()
    voice = voice or FakeRerunVoiceService()
    app_settings = app_settings or Settings(_env_file=None)

    def principal_override() -> AuthPrincipal:
        return AuthPrincipal(
            subject=f"user:{user_id}",
            principal_type="clerk_user",
            user_id=user_id,
        )

    app.dependency_overrides[require_clerk_user] = principal_override
    app.dependency_overrides[get_prediction_history_service] = lambda: (
        PredictionHistoryService(
            repository=repository,
            storage=storage,
            xai_repository=xai_repository,
            xai_artifact_store=xai_artifact_store,
        )
    )
    app.dependency_overrides[get_prediction_rerun_service] = lambda: (
        PredictionRerunService(
            repository=repository,
            persistence=PredictionPersistenceService(repository),
            storage=storage,
            voice_service=voice,
            job_runner=InlinePredictionJobRunner(),
            xai_orchestrator=xai_orchestrator,
            app_settings=app_settings,
        )
    )
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def prediction_document(
    *,
    prediction_id: str = "prediction_1",
    owner_user_id: str = "user_123",
    status: str = "completed",
    source_type: str = "dashboard_upload",
    final_prediction: str = "spoof",
    created_at: datetime | None = None,
) -> dict[str, Any]:
    now = created_at or datetime.now(UTC)
    return {
        "id": prediction_id,
        "request_id": f"request-{prediction_id}",
        "owner_user_id": owner_user_id,
        "source_type": source_type,
        "status": status,
        "original_filename": "sample.wav",
        "original_extension": "wav",
        "detected_container": "wav",
        "detected_codec": "pcm_s16le",
        "duration_seconds": 1.0,
        "sample_rate": 16000,
        "channels": 1,
        "size_bytes": 1024,
        "cloudinary_asset": {
            "asset_id": "asset_123",
            "public_id": "multiscope/audio/user_123/audio_123",
            "resource_type": "video",
            "version": 1,
            "format": "wav",
            "bytes": 1024,
            "duration": 1.0,
            "created_at": now,
        },
        "preprocessing": {
            "shared_representation": "mono_float32_waveform",
            "target_sample_rate": 16000,
        },
        "branches": [
            {
                "model_name": "cnn_acoustic",
                "display_name": "CNN Acoustic",
                "status": "success",
                "mode": "dummy",
                "prediction": final_prediction,
                "confidence": 0.7,
                "probabilities": {
                    "bonafide": 0.3 if final_prediction == "spoof" else 0.7,
                    "spoof": 0.7 if final_prediction == "spoof" else 0.3,
                },
                "processing_time_ms": 1.0,
                "metadata": {
                    "development_placeholder": True,
                    "research_result": False,
                },
            }
        ],
        "fusion": {
            "status": "success",
            "prediction": final_prediction,
            "confidence": 0.7,
            "probabilities": {
                "bonafide": 0.3 if final_prediction == "spoof" else 0.7,
                "spoof": 0.7 if final_prediction == "spoof" else 0.3,
            },
            "method": "weighted_average",
            "branch_weights": {"cnn_acoustic": 1.0},
            "contains_dummy_branches": True,
            "eligible_for_research_evaluation": False,
            "warning": "Dummy result.",
        },
        "research_eligible": False,
        "error_summary": [],
        "total_processing_time_ms": 2.0,
        "created_at": now,
        "updated_at": now,
        "completed_at": now,
        "deleted_at": now if status == "deleted" else None,
    }


class FakeHistoryRepository:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents
        self.statuses = []

    async def create_prediction(
        self,
        *,
        request_id,
        owner_user_id,
        source_type,
        client_correlation_id=None,
        idempotency_key=None,
        logical_request=None,
        parent_prediction_id=None,
        rerun_reason=None,
        preprocessing_version=None,
        model_versions=None,
    ):
        prediction_id = f"prediction_{len(self.documents) + 1}"
        now = datetime.now(UTC)
        self.documents.append(
            {
                "id": prediction_id,
                "request_id": request_id,
                "owner_user_id": owner_user_id,
                "source_type": source_type.value,
                "status": "queued",
                "client_correlation_id": client_correlation_id,
                "idempotency_key": idempotency_key,
                "idempotency_logical_request": logical_request,
                "parent_prediction_id": parent_prediction_id,
                "rerun_reason": rerun_reason,
                "preprocessing_version": preprocessing_version,
                "model_versions": model_versions or {},
                "original_filename": None,
                "original_extension": None,
                "detected_container": None,
                "detected_codec": None,
                "duration_seconds": None,
                "sample_rate": None,
                "channels": None,
                "size_bytes": None,
                "cloudinary_asset": None,
                "preprocessing": {},
                "branches": [],
                "fusion": None,
                "research_eligible": False,
                "error_summary": [],
                "total_processing_time_ms": None,
                "created_at": now,
                "updated_at": now,
                "completed_at": None,
                "deleted_at": None,
            }
        )
        self.statuses.append("queued")
        return prediction_id

    async def list_predictions_for_owner(
        self,
        *,
        owner_user_id,
        page,
        limit,
        status=None,
        source_type=None,
        prediction_label=None,
        created_from=None,
        created_to=None,
    ):
        documents = [
            document
            for document in self.documents
            if document["owner_user_id"] == owner_user_id
            and document["status"] != "deleted"
        ]
        if status is not None:
            documents = [
                document for document in documents if document["status"] == status.value
            ]
        if source_type is not None:
            documents = [
                document
                for document in documents
                if document["source_type"] == source_type.value
            ]
        if prediction_label is not None:
            documents = [
                document
                for document in documents
                if document["fusion"]["prediction"] == prediction_label.value
            ]
        if created_from is not None:
            documents = [
                document
                for document in documents
                if document["created_at"] >= created_from
            ]
        if created_to is not None:
            documents = [
                document for document in documents if document["created_at"] <= created_to
            ]
        documents.sort(key=lambda document: document["created_at"], reverse=True)
        start = (page - 1) * limit
        return documents[start : start + limit + 1]

    async def get_prediction_for_owner(
        self,
        *,
        prediction_id,
        owner_user_id,
        include_deleted=False,
    ):
        for document in self.documents:
            if document["id"] != prediction_id:
                continue
            if document["owner_user_id"] != owner_user_id:
                continue
            if not include_deleted and document["status"] == "deleted":
                return None
            return document
        return None

    async def soft_delete_prediction_for_owner(
        self,
        *,
        prediction_id,
        owner_user_id,
    ):
        document = await self.get_prediction_for_owner(
            prediction_id=prediction_id,
            owner_user_id=owner_user_id,
            include_deleted=True,
        )
        if document is None:
            return False
        if document["status"] == "deleted":
            return True
        assert document["status"] == "deleting"
        document["status"] = "deleted"
        document["cloudinary_asset"] = None
        document["deleted_at"] = datetime.now(UTC)
        document["audio_deleted_at"] = document["deleted_at"]
        document["playback_revoked_at"] = document["deleted_at"]
        return True

    async def update_prediction_status(
        self,
        request_id,
        status,
        *,
        error_code=None,
        error_stage=None,
    ):
        document = self.document_for_request(request_id)
        assert is_valid_prediction_status_transition(
            PredictionStatus(document["status"]),
            status,
        )
        document["status"] = status.value
        document["updated_at"] = datetime.now(UTC)
        if status in {PredictionStatus.completed, PredictionStatus.failed}:
            document["completed_at"] = datetime.now(UTC)
        if error_code:
            document["error_summary"].append(
                {"stage": error_stage, "code": error_code}
            )
        self.statuses.append(status.value)

    async def attach_upload_metadata(self, request_id, upload_metadata):
        self.document_for_request(request_id).update(
            {
                "original_filename": upload_metadata.sanitized_filename,
                "original_extension": upload_metadata.original_extension,
                "detected_container": upload_metadata.detected_container,
                "detected_codec": upload_metadata.detected_codec,
                "duration_seconds": upload_metadata.duration_seconds,
                "sample_rate": upload_metadata.sample_rate,
                "channels": upload_metadata.channels,
                "size_bytes": upload_metadata.size_bytes,
            }
        )

    async def attach_cloudinary_asset(self, request_id, storage_metadata):
        self.document_for_request(request_id)["cloudinary_asset"] = (
            storage_metadata.model_dump(mode="python")
            if storage_metadata.status == BranchStatus.success
            else None
        )

    async def save_prediction_result(
        self,
        request_id,
        prediction,
        *,
        status,
        error_codes,
    ):
        document = self.document_for_request(request_id)
        assert is_valid_prediction_status_transition(
            PredictionStatus(document["status"]),
            status,
        )
        document.update(
            {
                "status": status.value,
                "branches": [
                    branch.model_dump(mode="python") for branch in prediction.branches
                ],
                "fusion": prediction.fusion.model_dump(mode="python"),
                "research_eligible": False,
                "total_processing_time_ms": prediction.total_processing_time_ms,
                "completed_at": datetime.now(UTC),
            }
        )
        document["error_summary"].extend(error_codes)
        self.statuses.append(status.value)

    def document_for_request(self, request_id):
        for document in self.documents:
            if document["request_id"] == request_id:
                return document
        raise AssertionError(f"Missing document for request {request_id}")


class DeletePersistenceFailureRepository(FakeHistoryRepository):
    async def soft_delete_prediction_for_owner(
        self,
        *,
        prediction_id,
        owner_user_id,
    ):
        return False


class FakeHistoryStorage:
    def __init__(self) -> None:
        self.deleted_public_ids = []
        self.signed_requests = []
        self.download_requests = []
        self.upload_count = 0

    async def upload_audio(self, source_path, *, owner_user_id, audio_id, **kwargs):
        self.upload_count += 1
        return AudioStorageMetadata(
            status=BranchStatus.success,
            asset_id="asset-rerun",
            public_id=f"multiscope/audio/{owner_user_id}/{audio_id}",
            resource_type="video",
            version=2,
            format="wav",
            bytes=1024,
            duration=1.0,
            created_at=datetime.now(UTC),
        )

    async def delete_audio(self, public_id: str) -> None:
        self.deleted_public_ids.append(public_id)

    async def download_audio(
        self,
        public_id: str,
        destination_path,
        *,
        owner_user_id: str,
        max_bytes: int,
    ) -> None:
        self.download_requests.append((public_id, owner_user_id))
        destination_path.write_bytes(b"downloaded")

    async def generate_signed_playback_url(
        self,
        public_id: str,
        *,
        owner_user_id: str,
        expires_in_seconds: int = 300,
    ) -> str:
        self.signed_requests.append((public_id, owner_user_id, expires_in_seconds))
        return "https://signed.example/audio"

    async def health(self):
        return {"storage_enabled": True, "storage_available": True}


class FakeHistoryXaiRepository:
    def __init__(self, documents: list[dict[str, str]]) -> None:
        self.documents = documents

    async def list_explanations_for_prediction_for_owner(
        self,
        *,
        prediction_id: str,
        owner_user_id: str,
    ) -> list[dict[str, str]]:
        return [
            document
            for document in self.documents
            if document["prediction_id"] == prediction_id
            and document["owner_user_id"] == owner_user_id
        ]

    async def delete_explanations_for_prediction_for_owner(
        self,
        *,
        prediction_id: str,
        owner_user_id: str,
    ) -> int:
        matches = await self.list_explanations_for_prediction_for_owner(
            prediction_id=prediction_id,
            owner_user_id=owner_user_id,
        )
        self.documents = [document for document in self.documents if document not in matches]
        return len(matches)


class FakeHistoryArtifactStore:
    def __init__(self) -> None:
        self.deleted_explanations: list[str] = []

    def delete_explanation_artifacts(self, explanation_id: str) -> int:
        self.deleted_explanations.append(explanation_id)
        return 1


class CancelingDownloadStorage(FakeHistoryStorage):
    async def download_audio(
        self,
        public_id: str,
        destination_path,
        *,
        owner_user_id: str,
        max_bytes: int,
    ) -> None:
        destination_path.write_bytes(b"partial")
        raise asyncio.CancelledError


def rerun_upload_metadata(saved_path) -> AudioUploadMetadata:
    return AudioUploadMetadata(
        original_filename="sample.wav",
        sanitized_filename="sample.wav",
        saved_filename="validated.wav",
        saved_path=saved_path,
        content_type="audio/wav",
        file_size_bytes=saved_path.stat().st_size,
        duration_seconds=1.0,
        sample_rate=16000,
        channels=1,
        original_extension="wav",
        detected_container="wav",
        detected_codec="pcm_s16le",
    )


class FakeRerunVoiceService:
    def __init__(self) -> None:
        self.call_count = 0

    def predict_from_validated_upload(
        self,
        upload_metadata,
        *,
        request_id=None,
        storage_metadata=None,
    ):
        self.call_count += 1
        branch = BranchPrediction(
            model_name="cnn_acoustic",
            display_name="CNN Acoustic",
            status=BranchStatus.success,
            mode=ModelMode.dummy,
            prediction=PredictionLabel.spoof,
            confidence=0.7,
            probabilities=ProbabilityScores(bonafide=0.3, spoof=0.7),
            processing_time_ms=1.0,
            metadata={
                "development_placeholder": True,
                "research_result": False,
            },
        )
        return VoicePredictionResponse(
            request_id=request_id or "rerun-request",
            audio=AudioMetadata(
                original_filename=upload_metadata.original_filename,
                content_type=upload_metadata.content_type or "",
                original_extension=upload_metadata.original_extension,
                detected_container=upload_metadata.detected_container,
                detected_codec=upload_metadata.detected_codec,
                file_size_bytes=upload_metadata.file_size_bytes,
                duration_seconds=upload_metadata.duration_seconds,
                sample_rate=upload_metadata.sample_rate,
                channels=upload_metadata.channels,
                storage=storage_metadata,
            ),
            branches=[branch],
            fusion=FusionResult(
                status=BranchStatus.success,
                prediction=PredictionLabel.spoof,
                confidence=0.7,
                probabilities=ProbabilityScores(bonafide=0.3, spoof=0.7),
                method="weighted_average",
                branch_weights={"cnn_acoustic": 1.0},
                contains_dummy_branches=True,
                eligible_for_research_evaluation=False,
                warning="Dummy result.",
            ),
            total_processing_time_ms=2.0,
            created_at=datetime.now(UTC),
        )

    def model_health(self):
        return [
            {
                "model_name": "cnn_acoustic",
                "display_name": "CNN Acoustic",
                "mode": "dummy",
                "is_loaded": True,
            }
        ]
