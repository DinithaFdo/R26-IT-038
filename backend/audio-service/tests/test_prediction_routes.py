import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any
from uuid import UUID

import httpx
import numpy as np
from fastapi.testclient import TestClient
import pytest

from app.api.dependencies import get_prediction_submission_service
from app.auth.clerk_auth import require_clerk_user
from app.auth.schemas import AuthPrincipal
from app.config.settings import Settings
from app.ingestion.audio import AudioUploadMetadata, ProcessedAudio
from app.main import app
from app.schemas.common import (
    BranchStatus,
    ModelMode,
    PredictionLabel,
    PredictionStatus,
    SourceType,
)
from app.schemas.prediction import (
    AudioMetadata,
    AudioStorageMetadata,
    BranchPrediction,
    FusionResult,
    ProbabilityScores,
    VoicePredictionResponse,
)
from app.services.prediction_persistence_service import PredictionPersistenceService
from app.services.prediction_job_runner import InlinePredictionJobRunner
from app.services.prediction_submission_service import PredictionSubmissionService
from app.services.voice_service import PredictionExecutionResult
from app.voice_xai.temporal.contracts import TemporalAttentionWindowInput

app.state.settings = app.state.settings.model_copy(
    update={"mongodb_required": False}
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
AUDIO_TOOLS_AVAILABLE = bool(FFMPEG and FFPROBE)
requires_audio_tools = pytest.mark.skipif(
    not AUDIO_TOOLS_AVAILABLE,
    reason="FFmpeg and ffprobe are not available in the test environment.",
)


def test_unified_prediction_requires_authenticated_clerk_user() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predictions",
            data={"source_type": "dashboard_upload"},
            files={"file": ("sample.wav", b"placeholder", "audio/wav")},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_failed"


def test_dashboard_upload_uses_unified_pipeline(monkeypatch, tmp_path) -> None:
    repository = FakeSubmissionRepository()
    storage = FakeSubmissionStorage()
    voice = FakeVoiceService()
    upload_path = tmp_path / "validated.wav"
    upload_path.write_bytes(b"validated")

    def validated_upload(file, **kwargs):
        assert kwargs["validation_filename"] is None
        assert kwargs["display_filename"] == "client.wav"
        return upload_metadata(upload_path, "client.wav", "wav")

    monkeypatch.setattr(
        "app.services.prediction_submission_service.save_validated_audio_upload_blocking",
        validated_upload,
    )

    with prediction_client(repository, storage=storage, voice=voice) as client:
        response = client.post(
            "/api/v1/predictions",
            data={
                "source_type": "dashboard_upload",
                "client_filename": "client.wav",
            },
            files={"file": ("sample.wav", b"placeholder", "audio/wav")},
            headers={"X-Request-ID": "request-dashboard"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["prediction_id"] == "prediction-1"
    UUID(body["request_id"])
    assert body["request_id"] != "request-dashboard"
    assert repository.documents[0]["client_correlation_id"] == "request-dashboard"
    assert body["source_type"] == "dashboard_upload"
    assert body["status"] == "completed"
    assert body["audio"]["original_filename"] == "client.wav"
    assert body["research_eligible"] is False
    assert repository.statuses == [
        "queued",
        "validating",
        "storing",
        "processing",
        "completed",
    ]
    assert voice.call_count == 1
    assert storage.upload_count == 1
    assert not upload_path.exists()


def test_live_recording_generates_safe_display_name(monkeypatch, tmp_path) -> None:
    repository = FakeSubmissionRepository()
    voice = FakeVoiceService()
    upload_path = tmp_path / "validated.webm"
    upload_path.write_bytes(b"validated")

    def validated_upload(file, **kwargs):
        assert kwargs["validation_filename"] == "browser-recording.webm"
        assert kwargs["display_filename"].startswith("Recording ")
        assert kwargs["display_filename"].endswith(".webm")
        return upload_metadata(
            upload_path,
            kwargs["display_filename"],
            "webm",
            detected_container="matroska,webm",
            detected_codec="opus",
        )

    monkeypatch.setattr(
        "app.services.prediction_submission_service.save_validated_audio_upload_blocking",
        validated_upload,
    )

    with prediction_client(repository, voice=voice) as client:
        response = client.post(
            "/api/v1/predictions",
            data={"source_type": "live_recording"},
            files={"file": ("blob", b"placeholder", "audio/webm")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["source_type"] == "live_recording"
    assert body["audio"]["original_filename"].startswith("Recording ")
    assert body["audio"]["detected_codec"] == "opus"
    assert voice.call_count == 1


def test_invalid_source_type_is_rejected(monkeypatch, tmp_path) -> None:
    repository = FakeSubmissionRepository()
    upload_path = tmp_path / "validated.wav"
    upload_path.write_bytes(b"validated")

    def validated_upload(*args, **kwargs):
        raise AssertionError("Validation must not run for invalid source_type.")

    monkeypatch.setattr(
        "app.services.prediction_submission_service.save_validated_audio_upload_blocking",
        validated_upload,
    )

    with prediction_client(repository) as client:
        response = client.post(
            "/api/v1/predictions",
            data={"source_type": "public_api"},
            files={"file": ("sample.wav", b"placeholder", "audio/wav")},
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "http_error"


def test_storage_failure_stops_inference(monkeypatch, tmp_path) -> None:
    repository = FakeSubmissionRepository()
    storage = FakeSubmissionStorage(status=BranchStatus.failed)
    voice = FakeVoiceService()
    upload_path = tmp_path / "validated.wav"
    upload_path.write_bytes(b"validated")

    monkeypatch.setattr(
        "app.services.prediction_submission_service.save_validated_audio_upload_blocking",
        lambda *_args, **_kwargs: upload_metadata(upload_path, "sample.wav", "wav"),
    )

    with prediction_client(repository, storage=storage, voice=voice) as client:
        response = client.post(
            "/api/v1/predictions",
            data={"source_type": "dashboard_upload"},
            files={"file": ("sample.wav", b"placeholder", "audio/wav")},
        )

    assert response.status_code == 503
    assert voice.call_count == 0
    assert repository.statuses == ["queued", "validating", "storing", "failed"]
    assert not upload_path.exists()


def test_skipped_storage_stops_authenticated_inference(monkeypatch, tmp_path) -> None:
    repository = FakeSubmissionRepository()
    storage = FakeSubmissionStorage(status=BranchStatus.skipped)
    voice = FakeVoiceService()
    upload_path = tmp_path / "validated.wav"
    upload_path.write_bytes(b"validated")

    monkeypatch.setattr(
        "app.services.prediction_submission_service.save_validated_audio_upload_blocking",
        lambda *_args, **_kwargs: upload_metadata(upload_path, "sample.wav", "wav"),
    )

    with prediction_client(repository, storage=storage, voice=voice) as client:
        response = client.post(
            "/api/v1/predictions",
            data={"source_type": "dashboard_upload"},
            files={"file": ("sample.wav", b"placeholder", "audio/wav")},
        )

    assert response.status_code == 503
    assert voice.call_count == 0
    assert repository.statuses == ["queued", "validating", "storing", "failed"]


def test_optional_storage_failure_allows_authenticated_inference(
    monkeypatch,
    tmp_path,
) -> None:
    repository = FakeSubmissionRepository()
    storage = FakeSubmissionStorage(status=BranchStatus.failed)
    voice = FakeVoiceService()
    upload_path = tmp_path / "validated.wav"
    upload_path.write_bytes(b"validated")

    monkeypatch.setattr(
        "app.services.prediction_submission_service.save_validated_audio_upload_blocking",
        lambda *_args, **_kwargs: upload_metadata(upload_path, "sample.wav", "wav"),
    )

    with prediction_client(
        repository,
        storage=storage,
        voice=voice,
        app_settings=Settings(_env_file=None, storage_policy="optional"),
    ) as client:
        response = client.post(
            "/api/v1/predictions",
            data={"source_type": "dashboard_upload"},
            files={"file": ("sample.wav", b"placeholder", "audio/wav")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["audio"]["storage_status"] == "failed"
    assert body["audio"]["playback_available"] is False
    assert voice.call_count == 1
    assert repository.statuses == [
        "queued",
        "validating",
        "storing",
        "processing",
        "completed",
    ]
    assert not upload_path.exists()


def test_idempotency_replays_existing_prediction_without_running_twice(
    monkeypatch,
    tmp_path,
) -> None:
    repository = FakeSubmissionRepository()
    storage = FakeSubmissionStorage()
    voice = FakeVoiceService()
    upload_path = tmp_path / "validated.wav"
    upload_path.write_bytes(b"validated")
    validation_calls = 0

    def validated_upload(file, **kwargs):
        nonlocal validation_calls
        validation_calls += 1
        return upload_metadata(upload_path, "idempotent.wav", "wav")

    monkeypatch.setattr(
        "app.services.prediction_submission_service.save_validated_audio_upload_blocking",
        validated_upload,
    )

    with prediction_client(repository, storage=storage, voice=voice) as client:
        first = client.post(
            "/api/v1/predictions",
            data={
                "source_type": "dashboard_upload",
                "client_filename": "idempotent.wav",
                "idempotency_key": "same-key",
            },
            files={"file": ("sample.wav", b"placeholder", "audio/wav")},
            headers={"X-Request-ID": "first-request"},
        )
        second = client.post(
            "/api/v1/predictions",
            data={
                "source_type": "dashboard_upload",
                "client_filename": "idempotent.wav",
                "idempotency_key": "same-key",
            },
            files={"file": ("sample.wav", b"placeholder", "audio/wav")},
            headers={"X-Request-ID": "second-request"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["prediction_id"] == first.json()["prediction_id"]
    assert second.json()["request_id"] == first.json()["request_id"]
    assert repository.documents[0]["client_correlation_id"] == "first-request"
    assert validation_calls == 1
    assert voice.call_count == 1
    assert storage.upload_count == 1


def test_idempotency_key_conflict_returns_409(monkeypatch, tmp_path) -> None:
    repository = FakeSubmissionRepository()
    upload_path = tmp_path / "validated.wav"
    upload_path.write_bytes(b"validated")

    def validated_upload(file, **kwargs):
        return upload_metadata(upload_path, "first.wav", "wav")

    monkeypatch.setattr(
        "app.services.prediction_submission_service.save_validated_audio_upload_blocking",
        validated_upload,
    )

    with prediction_client(repository) as client:
        first = client.post(
            "/api/v1/predictions",
            data={
                "source_type": "dashboard_upload",
                "client_filename": "first.wav",
                "idempotency_key": "same-key",
            },
            files={"file": ("sample.wav", b"placeholder", "audio/wav")},
        )
        second = client.post(
            "/api/v1/predictions",
            data={
                "source_type": "live_recording",
                "idempotency_key": "same-key",
            },
            files={"file": ("blob", b"placeholder", "audio/webm")},
        )

    assert first.status_code == 200
    assert second.status_code == 409


@requires_audio_tools
@pytest.mark.parametrize(
    ("extension", "content_type", "codec", "container", "filename"),
    [
        ("webm", "audio/webm", "libopus", "webm", "blob"),
        ("m4a", "audio/mp4", "aac", None, "recording.m4a"),
    ],
)
def test_browser_recording_formats_work_with_real_ingestion(
    tmp_path,
    extension,
    content_type,
    codec,
    container,
    filename,
) -> None:
    encoded_path = encode_audio_fixture(
        tmp_path,
        extension=extension,
        codec=codec,
        container=container,
    )
    repository = FakeSubmissionRepository()

    with prediction_client(repository) as client:
        response = client.post(
            "/api/v1/predictions",
            data={"source_type": "live_recording"},
            files={"file": (filename, encoded_path.read_bytes(), content_type)},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["source_type"] == "live_recording"
    assert body["audio"]["detected_container"]
    assert body["audio"]["detected_codec"]


def test_prediction_status_endpoint_returns_owner_status() -> None:
    repository = FakeSubmissionRepository()
    repository.documents.append(
        new_document(
            prediction_id="prediction-status",
            request_id="request-status",
            owner_user_id="user_123",
            source_type=SourceType.dashboard_upload,
        )
    )

    with prediction_client(repository) as client:
        response = client.get("/api/v1/predictions/prediction-status/status")

    assert response.status_code == 200
    body = response.json()
    assert body["prediction_id"] == "prediction-status"
    assert body["request_id"] == "request-status"
    assert body["status"] == "queued"
    assert body["source_type"] == "dashboard_upload"
    assert body["created_at"]
    assert body["updated_at"]
    assert body["completed_at"] is None
    assert body["error_summary"] == []


def test_prediction_status_endpoint_hides_cross_user_prediction() -> None:
    repository = FakeSubmissionRepository()
    repository.documents.append(
        new_document(
            prediction_id="prediction-status",
            request_id="request-status",
            owner_user_id="other-user",
            source_type=SourceType.dashboard_upload,
        )
    )

    with prediction_client(repository, user_id="user_123") as client:
        response = client.get("/api/v1/predictions/prediction-status/status")

    assert response.status_code == 404


@pytest.mark.anyio
async def test_health_endpoint_remains_responsive_while_prediction_work_runs(
    monkeypatch,
    tmp_path,
) -> None:
    repository = FakeSubmissionRepository()
    storage = FakeSubmissionStorage()
    upload_path = tmp_path / "validated.wav"
    upload_path.write_bytes(b"validated")
    loop = asyncio.get_running_loop()
    prediction_started = asyncio.Event()
    release_prediction = threading.Event()

    def validated_upload(file, **kwargs):
        return upload_metadata(upload_path, "responsive.wav", "wav")

    monkeypatch.setattr(
        "app.services.prediction_submission_service.save_validated_audio_upload_blocking",
        validated_upload,
    )

    voice = BlockingVoiceService(
        loop=loop,
        started=prediction_started,
        release=release_prediction,
    )
    persistence = PredictionPersistenceService(repository)
    service = PredictionSubmissionService(
        repository=repository,
        persistence=persistence,
        storage=storage,
        voice_service=voice,
        job_runner=InlinePredictionJobRunner(
            Settings(
                _env_file=None,
                prediction_job_timeout_seconds=5,
                max_concurrent_predictions=1,
            )
        ),
    )
    app.dependency_overrides[require_clerk_user] = lambda: AuthPrincipal(
        subject="user:user_123",
        principal_type="clerk_user",
        user_id="user_123",
    )
    app.dependency_overrides[get_prediction_submission_service] = lambda: service

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            prediction_task = asyncio.create_task(
                client.post(
                    "/api/v1/predictions",
                    data={"source_type": "dashboard_upload"},
                    files={"file": ("sample.wav", b"placeholder", "audio/wav")},
                )
            )
            await asyncio.wait_for(prediction_started.wait(), timeout=1)

            health_response = await client.get("/health")

            release_prediction.set()
            prediction_response = await prediction_task
    finally:
        app.dependency_overrides.clear()

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "healthy"}
    assert prediction_response.status_code == 200


@pytest.mark.anyio
async def test_xai_enqueue_failure_does_not_change_completed_prediction(
    monkeypatch,
    tmp_path,
) -> None:
    class FailingXaiOrchestrator:
        async def enqueue(self, _bundle) -> None:
            raise RuntimeError("queue unavailable")

    upload_path = tmp_path / "validated.wav"
    upload_path.write_bytes(b"validated")
    repository = FakeSubmissionRepository()
    await repository.create_prediction(
        request_id="request-xai-failure",
        owner_user_id="user-123",
        source_type=SourceType.dashboard_upload,
    )
    monkeypatch.setattr(
        "app.services.prediction_submission_service.save_validated_audio_upload_blocking",
        lambda _file, **_kwargs: upload_metadata(upload_path, "sample.wav", "wav"),
    )
    app_settings = Settings(_env_file=None, xai_enabled=True, xai_mode="mock")
    service = PredictionSubmissionService(
        repository=repository,
        persistence=PredictionPersistenceService(repository),
        storage=FakeSubmissionStorage(),
        voice_service=FakeVoiceService(),
        job_runner=InlinePredictionJobRunner(app_settings),
        xai_orchestrator=FailingXaiOrchestrator(),
        app_settings=app_settings,
    )

    class UploadedFile:
        filename = "sample.wav"
        content_type = "audio/wav"

    response = await service.execute_prediction(
        file=UploadedFile(),
        principal=AuthPrincipal(
            subject="user:user-123",
            principal_type="clerk_user",
            user_id="user-123",
        ),
        source_type=SourceType.dashboard_upload,
        prediction_id="prediction-1",
        request_id="request-xai-failure",
    )

    assert response.status.value == "completed"
    assert repository.document_for_request("request-xai-failure")["status"] == "completed"


@pytest.mark.anyio
async def test_completed_prediction_automatically_enqueues_xai(
    monkeypatch,
    tmp_path,
) -> None:
    class RecordingXaiOrchestrator:
        def __init__(self) -> None:
            self.bundle = None

        async def enqueue(self, bundle) -> None:
            self.bundle = bundle

    upload_path = tmp_path / "validated.wav"
    upload_path.write_bytes(b"validated")
    repository = FakeSubmissionRepository()
    await repository.create_prediction(
        request_id="request-xai-success",
        owner_user_id="user-123",
        source_type=SourceType.dashboard_upload,
    )
    monkeypatch.setattr(
        "app.services.prediction_submission_service.save_validated_audio_upload_blocking",
        lambda _file, **_kwargs: upload_metadata(upload_path, "sample.wav", "wav"),
    )
    settings = Settings(_env_file=None, xai_enabled=True, xai_mode="mock")
    xai = RecordingXaiOrchestrator()
    service = PredictionSubmissionService(
        repository=repository,
        persistence=PredictionPersistenceService(repository),
        storage=FakeSubmissionStorage(),
        voice_service=FakeVoiceService(),
        job_runner=InlinePredictionJobRunner(settings),
        xai_orchestrator=xai,
        app_settings=settings,
    )

    class UploadedFile:
        filename = "sample.wav"
        content_type = "audio/wav"

    response = await service.execute_prediction(
        file=UploadedFile(),
        principal=AuthPrincipal(
            subject="user:user-123",
            principal_type="clerk_user",
            user_id="user-123",
        ),
        source_type=SourceType.dashboard_upload,
        prediction_id="prediction-1",
        request_id="request-xai-success",
    )

    assert response.status.value == "completed"
    assert xai.bundle is not None
    assert xai.bundle.prediction_id == "prediction-1"
    assert xai.bundle.extraction is None


@pytest.mark.anyio
async def test_automatic_xai_handoff_reuses_the_same_processed_audio(
    monkeypatch,
    tmp_path,
) -> None:
    """INT-1 regression test.

    When the voice service exposes ``predict_from_validated_upload_with_processed_audio``
    (the real ``VoiceService`` does), the automatic post-persistence XAI
    handoff must receive the exact same ``ProcessedAudio`` instance used for
    classifier inference -- not None, and not a second decode -- while
    ``extraction`` still correctly stays None until real model capture is
    wired, and request IDs still match across prediction/bundle.
    """

    class RecordingXaiOrchestrator:
        def __init__(self) -> None:
            self.bundle = None

        async def enqueue(self, bundle) -> None:
            self.bundle = bundle

    upload_path = tmp_path / "validated.wav"
    upload_path.write_bytes(b"validated")
    repository = FakeSubmissionRepository()
    await repository.create_prediction(
        request_id="request-xai-handoff",
        owner_user_id="user-123",
        source_type=SourceType.dashboard_upload,
    )
    monkeypatch.setattr(
        "app.services.prediction_submission_service.save_validated_audio_upload_blocking",
        lambda _file, **_kwargs: upload_metadata(upload_path, "sample.wav", "wav"),
    )
    settings = Settings(_env_file=None, xai_enabled=True, xai_mode="mock")
    xai = RecordingXaiOrchestrator()
    service = PredictionSubmissionService(
        repository=repository,
        persistence=PredictionPersistenceService(repository),
        storage=FakeSubmissionStorage(),
        voice_service=FakeVoiceServiceWithProcessedAudio(),
        job_runner=InlinePredictionJobRunner(settings),
        xai_orchestrator=xai,
        app_settings=settings,
    )

    class UploadedFile:
        filename = "sample.wav"
        content_type = "audio/wav"

    response = await service.execute_prediction(
        file=UploadedFile(),
        principal=AuthPrincipal(
            subject="user:user-123",
            principal_type="clerk_user",
            user_id="user-123",
        ),
        source_type=SourceType.dashboard_upload,
        prediction_id="prediction-1",
        request_id="request-xai-handoff",
    )

    assert response.status.value == "completed"
    assert xai.bundle is not None
    assert xai.bundle.extraction is None
    assert xai.bundle.processed_audio is FakeVoiceServiceWithProcessedAudio.PROCESSED_AUDIO
    assert xai.bundle.temporal_evidence is FakeVoiceServiceWithProcessedAudio.TEMPORAL_EVIDENCE
    assert xai.bundle.request_id == xai.bundle.prediction.request_id == "request-xai-handoff"


@contextmanager
def prediction_client(
    repository,
    *,
    storage=None,
    voice=None,
    user_id: str = "user_123",
    app_settings: Settings | None = None,
) -> Generator[TestClient, None, None]:
    storage = storage or FakeSubmissionStorage()
    voice = voice or None

    def principal_override() -> AuthPrincipal:
        return AuthPrincipal(
            subject=f"user:{user_id}",
            principal_type="clerk_user",
            user_id=user_id,
        )

    persistence = PredictionPersistenceService(repository)
    service = PredictionSubmissionService(
        repository=repository,
        persistence=persistence,
        storage=storage,
        voice_service=voice or FakeVoiceService(),
        job_runner=InlinePredictionJobRunner(),
        app_settings=app_settings or Settings(_env_file=None),
    )
    app.dependency_overrides[require_clerk_user] = principal_override
    app.dependency_overrides[get_prediction_submission_service] = lambda: service
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


def upload_metadata(
    saved_path: Path,
    original_filename: str,
    extension: str,
    *,
    detected_container: str = "wav",
    detected_codec: str = "pcm_s16le",
) -> AudioUploadMetadata:
    return AudioUploadMetadata(
        original_filename=original_filename,
        sanitized_filename=original_filename,
        saved_filename=f"validated.{extension}",
        saved_path=saved_path,
        content_type=f"audio/{extension}",
        file_size_bytes=saved_path.stat().st_size,
        duration_seconds=0.1,
        sample_rate=16000,
        channels=1,
        original_extension=extension,
        detected_container=detected_container,
        detected_codec=detected_codec,
    )


def prediction_response(
    upload: AudioUploadMetadata,
    *,
    request_id: str,
    storage_metadata: AudioStorageMetadata,
) -> VoicePredictionResponse:
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
        request_id=request_id,
        audio=AudioMetadata(
            original_filename=upload.original_filename,
            content_type=upload.content_type or "",
            original_extension=upload.original_extension,
            detected_container=upload.detected_container,
            detected_codec=upload.detected_codec,
            file_size_bytes=upload.file_size_bytes,
            duration_seconds=upload.duration_seconds,
            sample_rate=upload.sample_rate,
            channels=upload.channels,
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
            warning="Dummy result is not a research result.",
        ),
        total_processing_time_ms=2.0,
        created_at=datetime.now(UTC),
    )


class FakeSubmissionRepository:
    def __init__(self) -> None:
        self.documents = []
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
        prediction_id = f"prediction-{len(self.documents) + 1}"
        document = new_document(
            prediction_id=prediction_id,
            request_id=request_id,
            owner_user_id=owner_user_id,
            source_type=source_type,
            client_correlation_id=client_correlation_id,
            idempotency_key=idempotency_key,
            logical_request=logical_request,
            parent_prediction_id=parent_prediction_id,
            rerun_reason=rerun_reason,
            preprocessing_version=preprocessing_version,
            model_versions=model_versions,
        )
        self.documents.append(document)
        self.statuses.append("queued")
        return prediction_id

    async def reserve_prediction(
        self,
        *,
        request_id,
        owner_user_id,
        source_type,
        client_correlation_id,
        idempotency_key,
        logical_request,
    ):
        for document in self.documents:
            if (
                document.get("owner_user_id") == owner_user_id
                and document.get("idempotency_key") == idempotency_key
            ):
                return {"created": False, "document": document}
        prediction_id = f"prediction-{len(self.documents) + 1}"
        document = new_document(
            prediction_id=prediction_id,
            request_id=request_id,
            owner_user_id=owner_user_id,
            source_type=source_type,
            client_correlation_id=client_correlation_id,
            idempotency_key=idempotency_key,
            logical_request=logical_request,
        )
        self.documents.append(document)
        self.statuses.append("queued")
        return {"created": True, "document": document}

    async def update_prediction_status(
        self,
        request_id,
        status,
        *,
        error_code=None,
        error_stage=None,
    ):
        document = self.document_for_request(request_id)
        document["status"] = status.value
        self.statuses.append(status.value)
        if error_code:
            document["error_summary"].append(
                {"stage": error_stage, "code": error_code}
            )

    async def attach_upload_metadata(self, request_id, upload):
        self.document_for_request(request_id).update(
            {
                "original_filename": upload.sanitized_filename,
                "original_extension": upload.original_extension,
                "detected_container": upload.detected_container,
                "detected_codec": upload.detected_codec,
                "duration_seconds": upload.duration_seconds,
                "sample_rate": upload.sample_rate,
                "channels": upload.channels,
                "size_bytes": upload.size_bytes,
            }
        )

    async def attach_cloudinary_asset(self, request_id, storage):
        self.document_for_request(request_id)["cloudinary_asset"] = (
            storage.model_dump(mode="python")
            if storage.status == BranchStatus.success
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

    async def get_prediction_for_owner(
        self,
        *,
        prediction_id,
        owner_user_id,
        include_deleted=False,
    ):
        for document in self.documents:
            if (
                document["id"] == prediction_id
                and document["owner_user_id"] == owner_user_id
                and (
                    include_deleted
                    or document["status"] != PredictionStatus.deleted.value
                )
            ):
                return document
        return None


def new_document(
    *,
    prediction_id,
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
    now = datetime.now(UTC)
    return {
        "id": prediction_id,
        "request_id": request_id,
        "client_correlation_id": client_correlation_id,
        "owner_user_id": owner_user_id,
        "source_type": source_type.value,
        "status": "queued",
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


class FakeSubmissionStorage:
    def __init__(self, *, status: BranchStatus = BranchStatus.success) -> None:
        self.upload_count = 0
        self.status = status

    async def upload_audio(self, source_path, *, owner_user_id, audio_id, **kwargs):
        self.upload_count += 1
        if self.status != BranchStatus.success:
            return AudioStorageMetadata(
                status=self.status,
                error="Storage unavailable.",
            )
        return AudioStorageMetadata(
            status=BranchStatus.success,
            asset_id="asset-123",
            public_id=f"multiscope/audio/{owner_user_id}/{audio_id}",
            resource_type="video",
            version=123,
            format="wav",
            bytes=1024,
            duration=0.1,
            created_at=datetime.now(UTC),
        )

    async def delete_audio(self, public_id: str) -> None:
        return None

    async def generate_signed_playback_url(self, *args, **kwargs) -> str:
        return "https://example.invalid/audio"

    async def health(self):
        return {"storage_enabled": True, "storage_available": True}


class FakeVoiceService:
    def __init__(self) -> None:
        self.call_count = 0

    def predict_from_validated_upload(
        self,
        upload,
        *,
        request_id=None,
        storage_metadata=None,
    ):
        self.call_count += 1
        return prediction_response(
            upload,
            request_id=request_id or "request-id",
            storage_metadata=storage_metadata,
        )


class FakeVoiceServiceWithProcessedAudio(FakeVoiceService):
    """Models the real VoiceService's enhanced handoff-capable method."""

    PROCESSED_AUDIO = ProcessedAudio(
        waveform=np.zeros(100, dtype=np.float32),
        sample_rate=16000,
        original_sample_rate=16000,
        original_channels=1,
        duration_seconds=0.00625,
        was_resampled=False,
        was_converted_to_mono=False,
        normalisation_applied=True,
        peak_amplitude=1.0,
        rms_energy=0.0,
    )
    TEMPORAL_EVIDENCE = (
        TemporalAttentionWindowInput(
            start_seconds=0.0,
            end_seconds=0.00625,
            token_times_seconds=np.array([0.003125], dtype=np.float32),
            attention_density=np.array([1.0], dtype=np.float32),
            spoof_probability=0.8,
        ),
    )

    def predict_from_validated_upload_with_processed_audio(
        self,
        upload,
        *,
        request_id=None,
        storage_metadata=None,
        cleanup_upload=True,
    ):
        prediction = self.predict_from_validated_upload(
            upload,
            request_id=request_id,
            storage_metadata=storage_metadata,
        )
        return PredictionExecutionResult(
            prediction=prediction,
            processed_audio=self.PROCESSED_AUDIO,
            temporal_evidence=self.TEMPORAL_EVIDENCE,
        )


class BlockingVoiceService(FakeVoiceService):
    def __init__(
        self,
        *,
        loop,
        started: asyncio.Event,
        release: threading.Event,
    ) -> None:
        super().__init__()
        self._loop = loop
        self._started = started
        self._release = release

    def predict_from_validated_upload(
        self,
        upload,
        *,
        request_id=None,
        storage_metadata=None,
    ):
        self._loop.call_soon_threadsafe(self._started.set)
        self._release.wait(timeout=2)
        return super().predict_from_validated_upload(
            upload,
            request_id=request_id,
            storage_metadata=storage_metadata,
        )


def encode_audio_fixture(
    tmp_path: Path,
    *,
    extension: str,
    codec: str,
    container: str | None,
) -> Path:
    output_path = tmp_path / f"recording.{extension}"
    command = [
        str(FFMPEG),
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=1.1",
        "-ac",
        "2",
        "-c:a",
        codec,
    ]
    if container is not None:
        command.extend(["-f", container])
    command.append(str(output_path))
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.skip(f"Installed FFmpeg cannot generate {codec} test fixtures.")
    return output_path
