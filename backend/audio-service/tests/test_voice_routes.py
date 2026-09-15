import asyncio
from collections.abc import Generator
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
import wave
from uuid import UUID

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_prediction_submission_service,
    get_voice_service,
)
from app.auth.clerk_auth import require_clerk_user
from app.auth.schemas import AuthPrincipal
from app.config.settings import Settings, settings
from app.main import app
from app.schemas.common import PredictionStatus, SourceType
from app.schemas.prediction_history import PredictionHistoryAudioMetadata
from app.schemas.prediction_submission import PredictionSubmissionResponse
from app.services.prediction_job_runner import InlinePredictionJobRunner


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def cleanup_overrides() -> Generator[None, None, None]:
    yield
    app.dependency_overrides.clear()


def make_wav_bytes() -> bytes:
    sample_rate = 16000
    frame_count = int(sample_rate * 0.1)
    time = np.arange(frame_count, dtype=np.float32) / sample_rate
    waveform = (0.25 * np.sin(2 * np.pi * 440 * time) * 32767).astype(np.int16)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(waveform.tobytes())
    return buffer.getvalue()


def uploaded_files() -> list[Path]:
    uploads_dir = Path("uploads")
    if not uploads_dir.exists():
        return []
    return [path for path in uploads_dir.rglob("*") if path.is_file()]


def authenticated_principal() -> AuthPrincipal:
    return AuthPrincipal(
        subject="user:user_123",
        principal_type="clerk_user",
        user_id="user_123",
    )


def enable_legacy_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "enable_legacy_anonymous_prediction", True)
    monkeypatch.setattr(settings, "app_env", "development")


def test_legacy_prediction_route_is_disabled_by_default(client: TestClient) -> None:
    response = client.post(
        "/api/v1/voice/predict",
        files={"file": ("sample.wav", make_wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 404


def test_legacy_prediction_route_is_disabled_in_production(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "enable_legacy_anonymous_prediction", True)
    monkeypatch.setattr(settings, "app_env", "production")

    response = client.post(
        "/api/v1/voice/predict",
        files={"file": ("sample.wav", make_wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 404


def test_enabled_legacy_prediction_route_rejects_anonymous_requests(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enable_legacy_route(monkeypatch)

    response = client.post(
        "/api/v1/voice/predict",
        files={"file": ("sample.wav", make_wav_bytes(), "audio/wav")},
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_failed"


def test_authenticated_legacy_prediction_route_uses_submission_service(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enable_legacy_route(monkeypatch)
    service = RecordingSubmissionService()
    app.dependency_overrides[require_clerk_user] = authenticated_principal
    app.dependency_overrides[get_prediction_submission_service] = lambda: service

    response = client.post(
        "/api/v1/voice/predict",
        files={"file": ("sample.wav", make_wav_bytes(), "audio/wav")},
        headers={"X-Request-ID": "legacy-authenticated"},
    )

    assert response.status_code == 200
    response_request_id = response.json()["request_id"]
    UUID(response_request_id)
    assert response_request_id != "legacy-authenticated"
    assert response.headers["X-Client-Correlation-ID"] == "legacy-authenticated"
    assert response.json()["source_type"] == "dashboard_upload"
    assert service.calls == [
        {
            "principal_user_id": "user_123",
            "source_type": SourceType.dashboard_upload,
            "request_id": response_request_id,
            "client_correlation_id": "legacy-authenticated",
        }
    ]


@pytest.mark.anyio
async def test_enabled_legacy_prediction_route_runs_behind_job_runner_semaphore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enable_legacy_route(monkeypatch)
    service = SemaphoreBackedSubmissionService()
    app.dependency_overrides[require_clerk_user] = authenticated_principal
    app.dependency_overrides[get_prediction_submission_service] = lambda: service

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        responses = await asyncio.gather(
            client.post(
                "/api/v1/voice/predict",
                files={"file": ("first.wav", make_wav_bytes(), "audio/wav")},
                headers={"X-Request-ID": "legacy-concurrency-1"},
            ),
            client.post(
                "/api/v1/voice/predict",
                files={"file": ("second.wav", make_wav_bytes(), "audio/wav")},
                headers={"X-Request-ID": "legacy-concurrency-2"},
            ),
        )

    assert [response.status_code for response in responses] == [200, 200]
    assert service.max_active_jobs == 1


def test_legacy_prediction_route_is_marked_deprecated_in_openapi(
    client: TestClient,
) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    operation = response.json()["paths"]["/api/v1/voice/predict"]["post"]
    assert operation["deprecated"] is True
    assert "POST /api/v1/predictions" in operation["description"]


def test_canonical_prediction_route_remains_unchanged(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    operation = response.json()["paths"]["/api/v1/predictions"]["post"]
    assert operation.get("deprecated") is not True
    assert operation["requestBody"]["content"]["multipart/form-data"]


def test_model_health_endpoint_reports_dummy_mode(client: TestClient) -> None:
    response = client.get("/api/v1/voice/models/health")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 4
    for branch in payload:
        assert {"model_name", "display_name", "mode", "is_loaded"} <= branch.keys()
        assert branch["mode"] == "dummy"
        assert branch["uses_dummy_mode"] is True
        assert "not a research result" in branch["warning"]


class RecordingSubmissionService:
    def __init__(self) -> None:
        self.calls = []

    async def submit(
        self,
        *,
        file,
        principal: AuthPrincipal,
        source_type: SourceType,
        request_id: str,
        client_correlation_id: str | None = None,
        client_filename: str | None = None,
        idempotency_key: str | None = None,
    ) -> PredictionSubmissionResponse:
        self.calls.append(
            {
                "principal_user_id": principal.user_id,
                "source_type": source_type,
                "request_id": request_id,
                "client_correlation_id": client_correlation_id,
            }
        )
        return make_submission_response(
            request_id=request_id,
            prediction_id="legacy-prediction-1",
        )


class SemaphoreBackedSubmissionService:
    def __init__(self) -> None:
        self._runner = InlinePredictionJobRunner(
            Settings(
                _env_file=None,
                prediction_job_concurrency=1,
                prediction_job_timeout_seconds=5,
            )
        )
        self._persistence = FakePersistence()
        self._active_jobs = 0
        self.max_active_jobs = 0

    async def submit(
        self,
        *,
        file,
        principal: AuthPrincipal,
        source_type: SourceType,
        request_id: str,
        client_correlation_id: str | None = None,
        client_filename: str | None = None,
        idempotency_key: str | None = None,
    ) -> PredictionSubmissionResponse:
        return await self._runner.run(
            prediction_id=f"prediction-{request_id}",
            request_id=request_id,
            persistence=self._persistence,
            execute=lambda: self._execute(request_id=request_id),
        )

    async def _execute(self, *, request_id: str) -> PredictionSubmissionResponse:
        self._active_jobs += 1
        self.max_active_jobs = max(self.max_active_jobs, self._active_jobs)
        await asyncio.sleep(0.05)
        self._active_jobs -= 1
        return make_submission_response(
            request_id=request_id,
            prediction_id=f"prediction-{request_id}",
        )


class FakePersistence:
    async def mark_failed(
        self,
        request_id: str,
        *,
        stage: str,
        code: str,
    ) -> None:
        return None


def make_submission_response(
    *,
    request_id: str,
    prediction_id: str,
) -> PredictionSubmissionResponse:
    return PredictionSubmissionResponse(
        prediction_id=prediction_id,
        request_id=request_id,
        status=PredictionStatus.completed,
        source_type=SourceType.dashboard_upload,
        audio=PredictionHistoryAudioMetadata(
            original_filename="sample.wav",
            original_extension="wav",
            detected_container="wav",
            detected_codec="pcm_s16le",
            duration_seconds=0.1,
            sample_rate=16000,
            channels=1,
            size_bytes=1024,
        ),
        branches=[],
        fusion=None,
        research_eligible=False,
        created_at=datetime.now(UTC),
    )
