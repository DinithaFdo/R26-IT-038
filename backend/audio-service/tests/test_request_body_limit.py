from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_prediction_submission_service
from app.auth.clerk_auth import require_clerk_user
from app.auth.schemas import AuthPrincipal
from app.config.settings import settings
from app.main import app
from app.schemas.common import PredictionStatus, SourceType
from app.schemas.prediction_history import PredictionHistoryAudioMetadata
from app.schemas.prediction_submission import PredictionSubmissionResponse
from tests.test_voice_routes import make_wav_bytes


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def cleanup_state():
    original_limit = settings.max_request_body_mb
    original_enabled = settings.enable_legacy_anonymous_prediction
    original_env = settings.app_env
    for path in uploaded_files():
        path.unlink(missing_ok=True)
    yield
    settings.max_request_body_mb = original_limit
    settings.enable_legacy_anonymous_prediction = original_enabled
    settings.app_env = original_env
    app.dependency_overrides.clear()
    for path in uploaded_files():
        path.unlink(missing_ok=True)


def test_small_authenticated_prediction_request_passes() -> None:
    service = RecordingSubmissionService()
    app.dependency_overrides[require_clerk_user] = authenticated_principal
    app.dependency_overrides[get_prediction_submission_service] = lambda: service

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predictions",
            data={"source_type": "dashboard_upload"},
            files={"file": ("sample.wav", make_wav_bytes(), "audio/wav")},
            headers={"X-Request-ID": "small-body"},
        )

    assert response.status_code == 200
    UUID(response.json()["request_id"])
    assert response.json()["request_id"] != "small-body"
    assert response.headers["X-Client-Correlation-ID"] == "small-body"
    assert response.json()["status"] == "completed"
    assert service.invocations == 1


def test_oversized_content_length_is_rejected_before_endpoint_service() -> None:
    service = RecordingSubmissionService()
    app.dependency_overrides[require_clerk_user] = authenticated_principal
    app.dependency_overrides[get_prediction_submission_service] = lambda: service

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predictions",
            content=b"",
            headers={
                "Content-Length": str(_limit_bytes() + 1),
                "Content-Type": "multipart/form-data; boundary=bodylimit",
                "X-Request-ID": "content-length-too-large",
            },
        )

    assert response.status_code == 413
    request_id = response.headers["X-Request-ID"]
    UUID(request_id)
    assert response.headers["X-Client-Correlation-ID"] == "content-length-too-large"
    assert response.json() == _request_too_large_payload(request_id)
    assert service.invocations == 0
    assert uploaded_files() == []


@pytest.mark.anyio
async def test_chunked_oversized_request_is_rejected_before_endpoint_service() -> None:
    service = RecordingSubmissionService()
    app.dependency_overrides[require_clerk_user] = authenticated_principal
    app.dependency_overrides[get_prediction_submission_service] = lambda: service

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/v1/predictions",
            content=oversized_multipart_stream(),
            headers={
                "Content-Type": "multipart/form-data; boundary=bodylimit",
                "X-Request-ID": "chunked-too-large",
            },
        )

    assert response.status_code == 413
    request_id = response.headers["X-Request-ID"]
    UUID(request_id)
    assert response.headers["X-Client-Correlation-ID"] == "chunked-too-large"
    assert response.json() == _request_too_large_payload(request_id)
    assert service.invocations == 0
    assert uploaded_files() == []


def test_external_prediction_request_is_limited_before_api_key_auth() -> None:
    service = RecordingSubmissionService()
    app.dependency_overrides[get_prediction_submission_service] = lambda: service

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/external/predictions",
            content=b"",
            headers={
                "Content-Length": str(_limit_bytes() + 1),
                "Content-Type": "multipart/form-data; boundary=bodylimit",
                "X-Request-ID": "external-too-large",
            },
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_body_too_large"
    assert service.invocations == 0


def test_enabled_legacy_prediction_request_is_limited_before_endpoint_service() -> None:
    settings.enable_legacy_anonymous_prediction = True
    settings.app_env = "development"
    service = RecordingSubmissionService()
    app.dependency_overrides[require_clerk_user] = authenticated_principal
    app.dependency_overrides[get_prediction_submission_service] = lambda: service

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/voice/predict",
            content=b"",
            headers={
                "Content-Length": str(_limit_bytes() + 1),
                "Content-Type": "multipart/form-data; boundary=bodylimit",
                "X-Request-ID": "legacy-too-large",
            },
        )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_body_too_large"
    assert service.invocations == 0


def test_disabled_legacy_prediction_still_returns_not_found_for_oversized_header() -> None:
    service = RecordingSubmissionService()
    app.dependency_overrides[require_clerk_user] = authenticated_principal
    app.dependency_overrides[get_prediction_submission_service] = lambda: service

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/voice/predict",
            content=b"",
            headers={
                "Content-Length": str(_limit_bytes() + 1),
                "Content-Type": "multipart/form-data; boundary=bodylimit",
                "X-Request-ID": "legacy-disabled-too-large",
            },
        )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert service.invocations == 0


async def oversized_multipart_stream() -> AsyncIterator[bytes]:
    yield (
        b"--bodylimit\r\n"
        b'Content-Disposition: form-data; name="source_type"\r\n\r\n'
        b"dashboard_upload\r\n"
        b"--bodylimit\r\n"
        b'Content-Disposition: form-data; name="file"; filename="sample.wav"\r\n'
        b"Content-Type: audio/wav\r\n\r\n"
    )
    remaining = _limit_bytes() + 1024
    chunk = b"x" * 8192
    while remaining > 0:
        next_size = min(len(chunk), remaining)
        yield chunk[:next_size]
        remaining -= next_size
    yield b"\r\n--bodylimit--\r\n"


def authenticated_principal() -> AuthPrincipal:
    return AuthPrincipal(
        subject="user:user_123",
        principal_type="clerk_user",
        user_id="user_123",
    )


class RecordingSubmissionService:
    def __init__(self) -> None:
        self.invocations = 0

    async def submit(self, **kwargs):
        self.invocations += 1
        return PredictionSubmissionResponse(
            prediction_id="prediction-body-limit",
            request_id=kwargs["request_id"],
            status=PredictionStatus.completed,
            source_type=kwargs.get("source_type") or SourceType.dashboard_upload,
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


def uploaded_files() -> list[Path]:
    uploads_dir = Path("uploads")
    if not uploads_dir.exists():
        return []
    return [path for path in uploads_dir.rglob("*") if path.is_file()]


def _limit_bytes() -> int:
    return settings.max_request_body_mb * 1024 * 1024


def _request_too_large_payload(request_id: str) -> dict:
    return {
        "request_id": request_id,
        "error": {
            "code": "request_body_too_large",
            "message": "Request body exceeds the configured size limit.",
            "details": None,
        },
    }
