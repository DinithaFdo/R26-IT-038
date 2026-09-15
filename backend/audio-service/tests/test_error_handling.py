import pytest
from fastapi.testclient import TestClient
from uuid import UUID

from app.api.dependencies import get_prediction_submission_service
from app.auth.clerk_auth import require_clerk_user
from app.auth.schemas import AuthPrincipal
from app.config.settings import settings
from app.core.exceptions import (
    AudioProcessingUnavailableError,
    CorruptedAudioError,
    NoUsableModelBranchesError,
)
from app.main import app
from tests.test_voice_routes import make_wav_bytes


@pytest.fixture(autouse=True)
def cleanup_legacy_route_state():
    original_enabled = settings.enable_legacy_anonymous_prediction
    original_env = settings.app_env
    yield
    settings.enable_legacy_anonymous_prediction = original_enabled
    settings.app_env = original_env
    app.dependency_overrides.clear()


def assert_error_envelope(payload: dict, *, request_id: str, code: str) -> None:
    assert payload["request_id"] == request_id
    assert payload["error"]["code"] == code
    assert isinstance(payload["error"]["message"], str)
    assert payload["error"]["details"] is None


def test_request_id_middleware_generates_server_id_and_preserves_client_correlation(
    client: TestClient,
) -> None:
    response = client.get("/", headers={"X-Request-ID": "req-abc-123"})

    UUID(response.headers["X-Request-ID"])
    assert response.headers["X-Request-ID"] != "req-abc-123"
    assert response.headers["X-Client-Correlation-ID"] == "req-abc-123"


def test_request_id_middleware_generates_missing_header(client: TestClient) -> None:
    response = client.get("/")

    assert response.headers["X-Request-ID"]


def test_validation_error_uses_global_error_schema(client: TestClient) -> None:
    enable_legacy_route()
    app.dependency_overrides[require_clerk_user] = authenticated_principal
    response = client.post(
        "/api/v1/voice/predict",
        files={},
        headers={"X-Request-ID": "validation-global"},
    )

    assert response.status_code == 422
    request_id = response.headers["X-Request-ID"]
    UUID(request_id)
    assert response.headers["X-Client-Correlation-ID"] == "validation-global"
    assert_error_envelope(
        response.json(),
        request_id=request_id,
        code="request_validation_error",
    )


def test_domain_audio_error_uses_global_error_schema(client: TestClient) -> None:
    enable_legacy_route()
    app.dependency_overrides[require_clerk_user] = authenticated_principal
    app.dependency_overrides[get_prediction_submission_service] = lambda: (
        RaisingSubmissionService(CorruptedAudioError("Uploaded audio is corrupted."))
    )

    response = client.post(
        "/api/v1/voice/predict",
        files={"file": ("bad.wav", b"not audio", "audio/wav")},
        headers={"X-Request-ID": "audio-global"},
    )

    assert response.status_code == 400
    request_id = response.headers["X-Request-ID"]
    UUID(request_id)
    assert response.headers["X-Client-Correlation-ID"] == "audio-global"
    assert_error_envelope(
        response.json(),
        request_id=request_id,
        code="corrupted_audio",
    )
    assert "Traceback" not in response.text
    assert "/Volumes/" not in response.text


def test_unavailable_audio_tools_use_global_error_schema(
    client: TestClient,
) -> None:
    enable_legacy_route()
    app.dependency_overrides[require_clerk_user] = authenticated_principal
    app.dependency_overrides[get_prediction_submission_service] = lambda: (
        RaisingSubmissionService(
            AudioProcessingUnavailableError("Audio processing is unavailable.")
        )
    )

    response = client.post(
        "/api/v1/voice/predict",
        files={"file": ("sample.wav", make_wav_bytes(), "audio/wav")},
        headers={"X-Request-ID": "tools-unavailable"},
    )

    assert response.status_code == 503
    request_id = response.headers["X-Request-ID"]
    UUID(request_id)
    assert response.headers["X-Client-Correlation-ID"] == "tools-unavailable"
    assert_error_envelope(
        response.json(),
        request_id=request_id,
        code="audio_processing_unavailable",
    )
    assert "ffmpeg" not in response.text.lower()
    assert "ffprobe" not in response.text.lower()


def test_model_unavailable_error_uses_global_error_schema(client: TestClient) -> None:
    enable_legacy_route()
    app.dependency_overrides[require_clerk_user] = authenticated_principal
    app.dependency_overrides[get_prediction_submission_service] = lambda: (
        RaisingSubmissionService(
            NoUsableModelBranchesError("No usable model branches are available.")
        )
    )

    response = client.post(
        "/api/v1/voice/predict",
        files={"file": ("sample.wav", make_wav_bytes(), "audio/wav")},
        headers={"X-Request-ID": "model-global"},
    )

    assert response.status_code == 503
    request_id = response.headers["X-Request-ID"]
    UUID(request_id)
    assert response.headers["X-Client-Correlation-ID"] == "model-global"
    assert_error_envelope(
        response.json(),
        request_id=request_id,
        code="model_unavailable",
    )


def test_unexpected_error_uses_sanitized_global_error_schema() -> None:
    enable_legacy_route()
    app.dependency_overrides[require_clerk_user] = authenticated_principal
    app.dependency_overrides[get_prediction_submission_service] = lambda: (
        RaisingSubmissionService(RuntimeError("secret path /tmp/model-weights.bin"))
    )
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post(
                "/api/v1/voice/predict",
                files={"file": ("sample.wav", make_wav_bytes(), "audio/wav")},
                headers={"X-Request-ID": "unexpected-global"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 500
    request_id = response.headers["X-Request-ID"]
    UUID(request_id)
    assert response.headers["X-Client-Correlation-ID"] == "unexpected-global"
    assert_error_envelope(
        response.json(),
        request_id=request_id,
        code="internal_server_error",
    )
    assert "secret path" not in response.text
    assert "Traceback" not in response.text


def enable_legacy_route() -> None:
    settings.enable_legacy_anonymous_prediction = True
    settings.app_env = "development"


def authenticated_principal() -> AuthPrincipal:
    return AuthPrincipal(
        subject="user:user_123",
        principal_type="clerk_user",
        user_id="user_123",
    )


class RaisingSubmissionService:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def submit(self, **kwargs):
        raise self._error
