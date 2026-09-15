"""Focused contract tests for POST /api/v1/predictions idempotency handling.

The idempotency key is an OPTIONAL owner-scoped multipart form field. These
tests pin the four contract cases (missing key, replay, conflict, new key),
the Swagger/OpenAPI surface (no static default key that clients would resubmit),
and the guarantee that local development convenience never weakens the
production duplicate-request protection.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from app.config.settings import Settings
from app.main import app
from tests.test_prediction_routes import (
    FakeSubmissionRepository,
    FakeSubmissionStorage,
    FakeVoiceService,
    prediction_client,
    upload_metadata,
)

WAV = ("sample.wav", b"placeholder", "audio/wav")
OTHER_WAV = ("other-sample.wav", b"different-bytes", "audio/wav")

DEV_BYPASS_SETTINGS = Settings(
    _env_file=None,
    app_env="development",
    dev_auth_bypass=True,
)

PRODUCTION_SETTINGS = Settings(
    _env_file=None,
    app_env="production",
    debug=False,
    # Production settings reject the development-only bypass at validation time.
    dev_auth_bypass=False,
    api_key_hash_secret="production-api-key-secret",
    mongodb_uri="mongodb://localhost:27017",
    mongodb_database="multiscope",
    storage_policy="required",
    cloudinary_cloud_name="cloud",
    cloudinary_api_key="key",
    cloudinary_api_secret="secret",
    clerk_issuer="https://clerk.example.com",
    clerk_jwks_url="https://clerk.example.com/.well-known/jwks.json",
    clerk_audience="multiscope",
    clerk_secret_key="sk_test_production_placeholder",
    clerk_authorized_parties="https://app.example.com",
    allowed_origins="https://app.example.com",
)


@pytest.fixture(autouse=True)
def reset_prediction_rate_limit():
    """Keep the shared per-process limiter from bleeding across these tests.

    This module intentionally submits many predictions from one client IP,
    which would otherwise trip the 20-per-window prediction rate limit and
    mask the idempotency assertions.
    """

    app.state.rate_limiter.reset()
    yield
    app.state.rate_limiter.reset()


@pytest.fixture
def stub_upload(monkeypatch, tmp_path):
    """Stub validated ingestion so tests exercise reservation, not FFmpeg."""

    saved = tmp_path / "validated.wav"

    def validated_upload(file, **kwargs):
        # The pipeline unlinks the validated upload after every run, so each
        # call re-materialises it exactly like a real ingestion would.
        saved.write_bytes(b"validated")
        return upload_metadata(saved, file.filename or "sample.wav", "wav")

    monkeypatch.setattr(
        "app.services.prediction_submission_service."
        "save_validated_audio_upload_blocking",
        validated_upload,
    )
    return saved


def submit(
    client: TestClient,
    *,
    file=WAV,
    source_type: str = "dashboard_upload",
    client_filename: str | None = None,
    idempotency_key: str | None = None,
):
    data: dict[str, str] = {"source_type": source_type}
    if client_filename is not None:
        data["client_filename"] = client_filename
    if idempotency_key is not None:
        data["idempotency_key"] = idempotency_key
    return client.post(
        "/api/v1/predictions",
        data=data,
        files={"file": file},
    )


# 1. Missing Idempotency key -------------------------------------------------


@pytest.mark.parametrize("blank_key", [None, "", "   "])
def test_missing_or_blank_key_creates_a_new_prediction_every_time(
    stub_upload,
    blank_key,
) -> None:
    repository = FakeSubmissionRepository()
    voice = FakeVoiceService()

    with prediction_client(repository, voice=voice) as client:
        first = submit(client, idempotency_key=blank_key)
        second = submit(client, idempotency_key=blank_key)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["prediction_id"] != second.json()["prediction_id"]
    assert voice.call_count == 2
    assert len(repository.documents) == 2
    # No key is invented server-side for a keyless request.
    assert [document["idempotency_key"] for document in repository.documents] == [
        None,
        None,
    ]


# 2. New key + request A -----------------------------------------------------


def test_new_key_processes_request_normally(stub_upload) -> None:
    repository = FakeSubmissionRepository()
    voice = FakeVoiceService()

    with prediction_client(repository, voice=voice) as client:
        response = submit(client, idempotency_key="key-a")

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert voice.call_count == 1
    assert repository.documents[0]["idempotency_key"] == "key-a"


# 3. Same key + identical request A ------------------------------------------


def test_same_key_and_identical_request_replays_without_running_twice(
    stub_upload,
) -> None:
    repository = FakeSubmissionRepository()
    storage = FakeSubmissionStorage()
    voice = FakeVoiceService()

    with prediction_client(repository, storage=storage, voice=voice) as client:
        first = submit(client, client_filename="a.wav", idempotency_key="key-a")
        second = submit(client, client_filename="a.wav", idempotency_key="key-a")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["prediction_id"] == first.json()["prediction_id"]
    assert second.json()["request_id"] == first.json()["request_id"]
    assert voice.call_count == 1
    assert storage.upload_count == 1
    assert len(repository.documents) == 1


# 4. Same key + different request B ------------------------------------------


@pytest.mark.parametrize(
    ("second_kwargs", "changed_field"),
    [
        ({"file": OTHER_WAV}, "submitted_filename"),
        ({"client_filename": "renamed.wav"}, "client_filename"),
        ({"source_type": "live_recording"}, "source_type"),
    ],
)
def test_same_key_with_a_different_request_returns_409_conflict(
    stub_upload,
    second_kwargs,
    changed_field,
) -> None:
    repository = FakeSubmissionRepository()
    voice = FakeVoiceService()

    with prediction_client(repository, voice=voice) as client:
        first = submit(client, client_filename="a.wav", idempotency_key="key-a")
        second = submit(
            client,
            **{"client_filename": "a.wav", "idempotency_key": "key-a", **second_kwargs},
        )

    assert first.status_code == 200
    assert second.status_code == 409
    error = second.json()["error"]
    assert error["code"] == "idempotency_conflict"
    assert "already used for a different request" in error["message"]
    # The conflicting request must not have produced a second prediction.
    assert voice.call_count == 1
    assert len(repository.documents) == 1
    assert changed_field  # documents which field drove the conflict


# 5. Different key + request B -----------------------------------------------


def test_different_key_for_a_different_request_succeeds(stub_upload) -> None:
    repository = FakeSubmissionRepository()
    voice = FakeVoiceService()

    with prediction_client(repository, voice=voice) as client:
        first = submit(client, client_filename="a.wav", idempotency_key="key-a")
        second = submit(
            client,
            file=OTHER_WAV,
            client_filename="b.wav",
            idempotency_key="key-b",
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["prediction_id"] != second.json()["prediction_id"]
    assert voice.call_count == 2
    assert len(repository.documents) == 2


# 6. Same audio + different key ----------------------------------------------


def test_same_audio_with_a_different_key_creates_a_new_prediction(
    stub_upload,
) -> None:
    repository = FakeSubmissionRepository()
    voice = FakeVoiceService()

    with prediction_client(repository, voice=voice) as client:
        first = submit(client, client_filename="a.wav", idempotency_key="key-a")
        second = submit(client, client_filename="a.wav", idempotency_key="key-b")

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["prediction_id"] != second.json()["prediction_id"]
    assert voice.call_count == 2


def test_replay_matching_is_filename_based_not_content_based(stub_upload) -> None:
    """Documents the current (intentional) fingerprint semantics.

    The reservation happens before the upload stream is read, so the logical
    request is source_type + client_filename + submitted filename. Two
    different audio payloads sent under the same filename and the same key
    therefore replay rather than conflict. Changing this would require hashing
    the upload before reservation; see the idempotency report.
    """

    repository = FakeSubmissionRepository()
    voice = FakeVoiceService()

    with prediction_client(repository, voice=voice) as client:
        first = submit(
            client,
            file=("sample.wav", b"payload-one", "audio/wav"),
            client_filename="a.wav",
            idempotency_key="key-a",
        )
        second = submit(
            client,
            file=("sample.wav", b"payload-two-different", "audio/wav"),
            client_filename="a.wav",
            idempotency_key="key-a",
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["prediction_id"] == first.json()["prediction_id"]
    assert voice.call_count == 1


# 7. Swagger / OpenAPI surface -----------------------------------------------


def test_openapi_does_not_force_a_static_idempotency_key_default(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/predictions"]["post"]
    multipart = operation["requestBody"]["content"]["multipart/form-data"]
    body_schema_name = multipart["schema"]["$ref"].rsplit("/", 1)[-1]
    body_schema = schema["components"]["schemas"][body_schema_name]
    field = body_schema["properties"]["idempotency_key"]

    assert "idempotency_key" not in body_schema.get("required", [])
    assert "default" not in field
    assert "example" not in field
    assert "examples" not in field
    # Multipart examples must stay documentation-only (no submitted values).
    for example in multipart["examples"].values():
        assert "value" not in example
    assert "unique" in field["description"].lower()
    assert "idempotency_conflict" in operation["responses"]["409"]["description"]
    assert (
        "idempotency_conflict"
        in operation["responses"]["409"]["content"]["application/json"]["examples"]
    )


def test_openapi_documents_when_to_reuse_an_idempotency_key(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    description = schema["paths"]["/api/v1/predictions"]["post"]["description"]

    assert "Idempotency" in description
    assert "optional" in description
    assert "409" in description


# 8-10. Development convenience must not weaken production -------------------


def test_development_bypass_still_rejects_a_reused_key(stub_upload) -> None:
    repository = FakeSubmissionRepository()
    voice = FakeVoiceService()

    with prediction_client(
        repository,
        voice=voice,
        app_settings=DEV_BYPASS_SETTINGS,
    ) as client:
        first = submit(client, client_filename="a.wav", idempotency_key="key-a")
        second = submit(client, file=OTHER_WAV, idempotency_key="key-a")

    assert first.status_code == 200
    assert second.status_code == 409
    body = second.json()
    assert body["error"]["code"] == "idempotency_conflict"
    details = body["error"]["details"]
    assert details is not None
    assert details["existing_prediction_id"] == first.json()["prediction_id"]
    assert "submitted_filename" in details["changed_fields"]
    # The reused key is never silently replaced with a generated one.
    assert len(repository.documents) == 1
    assert voice.call_count == 1


def test_development_bypass_does_not_generate_a_key_for_keyless_requests(
    stub_upload,
) -> None:
    repository = FakeSubmissionRepository()
    voice = FakeVoiceService()

    with prediction_client(
        repository,
        voice=voice,
        app_settings=DEV_BYPASS_SETTINGS,
    ) as client:
        first = submit(client)
        second = submit(client, file=OTHER_WAV)

    assert first.status_code == 200
    assert second.status_code == 200
    assert [document["idempotency_key"] for document in repository.documents] == [
        None,
        None,
    ]
    assert voice.call_count == 2


def test_production_rejects_a_reused_key_and_hides_diagnostics(stub_upload) -> None:
    repository = FakeSubmissionRepository()
    voice = FakeVoiceService()

    with prediction_client(
        repository,
        voice=voice,
        app_settings=PRODUCTION_SETTINGS,
    ) as client:
        first = submit(client, client_filename="a.wav", idempotency_key="key-a")
        second = submit(client, file=OTHER_WAV, idempotency_key="key-a")
        replay = submit(client, client_filename="a.wav", idempotency_key="key-a")

    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["error"]["code"] == "idempotency_conflict"
    # Production never leaks another request's stored shape.
    assert second.json()["error"]["details"] is None
    assert replay.status_code == 200
    assert replay.json()["prediction_id"] == first.json()["prediction_id"]
    assert voice.call_count == 1
    assert len(repository.documents) == 1
