from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
import logging
from typing import Any

from fastapi.testclient import TestClient
import pytest

from app.api.dependencies import get_api_key_service
from app.api.dependencies import get_prediction_submission_service
from app.auth.api_keys import get_api_key_repository
from app.auth.api_keys import generate_api_key, hash_api_key, key_prefix
from app.auth.clerk_auth import require_clerk_user
from app.auth.schemas import AuthPrincipal
from app.main import app
from app.schemas.common import ModelMode, PredictionLabel, PredictionStatus
from app.schemas.common import SourceType
from app.schemas.prediction import BranchPrediction, FusionResult
from app.schemas.prediction import ProbabilityScores
from app.schemas.prediction_history import PredictionHistoryAudioMetadata
from app.schemas.prediction_submission import PredictionSubmissionResponse
from app.services.api_key_service import ApiKeyService


def test_clerk_user_can_create_key_and_full_key_is_displayed_once() -> None:
    repository = FakeApiKeyRepository()
    with api_key_client(repository) as client:
        create_response = client.post(
            "/api/v1/me/api-keys",
            json={
                "name": "Partner",
                "scopes": ["prediction:create", "prediction:read"],
            },
        )
        list_response = client.get("/api/v1/me/api-keys")

    assert create_response.status_code == 200
    created = create_response.json()
    assert created["api_key"].startswith("msk_live_")
    assert created["key_prefix"] == created["api_key"][:24]
    assert "api_key" not in list_response.json()["items"][0]
    assert repository.documents[0]["key_hash"] != created["api_key"]
    assert repository.documents[0]["key_prefix"] == created["key_prefix"]


def test_clerk_user_can_revoke_own_api_key() -> None:
    repository = FakeApiKeyRepository()
    document = repository.add_key(owner_user_id="user_123")

    with api_key_client(repository) as client:
        response = client.delete(f"/api/v1/me/api-keys/{document['id']}")

    assert response.status_code == 200
    assert response.json() == {"api_key_id": document["id"], "revoked": True}
    assert repository.documents[0]["revoked_at"] is not None


def test_valid_api_key_can_create_external_prediction() -> None:
    repository = FakeApiKeyRepository()
    full_key = repository.add_key(
        owner_user_id="user_123",
        scopes=["prediction:create"],
    )["full_key"]
    submission = FakeExternalSubmissionService()

    with external_client(repository, submission) as client:
        response = client.post(
            "/api/v1/external/predictions",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            headers={"Authorization": f"Bearer {full_key}"},
        )

    assert response.status_code == 200
    assert response.json()["source_type"] == "public_api"
    assert submission.calls[0]["principal"].principal_type == "api_key"
    assert submission.calls[0]["principal"].user_id == "user_123"
    assert repository.documents[0]["usage_count"] == 1
    assert repository.documents[0]["last_used_at"] is not None
    assert any(event["event_type"] == "api_key_auth_success" for event in repository.audit_events)


def test_invalid_api_key_is_rejected_and_not_logged(caplog) -> None:
    repository = FakeApiKeyRepository()
    leaked_key = "msk_live_not-a-real-key"
    caplog.set_level(logging.INFO)

    with external_client(repository, FakeExternalSubmissionService()) as client:
        response = client.post(
            "/api/v1/external/predictions",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            headers={"Authorization": f"Bearer {leaked_key}"},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "http_error"
    assert leaked_key not in caplog.text


def test_expired_api_key_is_rejected() -> None:
    repository = FakeApiKeyRepository()
    full_key = repository.add_key(
        owner_user_id="user_123",
        scopes=["prediction:create"],
        expires_at=datetime.now(UTC) - timedelta(days=1),
    )["full_key"]

    with external_client(repository, FakeExternalSubmissionService()) as client:
        response = client.post(
            "/api/v1/external/predictions",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            headers={"Authorization": f"Bearer {full_key}"},
        )

    assert response.status_code == 401
    assert any(event["event_type"] == "api_key_auth_expired" for event in repository.audit_events)


def test_revoked_api_key_is_rejected() -> None:
    repository = FakeApiKeyRepository()
    full_key = repository.add_key(
        owner_user_id="user_123",
        scopes=["prediction:create"],
        revoked_at=datetime.now(UTC),
    )["full_key"]

    with external_client(repository, FakeExternalSubmissionService()) as client:
        response = client.post(
            "/api/v1/external/predictions",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            headers={"Authorization": f"Bearer {full_key}"},
        )

    assert response.status_code == 401
    assert any(event["event_type"] == "api_key_auth_revoked" for event in repository.audit_events)


def test_api_key_missing_scope_is_rejected() -> None:
    repository = FakeApiKeyRepository()
    full_key = repository.add_key(
        owner_user_id="user_123",
        scopes=["prediction:read"],
    )["full_key"]

    with external_client(repository, FakeExternalSubmissionService()) as client:
        response = client.post(
            "/api/v1/external/predictions",
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            headers={"Authorization": f"Bearer {full_key}"},
        )

    assert response.status_code == 403
    assert any(event["event_type"] == "api_key_missing_scope" for event in repository.audit_events)


def test_api_key_owner_is_used_for_external_prediction_not_client_input() -> None:
    repository = FakeApiKeyRepository()
    full_key = repository.add_key(
        owner_user_id="owner_user",
        scopes=["prediction:create"],
    )["full_key"]
    submission = FakeExternalSubmissionService()

    with external_client(repository, submission) as client:
        response = client.post(
            "/api/v1/external/predictions",
            data={"owner_user_id": "attacker"},
            files={"file": ("sample.wav", b"audio", "audio/wav")},
            headers={"Authorization": f"Bearer {full_key}"},
        )

    assert response.status_code == 200
    assert submission.calls[0]["principal"].user_id == "owner_user"


@contextmanager
def api_key_client(repository) -> Generator[TestClient, None, None]:
    def principal_override() -> AuthPrincipal:
        return AuthPrincipal(
            subject="user:user_123",
            principal_type="clerk_user",
            user_id="user_123",
        )

    app.dependency_overrides[require_clerk_user] = principal_override
    app.dependency_overrides[get_api_key_service] = lambda: ApiKeyService(repository)
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


@contextmanager
def external_client(
    repository,
    submission_service,
) -> Generator[TestClient, None, None]:
    app.dependency_overrides[get_api_key_repository] = lambda: repository
    app.dependency_overrides[get_prediction_submission_service] = lambda: (
        submission_service
    )
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


class FakeApiKeyRepository:
    def __init__(self) -> None:
        self.documents = []
        self.audit_events = []

    def add_key(
        self,
        *,
        owner_user_id: str,
        scopes: list[str] | None = None,
        expires_at=None,
        revoked_at=None,
    ) -> dict[str, Any]:
        full_key = generate_api_key()
        document = {
            "id": f"api_key_{len(self.documents) + 1}",
            "key_hash": hash_api_key(full_key),
            "key_prefix": key_prefix(full_key),
            "owner_user_id": owner_user_id,
            "name": "Test key",
            "scopes": scopes or ["prediction:create"],
            "created_at": datetime.now(UTC),
            "last_used_at": None,
            "expires_at": expires_at,
            "revoked_at": revoked_at,
            "usage_count": 0,
            "full_key": full_key,
        }
        self.documents.append(document)
        return document

    async def create_api_key(self, document):
        self.documents.append(document)
        return document["id"]

    async def list_api_keys_for_owner(self, owner_user_id):
        return [
            document
            for document in self.documents
            if document["owner_user_id"] == owner_user_id
        ]

    async def get_api_key_by_prefix(self, prefix):
        for document in self.documents:
            if document["key_prefix"] == prefix:
                return document
        return None

    async def revoke_api_key_for_owner(self, *, api_key_id, owner_user_id):
        for document in self.documents:
            if document["id"] == api_key_id and document["owner_user_id"] == owner_user_id:
                if document["revoked_at"] is None:
                    document["revoked_at"] = datetime.now(UTC)
                    return True
                return False
        return False

    async def mark_revoked(self, key_prefix):
        for document in self.documents:
            if document["key_prefix"] == key_prefix and document["revoked_at"] is None:
                document["revoked_at"] = datetime.now(UTC)
                return True
        return False

    async def mark_api_key_used(self, api_key_id):
        for document in self.documents:
            if document["id"] == api_key_id:
                document["last_used_at"] = datetime.now(UTC)
                document["usage_count"] += 1

    async def record_audit_event(self, event):
        self.audit_events.append(event)


class FakeExternalSubmissionService:
    def __init__(self) -> None:
        self.calls = []

    async def submit(self, **kwargs):
        self.calls.append(kwargs)
        return PredictionSubmissionResponse(
            prediction_id="prediction_123",
            request_id=kwargs["request_id"],
            status=PredictionStatus.completed,
            source_type=SourceType.public_api,
            audio=PredictionHistoryAudioMetadata(
                original_filename="sample.wav",
                original_extension="wav",
                detected_container="wav",
                detected_codec="pcm_s16le",
                duration_seconds=1.0,
                sample_rate=16000,
                channels=1,
                size_bytes=1024,
            ),
            branches=[
                BranchPrediction(
                    model_name="cnn_acoustic",
                    display_name="CNN Acoustic",
                    status="success",
                    mode=ModelMode.dummy,
                    prediction=PredictionLabel.spoof,
                    confidence=0.7,
                    probabilities=ProbabilityScores(bonafide=0.3, spoof=0.7),
                    processing_time_ms=1.0,
                )
            ],
            fusion=FusionResult(
                status="success",
                prediction=PredictionLabel.spoof,
                confidence=0.7,
                probabilities=ProbabilityScores(bonafide=0.3, spoof=0.7),
                method="weighted_average",
                branch_weights={"cnn_acoustic": 1.0},
                contains_dummy_branches=True,
                eligible_for_research_evaluation=False,
                warning="Dummy result.",
            ),
            research_eligible=False,
            created_at=datetime.now(UTC),
        )
