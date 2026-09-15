import base64
from datetime import UTC, datetime
import os
from typing import Any
from uuid import UUID

import pytest

from app.auth.api_keys import generate_api_key, hash_api_key, key_prefix
from app.config.settings import Settings
from app.schemas.common import (
    BranchStatus,
    ModelMode,
    PredictionLabel,
    PredictionStatus,
    SourceType,
)
from app.schemas.prediction import (
    BranchPrediction,
    FusionResult,
    ProbabilityScores,
)
from app.schemas.prediction_history import (
    PredictionDeleteResponse,
    PredictionDetailResponse,
    PredictionHistoryAudioMetadata,
    PredictionHistoryItem,
    PredictionHistoryListResponse,
    PredictionModeSummary,
)
from app.schemas.prediction_submission import PredictionSubmissionResponse
from app.services.voice_service import ModelRegistry
from mcp_server import lifecycle
from mcp_server import tools as mcp_tools
from mcp_server.server import MCP_TOOL_NAMES
from mcp_server.tools import (
    MCPToolContext,
    multiscope_create_prediction,
    multiscope_delete_prediction,
    multiscope_get_model_status,
    multiscope_get_prediction,
    multiscope_list_predictions,
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def reset_mcp_lifecycle_state():
    mcp_tools.clear_shared_context()
    lifecycle._started = False
    yield
    mcp_tools.clear_shared_context()
    lifecycle._started = False


@pytest.mark.anyio
async def test_mcp_create_prediction_uses_api_key_owner_and_mcp_source() -> None:
    api_repository = FakeApiKeyRepository()
    full_key = api_repository.add_key(
        owner_user_id="owner-user",
        scopes=["prediction:create"],
    )["full_key"]
    submission = FakeSubmissionService()
    context = fake_context(api_repository, submission_service=submission)

    response = await multiscope_create_prediction(
        api_token=full_key,
        filename="sample.wav",
        content_type="audio/wav",
        small_audio_payload_base64=_payload(b"fake-audio"),
        request_id="mcp-create",
        context=context,
    )

    assert response["ok"] is True
    UUID(response["request_id"])
    assert response["request_id"] != "mcp-create"
    assert response["data"]["source_type"] == "mcp"
    assert response["data"]["mode_summary"]["contains_dummy"] is True
    assert submission.calls[0]["principal"].user_id == "owner-user"
    assert submission.calls[0]["source_type"] == SourceType.mcp
    assert submission.calls[0]["payload_bytes"] == b"fake-audio"
    assert api_repository.documents[0]["usage_count"] == 1


@pytest.mark.anyio
async def test_mcp_create_prediction_rejects_missing_scope_safely() -> None:
    api_repository = FakeApiKeyRepository()
    full_key = api_repository.add_key(
        owner_user_id="owner-user",
        scopes=["prediction:read"],
    )["full_key"]

    response = await multiscope_create_prediction(
        api_token=full_key,
        filename="sample.wav",
        small_audio_payload_base64=_payload(b"fake-audio"),
        context=fake_context(api_repository),
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "permission_denied"
    assert "msk_live_" not in str(response)


@pytest.mark.anyio
async def test_mcp_invalid_token_is_not_exposed(caplog) -> None:
    leaked_token = "msk_live_invalid-token"

    response = await multiscope_get_prediction(
        api_token=leaked_token,
        prediction_id="prediction-1",
        context=fake_context(FakeApiKeyRepository()),
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "authentication_failed"
    assert leaked_token not in str(response)
    assert leaked_token not in caplog.text


@pytest.mark.anyio
async def test_mcp_history_tools_use_existing_history_service() -> None:
    api_repository = FakeApiKeyRepository()
    full_key = api_repository.add_key(
        owner_user_id="owner-user",
        scopes=["prediction:read", "prediction:list", "prediction:delete"],
    )["full_key"]
    history = FakeHistoryService()
    context = fake_context(api_repository, history_service=history)

    detail = await multiscope_get_prediction(
        api_token=full_key,
        prediction_id="prediction-1",
        context=context,
    )
    listing = await multiscope_list_predictions(
        api_token=full_key,
        limit=10,
        source_type=SourceType.mcp,
        context=context,
    )
    deletion = await multiscope_delete_prediction(
        api_token=full_key,
        prediction_id="prediction-1",
        context=context,
    )

    assert detail["ok"] is True
    assert detail["data"]["prediction_id"] == "prediction-1"
    assert listing["ok"] is True
    assert listing["data"]["items"][0]["source_type"] == "mcp"
    assert deletion["ok"] is True
    assert deletion["data"] == {
        "prediction_id": "prediction-1",
        "status": "deleted",
    }
    assert history.principals == ["owner-user", "owner-user", "owner-user"]


@pytest.mark.anyio
async def test_mcp_model_status_reports_dummy_mode() -> None:
    api_repository = FakeApiKeyRepository()
    full_key = api_repository.add_key(
        owner_user_id="owner-user",
        scopes=["prediction:read"],
    )["full_key"]

    response = await multiscope_get_model_status(
        api_token=full_key,
        context=fake_context(api_repository),
    )

    assert response["ok"] is True
    assert response["data"]["models"][0] == {
        "model_name": "cnn_acoustic",
        "display_name": "CNN Acoustic",
        "mode": "dummy",
        "is_loaded": True,
        "development_placeholder": True,
        # A dummy branch exists in the deployment but is never a research
        # result, and carries no verification attestations.
        "available": True,
        "research_ready": False,
        "preprocessing_verified": False,
        "class_mapping_verified": False,
    }


@pytest.mark.anyio
async def test_mcp_model_status_distinguishes_disabled_from_unverified_real() -> None:
    """An MCP client must be able to tell 'no model here' from 'unvalidated model'."""

    api_repository = FakeApiKeyRepository()
    full_key = api_repository.add_key(
        owner_user_id="owner-user",
        scopes=["prediction:read"],
    )["full_key"]

    context = fake_context(api_repository)
    context.voice_service.model_health = lambda: [  # type: ignore[method-assign]
        {
            "model_name": "cnn_acoustic",
            "display_name": "LFCC CNN/TCN",
            "mode": "real",
            "is_loaded": True,
            "research_ready": False,
            "preprocessing_verified": False,
            "class_mapping_verified": False,
        },
        {
            "model_name": "ssl_wavlm_xlsr",
            "display_name": "SSL Sequence",
            "mode": "disabled",
            "is_loaded": False,
            "research_ready": False,
        },
    ]

    response = await multiscope_get_model_status(api_token=full_key, context=context)
    models = response["data"]["models"]

    assert models[0]["available"] is True
    assert models[0]["development_placeholder"] is False
    assert models[0]["research_ready"] is False
    assert models[0]["preprocessing_verified"] is False

    assert models[1]["available"] is False
    assert models[1]["development_placeholder"] is False
    assert models[1]["is_loaded"] is False


@pytest.mark.anyio
async def test_mcp_create_prediction_rejects_oversized_small_payload(
    monkeypatch,
) -> None:
    import mcp_server.tools as mcp_tools

    api_repository = FakeApiKeyRepository()
    full_key = api_repository.add_key(
        owner_user_id="owner-user",
        scopes=["prediction:create"],
    )["full_key"]
    monkeypatch.setattr(mcp_tools.settings, "mcp_small_payload_max_bytes", 4)

    response = await multiscope_create_prediction(
        api_token=full_key,
        filename="sample.wav",
        small_audio_payload_base64=_payload(b"too-large"),
        context=fake_context(api_repository),
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "audio_payload_too_large"


@pytest.mark.anyio
async def test_mcp_create_prediction_rejects_asset_reference_until_handoff_exists() -> None:
    api_repository = FakeApiKeyRepository()
    full_key = api_repository.add_key(
        owner_user_id="owner-user",
        scopes=["prediction:create"],
    )["full_key"]

    response = await multiscope_create_prediction(
        api_token=full_key,
        secure_audio_asset_id="asset-123",
        context=fake_context(api_repository),
    )

    assert response["ok"] is False
    assert response["error"]["code"] == "unsupported_audio_reference"


def test_mcp_server_declares_initial_tool_set() -> None:
    assert MCP_TOOL_NAMES == (
        "multiscope_create_prediction",
        "multiscope_get_prediction",
        "multiscope_list_predictions",
        "multiscope_get_model_status",
        "multiscope_delete_prediction",
    )


@pytest.mark.anyio
async def test_mcp_startup_connects_once(monkeypatch) -> None:
    connect_counter = AsyncCallCounter()
    close_counter = AsyncCallCounter()
    context = fake_context(FakeApiKeyRepository())

    async def fake_mongodb_readiness(app_settings):
        return {"mongodb_configured": True, "mongodb_available": True}

    async def fake_storage_readiness(app_settings):
        return {"storage_enabled": False, "storage_available": False}

    monkeypatch.setattr(lifecycle, "connect_to_mongodb", connect_counter)
    monkeypatch.setattr(lifecycle, "close_mongodb", close_counter)
    monkeypatch.setattr(lifecycle, "mongodb_readiness", fake_mongodb_readiness)
    monkeypatch.setattr(lifecycle, "storage_readiness", fake_storage_readiness)
    monkeypatch.setattr(mcp_tools, "create_default_context", lambda _settings: context)

    first = await lifecycle.startup_mcp_server(
        make_mcp_settings(mongodb_uri="mongodb://configured")
    )
    second = await lifecycle.startup_mcp_server(
        make_mcp_settings(mongodb_uri="mongodb://configured")
    )

    assert first is context
    assert second is context
    assert connect_counter.calls == 1

    await lifecycle.shutdown_mcp_server()
    assert close_counter.calls == 1


@pytest.mark.anyio
async def test_mcp_tools_share_one_lifecycle_context() -> None:
    api_repository = FakeApiKeyRepository()
    full_key = api_repository.add_key(
        owner_user_id="owner-user",
        scopes=[
            "prediction:create",
            "prediction:read",
            "prediction:list",
            "prediction:delete",
        ],
    )["full_key"]
    history = FakeHistoryService()
    submission = FakeSubmissionService()
    context = fake_context(
        api_repository,
        submission_service=submission,
        history_service=history,
    )
    mcp_tools.set_shared_context(context)

    create = await multiscope_create_prediction(
        api_token=full_key,
        filename="sample.wav",
        small_audio_payload_base64=_payload(b"fake-audio"),
        request_id="shared-create",
    )
    detail = await multiscope_get_prediction(
        api_token=full_key,
        prediction_id="prediction-1",
    )
    listing = await multiscope_list_predictions(api_token=full_key)
    status_response = await multiscope_get_model_status(api_token=full_key)
    deletion = await multiscope_delete_prediction(
        api_token=full_key,
        prediction_id="prediction-1",
    )

    assert [
        create["ok"],
        detail["ok"],
        listing["ok"],
        status_response["ok"],
        deletion["ok"],
    ] == [True, True, True, True, True]
    assert mcp_tools.get_shared_context() is context
    assert submission.calls[0]["source_type"] == SourceType.mcp
    assert history.principals == ["owner-user", "owner-user", "owner-user"]


@pytest.mark.anyio
async def test_mcp_model_registry_and_semaphore_are_reused(monkeypatch) -> None:
    context = fake_context(FakeApiKeyRepository())

    async def fake_mongodb_readiness(app_settings):
        return {"mongodb_configured": True, "mongodb_available": True}

    async def fake_storage_readiness(app_settings):
        return {"storage_enabled": False, "storage_available": False}

    monkeypatch.setattr(lifecycle, "connect_to_mongodb", AsyncCallCounter())
    monkeypatch.setattr(lifecycle, "close_mongodb", AsyncCallCounter())
    monkeypatch.setattr(lifecycle, "mongodb_readiness", fake_mongodb_readiness)
    monkeypatch.setattr(lifecycle, "storage_readiness", fake_storage_readiness)
    monkeypatch.setattr(mcp_tools, "create_default_context", lambda _settings: context)

    started = await lifecycle.startup_mcp_server(
        make_mcp_settings(mongodb_uri="mongodb://configured")
    )
    shared = mcp_tools.get_shared_context()

    assert started is shared
    assert started.model_registry is shared.model_registry
    assert started.voice_service._model_registry is started.model_registry
    assert started.job_runner._semaphore is shared.job_runner._semaphore


@pytest.mark.anyio
async def test_mcp_shutdown_closes_mongodb_once(monkeypatch) -> None:
    close_counter = AsyncCallCounter()
    context = fake_context(FakeApiKeyRepository())
    mcp_tools.set_shared_context(context)
    lifecycle._started = True

    monkeypatch.setattr(lifecycle, "close_mongodb", close_counter)

    await lifecycle.shutdown_mcp_server()
    await lifecycle.shutdown_mcp_server()

    assert close_counter.calls == 1
    assert context.job_runner.shutdown_count == 1
    assert mcp_tools.get_optional_shared_context() is None


@pytest.mark.anyio
async def test_mcp_startup_failure_is_clear() -> None:
    with pytest.raises(lifecycle.MCPServerLifecycleError) as error:
        await lifecycle.startup_mcp_server(make_mcp_settings(mongodb_uri=""))

    assert "MONGODB_URI" in str(error.value)
    assert "mongodb://" not in str(error.value)
    assert mcp_tools.get_optional_shared_context() is None


@pytest.mark.anyio
async def test_mcp_tool_without_startup_returns_sanitized_not_ready_response() -> None:
    response = await multiscope_get_model_status(api_token="msk_live_missing")

    assert response["ok"] is False
    assert response["error"] == {
        "code": "mcp_server_not_ready",
        "message": "MCP server is not ready.",
        "details": None,
    }


@pytest.mark.anyio
async def test_mcp_live_mongodb_startup_is_skipped_without_test_uri() -> None:
    test_uri = os.getenv("TEST_MONGODB_URI")
    if not test_uri:
        pytest.skip("TEST_MONGODB_URI is not configured.")

    try:
        __import__("pymongo")
    except ImportError:
        pytest.skip("PyMongo is not installed in the test environment.")

    context = await lifecycle.startup_mcp_server(
        make_mcp_settings(
            mongodb_uri=test_uri,
            mongodb_database=os.getenv("TEST_MONGODB_DATABASE", "multiscope_test"),
        )
    )

    assert context is mcp_tools.get_shared_context()
    await lifecycle.shutdown_mcp_server()


def fake_context(
    api_repository,
    *,
    submission_service=None,
    history_service=None,
    voice_service=None,
) -> MCPToolContext:
    model_registry = ModelRegistry(models=[FakeModel()])
    job_runner = FakeJobRunner()
    voice_service = voice_service or FakeVoiceService(model_registry)
    return MCPToolContext(
        api_key_repository=api_repository,
        prediction_repository=FakePredictionRepository(),
        storage=FakeStorage(),
        model_registry=model_registry,
        job_runner=job_runner,
        submission_service=submission_service or FakeSubmissionService(),
        history_service=history_service or FakeHistoryService(),
        voice_service=voice_service,
    )


def _payload(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


class FakeApiKeyRepository:
    def __init__(self) -> None:
        self.documents = []
        self.audit_events = []

    def add_key(
        self,
        *,
        owner_user_id: str,
        scopes: list[str],
        expires_at=None,
        revoked_at=None,
    ) -> dict[str, Any]:
        full_key = generate_api_key()
        document = {
            "id": f"api_key_{len(self.documents) + 1}",
            "key_hash": hash_api_key(full_key),
            "key_prefix": key_prefix(full_key),
            "owner_user_id": owner_user_id,
            "name": "MCP key",
            "scopes": scopes,
            "created_at": datetime.now(UTC),
            "last_used_at": None,
            "expires_at": expires_at,
            "revoked_at": revoked_at,
            "usage_count": 0,
            "full_key": full_key,
        }
        self.documents.append(document)
        return document

    async def get_api_key_by_prefix(self, prefix):
        for document in self.documents:
            if document["key_prefix"] == prefix:
                return document
        return None

    async def mark_api_key_used(self, api_key_id):
        for document in self.documents:
            if document["id"] == api_key_id:
                document["usage_count"] += 1
                document["last_used_at"] = datetime.now(UTC)

    async def record_audit_event(self, event):
        self.audit_events.append(event)


class FakePredictionRepository:
    pass


class FakeStorage:
    pass


class FakeJobRunner:
    def __init__(self) -> None:
        self._semaphore = object()
        self.shutdown_count = 0

    async def run(self, **kwargs):
        return await kwargs["execute"]()

    async def run_blocking(self, func):
        return func()

    def shutdown(self) -> None:
        self.shutdown_count += 1


class FakeModel:
    pass


class FakeSubmissionService:
    def __init__(self) -> None:
        self.calls = []

    async def submit(self, **kwargs):
        payload_bytes = await kwargs["file"].read()
        kwargs["payload_bytes"] = payload_bytes
        self.calls.append(kwargs)
        return PredictionSubmissionResponse(
            prediction_id="prediction-1",
            request_id=kwargs["request_id"],
            status=PredictionStatus.completed,
            source_type=kwargs["source_type"],
            audio=PredictionHistoryAudioMetadata(
                original_filename=kwargs["file"].filename,
                original_extension="wav",
                detected_container="wav",
                detected_codec="pcm_s16le",
                duration_seconds=0.1,
                sample_rate=16000,
                channels=1,
                size_bytes=len(payload_bytes),
            ),
            branches=[dummy_branch()],
            fusion=dummy_fusion(),
            research_eligible=False,
            created_at=datetime.now(UTC),
        )


class FakeHistoryService:
    def __init__(self) -> None:
        self.principals = []

    async def get_prediction_detail(self, *, principal, prediction_id):
        self.principals.append(principal.user_id)
        return PredictionDetailResponse(
            prediction_id=prediction_id,
            request_id="request-1",
            source_type=SourceType.mcp,
            status=PredictionStatus.completed,
            audio=PredictionHistoryAudioMetadata(
                original_filename="sample.wav",
                original_extension="wav",
                detected_container="wav",
                detected_codec="pcm_s16le",
                duration_seconds=0.1,
                sample_rate=16000,
                channels=1,
                size_bytes=64,
            ),
            branches=[dummy_branch()],
            fusion=dummy_fusion(),
            preprocessing={},
            total_processing_time_ms=1.0,
            warnings=["Dummy branch outputs are development placeholders."],
            research_eligible=False,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            completed_at=datetime.now(UTC),
        )

    async def list_predictions(self, *, principal, **kwargs):
        self.principals.append(principal.user_id)
        return PredictionHistoryListResponse(
            items=[
                PredictionHistoryItem(
                    prediction_id="prediction-1",
                    filename="sample.wav",
                    source_type=SourceType.mcp,
                    status=PredictionStatus.completed,
                    duration_seconds=0.1,
                    final_prediction=PredictionLabel.spoof,
                    confidence=0.7,
                    mode_summary=PredictionModeSummary(
                        dummy=1,
                        real=0,
                        contains_dummy=True,
                    ),
                    created_at=datetime.now(UTC),
                )
            ],
            page=kwargs["page"],
            limit=kwargs["limit"],
            has_next=False,
        )

    async def delete_prediction(self, *, principal, prediction_id):
        self.principals.append(principal.user_id)
        return PredictionDeleteResponse(
            prediction_id=prediction_id,
            status=PredictionStatus.deleted,
        )


class FakeVoiceService:
    def __init__(self, model_registry: ModelRegistry | None = None) -> None:
        self._model_registry = model_registry or ModelRegistry(models=[FakeModel()])

    def model_health(self):
        return [
            {
                "model_name": "cnn_acoustic",
                "display_name": "CNN Acoustic",
                "mode": "dummy",
                "is_loaded": True,
            }
        ]


def dummy_branch() -> BranchPrediction:
    return BranchPrediction(
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


def dummy_fusion() -> FusionResult:
    return FusionResult(
        status=BranchStatus.success,
        prediction=PredictionLabel.spoof,
        confidence=0.7,
        probabilities=ProbabilityScores(bonafide=0.3, spoof=0.7),
        method="weighted_average",
        branch_weights={"cnn_acoustic": 1.0},
        contains_dummy_branches=True,
        eligible_for_research_evaluation=False,
        warning="Dummy result.",
    )


class AsyncCallCounter:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, *args, **kwargs):
        self.calls += 1


def make_mcp_settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "api_key_hash_secret": "test-secret-value",
        "cloudinary_storage_enabled": False,
    }
    values.update(overrides)
    return Settings(**values)
