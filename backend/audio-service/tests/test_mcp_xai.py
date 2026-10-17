"""Exercise MCP authentication, real XAI orchestration and SDK tool discovery."""

import asyncio
import base64
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from app.config.settings import Settings
from app.voice_xai.artifacts.service import LocalExplanationArtifactStore
from app.voice_xai.persistence.mongodb import MongoXaiExplanationRepository
from app.voice_xai.workflow.orchestrator import VoiceXaiOrchestrator
from mcp_server import lifecycle, server, tools, xai_tools
from mcp_server.auth import MCP_SCOPE_MAP
from tests.test_mcp_adapter import FakeApiKeyRepository, fake_context, make_mcp_settings
from tests.test_mongodb import FakeDatabase
from tests.test_voice_xai_orchestrator import (
    FailingNarrativeService, ReplayService, StaticTemporalService,
)
from tests.test_voice_xai_routes import (
    PausedQueue, PredictionRepository, StaticNarrativeService, _prediction_document,
)
from tests.xai_support import StaticSemanticService

XAI_TOOLS = tuple(name for name in server.MCP_TOOL_NAMES if name in vars(xai_tools))


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def xai(tmp_path):
    app_settings = Settings(
        _env_file=None, xai_enabled=True, xai_mode="real",
        xai_narrative_enabled=True, xai_artifact_root=str(tmp_path / "artifacts"),
        xai_narrative_api_key="test-only-key",
        xai_narrative_base_url="https://example.invalid/compatible-mode/v1",
    )
    api_repository = FakeApiKeyRepository()
    key = api_repository.add_key(
        owner_user_id="user-123", scopes=["xai:create", "xai:read", "xai:retry"],
    )["full_key"]
    context = fake_context(api_repository)
    context.app_settings = app_settings
    repository = MongoXaiExplanationRepository(FakeDatabase("mcp_xai"))
    queue = PausedQueue()
    store = LocalExplanationArtifactStore(tmp_path / "artifacts")
    prediction_repository = PredictionRepository(_prediction_document())
    orchestrator = VoiceXaiOrchestrator(
        repository=repository, prediction_repository=prediction_repository,
        queue=queue, app_settings=app_settings, artifact_store=store,
        temporal_service=StaticTemporalService(), semantic_service=StaticSemanticService(),
        narrative_service=StaticNarrativeService(), replay_service=ReplayService(),
    )
    context.xai_orchestrator = orchestrator
    context.xai_repository = repository
    context.xai_artifact_store = store
    context.xai_queue = queue
    return SimpleNamespace(
        context=context, key=key, queue=queue, repository=repository,
        predictions=prediction_repository, api_repository=api_repository,
    )


async def _call(xai, name, **kwargs):
    return await getattr(xai_tools, name)(
        api_token=xai.key, prediction_id="prediction-123", context=xai.context, **kwargs,
    )


async def _complete(xai):
    response = await _call(xai, "multiscope_create_explanation")
    assert response["ok"], response
    await xai.queue.task(xai.queue.bundle)
    return await _call(xai, "multiscope_get_explanation")


@pytest.mark.anyio
async def test_mcp_xai_full_workflow_preserves_prediction_and_reads_all_components(xai):
    before = json.dumps(xai.predictions.document, default=str, sort_keys=True)
    queued = await _call(xai, "multiscope_create_explanation")
    assert queued["ok"] and queued["data"]["status"] == "queued"
    reused = await _call(xai, "multiscope_create_explanation")
    assert reused["data"]["explanation_id"] == queued["data"]["explanation_id"]
    pending = await _call(xai, "multiscope_get_semantic_explanation")
    assert pending["error"]["code"] == "conflict"
    await xai.queue.task(xai.queue.bundle)
    completed = await _call(xai, "multiscope_get_explanation")
    assert completed["data"]["status"] == "completed", completed
    for tool_name, field in (
        ("multiscope_get_temporal_explanation", "temporal"),
        ("multiscope_get_semantic_explanation", "semantic"),
        ("multiscope_get_explanation_report", "combined_report"),
        ("multiscope_get_explanation_narrative", "narrative"),
    ):
        response = await _call(xai, tool_name)
        assert response["ok"], response
        assert response["data"] == completed["data"][field]
    assert json.dumps(xai.predictions.document, default=str, sort_keys=True) == before
    assert completed["data"]["research_eligible"] is False


@pytest.mark.anyio
@pytest.mark.parametrize("tool_name", XAI_TOOLS)
async def test_all_xai_tools_require_their_scope_before_accessing_evidence(xai, tool_name):
    xai.api_repository.documents[0]["scopes"] = ["prediction:create", "prediction:read"]
    kwargs = {"artifact_id": "private-artifact"} if tool_name.endswith("_artifact") else {}
    response = await _call(xai, tool_name, **kwargs)
    assert response["error"]["code"] == "permission_denied"
    assert "bundle" not in vars(xai.queue)
    assert xai.key not in json.dumps(response)


@pytest.mark.anyio
@pytest.mark.parametrize("tool_name", XAI_TOOLS)
async def test_all_xai_tools_hide_another_owners_data(xai, tool_name):
    await _complete(xai)
    xai.api_repository.documents[0]["owner_user_id"] = "different-user"
    kwargs = {"artifact_id": "private-artifact"} if tool_name.endswith("_artifact") else {}
    response = await _call(xai, tool_name, **kwargs)
    assert response["error"]["code"] == "not_found", response
    assert "user-123" not in json.dumps(response)


@pytest.mark.anyio
async def test_mcp_artifact_uses_owner_verified_reference_and_bounded_base64(xai, monkeypatch):
    response = await _complete(xai)
    reference = response["data"]["artifacts"][0]
    artifact = await _call(
        xai, "multiscope_get_explanation_artifact", artifact_id=reference["artifact_id"],
    )
    assert artifact["ok"], artifact
    assert artifact["data"]["artifact"] == reference
    assert base64.b64decode(artifact["data"]["content_base64"], validate=True)
    assert str(xai.context.xai_artifact_store._root) not in json.dumps(artifact)
    monkeypatch.setattr(xai_tools, "MAX_MCP_ARTIFACT_BYTES", 1)
    too_large = await _call(
        xai, "multiscope_get_explanation_artifact", artifact_id=reference["artifact_id"],
    )
    assert too_large["error"]["code"] == "payload_too_large"
    missing = await _call(
        xai, "multiscope_get_explanation_artifact", artifact_id="../private-artifact",
    )
    assert missing["error"]["code"] == "not_found"


@pytest.mark.anyio
async def test_mcp_retry_creates_new_run_without_changing_classifier(xai):
    completed = await _complete(xai)
    before = json.dumps(xai.predictions.document, default=str, sort_keys=True)
    retry = await _call(xai, "multiscope_retry_explanation")
    assert retry["ok"], retry
    assert retry["data"]["explanation_id"] != completed["data"]["explanation_id"]
    previous = await xai.context.xai_orchestrator.get_for_owner(
        explanation_id=completed["data"]["explanation_id"], owner_user_id="user-123",
    )
    assert previous.model_dump(mode="json") == completed["data"]
    assert json.dumps(xai.predictions.document, default=str, sort_keys=True) == before


@pytest.mark.anyio
async def test_mcp_narrative_retry_does_not_create_new_explanation(xai):
    completed = await _complete(xai)
    # Successful narratives cannot be retried; an invalid retry remains controlled.
    response = await _call(xai, "multiscope_retry_explanation_narrative")
    assert response["ok"] is False
    assert response["error"]["code"] == "conflict"
    latest = await _call(xai, "multiscope_get_explanation")
    assert latest["data"]["explanation_id"] == completed["data"]["explanation_id"]


@pytest.mark.anyio
async def test_mcp_narrative_only_retry_preserves_persisted_evidence(xai):
    orchestrator = xai.context.xai_orchestrator
    orchestrator._narrative_service = FailingNarrativeService()
    completed = await _complete(xai)
    assert completed["data"]["component_statuses"]["narrative"] == "failed"
    orchestrator._narrative_service = StaticNarrativeService()
    response = await _call(xai, "multiscope_retry_explanation_narrative")
    assert response["ok"], response
    await xai.queue.task(xai.queue.bundle)
    retried = await _call(xai, "multiscope_get_explanation")
    assert retried["data"]["explanation_id"] == completed["data"]["explanation_id"]
    assert retried["data"]["component_statuses"]["narrative"] == "completed"
    for field in ("classifier_snapshot", "temporal", "semantic", "combined_report"):
        assert retried["data"][field] == completed["data"][field]


@pytest.mark.anyio
@pytest.mark.parametrize("queue_failure", [False, True])
async def test_default_mcp_prediction_hands_original_evidence_to_xai_after_persistence(
    monkeypatch, tmp_path, queue_failure,
):
    from tests.test_prediction_routes import (
        FakeSubmissionRepository, FakeSubmissionStorage,
        FakeVoiceServiceWithProcessedAudio, upload_metadata,
    )

    predictions = FakeSubmissionRepository()
    api_repository = FakeApiKeyRepository()
    key = api_repository.add_key(owner_user_id="user-123", scopes=["prediction:create"])["full_key"]
    xai_repository = MongoXaiExplanationRepository(FakeDatabase("mcp_auto_xai"))
    voice = FakeVoiceServiceWithProcessedAudio()
    queue = PausedQueue()
    path = tmp_path / "validated.wav"
    path.write_bytes(b"validated")

    def submit(bundle, task):
        assert predictions.documents[0]["status"] == "completed"
        queue.bundle = bundle
        queue.task = task
        if queue_failure:
            raise RuntimeError("XAI queue unavailable")
        return object()

    monkeypatch.setattr(queue, "submit", submit)
    monkeypatch.setattr(tools, "MongoPredictionRepository", lambda: predictions)
    monkeypatch.setattr(tools, "MongoApiKeyRepository", lambda: api_repository)
    monkeypatch.setattr(tools, "MongoXaiExplanationRepository", lambda *args: xai_repository)
    monkeypatch.setattr(tools, "AsynchronousExplanationQueue", lambda **kwargs: queue)
    monkeypatch.setattr(tools, "get_audio_storage", lambda _: FakeSubmissionStorage())
    monkeypatch.setattr(tools, "VoiceService", lambda **kwargs: voice)
    monkeypatch.setattr(tools, "InlinePredictionJobRunner", lambda _: fake_context(api_repository).job_runner)
    monkeypatch.setattr(
        "app.services.prediction_submission_service.save_validated_audio_upload_blocking",
        lambda *args, **kwargs: upload_metadata(path, "sample.wav", "wav"),
    )
    context = tools.create_default_context(Settings(
        _env_file=None, xai_enabled=True, xai_mode="real",
        xai_artifact_root=str(tmp_path / "artifacts"),
    ))
    response = await tools.multiscope_create_prediction(
        api_token=key, filename="sample.wav", content_type="audio/wav",
        small_audio_payload_base64=base64.b64encode(b"test-audio").decode(),
        context=context,
    )
    assert response["ok"] and response["data"]["status"] == "completed", response
    assert voice.call_count == 1
    assert queue.bundle.processed_audio is voice.PROCESSED_AUDIO
    assert queue.bundle.temporal_evidence is voice.TEMPORAL_EVIDENCE
    assert queue.bundle.source_type.value == "mcp"
    assert queue.bundle.owner_user_id == "user-123"
    assert predictions.documents[0]["status"] == "completed"


@pytest.mark.anyio
async def test_mcp_xai_rate_limit_is_per_owner_across_predictions(xai):
    xai.context.app_settings.xai_trigger_rate_limit_requests = 1
    assert (await _call(xai, "multiscope_create_explanation"))["ok"]
    limited = await xai_tools.multiscope_create_explanation(
        api_token=xai.key, prediction_id="another-prediction", context=xai.context,
    )
    assert limited["error"]["code"] == "rate_limited"
    # Polling reads do not consume the trigger budget.
    assert (await _call(xai, "multiscope_get_explanation"))["ok"]


@pytest.mark.anyio
async def test_mcp_xai_disabled_and_unfinished_prediction_are_controlled(xai):
    xai.context.app_settings.xai_enabled = False
    assert (await _call(xai, "multiscope_create_explanation"))["error"]["code"] == "service_unavailable"
    xai.context.app_settings.xai_enabled = True
    xai.predictions.document["status"] = "running"
    assert (await _call(xai, "multiscope_create_explanation"))["error"]["code"] == "conflict"


@pytest.mark.anyio
async def test_sdk_discovers_xai_tools_without_internal_context_and_calls_them(xai):
    from mcp.shared.memory import create_connected_server_and_client_session

    tools.set_shared_context(xai.context)
    try:
        mcp_server = server.create_mcp_server()
        async with create_connected_server_and_client_session(mcp_server._mcp_server) as client:
            discovered = await client.list_tools()
            assert {item.name for item in discovered.tools} == set(server.MCP_TOOL_NAMES)
            assert len(XAI_TOOLS) == 9
            for item in discovered.tools:
                assert "context" not in item.inputSchema.get("properties", {})
                assert "owner_user_id" not in item.inputSchema.get("properties", {})
                assert item.name in MCP_SCOPE_MAP
            result = await client.call_tool("multiscope_create_explanation", {
                "api_token": xai.key, "prediction_id": "prediction-123",
                "context": {"owner_user_id": "attacker"},
            })
            response = json.loads(result.content[0].text)
            assert response["ok"], response
            assert xai.queue.bundle.owner_user_id == "user-123"
            result = await client.call_tool("multiscope_get_explanation", {
                "api_token": xai.key, "prediction_id": "prediction-123",
            })
            assert json.loads(result.content[0].text)["data"]["status"] == "queued"
    finally:
        tools.clear_shared_context()


@pytest.mark.anyio
async def test_real_stdio_transport_keeps_application_logs_out_of_protocol():
    import anyio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    # Stub external resources only; run the actual entrypoint, transport and tools
    # in a separate process so stdout contamination would break the SDK client.
    script = """
import logging
from mcp_server import server, tools
from tests.test_mcp_adapter import FakeApiKeyRepository, fake_context
async def startup():
    tools.set_shared_context(fake_context(FakeApiKeyRepository()))
    logging.getLogger('mcp.test').info('startup log on stderr')
async def shutdown():
    tools.clear_shared_context()
server.startup_mcp_server = startup
server.shutdown_mcp_server = shutdown
server.main()
"""
    parameters = StdioServerParameters(
        command=sys.executable, args=["-u", "-c", script],
        env={**os.environ, "MULTISCOPE_DISABLE_DOTENV": "1"},
    )
    with anyio.fail_after(30):
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=15),
            ) as client:
                await client.initialize()
                discovered = await client.list_tools()
                assert {item.name for item in discovered.tools} == set(server.MCP_TOOL_NAMES)
                response = await client.call_tool("multiscope_get_explanation", {
                    "api_token": "msk_live_testonly", "prediction_id": "prediction-123",
                })
                assert json.loads(response.content[0].text)["error"]["code"] == "authentication_failed"


def test_mcp_entrypoint_imports_in_fresh_process_without_http_app():
    environment = {**os.environ, "MULTISCOPE_DISABLE_DOTENV": "1"}
    result = subprocess.run(
        [sys.executable, "-c", "from mcp_server.server import create_mcp_server; create_mcp_server()"],
        cwd=Path(__file__).resolve().parents[1], env=environment,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stdout


@pytest.mark.anyio
async def test_mcp_transport_and_lifecycle_share_one_event_loop(monkeypatch):
    events = []

    async def startup():
        events.append(("startup", asyncio.get_running_loop()))

    async def serve():
        events.append(("serve", asyncio.get_running_loop()))
        raise RuntimeError("transport ended")

    async def shutdown():
        events.append(("shutdown", asyncio.get_running_loop()))

    monkeypatch.setattr(server, "startup_mcp_server", startup)
    monkeypatch.setattr(server, "shutdown_mcp_server", shutdown)
    with pytest.raises(RuntimeError, match="transport ended"):
        await server.run_mcp_server(SimpleNamespace(run_stdio_async=serve))
    assert [name for name, _ in events] == ["startup", "serve", "shutdown"]
    assert all(loop is events[0][1] for _, loop in events)


@pytest.mark.anyio
@pytest.mark.parametrize("fail_semantic_load", [False, True])
async def test_mcp_xai_startup_loads_real_service_and_cleans_partial_startup(
    xai, monkeypatch, fail_semantic_load,
):
    calls = []
    semantic_service = object()
    lifecycle._started = False
    tools.clear_shared_context()

    async def connect(_settings):
        calls.append("connect")

    async def close():
        calls.append("close")

    async def ready(_settings):
        return {"mongodb_configured": True, "mongodb_available": True}

    async def storage_ready(_settings):
        return {"storage_enabled": False, "storage_available": False}

    async def recover(*args, **kwargs):
        calls.append("recover")

    def load(**kwargs):
        calls.append("load")
        if fail_semantic_load:
            raise ValueError("private-model-path")
        return semantic_service

    async def runner_shutdown():
        await asyncio.sleep(0)
        calls.append("runner_shutdown")

    def queue_close():
        calls.append("queue_close")

    monkeypatch.setattr(lifecycle, "connect_to_mongodb", connect)
    monkeypatch.setattr(lifecycle, "close_mongodb", close)
    monkeypatch.setattr(lifecycle, "mongodb_readiness", ready)
    monkeypatch.setattr(lifecycle, "recover_stale_explanations", recover)
    monkeypatch.setattr(lifecycle, "storage_readiness", storage_ready)
    monkeypatch.setattr(lifecycle.ProductionSemanticExplanationService, "load_once", load)
    monkeypatch.setattr(tools, "create_default_context", lambda _: xai.context)
    monkeypatch.setattr(xai.context.job_runner, "shutdown", runner_shutdown)
    monkeypatch.setattr(xai.queue, "close", queue_close)
    try:
        if fail_semantic_load:
            with pytest.raises(lifecycle.MCPServerLifecycleError) as error:
                await lifecycle.startup_mcp_server(make_mcp_settings(
                    mongodb_uri="mongodb://configured", xai_enabled=True, xai_mode="real",
                ))
            assert "private-model-path" not in str(error.value)
        else:
            await lifecycle.startup_mcp_server(make_mcp_settings(
                mongodb_uri="mongodb://configured", xai_enabled=True, xai_mode="real",
            ))
            assert xai.context.xai_orchestrator._semantic_service is semantic_service
            assert "recover" in calls
            assert xai.context.artifact_cleanup_task is not None
            await lifecycle.shutdown_mcp_server()
        assert calls.index("runner_shutdown") < calls.index("queue_close") < calls.index("close")
        assert tools.get_optional_shared_context() is None
        assert lifecycle._started is False
        assert xai.context.artifact_cleanup_task is None
    finally:
        await lifecycle.shutdown_mcp_server()
