import asyncio
from collections.abc import AsyncIterator
import contextlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api import dependencies as api_dependencies
from app.auth import clerk_auth
from app.config.settings import Settings, settings
from app.core.access_log import AccessLogMiddleware
from app.core.exception_handlers import register_exception_handlers
from app.core.logging import configure_logging
from app.core.rate_limit import InMemoryRateLimiter, RateLimitMiddleware
from app.core.request_body_limit import RequestBodyLimitMiddleware
from app.core.request_context import RequestIDMiddleware
from app.api.v1.external_routes import router as external_router
from app.api.v1.me_routes import router as me_router
from app.api.v1.prediction_routes import router as prediction_router
from app.api.v1.user_routes import router as user_router
from app.api.v1.voice_routes import router as voice_router
from app.api.v1.xai_routes import router as xai_router
from app.database.mongodb import (
    build_dedicated_async_mongo_client,
    close_mongodb,
    connect_to_mongodb,
    mongodb_readiness,
)
from app.ingestion.audio import audio_tool_status
from app.repositories.mongodb import (
    MongoApiKeyRepository,
    MongoPredictionRepository,
    MongoUserRepository,
)
from app.schemas.common import ReadinessResponse
from app.services.api_key_service import ApiKeyService
from app.services.prediction_history_service import PredictionHistoryService
from app.services.prediction_job_runner import InlinePredictionJobRunner
from app.services.prediction_persistence_service import PredictionPersistenceService
from app.services.prediction_rerun_service import PredictionRerunService
from app.services.prediction_submission_service import PredictionSubmissionService
from app.services.voice_service import VoiceService
from app.storage.factory import get_audio_storage, storage_readiness
from app.voice_xai.jobs.queue import AsynchronousExplanationQueue
from app.voice_xai.artifacts.service import LocalExplanationArtifactStore
from app.voice_xai.orchestrator import VoiceXaiOrchestrator
from app.voice_xai.persistence.mongodb import MongoXaiExplanationRepository
from app.voice_xai.recovery import recover_stale_explanations
from app.voice_xai.replay import StoredAudioReplayService
from app.voice_xai.semantic.service import ProductionSemanticExplanationService
from app.voice_xai.semantic.windows import SemanticWindowConfig
from app.voice_xai.temporal.xlsr_attention_service import XLSRTemporalAttentionService

logger = logging.getLogger(__name__)


API_DESCRIPTION = (
    "Backend API for the MULTI-SCOPE deepfake voice classification research "
    "platform. The service accepts authenticated audio uploads, validates and "
    "preprocesses them with FFmpeg/ffprobe, runs the configured voice model "
    "branches, fuses branch-level scores, persists prediction records, and can "
    "enqueue post-prediction Voice XAI explanations. This is a research "
    "prototype: loaded/ready models are not automatically research-validated "
    "or production-certified, and dummy branches are development placeholders."
)

OPENAPI_TAGS = [
    {
        "name": "Health & Readiness",
        "description": (
            "Service liveness, dependency readiness, model readiness, and "
            "research-readiness signals."
        ),
    },
    {
        "name": "Predictions",
        "description": (
            "Authenticated prediction submission, job status, history, playback, "
            "rerun, and deletion workflows."
        ),
    },
    {
        "name": "Voice Classification",
        "description": (
            "Voice model branch health and the deprecated local compatibility "
            "prediction route."
        ),
    },
    {
        "name": "Explainability / XAI",
        "description": (
            "Owner-scoped post-prediction explanation runs, temporal evidence, "
            "semantic evidence, reports, retries, and private artifacts."
        ),
    },
    {
        "name": "Users",
        "description": "Authenticated user and principal metadata endpoints.",
    },
    {
        "name": "API Keys / External",
        "description": (
            "API-key management and third-party prediction submission using "
            "Authorization bearer API keys."
        ),
    },
]


@dataclass(slots=True)
class LifespanDependencies:
    connect_mongodb: Callable[[Settings], Awaitable[None]] = connect_to_mongodb
    close_mongodb: Callable[[], Awaitable[None]] = close_mongodb
    mongodb_readiness: Callable[[Settings], Awaitable[dict[str, Any]]] = (
        mongodb_readiness
    )
    storage_readiness: Callable[[Settings], Awaitable[dict[str, Any]]] = (
        storage_readiness
    )


@dataclass(slots=True)
class ApplicationDependencies:
    prediction_repository: Any | None = None
    api_key_repository: Any | None = None
    user_repository: Any | None = None
    storage: Any | None = None
    voice_service: VoiceService | None = None
    job_runner: Any | None = None
    rate_limiter: InMemoryRateLimiter | None = None
    xai_repository: Any | None = None
    xai_artifact_store: Any | None = None
    xai_queue: Any | None = None
    xai_orchestrator: Any | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app_settings: Settings = app.state.settings
    lifecycle: LifespanDependencies = app.state.lifespan_dependencies
    tool_status = audio_tool_status()
    logger.info(
        "Audio processing readiness checked: ffmpeg_available=%s, "
        "ffprobe_available=%s.",
        tool_status["ffmpeg_available"],
        tool_status["ffprobe_available"],
    )
    if clerk_auth.dev_auth_bypass_enabled(app_settings):
        logger.warning(
            "DEVELOPMENT AUTH BYPASS IS ENABLED. Authentication is disabled "
            "for local development only. Never enable DEV_AUTH_BYPASS in "
            "production, staging, test, or unknown environments."
        )
    await lifecycle.connect_mongodb(app_settings)
    voice_service = getattr(app.state, "voice_service", None)
    if voice_service is not None:
        load_startup_models = getattr(voice_service, "load_startup_models", None)
        if callable(load_startup_models):
            load_startup_models()
    cleanup_task: asyncio.Task | None = None
    try:
        if app_settings.xai_enabled:
            await _recover_stale_xai_runs(app, app_settings)
            cleanup_task = asyncio.create_task(
                _artifact_cleanup_loop(app, app_settings),
                name="voice-xai-artifact-cleanup",
            )
        if app_settings.xai_enabled and app_settings.xai_mode == "real":
            semantic_service = ProductionSemanticExplanationService.load_once(
                manifest_path=app_settings.resolve_xai_artifact_path(
                    app_settings.xai_semantic_manifest_path
                ),
                model_path=app_settings.resolve_xai_artifact_path(
                    app_settings.xai_semantic_model_path
                ),
                feature_columns_path=app_settings.resolve_xai_artifact_path(
                    app_settings.xai_semantic_feature_columns_path
                ),
                threshold_path=app_settings.resolve_xai_artifact_path(
                    app_settings.xai_semantic_threshold_path
                ),
                training_metadata_path=app_settings.resolve_xai_artifact_path(
                    app_settings.xai_semantic_training_metadata_path
                ),
                imputer_path=app_settings.resolve_xai_artifact_path(
                    app_settings.xai_semantic_imputer_path
                ),
                shap_background_path=app_settings.resolve_xai_artifact_path(
                    app_settings.xai_semantic_shap_background_path
                ) if app_settings.xai_semantic_shap_background_path.strip() else None,
                window_config=SemanticWindowConfig(
                    duration_seconds=app_settings.xai_semantic_window_duration_seconds,
                    overlap_seconds=app_settings.xai_semantic_window_overlap_seconds,
                ),
            )
            app.state.xai_orchestrator.set_semantic_service(semantic_service)
        yield
    finally:
        if cleanup_task is not None:
            cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task
        voice_service = getattr(app.state, "voice_service", None)
        if voice_service is not None:
            unload_models = getattr(voice_service, "unload_models", None)
            if callable(unload_models):
                unload_models()
        runner = getattr(app.state, "prediction_job_runner", None)
        if runner is not None:
            shutdown = getattr(runner, "shutdown", None)
            if callable(shutdown):
                shutdown_result = shutdown(wait=True)
                if hasattr(shutdown_result, "__await__"):
                    await shutdown_result
        xai_queue = getattr(app.state, "xai_queue", None)
        if xai_queue is not None:
            close = getattr(xai_queue, "close", None)
            if callable(close):
                close(wait=True)
        await lifecycle.close_mongodb()
        api_dependencies.reset_dependency_caches()
        clerk_auth.reset_jwks_cache()


def create_app(
    app_settings: Settings | None = None,
    *,
    lifespan_dependencies: LifespanDependencies | None = None,
    dependencies: ApplicationDependencies | None = None,
) -> FastAPI:
    app_settings = app_settings or settings
    lifecycle_dependencies = lifespan_dependencies or LifespanDependencies()
    app_dependencies = dependencies or ApplicationDependencies()

    configure_logging()
    app = FastAPI(
        title="MULTI-SCOPE Deepfake Detection API",
        description=API_DESCRIPTION,
        version=app_settings.app_version,
        openapi_tags=OPENAPI_TAGS,
        lifespan=lifespan,
        docs_url="/docs" if app_settings.docs_enabled else None,
        redoc_url="/redoc" if app_settings.docs_enabled else None,
        openapi_url="/openapi.json" if app_settings.openapi_enabled else None,
    )
    app.state.settings = app_settings
    app.state.lifespan_dependencies = lifecycle_dependencies
    app.state.prediction_repository = (
        app_dependencies.prediction_repository or MongoPredictionRepository()
    )
    app.state.api_key_repository = (
        app_dependencies.api_key_repository or MongoApiKeyRepository()
    )
    app.state.user_repository = (
        app_dependencies.user_repository or MongoUserRepository()
    )
    app.state.storage = app_dependencies.storage or get_audio_storage(app_settings)
    app.state.voice_service = app_dependencies.voice_service or VoiceService(
        app_settings=app_settings
    )
    app.state.prediction_job_runner = (
        app_dependencies.job_runner or InlinePredictionJobRunner(app_settings)
    )
    app.state.rate_limiter = app_dependencies.rate_limiter or InMemoryRateLimiter()
    app.state.xai_repository = (
        app_dependencies.xai_repository or MongoXaiExplanationRepository()
    )
    app.state.xai_artifact_store = (
        app_dependencies.xai_artifact_store
        or LocalExplanationArtifactStore(
            app_settings.resolve_xai_artifact_path(app_settings.xai_artifact_root),
            retention_seconds=app_settings.xai_artifact_retention_seconds,
        )
    )
    app.state.xai_queue = app_dependencies.xai_queue or AsynchronousExplanationQueue(
        max_workers=app_settings.xai_max_workers,
        max_queued_jobs=app_settings.xai_max_queued_jobs,
        # Built lazily, from inside the queue's own dedicated worker thread,
        # the first time a job is actually submitted -- never shares the
        # main application's AsyncMongoClient (which is bound to the main
        # event loop) with jobs running on the queue's persistent worker
        # loop. See SEC-1 in the 2026-08-08 reliability fixes.
        mongodb_client_factory=lambda: build_dedicated_async_mongo_client(app_settings),
    )
    temporal_service = XLSRTemporalAttentionService(app_settings=app_settings)
    app.state.xai_orchestrator = app_dependencies.xai_orchestrator or VoiceXaiOrchestrator(
        repository=app.state.xai_repository,
        prediction_repository=app.state.prediction_repository,
        queue=app.state.xai_queue,
        app_settings=app_settings,
        artifact_store=app.state.xai_artifact_store,
        temporal_service=temporal_service,
        replay_service=StoredAudioReplayService(
            storage=app.state.storage,
            app_settings=app_settings,
        ),
        # Resolved once per background run, after the queue's dedicated
        # worker loop/client (above) is guaranteed to exist. Falls back to
        # the main-loop repository automatically if MongoDB is unconfigured.
        worker_repository_factory=lambda: MongoXaiExplanationRepository(
            app.state.xai_queue.worker_database
        ),
    )
    reconciliation_setter = getattr(
        app.state.storage,
        "set_reconciliation_recorder",
        None,
    )
    if callable(reconciliation_setter):
        reconciliation_setter(
            PredictionPersistenceService(
                app.state.prediction_repository
            ).record_storage_reconciliation_event
        )

    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(AccessLogMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        app_settings=app_settings,
        limiter=app.state.rate_limiter,
    )
    app.add_middleware(RequestBodyLimitMiddleware, app_settings=app_settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.allowed_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if app_settings.trusted_host_list:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=app_settings.trusted_host_list,
        )

    _register_dependency_overrides(app)
    register_exception_handlers(app)
    app.include_router(external_router, prefix=app_settings.api_v1_prefix)
    app.include_router(me_router, prefix=app_settings.api_v1_prefix)
    app.include_router(prediction_router, prefix=app_settings.api_v1_prefix)
    app.include_router(user_router, prefix=app_settings.api_v1_prefix)
    app.include_router(voice_router, prefix=app_settings.api_v1_prefix)
    app.include_router(xai_router, prefix=app_settings.api_v1_prefix)
    _install_openapi_documentation(app, app_settings)

    @app.get(
        "/",
        tags=["Health & Readiness"],
        summary="Get service metadata",
        description="Returns the API name, version, and a lightweight running status.",
    )
    def root() -> dict[str, str]:
        return {
            "name": app_settings.app_name,
            "version": app_settings.app_version,
            "status": "running",
        }

    @app.get(
        "/health",
        tags=["Health & Readiness"],
        summary="Check service liveness",
        description=(
            "Lightweight liveness probe. A healthy response means the HTTP "
            "process is alive; it does not prove MongoDB, storage, model, or "
            "research readiness."
        ),
    )
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    async def readiness_payload(response: Response) -> ReadinessResponse:
        current_tool_status = audio_tool_status()
        current_mongodb_status = await lifecycle_dependencies.mongodb_readiness(
            app_settings
        )
        current_storage_status = await lifecycle_dependencies.storage_readiness(
            app_settings
        )
        runner_status = _runner_readiness(app.state.prediction_job_runner)
        model_status = _model_readiness(app.state.voice_service, app_settings)
        xai_status = app.state.xai_orchestrator.readiness()
        storage_ready = _storage_policy_ready(
            current_storage_status,
            app_settings,
        )
        prediction_ready = all(current_tool_status.values()) and (
            current_mongodb_status["mongodb_configured"]
            and current_mongodb_status["mongodb_available"]
            and storage_ready
            and runner_status["ready"]
            and model_status["ready"]
        )
        if not prediction_ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="ready" if prediction_ready else "not_ready",
            prediction_ready=prediction_ready,
            research_ready=model_status["research_ready"] and prediction_ready,
            **current_tool_status,
            **current_mongodb_status,
            **current_storage_status,
            components={
                "audio_tools": {
                    "ready": all(current_tool_status.values()),
                    **current_tool_status,
                },
                "mongodb": {
                    "ready": current_mongodb_status["mongodb_configured"]
                    and current_mongodb_status["mongodb_available"],
                    **current_mongodb_status,
                },
                "storage": {
                    "ready": storage_ready,
                    "policy": app_settings.storage_policy,
                    **current_storage_status,
                },
                "prediction_runner": runner_status,
                "models": model_status,
                "voice_xai": xai_status,
            },
        )

    @app.get(
        "/ready",
        tags=["Health & Readiness"],
        response_model=ReadinessResponse,
        summary="Check dependency and prediction readiness",
        description=(
            "Readiness probe for real prediction work. It combines audio-tool, "
            "MongoDB, storage, prediction-runner, model, and Voice XAI component "
            "signals. `research_ready` is stricter than runtime readiness and "
            "does not become true merely because a checkpoint is loaded."
        ),
        responses={
            503: {
                "model": ReadinessResponse,
                "description": "One or more required backend dependencies are unavailable.",
            }
        },
    )
    async def ready(response: Response) -> ReadinessResponse:
        return await readiness_payload(response)

    @app.get(
        "/readiness",
        tags=["Health & Readiness"],
        response_model=ReadinessResponse,
        include_in_schema=False,
    )
    async def readiness(response: Response) -> ReadinessResponse:
        return await readiness_payload(response)

    return app


def _install_openapi_documentation(app: FastAPI, app_settings: Settings) -> None:
    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=OPENAPI_TAGS,
        )
        components = schema.setdefault("components", {})
        security_schemes = components.setdefault("securitySchemes", {})
        security_schemes.update(
            {
                "ClerkBearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT",
                    "description": (
                        "Paste a Clerk session JWT in Swagger's Authorize "
                        "dialog. Send it as `Authorization: Bearer <token>`."
                    ),
                },
                "ApiKeyBearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "msk_live API key",
                    "description": (
                        "Third-party integrations use the existing "
                        "`Authorization: Bearer msk_live_...` header. Do not "
                        "use an X-API-Key header."
                    ),
                },
            }
        )
        _apply_operation_security(schema)
        _add_openapi_examples(schema, app_settings)
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi


def _apply_operation_security(schema: dict[str, Any]) -> None:
    paths = schema.get("paths", {})
    clerk_paths = {
        "/api/v1/predictions",
        "/api/v1/predictions/{prediction_id}/status",
        "/api/v1/voice/predict",
        "/api/v1/users/me",
    }
    for path, path_item in paths.items():
        if path in clerk_paths or path.startswith("/api/v1/me/"):
            for operation in path_item.values():
                if isinstance(operation, dict):
                    operation["security"] = [{"ClerkBearerAuth": []}]
        if path == "/api/v1/external/predictions":
            for operation in path_item.values():
                if isinstance(operation, dict):
                    operation["security"] = [{"ApiKeyBearerAuth": []}]


def _add_openapi_examples(schema: dict[str, Any], app_settings: Settings) -> None:
    prediction = schema.get("paths", {}).get("/api/v1/predictions", {}).get("post")
    if not isinstance(prediction, dict):
        return
    prediction.setdefault(
        "description",
        "Submit audio for deepfake voice classification.",
    )
    prediction["summary"] = "Submit audio for deepfake voice classification"
    responses = prediction.setdefault("responses", {})
    responses.setdefault("200", {}).setdefault("content", {}).setdefault(
        "application/json",
        {},
    )["examples"] = {
        "completedPrediction": {
            "summary": "Completed prediction",
            "value": {
                "prediction_id": "4e2fd0a3-7f5b-41d9-98d5-f8f26f0f1e3c",
                "request_id": "b34b2e2f-8fd2-4781-87cf-2c96e01fc2f5",
                "status": "completed",
                "source_type": "dashboard_upload",
                "audio": {
                    "original_filename": "sample-voice.wav",
                    "original_extension": "wav",
                    "detected_container": "wav",
                    "detected_codec": "pcm_s16le",
                    "duration_seconds": 3.0,
                    "sample_rate": 16000,
                    "channels": 1,
                    "size_bytes": 96044,
                    "storage_status": "success",
                    "playback_available": True,
                },
                "branches": [
                    {
                        "model_name": "aasist",
                        "display_name": "AASIST",
                        "status": "success",
                        "mode": "real",
                        "prediction": "spoof",
                        "confidence": 0.91,
                        "probabilities": {"bonafide": 0.09, "spoof": 0.91},
                        "processing_time_ms": 84.2,
                        "error": None,
                        "metadata": {"research_result": False},
                    }
                ],
                "fusion": {
                    "status": "success",
                    "prediction": "spoof",
                    "confidence": 0.91,
                    "probabilities": {"bonafide": 0.09, "spoof": 0.91},
                    "method": "simple_average",
                    "decision_threshold": 0.5519237850482265,
                    "branch_weights": {"aasist": 1.0},
                    "contains_dummy_branches": False,
                    "eligible_for_research_evaluation": False,
                    "warning": None,
                    "config_version": "fusion-config-v1",
                    "minimum_successful_branches": 1,
                    "contributing_branches": ["aasist"],
                    "excluded_branches": {},
                },
                "research_eligible": False,
                "created_at": "2026-08-19T00:00:00Z",
            },
        }
    }
    for status_code, code, message in [
        ("401", "authentication_failed", "Authentication failed."),
        (
            "409",
            "idempotency_conflict",
            "Idempotency key was already used for a different request. Use a "
            "new unique idempotency_key for a new submission, or resend the "
            "exact original request to replay its result.",
        ),
        ("415", "unsupported_audio_format", "Unsupported audio format."),
        ("503", "model_unavailable", "No usable model branches are available."),
    ]:
        responses.setdefault(status_code, {}).setdefault("content", {}).setdefault(
            "application/json",
            {},
        )["examples"] = {
            code: {
                "summary": message,
                "value": {
                    "request_id": "b34b2e2f-8fd2-4781-87cf-2c96e01fc2f5",
                    "error": {
                        "code": code,
                        "message": message,
                        "details": None,
                    },
                },
            }
        }
    request_body = prediction.get("requestBody", {})
    content = request_body.get("content", {}).get("multipart/form-data", {})
    content["examples"] = {
        "humanVoice": {
            "summary": "Known human/bonafide sample",
            "description": "Choose a local file such as human_voice.wav.",
        },
        "syntheticVoice": {
            "summary": "Known synthetic/spoof sample",
            "description": "Choose a local file such as synthetic_voice.wav.",
        },
        "repoSample": {
            "summary": "Repository smoke-test sample",
            "description": "Choose frontend/e2e/fixtures/sample-voice.wav.",
        },
    }
    prediction["description"] = (
        "Submit an authenticated audio file for deepfake voice classification.\n\n"
        "Flow: upload -> validation -> audio preprocessing -> configured model "
        "branches -> score-level fusion -> persistence -> optional Voice XAI "
        "handoff.\n\n"
        "Accepted formats: WAV, FLAC, MP3, M4A, AAC, Opus, OGG, and audio-only "
        f"WebM. Maximum upload size: {app_settings.max_upload_size_mb} MB. "
        f"Maximum audio duration: {app_settings.max_audio_duration_seconds} "
        "seconds. Use Swagger's Authorize button with a Clerk bearer JWT before "
        "trying this endpoint. Common failure modes include authentication "
        "failure, unsupported or corrupted audio, near-silent audio, oversized "
        "uploads, unavailable storage when required, and unavailable model "
        "branches.\n\n"
        "Idempotency: the `idempotency_key` form field is optional and "
        "owner-scoped. Leave it empty to submit a normal new request. Send a "
        "new unique value (for example a UUID4) for every new audio "
        "submission, and reuse the same value only when retrying the exact "
        "same logical request, which replays the stored prediction instead of "
        "running inference again. Reusing one key for a different logical "
        "request returns HTTP 409 `idempotency_conflict`. Swagger never "
        "submits a default idempotency key; any key shown in this "
        "documentation is an example only."
    )


def _register_dependency_overrides(app: FastAPI) -> None:
    app.dependency_overrides[api_dependencies.get_app_settings] = (
        lambda: app.state.settings
    )
    app.dependency_overrides[api_dependencies.get_voice_service] = (
        lambda: app.state.voice_service
    )
    app.dependency_overrides[api_dependencies.get_prediction_job_runner] = (
        lambda: app.state.prediction_job_runner
    )
    app.dependency_overrides[api_dependencies.get_storage_service] = (
        lambda: app.state.storage
    )
    app.dependency_overrides[api_dependencies.get_prediction_repository] = (
        lambda: app.state.prediction_repository
    )
    app.dependency_overrides[api_dependencies.get_api_key_repository] = (
        lambda: app.state.api_key_repository
    )
    app.dependency_overrides[clerk_auth.get_user_repository] = (
        lambda: app.state.user_repository
    )
    app.dependency_overrides[api_dependencies.get_prediction_persistence_service] = (
        lambda: PredictionPersistenceService(app.state.prediction_repository)
    )
    app.dependency_overrides[api_dependencies.get_prediction_history_service] = (
        lambda: PredictionHistoryService(
            repository=app.state.prediction_repository,
            storage=app.state.storage,
            xai_repository=app.state.xai_repository,
            xai_artifact_store=app.state.xai_artifact_store,
        )
    )
    app.dependency_overrides[api_dependencies.get_prediction_submission_service] = (
        lambda: PredictionSubmissionService(
            repository=app.state.prediction_repository,
            persistence=PredictionPersistenceService(app.state.prediction_repository),
            storage=app.state.storage,
            voice_service=app.state.voice_service,
            job_runner=app.state.prediction_job_runner,
            xai_orchestrator=app.state.xai_orchestrator,
            app_settings=app.state.settings,
        )
    )
    app.dependency_overrides[api_dependencies.get_prediction_rerun_service] = (
        lambda: PredictionRerunService(
            repository=app.state.prediction_repository,
            persistence=PredictionPersistenceService(app.state.prediction_repository),
            storage=app.state.storage,
            voice_service=app.state.voice_service,
            job_runner=app.state.prediction_job_runner,
            xai_orchestrator=app.state.xai_orchestrator,
            app_settings=app.state.settings,
        )
    )
    app.dependency_overrides[api_dependencies.get_api_key_service] = (
        lambda: ApiKeyService(app.state.api_key_repository)
    )


async def _recover_stale_xai_runs(app: FastAPI, app_settings: Settings) -> None:
    """Best-effort startup reconciliation for runs stranded by a restart.

    Must never block or fail application startup -- XAI reliability work is
    itself subject to the same "XAI failure != prediction failure" rule.
    """

    repository = getattr(app.state, "xai_repository", None)
    if repository is None:
        return
    try:
        result = await recover_stale_explanations(
            repository,
            stale_after_seconds=app_settings.xai_stale_run_recovery_threshold_seconds,
        )
        if result.scanned:
            logger.warning(
                "Voice XAI stale-run recovery finished.",
                extra={
                    "scanned": result.scanned,
                    "recovered": result.recovered,
                    "skipped": result.skipped,
                },
            )
    except Exception:
        logger.exception("Voice XAI stale-run recovery failed at startup.")


async def _artifact_cleanup_loop(app: FastAPI, app_settings: Settings) -> None:
    """Periodically delete artifact files whose retention window has passed.

    Runs for the life of the process; each iteration is independently
    guarded so one failed pass (e.g. a transient filesystem error) does not
    kill the background task permanently.
    """

    store = getattr(app.state, "xai_artifact_store", None)
    cleanup = getattr(store, "cleanup_expired", None)
    if not callable(cleanup):
        return
    loop = asyncio.get_running_loop()
    try:
        while True:
            await asyncio.sleep(app_settings.xai_artifact_cleanup_interval_seconds)
            try:
                await loop.run_in_executor(None, cleanup)
            except Exception:
                logger.exception("Voice XAI artifact cleanup pass failed.")
    except asyncio.CancelledError:
        raise


def _runner_readiness(runner: Any) -> dict[str, Any]:
    health = getattr(runner, "health", None)
    if callable(health):
        return health()
    return {
        "ready": True,
        "accepting_jobs": True,
        "shutting_down": False,
        "active_jobs": 0,
        "max_concurrent_jobs": None,
        "available_capacity": None,
        "executor_healthy": True,
    }


def _model_readiness(voice_service: VoiceService, app_settings: Settings) -> dict[str, Any]:
    model_readiness = getattr(voice_service, "model_readiness", None)
    if callable(model_readiness):
        return model_readiness()
    branches = voice_service.model_health()
    dummy_modes = [branch for branch in branches if branch.get("mode") == "dummy"]
    production_like = app_settings.app_env.lower() in {"production", "research"}
    ready = bool(branches) and all(
        branch.get("mode") == "dummy" or branch.get("is_loaded")
        for branch in branches
    )
    if production_like and dummy_modes:
        ready = False
    return {
        "ready": ready,
        "research_ready": ready and not dummy_modes,
        "dummy_mode_allowed": app_settings.app_env.lower()
        not in {"production", "research"},
        "branches": branches,
    }


def _storage_policy_ready(
    storage_status: dict[str, Any],
    app_settings: Settings,
) -> bool:
    if app_settings.storage_policy == "required":
        return bool(storage_status.get("storage_available"))
    return app_settings.app_env.lower() in {"development", "local", "test", "testing"}


app = create_app()
