from functools import lru_cache

from fastapi import Depends, Request

from app.auth.clerk_auth import (
    get_optional_principal,
    require_authenticated_principal,
    require_clerk_user,
)
from app.config.settings import Settings, settings
from app.core.rate_limit import InMemoryRateLimiter
from app.repositories.mongodb import MongoApiKeyRepository, MongoPredictionRepository
from app.repositories.protocols import ApiKeyRepository, PredictionRepository
from app.storage.factory import get_audio_storage
from app.storage.protocols import AudioStorage
from app.services.api_key_service import ApiKeyService
from app.services.prediction_history_service import PredictionHistoryService
from app.services.prediction_job_runner import (
    InlinePredictionJobRunner,
    PredictionJobRunner,
)
from app.services.prediction_persistence_service import PredictionPersistenceService
from app.services.prediction_rerun_service import PredictionRerunService
from app.services.prediction_submission_service import PredictionSubmissionService
from app.services.voice_service import VoiceService
from app.voice_xai.jobs.queue import AsynchronousExplanationQueue
from app.voice_xai.artifacts.protocols import ExplanationArtifactStore
from app.voice_xai.artifacts.service import LocalExplanationArtifactStore
from app.voice_xai.orchestrator import VoiceXaiOrchestrator
from app.voice_xai.replay import StoredAudioReplayService
from app.voice_xai.persistence.mongodb import MongoXaiExplanationRepository
from app.voice_xai.persistence.protocols import XaiExplanationRepository


def get_app_settings(request: Request = None) -> Settings:
    if request is not None:
        return getattr(request.app.state, "settings", settings)
    return settings


@lru_cache
def get_voice_service() -> VoiceService:
    return VoiceService(app_settings=settings)


@lru_cache
def get_prediction_job_runner() -> PredictionJobRunner:
    return InlinePredictionJobRunner()


def get_storage_service() -> AudioStorage:
    return get_audio_storage()


def get_prediction_repository() -> PredictionRepository:
    return MongoPredictionRepository()


def get_api_key_repository() -> ApiKeyRepository:
    return MongoApiKeyRepository()


_default_rate_limiter = InMemoryRateLimiter()


def get_rate_limiter(request: Request) -> InMemoryRateLimiter:
    """The process-wide limiter, shared with ``RateLimitMiddleware`` so
    per-user checks (e.g. XAI trigger/retry) and the generic per-path
    middleware count against the same state."""

    limiter = getattr(request.app.state, "rate_limiter", None)
    return limiter if limiter is not None else _default_rate_limiter


def get_xai_repository(request: Request) -> XaiExplanationRepository:
    repository = getattr(request.app.state, "xai_repository", None)
    return repository if repository is not None else MongoXaiExplanationRepository()


def get_xai_artifact_store(request: Request) -> ExplanationArtifactStore:
    store = getattr(request.app.state, "xai_artifact_store", None)
    if store is not None:
        return store
    app_settings = get_app_settings(request)
    return LocalExplanationArtifactStore(
        app_settings.resolve_xai_artifact_path(app_settings.xai_artifact_root),
        retention_seconds=app_settings.xai_artifact_retention_seconds,
    )


def get_xai_orchestrator(request: Request) -> VoiceXaiOrchestrator:
    orchestrator = getattr(request.app.state, "xai_orchestrator", None)
    if orchestrator is not None:
        return orchestrator
    app_settings = get_app_settings(request)
    from app.voice_xai.temporal.xlsr_attention_service import XLSRTemporalAttentionService

    temporal_service = XLSRTemporalAttentionService(app_settings=app_settings)
    return VoiceXaiOrchestrator(
        repository=get_xai_repository(request),
        prediction_repository=get_prediction_repository(),
        queue=AsynchronousExplanationQueue(),
        app_settings=app_settings,
        artifact_store=get_xai_artifact_store(request),
        temporal_service=temporal_service,
        replay_service=StoredAudioReplayService(
            storage=getattr(request.app.state, "storage", None) or get_storage_service(),
            app_settings=app_settings,
        ),
    )


def get_api_key_service(
    repository: ApiKeyRepository = Depends(get_api_key_repository),
) -> ApiKeyService:
    return ApiKeyService(repository)


def get_prediction_persistence_service(
    repository: PredictionRepository = Depends(get_prediction_repository),
) -> PredictionPersistenceService:
    return PredictionPersistenceService(repository)


def get_prediction_history_service(
    repository: PredictionRepository = Depends(get_prediction_repository),
    storage: AudioStorage = Depends(get_storage_service),
    xai_repository: XaiExplanationRepository = Depends(get_xai_repository),
    xai_artifact_store: ExplanationArtifactStore = Depends(get_xai_artifact_store),
) -> PredictionHistoryService:
    return PredictionHistoryService(
        repository=repository,
        storage=storage,
        xai_repository=xai_repository,
        xai_artifact_store=xai_artifact_store,
    )


def get_prediction_submission_service(
    app_settings: Settings = Depends(get_app_settings),
    repository: PredictionRepository = Depends(get_prediction_repository),
    persistence: PredictionPersistenceService = Depends(
        get_prediction_persistence_service
    ),
    storage: AudioStorage = Depends(get_storage_service),
    voice_service: VoiceService = Depends(get_voice_service),
    job_runner: PredictionJobRunner = Depends(get_prediction_job_runner),
    xai_orchestrator: VoiceXaiOrchestrator = Depends(get_xai_orchestrator),
) -> PredictionSubmissionService:
    return PredictionSubmissionService(
        repository=repository,
        persistence=persistence,
        storage=storage,
        voice_service=voice_service,
        job_runner=job_runner,
        xai_orchestrator=xai_orchestrator,
        app_settings=app_settings,
    )


def get_prediction_rerun_service(
    app_settings: Settings = Depends(get_app_settings),
    repository: PredictionRepository = Depends(get_prediction_repository),
    persistence: PredictionPersistenceService = Depends(
        get_prediction_persistence_service
    ),
    storage: AudioStorage = Depends(get_storage_service),
    voice_service: VoiceService = Depends(get_voice_service),
    job_runner: PredictionJobRunner = Depends(get_prediction_job_runner),
    xai_orchestrator: VoiceXaiOrchestrator = Depends(get_xai_orchestrator),
) -> PredictionRerunService:
    return PredictionRerunService(
        repository=repository,
        persistence=persistence,
        storage=storage,
        voice_service=voice_service,
        job_runner=job_runner,
        xai_orchestrator=xai_orchestrator,
        app_settings=app_settings,
    )


async def shutdown_cached_prediction_job_runner(*, wait: bool = True) -> None:
    runner = get_prediction_job_runner()
    shutdown = getattr(runner, "shutdown", None)
    if shutdown is not None:
        result = shutdown(wait=wait)
        if hasattr(result, "__await__"):
            await result
    get_prediction_job_runner.cache_clear()


def reset_dependency_caches() -> None:
    get_voice_service.cache_clear()
    get_prediction_job_runner.cache_clear()


__all__ = [
    "get_app_settings",
    "get_api_key_repository",
    "get_api_key_service",
    "get_optional_principal",
    "get_prediction_history_service",
    "get_prediction_job_runner",
    "get_prediction_persistence_service",
    "get_prediction_repository",
    "get_prediction_rerun_service",
    "get_prediction_submission_service",
    "get_rate_limiter",
    "get_storage_service",
    "get_xai_orchestrator",
    "get_xai_artifact_store",
    "get_xai_repository",
    "get_voice_service",
    "require_authenticated_principal",
    "require_clerk_user",
    "reset_dependency_caches",
    "shutdown_cached_prediction_job_runner",
]
