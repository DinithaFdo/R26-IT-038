from datetime import UTC, datetime
import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from app.api.dependencies import get_xai_orchestrator
from app.auth.clerk_auth import require_clerk_user
from app.auth.schemas import AuthPrincipal
from app.config.settings import Settings
from app.main import ApplicationDependencies, LifespanDependencies, create_app
from app.schemas.common import BranchStatus, ModelMode, PredictionLabel, SourceType
from app.schemas.prediction import (
    AudioMetadata,
    BranchPrediction,
    FusionResult,
    ProbabilityScores,
    VoicePredictionResponse,
)
from app.voice_xai.orchestrator import VoiceXaiOrchestrator
from app.voice_xai.persistence.mongodb import MongoXaiExplanationRepository
from app.voice_xai.status import XaiStatusService
from app.schemas.xai import (
    ExplanationArtifactReference,
    ExplanationError,
    NarrativeExplanation,
)
from app.voice_xai.artifacts.service import LocalExplanationArtifactStore
from app.voice_xai.artifacts.service import ArtifactStorageError
from tests.test_mongodb import FakeDatabase


class PausedQueue:
    def submit(self, bundle, task):
        self.bundle = bundle
        self.task = task
        return object()

    def close(self, *, wait=True):
        return None


class PredictionRepository:
    def __init__(self, document: dict) -> None:
        self.document = document

    async def get_prediction_for_owner(
        self, *, prediction_id: str, owner_user_id: str, include_deleted: bool = False
    ) -> dict | None:
        if (
            prediction_id == self.document["id"]
            and owner_user_id == self.document["owner_user_id"]
        ):
            return self.document
        return None


class MultiPredictionRepository:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents

    async def get_prediction_for_owner(
        self, *, prediction_id: str, owner_user_id: str, include_deleted: bool = False
    ) -> dict | None:
        for document in self.documents:
            if (
                document["id"] == prediction_id
                and document["owner_user_id"] == owner_user_id
            ):
                return document
        return None


class ReplayService:
    async def load_processed_audio(self, *, prediction_document, owner_user_id):
        return None


class FailingArtifactStore:
    def read(self, _reference):
        raise ArtifactStorageError("internal artifact integrity details")


class StaticNarrativeService:
    async def generate(self, **_kwargs) -> NarrativeExplanation:
        return NarrativeExplanation(
            provider="alibaba_model_studio",
            model_id="qwen-test-snapshot",
            prompt_version="voice-xai-narrative-v1",
            input_sha256="0" * 64,
            summary="Validated evidence requires qualified human review.",
            detailed_explanation="The deterministic report remains authoritative.",
            evidence_references=["classifier:verdict"],
        )


def test_owner_scoped_xai_routes_hide_another_users_prediction() -> None:
    prediction_document = _prediction_document()
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_routes"))
    queue = PausedQueue()
    settings = Settings(_env_file=None, xai_enabled=True, xai_mode="mock")
    orchestrator = VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=PredictionRepository(prediction_document),
        queue=queue,
        app_settings=settings,
        temporal_service=object(),
        replay_service=ReplayService(),
        artifact_store=FailingArtifactStore(),
    )

    async def no_connect(_settings):
        return None

    async def no_close():
        return None

    app = create_app(
        settings,
        lifespan_dependencies=LifespanDependencies(
            connect_mongodb=no_connect,
            close_mongodb=no_close,
        ),
        dependencies=ApplicationDependencies(xai_orchestrator=orchestrator),
    )
    current_user = {"id": "user-123"}
    app.dependency_overrides[require_clerk_user] = lambda: AuthPrincipal(
        subject=f"user:{current_user['id']}",
        principal_type="clerk_user",
        user_id=current_user["id"],
    )
    app.dependency_overrides[get_xai_orchestrator] = lambda: orchestrator
    try:
        with TestClient(app) as client:
            created = client.post("/api/v1/me/predictions/prediction-123/explanation")
            repeated = client.post("/api/v1/me/predictions/prediction-123/explanation")
            assert created.status_code == 202
            assert repeated.status_code == 202
            assert repeated.json()["explanation_id"] == created.json()["explanation_id"]
            assert created.json()["development_placeholder"] is True
            assert created.json()["research_eligible"] is False

            pending_temporal = client.get(
                "/api/v1/me/predictions/prediction-123/explanation/temporal"
            )
            assert pending_temporal.status_code == 409

            pending_semantic = client.get(
                "/api/v1/me/predictions/prediction-123/explanation/semantic"
            )
            assert pending_semantic.status_code == 409
            assert "still being generated" in pending_semantic.json()["error"]["message"]

            explanation_id = created.json()["explanation_id"]
            asyncio.run(
                repository.append_artifact(
                    explanation_id,
                    ExplanationArtifactReference(
                        artifact_id="artifact-123",
                        kind="attention_visualization",
                        content_type="application/json",
                        size_bytes=1,
                        sha256="0" * 64,
                    ),
                )
            )
            unavailable_artifact = client.get(
                "/api/v1/me/predictions/prediction-123/explanation/artifacts/artifact-123"
            )
            assert unavailable_artifact.status_code == 409
            assert unavailable_artifact.json()["error"]["message"] == (
                "The explanation artifact is unavailable."
            )
            awaitable = XaiStatusService(repository).fail_run(
                explanation_id=explanation_id,
                owner_user_id="user-123",
                error=ExplanationError(
                    component=None,
                    code="xai_run_interrupted",
                    message="The explanation run was interrupted by a service restart.",
                ),
            )
            asyncio.run(awaitable)
            failed_temporal = client.get(
                "/api/v1/me/predictions/prediction-123/explanation/temporal"
            )
            assert failed_temporal.status_code == 409
            assert "interrupted" in failed_temporal.json()["error"]["message"]

            failed_semantic = client.get(
                "/api/v1/me/predictions/prediction-123/explanation/semantic"
            )
            assert failed_semantic.status_code == 409
            assert "interrupted" in failed_semantic.json()["error"]["message"]

            failed_trigger = client.post(
                "/api/v1/me/predictions/prediction-123/explanation"
            )
            assert failed_trigger.status_code == 409
            assert "retry endpoint" in failed_trigger.json()["error"]["message"]

            retried = client.post(
                "/api/v1/me/predictions/prediction-123/explanation/retry"
            )
            assert retried.status_code == 202
            assert retried.json()["explanation_id"] != explanation_id

            current_user["id"] = "another-user"
            hidden = client.get("/api/v1/me/predictions/prediction-123/explanation")
            assert hidden.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_narrative_route_is_owner_scoped_and_preserves_normal_report(tmp_path: Path) -> None:
    prediction_document = _prediction_document()
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_narrative_routes"))
    queue = PausedQueue()
    settings = Settings(
        _env_file=None,
        xai_enabled=True,
        xai_mode="mock",
        xai_narrative_enabled=True,
        xai_narrative_api_key="test-key",
        xai_narrative_base_url=(
            "https://workspace.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
        ),
    )
    orchestrator = VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=PredictionRepository(prediction_document),
        queue=queue,
        app_settings=settings,
        temporal_service=object(),
        replay_service=ReplayService(),
        artifact_store=LocalExplanationArtifactStore(tmp_path / "private"),
        narrative_service=StaticNarrativeService(),
    )

    async def no_connect(_settings):
        return None

    async def no_close():
        return None

    app = create_app(
        settings,
        lifespan_dependencies=LifespanDependencies(
            connect_mongodb=no_connect,
            close_mongodb=no_close,
        ),
        dependencies=ApplicationDependencies(xai_orchestrator=orchestrator),
    )
    current_user = {"id": "user-123"}
    app.dependency_overrides[require_clerk_user] = lambda: AuthPrincipal(
        subject=f"user:{current_user['id']}",
        principal_type="clerk_user",
        user_id=current_user["id"],
    )
    app.dependency_overrides[get_xai_orchestrator] = lambda: orchestrator
    try:
        with TestClient(app) as client:
            created = client.post("/api/v1/me/predictions/prediction-123/explanation")
            assert created.status_code == 202
            asyncio.run(orchestrator._run(created.json()["explanation_id"], queue.bundle))

            deterministic = client.get(
                "/api/v1/me/predictions/prediction-123/explanation/report"
            )
            narrative = client.get(
                "/api/v1/me/predictions/prediction-123/explanation/narrative"
            )
            assert deterministic.status_code == 200
            assert deterministic.json()["finding"]
            assert narrative.status_code == 200
            assert narrative.json()["model_id"] == "qwen-test-snapshot"

            current_user["id"] = "another-user"
            hidden = client.get(
                "/api/v1/me/predictions/prediction-123/explanation/narrative"
            )
            assert hidden.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_xai_trigger_rate_limit_is_per_user_not_per_prediction_id() -> None:
    """SEC-3 regression test.

    The generic RateLimitMiddleware keys by literal request path (which
    includes prediction_id), so triggering explanations for many different
    predictions previously was not bounded per user at all. This proves the
    dedicated per-user limiter now catches that, while a different
    authenticated user remains independently limited.
    """

    documents = [
        _prediction_document(prediction_id=f"prediction-{index}", owner_user_id="user-123")
        for index in range(1, 4)
    ] + [_prediction_document(prediction_id="prediction-other-user", owner_user_id="another-user")]
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_trigger_rate_limit"))
    queue = PausedQueue()
    settings = Settings(
        _env_file=None,
        xai_enabled=True,
        xai_mode="mock",
        xai_trigger_rate_limit_requests=2,
        xai_trigger_rate_limit_window_seconds=60,
    )
    orchestrator = VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=MultiPredictionRepository(documents),
        queue=queue,
        app_settings=settings,
        temporal_service=object(),
        replay_service=ReplayService(),
    )

    async def no_connect(_settings):
        return None

    async def no_close():
        return None

    app = create_app(
        settings,
        lifespan_dependencies=LifespanDependencies(
            connect_mongodb=no_connect,
            close_mongodb=no_close,
        ),
        dependencies=ApplicationDependencies(xai_orchestrator=orchestrator),
    )
    current_user = {"id": "user-123"}
    app.dependency_overrides[require_clerk_user] = lambda: AuthPrincipal(
        subject=f"user:{current_user['id']}",
        principal_type="clerk_user",
        user_id=current_user["id"],
    )
    app.dependency_overrides[get_xai_orchestrator] = lambda: orchestrator
    try:
        with TestClient(app) as client:
            first = client.post("/api/v1/me/predictions/prediction-1/explanation")
            second = client.post("/api/v1/me/predictions/prediction-2/explanation")
            # Same user, a third distinct prediction_id: still counts toward
            # the same per-user bucket and must now be refused.
            third = client.post("/api/v1/me/predictions/prediction-3/explanation")

            assert first.status_code == 202
            assert second.status_code == 202
            assert third.status_code == 429
            assert "Retry-After" in third.headers

            # A different authenticated user has their own independent budget.
            current_user["id"] = "another-user"
            other_user_first = client.post(
                "/api/v1/me/predictions/prediction-other-user/explanation"
            )
            assert other_user_first.status_code == 202
    finally:
        app.dependency_overrides.clear()


def _prediction_document(
    *, prediction_id: str = "prediction-123", owner_user_id: str = "user-123"
) -> dict:
    now = datetime.now(UTC)
    prediction = VoicePredictionResponse(
        request_id=f"request-for-{prediction_id}",
        audio=AudioMetadata(
            original_filename="sample.wav",
            content_type="audio/wav",
            file_size_bytes=100,
            duration_seconds=2.0,
            sample_rate=16000,
            channels=1,
        ),
        branches=[
            BranchPrediction(
                model_name="cnn_acoustic",
                display_name="CNN Acoustic",
                status=BranchStatus.success,
                mode=ModelMode.dummy,
                prediction=PredictionLabel.spoof,
                confidence=0.8,
                probabilities=ProbabilityScores(bonafide=0.2, spoof=0.8),
                processing_time_ms=1,
            )
        ],
        fusion=FusionResult(
            status=BranchStatus.success,
            prediction=PredictionLabel.spoof,
            confidence=0.8,
            probabilities=ProbabilityScores(bonafide=0.2, spoof=0.8),
            method="weighted_average",
            contains_dummy_branches=True,
            eligible_for_research_evaluation=False,
        ),
        total_processing_time_ms=1,
        created_at=now,
    )
    return {
        "id": prediction_id,
        "request_id": prediction.request_id,
        "owner_user_id": owner_user_id,
        "source_type": SourceType.dashboard_upload.value,
        "status": "completed",
        "original_filename": prediction.audio.original_filename,
        "original_extension": "wav",
        "detected_container": "wav",
        "detected_codec": "pcm_s16le",
        "size_bytes": prediction.audio.file_size_bytes,
        "duration_seconds": prediction.audio.duration_seconds,
        "sample_rate": prediction.audio.sample_rate,
        "channels": prediction.audio.channels,
        "branches": [branch.model_dump(mode="python") for branch in prediction.branches],
        "fusion": prediction.fusion.model_dump(mode="python"),
        "total_processing_time_ms": prediction.total_processing_time_ms,
        "created_at": now,
    }
