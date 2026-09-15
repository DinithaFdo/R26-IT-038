from datetime import UTC, datetime
import os
from pathlib import Path
import time

import numpy as np
import pytest

from fastapi.testclient import TestClient

from app.api.dependencies import get_xai_orchestrator
from app.auth.clerk_auth import require_clerk_user
from app.auth.schemas import AuthPrincipal
from app.config.settings import Settings
from app.main import ApplicationDependencies, LifespanDependencies, create_app
from app.schemas.common import BranchStatus, ModelMode, PredictionLabel, SourceType
from app.schemas.xai import ComponentStatus, TemporalExplanation
from app.schemas.prediction import (
    AudioMetadata,
    BranchPrediction,
    FusionResult,
    ProbabilityScores,
    VoicePredictionResponse,
)
from app.voice_xai.artifacts.service import LocalExplanationArtifactStore
from app.voice_xai.contracts import ClassifierInferenceBundle
from app.voice_xai.orchestrator import (
    VoiceXaiOrchestrator,
    classifier_snapshot_from_prediction,
)
from app.voice_xai.persistence.mongodb import MongoXaiExplanationRepository
from app.voice_xai.temporal.contracts import TemporalAnalysisResult, TemporalVisualization
from tests.test_mongodb import FakeDatabase


def test_local_artifact_store_keeps_json_and_binary_bytes_outside_mongodb(tmp_path: Path) -> None:
    store = LocalExplanationArtifactStore(tmp_path / "private")
    json_reference = store.store_json(
        explanation_id="explanation-123",
        payload={"scores": [0.1, 0.9], "threshold": 0.8},
    )
    npz_reference = store.store_bytes(
        explanation_id="explanation-123",
        content=b"npz-bytes",
        kind="attention_tensor",
        content_type="application/x-npz",
    )

    assert json_reference.download_path is None
    assert json_reference.kind == "attention_visualization"
    assert store.read(json_reference).content == b'{"scores":[0.1,0.9],"threshold":0.8}'
    assert store.read(npz_reference).content == b"npz-bytes"
    assert len(list((tmp_path / "private").rglob("*"))) == 3


def test_deleting_one_explanation_removes_only_its_private_artifacts(tmp_path: Path) -> None:
    store = LocalExplanationArtifactStore(tmp_path / "private")
    deleted_reference = store.store_bytes(
        explanation_id="explanation-delete",
        content=b"remove-me",
        kind="attention_tensor",
        content_type="application/x-npz",
    )
    retained_reference = store.store_bytes(
        explanation_id="explanation-retain",
        content=b"keep-me",
        kind="attention_tensor",
        content_type="application/x-npz",
    )

    assert store.delete_explanation_artifacts("explanation-delete") == 1
    assert store.delete_explanation_artifacts("explanation-delete") == 0
    assert store.read(deleted_reference) is None
    assert store.read(retained_reference).content == b"keep-me"


def test_cleanup_expired_deletes_only_expired_files_inside_the_root(tmp_path: Path) -> None:
    store = LocalExplanationArtifactStore(tmp_path / "private", retention_seconds=1)
    expired_reference = store.store_json(
        explanation_id="explanation-expired",
        payload={"scores": [0.1]},
    )
    # Backdate the file's mtime instead of sleeping, so this asserts on the
    # store's own retention math rather than real wall-clock timing.
    expired_path = next((tmp_path / "private").glob("*/*"))
    old_time = time.time() - 3600
    os.utime(expired_path, (old_time, old_time))

    fresh_reference = store.store_json(
        explanation_id="explanation-fresh",
        payload={"scores": [0.2]},
    )

    result = store.cleanup_expired()

    assert result.deleted == 1
    assert result.retained == 1
    assert result.errors == 0
    assert store.read(expired_reference) is None
    assert store.read(fresh_reference) is not None
    # The now-empty expired-explanation directory should also be removed.
    remaining_directories = [p for p in (tmp_path / "private").iterdir() if p.is_dir()]
    assert len(remaining_directories) == 1


def test_cleanup_expired_is_a_safe_no_op_when_root_does_not_exist(tmp_path: Path) -> None:
    store = LocalExplanationArtifactStore(tmp_path / "does-not-exist-yet")

    result = store.cleanup_expired()

    assert result == store.cleanup_expired()
    assert result.deleted == 0
    assert result.retained == 0
    assert result.errors == 0


def test_cleanup_expired_cannot_delete_outside_the_configured_root(tmp_path: Path) -> None:
    root = tmp_path / "private"
    store = LocalExplanationArtifactStore(root, retention_seconds=1)
    store.store_json(explanation_id="explanation-1", payload={"scores": [0.1]})

    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("must never be touched")

    store.cleanup_expired()

    assert outside_file.exists()
    assert outside_file.read_text() == "must never be touched"


def test_owner_scoped_artifact_route_requires_the_persisted_reference(tmp_path: Path) -> None:
    document = _prediction_document()
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_artifact_routes"))
    store = LocalExplanationArtifactStore(tmp_path / "private")
    settings = Settings(_env_file=None, xai_enabled=True, xai_mode="mock")
    orchestrator = VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=PredictionRepository(document),
        queue=PausedQueue(),
        app_settings=settings,
        artifact_store=store,
        temporal_service=object(),
        replay_service=ReplayService(),
    )

    async def seed():
        response = await orchestrator.trigger_for_prediction(
            prediction_id="prediction-123", owner_user_id="user-123"
        )
        reference = store.store_json(
            explanation_id=response.explanation_id,
            payload={"fixture": "private"},
        )
        await repository.append_artifact(response.explanation_id, reference)
        return reference

    import asyncio

    reference = asyncio.run(seed())

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
        dependencies=ApplicationDependencies(
            xai_orchestrator=orchestrator,
            xai_artifact_store=store,
        ),
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
            response = client.get(
                f"/api/v1/me/predictions/prediction-123/explanation/artifacts/{reference.artifact_id}"
            )
            assert response.status_code == 200
            assert response.headers["content-type"] == "application/json"
            assert response.json() == {"fixture": "private"}

            current_user["id"] = "another-user"
            assert client.get(
                f"/api/v1/me/predictions/prediction-123/explanation/artifacts/{reference.artifact_id}"
            ).status_code == 404
    finally:
        app.dependency_overrides.clear()


@pytest.mark.anyio
async def test_orchestrator_persists_only_a_temporal_json_reference(tmp_path: Path) -> None:
    document = _prediction_document()
    prediction = _prediction_response(document)
    repository = MongoXaiExplanationRepository(FakeDatabase("xai_artifact_run"))
    store = LocalExplanationArtifactStore(tmp_path / "private")
    settings = Settings(_env_file=None, xai_enabled=True, xai_mode="mock")
    orchestrator = VoiceXaiOrchestrator(
        repository=repository,
        prediction_repository=PredictionRepository(document),
        queue=PausedQueue(),
        app_settings=settings,
        artifact_store=store,
        temporal_service=StaticTemporalService(),
    )
    bundle = ClassifierInferenceBundle(
        prediction_id=document["id"],
        request_id=prediction.request_id,
        owner_user_id=document["owner_user_id"],
        source_type=SourceType.dashboard_upload,
        prediction=prediction,
    )
    explanation_id = await repository.create_explanation(
        prediction_id=bundle.prediction_id,
        request_id=bundle.request_id,
        owner_user_id=bundle.owner_user_id,
        source_type=bundle.source_type,
        classifier_snapshot=classifier_snapshot_from_prediction(prediction),
        pipeline_version=settings.xai_pipeline_version,
    )

    await orchestrator._run(explanation_id, bundle)

    response = await orchestrator.get_for_owner(
        explanation_id=explanation_id,
        owner_user_id="user-123",
    )
    assert len(response.artifacts) == 1
    reference = response.artifacts[0]
    assert reference.kind == "attention_visualization"
    assert reference.download_path is None
    assert response.temporal.artifacts == [reference]
    stored = store.read(reference)
    assert stored is not None
    assert b"attention_scores" in stored.content
    persisted = await repository.get_explanation_for_owner(
        explanation_id=explanation_id,
        owner_user_id="user-123",
    )
    assert "content" not in persisted["artifacts"][0]


class PausedQueue:
    def submit(self, bundle, task):
        return object()

    def close(self, *, wait=True):
        return None


class ReplayService:
    async def load_processed_audio(self, *, prediction_document, owner_user_id):
        return None


class StaticTemporalService:
    def analyze(self, inference):
        return TemporalAnalysisResult(
            explanation=TemporalExplanation(
                status=ComponentStatus.completed,
                method_version="test-xlsr-attention-v1",
                model_version="test-xlsr",
                development_placeholder=False,
                research_eligible=False,
                attention_score_peak=1.0,
                attention_threshold=0.5,
            ),
            visualization=TemporalVisualization(
                time_seconds=np.array([0.5], dtype=np.float32),
                attention_scores=np.array([1.0], dtype=np.float32),
                threshold=0.5,
                high_attention_mask=np.array([False]),
            ),
            development_placeholder=False,
            research_eligible=False,
            warnings=(),
        )


class PredictionRepository:
    def __init__(self, document: dict) -> None:
        self.document = document

    async def get_prediction_for_owner(self, *, prediction_id, owner_user_id, include_deleted=False):
        if prediction_id == self.document["id"] and owner_user_id == self.document["owner_user_id"]:
            return self.document
        return None


def _prediction_document() -> dict:
    now = datetime.now(UTC)
    probabilities = ProbabilityScores(bonafide=0.2, spoof=0.8)
    prediction = VoicePredictionResponse(
        request_id="request-123",
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
                probabilities=probabilities,
                processing_time_ms=1,
            )
        ],
        fusion=FusionResult(
            status=BranchStatus.success,
            prediction=PredictionLabel.spoof,
            confidence=0.8,
            probabilities=probabilities,
            method="weighted_average",
            contains_dummy_branches=True,
            eligible_for_research_evaluation=False,
        ),
        total_processing_time_ms=1,
        created_at=now,
    )
    return {
        "id": "prediction-123",
        "request_id": prediction.request_id,
        "owner_user_id": "user-123",
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


def _prediction_response(document: dict) -> VoicePredictionResponse:
    return VoicePredictionResponse(
        request_id=document["request_id"],
        audio=AudioMetadata(
            original_filename=document["original_filename"],
            content_type="audio/wav",
            file_size_bytes=document["size_bytes"],
            duration_seconds=document["duration_seconds"],
            sample_rate=document["sample_rate"],
            channels=document["channels"],
        ),
        branches=document["branches"],
        fusion=document["fusion"],
        total_processing_time_ms=document["total_processing_time_ms"],
        created_at=document["created_at"],
    )
