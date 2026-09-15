from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.api.dependencies import get_prediction_submission_service
from app.auth.clerk_auth import require_clerk_user
from app.auth.schemas import AuthPrincipal
from app.config.settings import Settings
from app.core.exceptions import NoUsableModelBranchesError
from app.ingestion.audio import ProcessedAudio
from app.main import app
from app.models.factory import ModelFactory
from app.models.registry import ModelRegistry
from app.models.runtime import PUBLIC_MODEL_NAMES, REAL_ARCHITECTURES
from app.schemas.common import BranchStatus, ModelMode, PredictionLabel, SourceType
from app.schemas.prediction import (
    AudioMetadata,
    AudioStorageMetadata,
    BranchPrediction,
    FusionResult,
    ProbabilityScores,
    VoicePredictionResponse,
)
from app.services.prediction_job_runner import InlinePredictionJobRunner
from app.services.prediction_persistence_service import PredictionPersistenceService
from app.services.prediction_submission_service import PredictionSubmissionService
from app.services.voice_service import PredictionExecutionResult, VoiceService
from app.utils.fusion import DEFAULT_DEVELOPMENT_WEIGHTS, FusionEngine
from scripts.validate_aasist_light_v2_inference import validation_settings
from tests.real_model_helpers import CHECKPOINT_ROOT, requires_torch
from tests.test_prediction_routes import (
    FakeSubmissionRepository,
    FakeSubmissionStorage,
    upload_metadata,
)

AASIST_V2_CHECKPOINT = CHECKPOINT_ROOT / "aasist" / "aasist_light_v2_best.pt"
PHASE3_AUDIO = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "e2e"
    / "fixtures"
    / "sample-voice.wav"
)

requires_public_aasist_smoke_assets = pytest.mark.skipif(
    not AASIST_V2_CHECKPOINT.is_file() or not PHASE3_AUDIO.is_file(),
    reason="AASIST-Light V2 checkpoint or public audio fixture is unavailable.",
)


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_phase6_public_prediction_openapi_contract_has_no_aasist_v2_fields(
    client: TestClient,
) -> None:
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/v1/predictions"]["post"]
    request_schema = operation["requestBody"]["content"]["multipart/form-data"][
        "schema"
    ]
    response_schema = schema["components"]["schemas"]["PredictionSubmissionResponse"]
    response_fields = set(response_schema["properties"])

    assert set(operation["responses"]) >= {
        "200",
        "400",
        "401",
        "409",
        "413",
        "415",
        "422",
        "500",
        "503",
    }
    assert request_schema["$ref"].startswith("#/components/schemas/")
    assert response_fields == {
        "prediction_id",
        "request_id",
        "status",
        "source_type",
        "audio",
        "branches",
        "fusion",
        "research_eligible",
        "created_at",
    }
    assert not any("aasist_light_v2" in field for field in response_fields)
    assert not any("checkpoint" in field for field in response_fields)


def test_phase6_public_prediction_unauthorized_response_is_unchanged() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/predictions",
            data={"source_type": "dashboard_upload"},
            files={"file": ("sample.wav", b"placeholder", "audio/wav")},
        )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_failed"
    assert response.json()["request_id"]


@requires_torch
@requires_public_aasist_smoke_assets
def test_phase6_real_aasist_v2_public_endpoint_smoke_exercises_existing_path() -> None:
    repository = FakeSubmissionRepository()
    storage = FakeSubmissionStorage()
    xai = RecordingXaiOrchestrator()
    settings = _phase6_real_endpoint_settings(xai_enabled=True)
    voice_service = VoiceService(app_settings=settings)

    with phase6_prediction_client(
        repository=repository,
        storage=storage,
        voice_service=voice_service,
        xai_orchestrator=xai,
        app_settings=settings,
    ) as client:
        with PHASE3_AUDIO.open("rb") as audio_file:
            started_at = perf_counter()
            response = client.post(
                "/api/v1/predictions",
                data={"source_type": "dashboard_upload"},
                files={"file": ("sample-voice.wav", audio_file, "audio/wav")},
                headers={"X-Request-ID": "phase6-public-smoke"},
            )
            latency_ms = (perf_counter() - started_at) * 1000

    assert response.status_code == 200
    body = response.json()
    aasist = _branch(body, "aasist")
    assert body["status"] == "completed"
    assert body["source_type"] == "dashboard_upload"
    assert body["fusion"]["prediction"] == "spoof"
    assert body["fusion"]["probabilities"]["spoof"] == pytest.approx(
        0.9999964237213135,
        abs=1e-8,
    )
    assert aasist["status"] == "success"
    assert aasist["mode"] == "real"
    assert aasist["prediction"] == "spoof"
    assert aasist["probabilities"]["spoof"] == pytest.approx(
        0.9999964237213135,
        abs=1e-8,
    )
    assert _finite_probability_pair(aasist["probabilities"])
    assert _finite_probability_pair(body["fusion"]["probabilities"])
    assert "aasist_light_v2" not in body
    assert "checkpoint" not in body

    document = repository.documents[0]
    assert document["status"] == "completed"
    assert _branch(document, "aasist")["model_name"] == "aasist"
    assert document["fusion"]["branch_weights"] == {"aasist": pytest.approx(1.0)}
    assert document["research_eligible"] is False
    assert document["cloudinary_asset"]["public_id"].startswith(
        "multiscope/audio/user_123/"
    )
    assert xai.bundle is not None
    assert xai.bundle.prediction_id == body["prediction_id"]
    assert xai.bundle.request_id == body["request_id"]
    assert xai.bundle.processed_audio is not None
    assert xai.bundle.prediction.fusion.contributing_branches == ["aasist"]
    print(
        "PHASE6_REAL_ENDPOINT_SMOKE "
        f"endpoint=/api/v1/predictions status={response.status_code} "
        f"prediction={body['fusion']['prediction']} "
        f"fused_spoof={body['fusion']['probabilities']['spoof']} "
        f"aasist_status={aasist['status']} request_id={body['request_id']} "
        f"persisted={document['status']} xai_enqueued={xai.bundle is not None} "
        f"latency_ms={latency_ms:.3f}"
    )


@pytest.mark.anyio
async def test_phase6_persistence_service_keeps_aasist_under_public_key() -> None:
    repository = FakeSubmissionRepository()
    persistence = PredictionPersistenceService(repository)
    storage = AudioStorageMetadata(status=BranchStatus.success, public_id="audio/1")
    prediction = voice_prediction_with_branches(
        [branch_prediction("aasist", 0.8, mode=ModelMode.real)],
        request_id="request-persist",
    )
    await persistence.create_queued(
        request_id="request-persist",
        principal=AuthPrincipal(
            subject="user:user_123",
            principal_type="clerk_user",
            user_id="user_123",
        ),
        source_type=SourceType.dashboard_upload,
    )
    await persistence.save_result(
        "request-persist",
        prediction,
        storage_metadata=storage,
        storage_required=False,
    )

    document = repository.document_for_request("request-persist")
    assert set(document) >= {
        "id",
        "request_id",
        "owner_user_id",
        "source_type",
        "status",
        "branches",
        "fusion",
        "research_eligible",
        "error_summary",
    }
    assert _branch(document, "aasist")["model_name"] == "aasist"
    assert document["fusion"]["branch_weights"] == {"aasist": pytest.approx(1.0)}
    assert "aasist-light-v2-finalized-baseline" not in document["fusion"][
        "branch_weights"
    ]


@requires_torch
@requires_public_aasist_smoke_assets
def test_phase6_model_health_ready_and_research_unverified_contract() -> None:
    settings = _phase6_real_endpoint_settings()
    model = ModelFactory(settings).create("aasist")
    model.predict_safe(_processed_audio())
    health = model.health()

    assert health["branch_name"] == "aasist"
    assert health["model_name"] == "aasist"
    assert health["mode"] == "real"
    assert health["ready"] is True
    assert health["checkpoint_configured"] is True
    assert health["checkpoint_valid"] is True
    assert health["research_ready"] is False
    assert health["model_version"] == "checkpoint-c944c69f05a0"
    assert health["architecture"] == "aasist-light-v2-finalized-baseline"


def test_phase6_missing_aasist_checkpoint_is_controlled_not_ready() -> None:
    settings = _phase6_real_endpoint_settings(aasist_model_path="missing.pt")
    readiness = ModelRegistry(app_settings=settings).readiness()
    aasist = _branch(readiness, "aasist")

    assert readiness["ready"] is False
    assert readiness["research_ready"] is False
    assert aasist["mode"] == "real"
    assert aasist["ready"] is False
    assert aasist["checkpoint_configured"] is True
    assert aasist["checkpoint_valid"] is False


@pytest.mark.anyio
async def test_phase6_xai_handoff_contract_preserves_processed_audio_and_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_prediction_routes import upload_metadata

    upload_path = tmp_path / "validated.wav"
    upload_path.write_bytes(b"validated")
    repository = FakeSubmissionRepository()
    await repository.create_prediction(
        request_id="request-xai",
        owner_user_id="user_123",
        source_type=SourceType.dashboard_upload,
    )
    monkeypatch.setattr(
        "app.services.prediction_submission_service.save_validated_audio_upload_blocking",
        lambda _file, **_kwargs: upload_metadata(upload_path, "sample.wav", "wav"),
    )
    xai = RecordingXaiOrchestrator(should_fail=True)
    settings = Settings(_env_file=None, xai_enabled=True, xai_mode="mock")
    service = PredictionSubmissionService(
        repository=repository,
        persistence=PredictionPersistenceService(repository),
        storage=FakeSubmissionStorage(),
        voice_service=ProcessedAudioVoiceService(),
        job_runner=InlinePredictionJobRunner(settings),
        xai_orchestrator=xai,
        app_settings=settings,
    )

    class UploadedFile:
        filename = "sample.wav"
        content_type = "audio/wav"

    response = await service.execute_prediction(
        file=UploadedFile(),
        principal=AuthPrincipal(
            subject="user:user_123",
            principal_type="clerk_user",
            user_id="user_123",
        ),
        source_type=SourceType.dashboard_upload,
        prediction_id="prediction-1",
        request_id="request-xai",
    )

    assert response.status.value == "completed"
    assert repository.document_for_request("request-xai")["status"] == "completed"
    assert xai.bundle is not None
    assert xai.bundle.prediction_id == "prediction-1"
    assert xai.bundle.processed_audio is ProcessedAudioVoiceService.PROCESSED_AUDIO


def test_phase6_branch_failure_combinations_remain_publicly_compatible() -> None:
    fusion = FusionEngine(minimum_successful_branches=1)
    aasist_success = fusion.fuse(
        [
            branch_prediction("aasist", 0.9, mode=ModelMode.real),
            failed_branch("cnn_acoustic"),
            failed_branch("ssl_wavlm_xlsr"),
            failed_branch("glottal_features"),
        ]
    )
    aasist_failed = fusion.fuse(
        [
            failed_branch("aasist", mode=ModelMode.real),
            branch_prediction("cnn_acoustic", 0.2),
            branch_prediction("ssl_wavlm_xlsr", 0.4),
        ]
    )
    all_failed = fusion.fuse(
        [
            failed_branch("aasist", mode=ModelMode.real),
            failed_branch("cnn_acoustic"),
        ]
    )

    assert aasist_success.status == BranchStatus.success
    assert aasist_success.contributing_branches == ["aasist"]
    assert aasist_failed.status == BranchStatus.success
    assert "aasist" in aasist_failed.excluded_branches
    assert all_failed.status == BranchStatus.failed


def test_phase6_other_branch_identifiers_and_weights_are_unchanged() -> None:
    settings = _phase6_real_endpoint_settings()

    assert PUBLIC_MODEL_NAMES == {
        "lfcc_cnn_tcn": "cnn_acoustic",
        "aasist": "aasist",
        "ssl_sequence": "ssl_wavlm_xlsr",
        "glottal": "glottal_features",
    }
    assert REAL_ARCHITECTURES["aasist"] == "aasist-light-v2-finalized-baseline"
    assert settings.fusion_weight_map == {
        "lfcc_cnn_tcn": pytest.approx(0.25),
        "aasist": pytest.approx(0.25),
        "ssl_sequence": pytest.approx(0.25),
        "glottal": pytest.approx(0.25),
    }
    assert DEFAULT_DEVELOPMENT_WEIGHTS == {
        "cnn": 0.25,
        "aasist": 0.25,
        "ssl": 0.25,
        "glottal": 0.25,
    }


def test_phase6_public_error_envelope_for_no_valid_branch_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = FakeSubmissionRepository()
    voice = RaisingVoiceService(NoUsableModelBranchesError("No usable branches."))
    upload_path = tmp_path / "validated.wav"
    upload_path.write_bytes(b"validated")
    monkeypatch.setattr(
        "app.services.prediction_submission_service.save_validated_audio_upload_blocking",
        lambda _file, **_kwargs: upload_metadata(upload_path, "sample.wav", "wav"),
    )

    with phase6_prediction_client(
        repository=repository,
        storage=FakeSubmissionStorage(),
        voice_service=voice,
    ) as client:
        response = client.post(
            "/api/v1/predictions",
            data={"source_type": "dashboard_upload"},
            files={"file": ("sample.wav", b"placeholder", "audio/wav")},
        )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"
    assert response.json()["request_id"]


def _phase6_real_endpoint_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "model_root_dir": str(CHECKPOINT_ROOT),
        "aasist_model_mode": "real",
        "aasist_model_path": "aasist/aasist_light_v2_best.pt",
        "cnn_model_mode": "disabled",
        "ssl_model_mode": "disabled",
        "glottal_model_mode": "disabled",
        "required_model_branches": "aasist",
        "fusion_min_successful_branches": 1,
        "model_device_policy": "cpu",
        "model_default_device": "cpu",
        "model_load_strategy": "lazy",
        "storage_policy": "optional",
        "xai_enabled": False,
        "xai_mode": "mock",
    }
    values.update(overrides)
    return validation_settings(**values)


@contextmanager
def phase6_prediction_client(
    *,
    repository: FakeSubmissionRepository,
    storage: FakeSubmissionStorage,
    voice_service: Any,
    xai_orchestrator: Any | None = None,
    app_settings: Settings | None = None,
) -> Generator[TestClient, None, None]:
    settings = app_settings or Settings(_env_file=None)
    service = PredictionSubmissionService(
        repository=repository,
        persistence=PredictionPersistenceService(repository),
        storage=storage,
        voice_service=voice_service,
        job_runner=InlinePredictionJobRunner(settings),
        xai_orchestrator=xai_orchestrator,
        app_settings=settings,
    )
    app.dependency_overrides[require_clerk_user] = lambda: AuthPrincipal(
        subject="user:user_123",
        principal_type="clerk_user",
        user_id="user_123",
    )
    app.dependency_overrides[get_prediction_submission_service] = lambda: service
    try:
        with TestClient(app) as client:
            yield client
    finally:
        app.dependency_overrides.clear()


class RecordingXaiOrchestrator:
    def __init__(self, *, should_fail: bool = False) -> None:
        self.bundle = None
        self.should_fail = should_fail

    async def enqueue(self, bundle):
        self.bundle = bundle
        if self.should_fail:
            raise RuntimeError("xai unavailable")


class ProcessedAudioVoiceService:
    PROCESSED_AUDIO = ProcessedAudio(
        waveform=np.zeros(16000, dtype=np.float32),
        sample_rate=16000,
        original_sample_rate=16000,
        original_channels=1,
        duration_seconds=1.0,
        was_resampled=False,
        was_converted_to_mono=False,
        normalisation_applied=True,
        peak_amplitude=1.0,
        rms_energy=0.0,
        unnormalised_waveform=np.zeros(16000, dtype=np.float32),
    )

    def predict_from_validated_upload_with_processed_audio(
        self,
        upload,
        *,
        request_id=None,
        storage_metadata=None,
        cleanup_upload=True,
        capture_extraction=False,
    ):
        return PredictionExecutionResult(
            prediction=voice_prediction_with_branches(
                [branch_prediction("aasist", 0.9, mode=ModelMode.real)],
                request_id=request_id,
                storage_metadata=storage_metadata,
            ),
            processed_audio=self.PROCESSED_AUDIO,
            extraction=None,
        )


class RaisingVoiceService:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def predict_from_validated_upload(self, *_args, **_kwargs):
        raise self.error


def voice_prediction_with_branches(
    branches: list[BranchPrediction],
    *,
    request_id: str | None = None,
    storage_metadata: AudioStorageMetadata | None = None,
) -> VoicePredictionResponse:
    fusion = FusionEngine(minimum_successful_branches=1).fuse(branches)
    return VoicePredictionResponse(
        request_id=request_id or "request-id",
        audio=AudioMetadata(
            original_filename="sample.wav",
            content_type="audio/wav",
            original_extension="wav",
            detected_container="wav",
            detected_codec="pcm_s16le",
            file_size_bytes=1024,
            duration_seconds=1.0,
            sample_rate=16000,
            channels=1,
            storage=storage_metadata,
        ),
        branches=branches,
        fusion=fusion,
        total_processing_time_ms=1.0,
        created_at=datetime.now(UTC),
    )


def branch_prediction(
    model_name: str,
    spoof_probability: float,
    *,
    mode: ModelMode = ModelMode.dummy,
) -> BranchPrediction:
    bonafide_probability = 1.0 - spoof_probability
    prediction = (
        PredictionLabel.spoof
        if spoof_probability >= bonafide_probability
        else PredictionLabel.bonafide
    )
    return BranchPrediction(
        model_name=model_name,
        display_name=model_name,
        status=BranchStatus.success,
        mode=mode,
        prediction=prediction,
        confidence=max(spoof_probability, bonafide_probability),
        probabilities=ProbabilityScores(
            bonafide=bonafide_probability,
            spoof=spoof_probability,
        ),
        processing_time_ms=1.0,
        metadata={"research_result": False},
    )


def failed_branch(model_name: str, *, mode: ModelMode = ModelMode.dummy) -> BranchPrediction:
    return BranchPrediction(
        model_name=model_name,
        display_name=model_name,
        status=BranchStatus.failed,
        mode=mode,
        processing_time_ms=1.0,
        error="branch failed",
        metadata={"error_code": "branch_failed", "research_result": False},
    )


def _branch(container: dict[str, Any], model_name: str) -> dict[str, Any]:
    for branch in container["branches"]:
        if branch.get("model_name") == model_name or branch.get("branch_name") == model_name:
            return branch
    raise AssertionError(f"Missing branch {model_name}")


def _finite_probability_pair(probabilities: dict[str, float]) -> bool:
    values = [probabilities["bonafide"], probabilities["spoof"]]
    return all(isfinite(value) for value in values) and sum(values) == pytest.approx(
        1.0,
        abs=1e-3,
    )


def _processed_audio() -> ProcessedAudio:
    waveform = np.zeros(64600, dtype=np.float32)
    waveform[1000:2000] = 0.1
    return ProcessedAudio(
        waveform=waveform,
        sample_rate=16000,
        original_sample_rate=16000,
        original_channels=1,
        duration_seconds=float(waveform.size / 16000),
        was_resampled=False,
        was_converted_to_mono=False,
        normalisation_applied=True,
        peak_amplitude=0.1,
        rms_energy=float(np.sqrt(np.mean(waveform**2))),
        unnormalised_waveform=waveform.copy(),
    )
