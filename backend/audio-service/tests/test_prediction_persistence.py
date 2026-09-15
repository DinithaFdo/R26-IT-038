from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.auth.schemas import AuthPrincipal
from app.schemas.common import (
    BranchStatus,
    ModelMode,
    PredictionLabel,
    PredictionStatus,
    SourceType,
    is_valid_prediction_status_transition,
)
from app.schemas.prediction import (
    AudioMetadata,
    AudioStorageMetadata,
    BranchPrediction,
    FusionResult,
    ProbabilityScores,
    VoicePredictionResponse,
)
from app.schemas.provenance import (
    BranchModelProvenance,
    FusionProvenance,
    PredictionProvenance,
    PreprocessingProvenance,
)
from app.services.prediction_persistence_service import (
    PredictionPersistenceError,
    PredictionPersistenceService,
    sanitized_error_code,
)


@pytest.mark.anyio
async def test_successful_prediction_status_path_is_persisted() -> None:
    repository = FakePredictionRepository()
    service = PredictionPersistenceService(repository)
    principal = AuthPrincipal(
        subject="user_123",
        principal_type="clerk_user",
        user_id="user_123",
    )
    storage_metadata = successful_storage()

    await service.create_uploaded(request_id="request-1", principal=principal)
    await service.mark_validating("request-1")
    await service.attach_validated_upload("request-1", FakeUploadMetadata())
    await service.mark_storing("request-1")
    await service.attach_storage("request-1", storage_metadata)
    await service.mark_processing("request-1")
    await service.save_result(
        "request-1",
        make_response(),
        storage_metadata=storage_metadata,
    )

    assert repository.statuses == [
        "queued",
        "validating",
        "storing",
        "processing",
        "completed",
    ]
    assert repository.document["status"] == "completed"
    assert repository.document["research_eligible"] is False
    assert repository.document["cloudinary_asset"]["public_id"].endswith("/audio_123")
    assert repository.document["preprocessing"]["input_sample_rate"] == 48000
    assert repository.document["preprocessing"]["target_sample_rate"] == 16000
    assert repository.document["preprocessing"]["target_channels"] == 1
    assert repository.document["preprocessing"]["resampled"] is True
    assert repository.document["preprocessing"]["mono_conversion_applied"] is True
    assert repository.document["provenance"]["fusion"]["fusion_method"] == (
        "weighted_average"
    )
    assert repository.document["provenance"]["models"][0]["mode"] == "dummy"
    assert repository.document["provenance"]["models"][0]["research_result"] is False


@pytest.mark.anyio
async def test_validation_failure_is_saved_without_storage_upload() -> None:
    repository = FakePredictionRepository()
    service = PredictionPersistenceService(repository)
    await service.create_uploaded(
        request_id="request-2",
        principal=AuthPrincipal(
            subject="user_123",
            principal_type="clerk_user",
            user_id="user_123",
        ),
    )
    await service.mark_validating("request-2")

    await service.mark_failed(
        "request-2",
        stage="validation",
        code=sanitized_error_code(ValueError("bad audio")),
    )

    assert repository.statuses == ["queued", "validating", "failed"]
    assert repository.document["cloudinary_asset"] is None
    assert repository.document["error_summary"][-1]["stage"] == "validation"
    assert repository.document["error_summary"][-1]["code"] == "value"


@pytest.mark.anyio
async def test_cloudinary_failure_marks_record_failed_without_model_results() -> None:
    repository = FakePredictionRepository()
    service = PredictionPersistenceService(repository)
    storage_metadata = AudioStorageMetadata(
        status=BranchStatus.failed,
        error="Audio storage failed.",
    )
    await service.create_uploaded(
        request_id="request-3",
        principal=AuthPrincipal(
            subject="user_123",
            principal_type="clerk_user",
            user_id="user_123",
        ),
    )

    await service.mark_validating("request-3")
    await service.mark_storing("request-3")
    await service.attach_storage("request-3", storage_metadata)

    assert repository.document["status"] == "failed"
    assert repository.document.get("branches", []) == []
    assert repository.document.get("fusion") is None
    assert _codes(repository) == ["cloudinary_upload_failed"]


@pytest.mark.anyio
async def test_cloudinary_timeout_keeps_mongodb_record_consistent() -> None:
    repository = FakePredictionRepository()
    service = PredictionPersistenceService(repository)
    storage_metadata = AudioStorageMetadata(
        status=BranchStatus.failed,
        error="Audio storage timed out; outcome is being reconciled.",
    )
    await service.create_uploaded(
        request_id="request-timeout",
        principal=AuthPrincipal(
            subject="user_123",
            principal_type="clerk_user",
            user_id="user_123",
        ),
    )

    await service.mark_validating("request-timeout")
    await service.mark_storing("request-timeout")
    await service.attach_storage("request-timeout", storage_metadata)

    assert repository.document["status"] == "failed"
    assert repository.document["cloudinary_asset"] is None
    assert repository.document.get("branches", []) == []
    assert _codes(repository) == ["cloudinary_upload_failed"]


@pytest.mark.anyio
async def test_model_branch_failure_preserves_partial_results_as_completed() -> None:
    repository = FakePredictionRepository()
    service = PredictionPersistenceService(repository)
    storage_metadata = successful_storage()
    await service.create_uploaded(
        request_id="request-4",
        principal=AuthPrincipal(
            subject="user_123",
            principal_type="clerk_user",
            user_id="user_123",
        ),
    )

    await service.mark_validating("request-4")
    await service.mark_storing("request-4")
    await service.attach_storage("request-4", storage_metadata)
    await service.mark_processing("request-4")
    await service.save_result(
        "request-4",
        make_response(branch_failed=True),
        storage_metadata=storage_metadata,
    )

    assert repository.document["status"] == "completed"
    assert repository.document["branches"][0]["status"] == "failed"
    assert _codes(repository) == ["cnn_acoustic_failed"]


@pytest.mark.anyio
async def test_fusion_failure_marks_record_failed_with_branch_results() -> None:
    repository = FakePredictionRepository()
    service = PredictionPersistenceService(repository)
    storage_metadata = successful_storage()
    await service.create_uploaded(
        request_id="request-5",
        principal=AuthPrincipal(
            subject="user_123",
            principal_type="clerk_user",
            user_id="user_123",
        ),
    )

    await service.mark_validating("request-5")
    await service.mark_storing("request-5")
    await service.attach_storage("request-5", storage_metadata)
    await service.mark_processing("request-5")
    await service.save_result(
        "request-5",
        make_response(fusion_failed=True),
        storage_metadata=storage_metadata,
    )

    assert repository.document["status"] == "failed"
    assert repository.document["branches"]
    assert _codes(repository) == ["fusion_failed"]


@pytest.mark.anyio
async def test_deleting_to_deleted_status_path_can_be_saved() -> None:
    repository = FakePredictionRepository()
    await repository.create_prediction(
        request_id="request-6",
        owner_user_id="user_123",
        source_type=repository.source_type,
    )

    await repository.update_prediction_status(
        "request-6",
        PredictionStatus.deleting,
    )
    await repository.update_prediction_status(
        "request-6",
        PredictionStatus.deleted,
    )

    assert repository.document["status"] == "deleted"
    assert repository.statuses == ["queued", "deleting", "deleted"]


def test_exact_canonical_prediction_state_machine() -> None:
    valid_path = [
        (PredictionStatus.queued, PredictionStatus.validating),
        (PredictionStatus.validating, PredictionStatus.storing),
        (PredictionStatus.storing, PredictionStatus.processing),
        (PredictionStatus.processing, PredictionStatus.completed),
        (PredictionStatus.completed, PredictionStatus.deleting),
        (PredictionStatus.deleting, PredictionStatus.deleted),
    ]
    for current_status, next_status in valid_path:
        assert is_valid_prediction_status_transition(current_status, next_status)

    assert not is_valid_prediction_status_transition(
        PredictionStatus.processing,
        PredictionStatus.validating,
    )
    assert not is_valid_prediction_status_transition(
        PredictionStatus.deleted,
        PredictionStatus.completed,
    )


@pytest.mark.anyio
async def test_persistence_failure_after_storage_attempts_compensating_delete() -> None:
    repository = FakePredictionRepository(raise_on_result=True)
    service = PredictionPersistenceService(repository)
    storage = FakeStorage()
    storage_metadata = successful_storage()
    await service.create_uploaded(
        request_id="request-7",
        principal=AuthPrincipal(
            subject="user_123",
            principal_type="clerk_user",
            user_id="user_123",
        ),
    )

    await service.mark_validating("request-7")
    await service.mark_storing("request-7")
    await service.attach_storage("request-7", storage_metadata)
    await service.mark_processing("request-7")
    with pytest.raises(PredictionPersistenceError):
        await service.save_result(
            "request-7",
            make_response(),
            storage_metadata=storage_metadata,
        )
    await service.compensate_cloudinary_upload(
        storage=storage,
        storage_metadata=storage_metadata,
    )

    assert storage.deleted_public_ids == [storage_metadata.public_id]


def make_response(
    *,
    storage_metadata: AudioStorageMetadata | None = None,
    branch_failed: bool = False,
    fusion_failed: bool = False,
) -> VoicePredictionResponse:
    branch = BranchPrediction(
        model_name="cnn_acoustic",
        display_name="CNN Acoustic",
        status=BranchStatus.failed if branch_failed else BranchStatus.success,
        mode=ModelMode.dummy,
        prediction=None if branch_failed else PredictionLabel.spoof,
        confidence=None if branch_failed else 0.7,
        probabilities=None
        if branch_failed
        else ProbabilityScores(bonafide=0.3, spoof=0.7),
        processing_time_ms=1.0,
        error="Branch failed." if branch_failed else None,
    )
    fusion = FusionResult(
        status=BranchStatus.failed if fusion_failed else BranchStatus.success,
        prediction=None if fusion_failed else PredictionLabel.spoof,
        confidence=None if fusion_failed else 0.7,
        probabilities=None
        if fusion_failed
        else ProbabilityScores(bonafide=0.3, spoof=0.7),
        method="weighted_average",
        branch_weights={"cnn_acoustic": 1.0} if not fusion_failed else {},
        contains_dummy_branches=True,
        eligible_for_research_evaluation=False,
        warning="Dummy result.",
    )
    return VoicePredictionResponse(
        request_id="request-123",
        audio=AudioMetadata(
            original_filename="sample.wav",
            content_type="audio/wav",
            original_extension="wav",
            detected_container="wav",
            detected_codec="pcm_s16le",
            size_bytes=1024,
            file_size_bytes=1024,
            duration_seconds=1.0,
            sample_rate=16000,
            channels=1,
            storage=storage_metadata,
        ),
        branches=[branch],
        fusion=fusion,
        provenance=make_provenance(fusion),
        total_processing_time_ms=2.0,
        created_at=datetime.now(UTC),
    )


def make_provenance(fusion: FusionResult) -> PredictionProvenance:
    return PredictionProvenance(
        preprocessing=PreprocessingProvenance(
            input_sample_rate=48000,
            input_channels=2,
            input_duration_seconds=1.0,
            target_sample_rate=16000,
            target_channels=1,
            resampled=True,
            mono_conversion_applied=True,
            normalisation_applied=True,
            preprocessing_version="shared-audio-ffmpeg-mono-16khz-v1",
            ffmpeg_version="ffmpeg version 6.1-test",
            ffprobe_version="ffprobe version 6.1-test",
        ),
        models=[
            BranchModelProvenance(
                model_name="cnn_acoustic",
                mode=ModelMode.dummy,
                model_version="dummy-v1",
                checkpoint_id=None,
                checkpoint_sha256=None,
                architecture_version="deterministic-placeholder-v1",
                class_mapping_version="binary-bonafide-spoof-v1",
                preprocessing_compatibility_version=(
                    "shared-audio-ffmpeg-mono-16khz-v1"
                ),
                framework_version="python-numpy-deterministic",
                device_type="cpu",
                research_result=False,
            )
        ],
        fusion=FusionProvenance(
            fusion_method=fusion.method,
            configured_weights={"cnn": 0.25, "aasist": 0.25},
            effective_weights=fusion.branch_weights,
            threshold=0.5,
            fusion_version="score-level-fusion-v1",
            contains_dummy_branches=fusion.contains_dummy_branches,
            eligible_for_research_evaluation=(
                fusion.eligible_for_research_evaluation
            ),
        ),
    )


def successful_storage() -> AudioStorageMetadata:
    return AudioStorageMetadata(
        status=BranchStatus.success,
        asset_id="asset-123",
        public_id="multiscope/audio/user_123/audio_123",
        resource_type="video",
        version=123,
        format="wav",
        bytes=1024,
        duration=1.0,
        created_at=datetime.now(UTC),
    )


def _codes(repository) -> list[str]:
    return [entry["code"] for entry in repository.document["error_summary"]]


class FakeUploadMetadata:
    sanitized_filename = "sample.wav"
    original_extension = "wav"
    detected_container = "wav"
    detected_codec = "pcm_s16le"
    duration_seconds = 1.0
    sample_rate = 16000
    channels = 1
    size_bytes = 1024
    saved_path = Path("/tmp/should-not-be-persisted.wav")


class FakePredictionRepository:
    source_type = SourceType.public_api
    def __init__(self, *, raise_on_result: bool = False) -> None:
        self.raise_on_result = raise_on_result
        self.document = {}
        self.statuses = []

    async def create_prediction(
        self,
        *,
        request_id,
        owner_user_id,
        source_type,
        client_correlation_id=None,
        idempotency_key=None,
        logical_request=None,
        parent_prediction_id=None,
        rerun_reason=None,
        preprocessing_version=None,
        model_versions=None,
    ):
        self.document = {
            "id": "prediction-id",
            "request_id": request_id,
            "client_correlation_id": client_correlation_id,
            "owner_user_id": owner_user_id,
            "source_type": source_type.value,
            "status": "queued",
            "idempotency_key": idempotency_key,
            "idempotency_logical_request": logical_request,
            "parent_prediction_id": parent_prediction_id,
            "rerun_reason": rerun_reason,
            "preprocessing_version": preprocessing_version,
            "model_versions": model_versions or {},
            "cloudinary_asset": None,
            "branches": [],
            "fusion": None,
            "error_summary": [],
        }
        self.statuses.append("queued")
        return "prediction-id"

    async def update_prediction_status(
        self,
        request_id,
        status,
        *,
        error_code=None,
        error_stage=None,
    ):
        self.document["status"] = status.value
        self.statuses.append(status.value)
        if error_code is not None:
            self.document["error_summary"].append(
                {"stage": error_stage, "code": error_code}
            )

    async def attach_upload_metadata(self, request_id, upload_metadata):
        self.document.update(
            {
                "original_filename": upload_metadata.sanitized_filename,
                "original_extension": upload_metadata.original_extension,
                "detected_container": upload_metadata.detected_container,
                "detected_codec": upload_metadata.detected_codec,
                "duration_seconds": upload_metadata.duration_seconds,
                "sample_rate": upload_metadata.sample_rate,
                "channels": upload_metadata.channels,
                "size_bytes": upload_metadata.size_bytes,
            }
        )

    async def attach_cloudinary_asset(self, request_id, storage_metadata):
        self.document["cloudinary_asset"] = (
            None
            if storage_metadata.status != BranchStatus.success
            else storage_metadata.model_dump(mode="python")
        )

    async def save_prediction_result(
        self,
        request_id,
        prediction,
        *,
        status,
        error_codes,
    ):
        if self.raise_on_result:
            raise RuntimeError("database unavailable")
        self.document.update(
            {
                "status": status.value,
                "branches": [
                    branch.model_dump(mode="python") for branch in prediction.branches
                ],
                "fusion": prediction.fusion.model_dump(mode="python"),
                "preprocessing": prediction.provenance.preprocessing.model_dump(
                    mode="python"
                )
                if prediction.provenance is not None
                else {},
                "provenance": prediction.provenance.model_dump(mode="python")
                if prediction.provenance is not None
                else None,
                "research_eligible": prediction.fusion.eligible_for_research_evaluation,
            }
        )
        self.statuses.append(status.value)
        self.document["error_summary"].extend(error_codes)


class FakeStorage:
    def __init__(self) -> None:
        self.deleted_public_ids = []

    async def delete_audio(self, public_id: str) -> None:
        self.deleted_public_ids.append(public_id)
