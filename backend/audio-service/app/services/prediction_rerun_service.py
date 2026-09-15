import inspect
import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, status

from app.auth.schemas import AuthPrincipal
from app.config.settings import Settings, settings
from app.core.exceptions import NoUsableModelBranchesError
from app.ingestion.audio import (
    AudioUploadMetadata,
    ProcessedAudio,
    save_validated_local_audio_file,
)
from app.repositories.protocols import PredictionRepository
from app.schemas.common import BranchStatus, PredictionStatus, SourceType
from app.schemas.prediction import AudioStorageMetadata, VoicePredictionResponse
from app.schemas.prediction_history import PredictionHistoryAudioMetadata
from app.schemas.prediction_submission import PredictionSubmissionResponse
from app.services.prediction_persistence_service import (
    PredictionPersistenceError,
    PredictionPersistenceService,
    sanitized_error_code,
)
from app.services.prediction_job_runner import PredictionJobRunner
from app.services.voice_service import VoiceService
from app.storage.protocols import AudioStorage
from app.voice_xai.capture.contracts import ExtractionBundle
from app.voice_xai.contracts import ClassifierInferenceBundle
from app.voice_xai.orchestrator import VoiceXaiOrchestrator
from app.voice_xai.temporal.contracts import TemporalAttentionWindowInput

logger = logging.getLogger(__name__)

PREPROCESSING_VERSION = "audio-preprocessing-v2"


class PredictionRerunService:
    def __init__(
        self,
        *,
        repository: PredictionRepository,
        persistence: PredictionPersistenceService,
        storage: AudioStorage,
        voice_service: VoiceService,
        job_runner: PredictionJobRunner,
        xai_orchestrator: VoiceXaiOrchestrator | None = None,
        app_settings: Settings = settings,
    ) -> None:
        self._repository = repository
        self._persistence = persistence
        self._storage = storage
        self._voice_service = voice_service
        self._job_runner = job_runner
        self._xai_orchestrator = xai_orchestrator
        self._settings = app_settings

    async def rerun_prediction(
        self,
        *,
        principal: AuthPrincipal,
        prediction_id: str,
        request_id: str,
        rerun_reason: str | None = None,
    ) -> PredictionSubmissionResponse:
        owner_user_id = _owner_user_id(principal)
        parent = await self._repository.get_prediction_for_owner(
            prediction_id=prediction_id,
            owner_user_id=owner_user_id,
        )
        if parent is None:
            _raise_not_found()

        public_id = _cloudinary_public_id(parent)
        if public_id is None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Source audio is unavailable for rerun.",
            )

        download_path = _safe_download_path()
        try:
            await self._download_source_audio(
                public_id=public_id,
                owner_user_id=owner_user_id,
                destination_path=download_path,
            )
        except BaseException:
            download_path.unlink(missing_ok=True)
            raise

        prediction_id_new = await self._persistence.create_queued(
            request_id=request_id,
            principal=principal,
            source_type=_source_type(parent),
            parent_prediction_id=prediction_id,
            rerun_reason=_safe_rerun_reason(rerun_reason),
            preprocessing_version=PREPROCESSING_VERSION,
            model_versions=_model_versions(self._voice_service),
        )

        return await self._job_runner.run(
            prediction_id=prediction_id_new,
            request_id=request_id,
            principal_key=_principal_key(principal),
            persistence=self._persistence,
            execute=lambda: self._execute_rerun(
                principal=principal,
                parent=parent,
                download_path=download_path,
                prediction_id_new=prediction_id_new,
                request_id=request_id,
            ),
        )

    async def _execute_rerun(
        self,
        *,
        principal: AuthPrincipal,
        parent: dict,
        download_path: Path,
        prediction_id_new: str,
        request_id: str,
    ) -> PredictionSubmissionResponse:
        upload_metadata = None
        storage_metadata = None

        try:
            await self._persistence.mark_validating(request_id)
            try:
                upload_metadata = await self._job_runner.run_blocking(
                    lambda: save_validated_local_audio_file(
                        download_path,
                        original_filename=_source_filename(parent),
                        content_type=None,
                        app_settings=self._settings,
                        display_filename=parent.get("original_filename"),
                    )
                )
            except Exception as error:
                await self._persistence.mark_failed(
                    request_id,
                    stage="validation",
                    code=sanitized_error_code(error),
                )
                raise

            await self._persistence.attach_validated_upload(
                request_id,
                upload_metadata,
            )
            await self._persistence.mark_storing(request_id)

            storage_metadata = await _store_audio(
                storage=self._storage,
                principal=principal,
                upload_metadata=upload_metadata,
                request_id=request_id,
            )
            storage_required = self._settings.storage_policy == "required"
            await self._persistence.attach_storage(
                request_id,
                storage_metadata,
                storage_required=storage_required,
            )
            _require_successful_storage(
                storage_metadata,
                storage_required=storage_required,
            )
            await self._persistence.mark_processing(request_id)

            (
                prediction,
                processed_audio,
                extraction,
                temporal_evidence,
            ) = await self._job_runner.run_blocking(
                lambda: _predict_without_upload_cleanup(
                    self._voice_service,
                    upload_metadata=upload_metadata,
                    request_id=request_id,
                    storage_metadata=storage_metadata,
                    capture_extraction=(
                        self._settings.xai_enabled and self._settings.xai_capture_enabled
                    ),
                )
            )
            await self._persistence.save_result(
                request_id,
                prediction,
                storage_metadata=storage_metadata,
                storage_required=self._settings.storage_policy == "required",
            )

            if not any(
                branch.status == BranchStatus.success
                for branch in prediction.branches
            ):
                raise NoUsableModelBranchesError(
                    "No usable model branches are available."
                )

            prediction_status = _final_status(
                prediction,
                storage_metadata,
                storage_required=self._settings.storage_policy == "required",
            )
            if prediction_status == PredictionStatus.completed:
                await self._enqueue_xai_without_affecting_prediction(
                    prediction_id=prediction_id_new,
                    request_id=request_id,
                    principal=principal,
                    source_type=_source_type(parent),
                    prediction=prediction,
                    processed_audio=processed_audio,
                    extraction=extraction,
                    temporal_evidence=temporal_evidence,
                )

            return _response_from_prediction(
                prediction_id=prediction_id_new,
                source_type=_source_type(parent),
                status=prediction_status,
                prediction=prediction,
            )
        except PredictionPersistenceError:
            await self._persistence.compensate_cloudinary_upload(
                storage=self._storage,
                storage_metadata=storage_metadata,
            )
            raise
        finally:
            download_path.unlink(missing_ok=True)
            if upload_metadata is not None:
                upload_metadata.saved_path.unlink(missing_ok=True)

    async def _enqueue_xai_without_affecting_prediction(
        self,
        *,
        prediction_id: str,
        request_id: str,
        principal: AuthPrincipal,
        source_type: SourceType,
        prediction: VoicePredictionResponse,
        processed_audio: ProcessedAudio | None,
        extraction: ExtractionBundle | None,
        temporal_evidence: tuple[TemporalAttentionWindowInput, ...] | None,
    ) -> None:
        """Start optional XAI after persistence without changing the rerun result."""

        if self._xai_orchestrator is None or not self._settings.xai_enabled:
            return
        try:
            await self._xai_orchestrator.enqueue(
                ClassifierInferenceBundle(
                    prediction_id=prediction_id,
                    request_id=request_id,
                    owner_user_id=_owner_user_id(principal),
                    source_type=source_type,
                    prediction=prediction,
                    # Preserve the exact decoded audio and bounded classifier
                    # evidence produced by this rerun. The orchestrator owns
                    # the async XAI work and never accesses the model registry.
                    processed_audio=processed_audio,
                    extraction=extraction,
                    temporal_evidence=temporal_evidence,
                )
            )
        except Exception:
            logger.exception(
                "Voice XAI enqueue failed after completed prediction rerun",
                extra={"request_id": request_id, "prediction_id": prediction_id},
            )

    async def _download_source_audio(
        self,
        *,
        public_id: str,
        owner_user_id: str,
        destination_path: Path,
    ) -> None:
        try:
            await self._storage.download_audio(
                public_id,
                destination_path,
                owner_user_id=owner_user_id,
                max_bytes=settings.max_upload_size_mb * 1024 * 1024,
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Source audio is unavailable for rerun.",
            )
        except PermissionError:
            _raise_not_found()
        except Exception:
            logger.exception(
                "Prediction source audio download failed.",
                extra={
                    "prediction_id": public_id.rsplit("/", 1)[-1],
                    "owner_user_id": owner_user_id,
                },
            )
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Source audio is unavailable for rerun.",
            )


async def _store_audio(
    *,
    storage: AudioStorage,
    principal: AuthPrincipal,
    upload_metadata: AudioUploadMetadata,
    request_id: str,
) -> AudioStorageMetadata:
    return await storage.upload_audio(
        upload_metadata.saved_path,
        owner_user_id=_owner_user_id(principal),
        audio_id=request_id or uuid4().hex,
        content_type=upload_metadata.content_type,
    )


def _predict_without_upload_cleanup(
    voice_service: VoiceService,
    *,
    upload_metadata: AudioUploadMetadata,
    request_id: str,
    storage_metadata: AudioStorageMetadata,
    capture_extraction: bool = False,
) -> tuple[
    VoicePredictionResponse,
    ProcessedAudio | None,
    ExtractionBundle | None,
    tuple[TemporalAttentionWindowInput, ...] | None,
]:
    """Run rerun inference and preserve the evidence needed by optional XAI."""

    with_audio = getattr(
        voice_service, "predict_from_validated_upload_with_processed_audio", None
    )
    if callable(with_audio):
        kwargs = {"request_id": request_id, "storage_metadata": storage_metadata}
        if "cleanup_upload" in inspect.signature(with_audio).parameters:
            kwargs["cleanup_upload"] = False
        if "capture_extraction" in inspect.signature(with_audio).parameters:
            kwargs["capture_extraction"] = capture_extraction
        result = with_audio(upload_metadata, **kwargs)
        return (
            result.prediction,
            result.processed_audio,
            getattr(result, "extraction", None),
            getattr(result, "temporal_evidence", None),
        )

    predict = voice_service.predict_from_validated_upload
    kwargs = {
        "request_id": request_id,
        "storage_metadata": storage_metadata,
    }
    if "cleanup_upload" in inspect.signature(predict).parameters:
        kwargs["cleanup_upload"] = False
    return predict(upload_metadata, **kwargs), None, None, None


def _require_successful_storage(
    storage_metadata: AudioStorageMetadata,
    *,
    storage_required: bool,
) -> None:
    if not storage_required:
        return
    if storage_metadata.status == BranchStatus.success:
        return
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Required audio storage is unavailable.",
    )


def _response_from_prediction(
    *,
    prediction_id: str,
    source_type: SourceType,
    status: PredictionStatus,
    prediction: VoicePredictionResponse,
) -> PredictionSubmissionResponse:
    return PredictionSubmissionResponse(
        prediction_id=prediction_id,
        request_id=prediction.request_id,
        status=status,
        source_type=source_type,
        audio=PredictionHistoryAudioMetadata(
            original_filename=prediction.audio.original_filename,
            original_extension=prediction.audio.original_extension,
            detected_container=prediction.audio.detected_container,
            detected_codec=prediction.audio.detected_codec,
            duration_seconds=prediction.audio.duration_seconds,
            sample_rate=prediction.audio.sample_rate,
            channels=prediction.audio.channels,
            size_bytes=prediction.audio.size_bytes,
        ),
        branches=prediction.branches,
        fusion=prediction.fusion,
        research_eligible=prediction.fusion.eligible_for_research_evaluation,
        created_at=prediction.created_at,
    )


def _final_status(
    prediction: VoicePredictionResponse,
    storage_metadata: AudioStorageMetadata,
    *,
    storage_required: bool = True,
) -> PredictionStatus:
    if (
        (storage_required and storage_metadata.status == BranchStatus.failed)
        or prediction.fusion.status == BranchStatus.failed
    ):
        return PredictionStatus.failed
    return PredictionStatus.completed


def _safe_download_path() -> Path:
    upload_dir = settings.resolved_upload_dir.resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir / f".rerun-{uuid4().hex}.tmp"


def _cloudinary_public_id(document: dict) -> str | None:
    asset = document.get("cloudinary_asset") or {}
    public_id = asset.get("public_id")
    return public_id if isinstance(public_id, str) and public_id.strip() else None


def _source_filename(document: dict) -> str:
    original_filename = document.get("original_filename")
    if isinstance(original_filename, str) and Path(original_filename).suffix:
        return original_filename
    extension = document.get("original_extension") or _cloudinary_format(document)
    if not isinstance(extension, str) or not extension.strip():
        extension = "wav"
    return f"rerun-source.{extension.strip().lower().lstrip('.')}"


def _cloudinary_format(document: dict) -> str | None:
    asset = document.get("cloudinary_asset") or {}
    value = asset.get("format")
    return value if isinstance(value, str) else None


def _source_type(document: dict) -> SourceType:
    return SourceType(document["source_type"])


def _safe_rerun_reason(value: str | None) -> str:
    if value is None or not value.strip():
        return "user_requested"
    return value.strip()[:300]


def _model_versions(voice_service: VoiceService) -> dict:
    versions = {}
    for branch in voice_service.model_health():
        model_name = branch.get("model_name")
        if not model_name:
            continue
        versions[str(model_name)] = {
            "display_name": branch.get("display_name"),
            "mode": branch.get("mode"),
            "version": branch.get("version", "not_available"),
        }
    return versions


def _owner_user_id(principal: AuthPrincipal) -> str:
    return principal.user_id or principal.subject


def _principal_key(principal: AuthPrincipal) -> str:
    if principal.api_key_id:
        return f"api_key:{principal.api_key_id}"
    return f"user:{_owner_user_id(principal)}"


def _raise_not_found() -> None:
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Prediction was not found.",
    )
