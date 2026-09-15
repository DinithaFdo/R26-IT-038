from datetime import datetime
import inspect
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from app.auth.schemas import AuthPrincipal
from app.config.settings import Settings, settings
from app.auth.clerk_auth import dev_auth_bypass_enabled
from app.core.exceptions import (
    IdempotencyConflictError,
    NoUsableModelBranchesError,
)
from app.core.timing import StageTimings, stage_timer
from app.ingestion.audio import (
    AudioUploadMetadata,
    ProcessedAudio,
    save_validated_audio_upload_blocking,
)
from app.repositories.protocols import PredictionRepository
from app.schemas.common import BranchStatus, PredictionStatus, SourceType
from app.schemas.prediction import AudioStorageMetadata, VoicePredictionResponse
from app.schemas.prediction_history import PredictionHistoryAudioMetadata
from app.schemas.prediction_submission import (
    PredictionJobStatusResponse,
    PredictionSubmissionResponse,
)
from app.services.prediction_job_runner import PredictionJobRunner
from app.services.prediction_persistence_service import (
    PredictionPersistenceError,
    PredictionPersistenceService,
    sanitized_error_code,
)
from app.services.voice_service import VoiceService
from app.storage.protocols import AudioStorage
from app.voice_xai.contracts import ClassifierInferenceBundle
from app.voice_xai.capture.contracts import ExtractionBundle
from app.voice_xai.orchestrator import VoiceXaiOrchestrator
from app.voice_xai.temporal.contracts import TemporalAttentionWindowInput

logger = logging.getLogger(__name__)

BROWSER_RECORDING_EXTENSION_BY_CONTENT_TYPE = {
    "audio/webm": "webm",
    "video/webm": "webm",
    "audio/ogg": "ogg",
    "audio/opus": "opus",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/aac": "aac",
}
MAX_IDEMPOTENCY_KEY_LENGTH = 120


class PredictionSubmissionService:
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

    async def submit(
        self,
        *,
        file: UploadFile,
        principal: AuthPrincipal,
        source_type: SourceType,
        request_id: str,
        client_correlation_id: str | None = None,
        client_filename: str | None = None,
        idempotency_key: str | None = None,
    ) -> PredictionSubmissionResponse:
        owner_user_id = _owner_user_id(principal)
        logical_request = _logical_request(
            source_type=source_type,
            client_filename=client_filename,
            upload_filename=file.filename,
        )
        safe_idempotency_key = _sanitize_idempotency_key(idempotency_key)
        reservation = await self._reserve_or_create_prediction(
            principal=principal,
            owner_user_id=owner_user_id,
            source_type=source_type,
            request_id=request_id,
            client_correlation_id=client_correlation_id,
            idempotency_key=safe_idempotency_key,
            logical_request=logical_request,
        )
        if not reservation["created"]:
            document = reservation["document"]
            _validate_replay_logical_request(
                document,
                logical_request,
                app_settings=self._settings,
            )
            return _response_from_document(document)

        prediction_id = reservation["prediction_id"]
        request_id = reservation["request_id"]
        return await self._job_runner.run(
            prediction_id=prediction_id,
            request_id=request_id,
            principal_key=_principal_key(principal),
            persistence=self._persistence,
            execute=lambda: self.execute_prediction(
                file=file,
                principal=principal,
                source_type=source_type,
                prediction_id=prediction_id,
                request_id=request_id,
                client_filename=client_filename,
            ),
        )

    async def execute_prediction(
        self,
        *,
        file: UploadFile,
        principal: AuthPrincipal,
        source_type: SourceType,
        prediction_id: str,
        request_id: str,
        client_filename: str | None = None,
    ) -> PredictionSubmissionResponse:
        upload_metadata = None
        storage_metadata = None
        timings = StageTimings()

        try:
            with stage_timer(
                "database_persistence",
                timings=timings,
                logger_name=__name__,
                request_id=request_id,
            ):
                await self._persistence.mark_validating(request_id)
            try:
                with stage_timer(
                    "upload_stream",
                    timings=timings,
                    logger_name=__name__,
                    request_id=request_id,
                ):
                    upload_metadata = await self._job_runner.run_blocking(
                        lambda: save_validated_audio_upload_blocking(
                            file,
                            app_settings=self._settings,
                            validation_filename=_validation_filename(
                                file=file,
                                source_type=source_type,
                            ),
                            display_filename=_display_filename(
                                source_type=source_type,
                                client_filename=client_filename,
                                extension=_display_extension(file),
                            ),
                        ),
                    )
            except Exception as error:
                await self._persistence.mark_failed(
                    request_id,
                    stage="validation",
                    code=sanitized_error_code(error),
                )
                raise

            with stage_timer(
                "database_persistence",
                timings=timings,
                logger_name=__name__,
                request_id=request_id,
            ):
                await self._persistence.attach_validated_upload(
                    request_id,
                    upload_metadata,
                )
                await self._persistence.mark_storing(request_id)

            with stage_timer(
                "storage_upload",
                timings=timings,
                logger_name=__name__,
                request_id=request_id,
            ):
                storage_metadata = await _store_audio(
                    storage=self._storage,
                    principal=principal,
                    upload_metadata=upload_metadata,
                    request_id=request_id,
                )
            storage_required = self._settings.storage_policy == "required"
            with stage_timer(
                "database_persistence",
                timings=timings,
                logger_name=__name__,
                request_id=request_id,
                storage_status=storage_metadata.status.value,
            ):
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
            with stage_timer(
                "database_persistence",
                timings=timings,
                logger_name=__name__,
                request_id=request_id,
            ):
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
                    prediction_id=prediction_id,
                    request_id=request_id,
                    principal=principal,
                    source_type=source_type,
                    prediction=prediction,
                    processed_audio=processed_audio,
                    extraction=extraction,
                    temporal_evidence=temporal_evidence,
                )

            return _response_from_prediction(
                prediction_id=prediction_id,
                source_type=source_type,
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
            if upload_metadata is not None:
                with stage_timer(
                    "cleanup",
                    timings=timings,
                    logger_name=__name__,
                    request_id=request_id,
                ):
                    upload_metadata.saved_path.unlink(missing_ok=True)

    async def get_status(
        self,
        *,
        principal: AuthPrincipal,
        prediction_id: str,
    ) -> PredictionJobStatusResponse:
        document = await self._repository.get_prediction_for_owner(
            prediction_id=prediction_id,
            owner_user_id=_owner_user_id(principal),
        )
        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Prediction not found.",
            )
        return _status_from_document(document)

    async def _reserve_or_create_prediction(
        self,
        *,
        principal: AuthPrincipal,
        owner_user_id: str,
        source_type: SourceType,
        request_id: str,
        client_correlation_id: str | None,
        idempotency_key: str | None,
        logical_request: dict[str, str | None],
    ) -> dict:
        if idempotency_key is None:
            prediction_id = await self._persistence.create_queued(
                request_id=request_id,
                principal=principal,
                source_type=source_type,
                client_correlation_id=client_correlation_id,
            )
            return {
                "created": True,
                "prediction_id": prediction_id,
                "request_id": request_id,
            }

        reservation = await self._repository.reserve_prediction(
            request_id=request_id,
            owner_user_id=owner_user_id,
            source_type=source_type,
            client_correlation_id=client_correlation_id,
            idempotency_key=idempotency_key,
            logical_request=logical_request,
        )
        if reservation["created"]:
            document = reservation["document"]
            return {
                "created": True,
                "prediction_id": document["id"],
                "request_id": document["request_id"],
            }
        return reservation

    async def _enqueue_xai_without_affecting_prediction(
        self,
        *,
        prediction_id: str,
        request_id: str,
        principal: AuthPrincipal,
        source_type: SourceType,
        prediction: VoicePredictionResponse,
        processed_audio: ProcessedAudio | None = None,
        extraction: ExtractionBundle | None = None,
        temporal_evidence: tuple[TemporalAttentionWindowInput, ...] | None = None,
    ) -> None:
        """XAI is post-persistence best effort and cannot fail the prediction."""

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
                    # Reuses the exact ProcessedAudio instance the classifier
                    # already decoded/normalised for this request -- XAI never
                    # re-reads or re-decodes the upload. `extraction` stays
                    # The optional bundle contains only bounded tensors captured
                    # during the exact classifier forward pass. It remains
                    # internal; the XAI worker serializes it as a private
                    # artifact rather than exposing it through prediction APIs.
                    extraction=extraction,
                    temporal_evidence=temporal_evidence,
                    processed_audio=processed_audio,
                )
            )
        except Exception:
            logger.exception(
                "Voice XAI enqueue failed after completed prediction persistence",
                extra={"request_id": request_id, "prediction_id": prediction_id},
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
    """Run classifier inference, returning the processed audio alongside it.

    Prefers ``predict_from_validated_upload_with_processed_audio`` so the
    same ``ProcessedAudio`` the classifier used can be reused for XAI without
    a second decode. Falls back to the plain public method (returning
    ``processed_audio=None``) for any voice-service test double/fake that
    only implements the original method -- the classifier prediction path
    itself is identical either way. Capture is passed only to implementations
    that opt into the additive internal argument.
    """

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
            storage_status=(
                prediction.audio.storage.status.value
                if prediction.audio.storage is not None
                else None
            ),
            playback_available=(
                prediction.audio.storage is not None
                and prediction.audio.storage.status == BranchStatus.success
            ),
        ),
        branches=prediction.branches,
        fusion=prediction.fusion,
        research_eligible=prediction.fusion.eligible_for_research_evaluation,
        created_at=prediction.created_at,
    )


def _response_from_document(document: dict) -> PredictionSubmissionResponse:
    audio = PredictionHistoryAudioMetadata(
        original_filename=document.get("original_filename"),
        original_extension=document.get("original_extension"),
        detected_container=document.get("detected_container"),
        detected_codec=document.get("detected_codec"),
        duration_seconds=document.get("duration_seconds"),
        sample_rate=document.get("sample_rate"),
        channels=document.get("channels"),
        size_bytes=document.get("size_bytes"),
        storage_status=document.get("storage_status"),
        playback_available=_document_playback_available(document),
    )
    branches = document.get("branches") or []
    fusion = document.get("fusion")
    return PredictionSubmissionResponse(
        prediction_id=document["id"],
        request_id=document["request_id"],
        status=PredictionStatus(document["status"]),
        source_type=SourceType(document["source_type"]),
        audio=audio,
        branches=branches,
        fusion=fusion,
        research_eligible=bool(document.get("research_eligible", False)),
        created_at=_datetime_value(document.get("created_at")),
    )


def _status_from_document(document: dict) -> PredictionJobStatusResponse:
    return PredictionJobStatusResponse(
        prediction_id=document["id"],
        request_id=document["request_id"],
        status=PredictionStatus(document["status"]),
        source_type=SourceType(document["source_type"]),
        created_at=_datetime_value(document.get("created_at")),
        updated_at=_datetime_value(document.get("updated_at")),
        completed_at=document.get("completed_at"),
        error_summary=document.get("error_summary") or [],
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


def _validation_filename(
    *,
    file: UploadFile,
    source_type: SourceType,
) -> str | None:
    if source_type == SourceType.dashboard_upload:
        return None
    filename = file.filename or ""
    if Path(filename).suffix:
        return filename
    extension = _extension_from_content_type(file.content_type)
    if extension is None:
        return None
    return f"browser-recording.{extension}"


def _display_filename(
    *,
    source_type: SourceType,
    client_filename: str | None,
    extension: str,
) -> str | None:
    if source_type == SourceType.dashboard_upload:
        return client_filename
    if client_filename and client_filename.strip():
        return client_filename
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M")
    return f"Recording {timestamp}.{extension}"


def _display_extension(file: UploadFile) -> str:
    filename = file.filename or ""
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension:
        return extension
    return _extension_from_content_type(file.content_type) or "webm"


def _extension_from_content_type(content_type: str | None) -> str | None:
    if content_type is None:
        return None
    return BROWSER_RECORDING_EXTENSION_BY_CONTENT_TYPE.get(
        content_type.split(";")[0].strip().lower()
    )


def _sanitize_idempotency_key(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    sanitized = value.strip()
    if len(sanitized) > MAX_IDEMPOTENCY_KEY_LENGTH:
        raise HTTPException(
            status_code=422,
            detail="idempotency_key is too long.",
        )
    return sanitized


def _logical_request(
    *,
    source_type: SourceType,
    client_filename: str | None,
    upload_filename: str | None = None,
) -> dict[str, str | None]:
    return {
        "source_type": source_type.value,
        "client_filename": client_filename.strip() if client_filename else None,
        "submitted_filename": _safe_logical_filename(upload_filename),
    }


def _validate_replay_logical_request(
    document: dict,
    logical_request: dict[str, str | None],
    *,
    app_settings: Settings = settings,
) -> None:
    """Reject a key that is being reused for a different logical request.

    Idempotency is owner-scoped, so a stored key only ever replays when the
    logical request matches exactly. Anything else is a client bug (typically a
    key copied from a previous upload) and must never silently return another
    submission's prediction.
    """

    stored = document.get("idempotency_logical_request")
    if stored == logical_request:
        return
    raise IdempotencyConflictError(
        details=_idempotency_conflict_details(
            document=document,
            stored=stored,
            logical_request=logical_request,
            app_settings=app_settings,
        )
    )


def _idempotency_conflict_details(
    *,
    document: dict,
    stored: object,
    logical_request: dict[str, str | None],
    app_settings: Settings,
) -> dict | None:
    """Local-development-only diagnostics for a rejected key reuse.

    Production responses keep ``details: null`` -- the conflict itself is the
    contract. Under the local Swagger auth bypass every request shares one
    synthetic owner, so showing which field changed (and the prediction the key
    is already bound to) is what makes a stale key obvious instead of baffling.
    """

    if not dev_auth_bypass_enabled(app_settings):
        return None
    stored_request = stored if isinstance(stored, dict) else {}
    changed = sorted(
        {
            field
            for field in set(stored_request) | set(logical_request)
            if stored_request.get(field) != logical_request.get(field)
        }
    )
    return {
        "development_hint": (
            "This idempotency_key is already bound to a different logical "
            "request. Leave idempotency_key blank or send a new UUID."
        ),
        "changed_fields": changed,
        "existing_prediction_id": document.get("id"),
        "existing_logical_request": stored_request or None,
        "submitted_logical_request": logical_request,
    }


def _owner_user_id(principal: AuthPrincipal) -> str:
    return principal.user_id or principal.subject


def _principal_key(principal: AuthPrincipal) -> str:
    if principal.api_key_id:
        return f"api_key:{principal.api_key_id}"
    return f"user:{_owner_user_id(principal)}"


def _safe_logical_filename(filename: str | None) -> str | None:
    if filename is None or not filename.strip():
        return None
    name = Path(filename.strip()).name
    if not name or name != filename.strip():
        return None
    return name[:120]


def _datetime_value(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.now()


def _document_playback_available(document: dict) -> bool:
    asset = document.get("cloudinary_asset") or {}
    return bool(asset.get("public_id"))
