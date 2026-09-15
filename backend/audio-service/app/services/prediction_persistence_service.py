import logging
from typing import Any

from app.auth.schemas import AuthPrincipal
from app.ingestion.audio import AudioUploadMetadata
from app.schemas.common import BranchStatus, PredictionStatus, SourceType
from app.schemas.prediction import AudioStorageMetadata, VoicePredictionResponse
from app.storage.protocols import AudioStorage
from app.storage.protocols import StorageReconciliationEvent

logger = logging.getLogger(__name__)


class PredictionPersistenceError(Exception):
    """Raised when prediction persistence fails after external side effects."""


class PredictionPersistenceService:
    def __init__(self, repository) -> None:
        self._repository = repository

    async def create_queued(
        self,
        *,
        request_id: str,
        principal: AuthPrincipal,
        source_type: SourceType = SourceType.dashboard_upload,
        client_correlation_id: str | None = None,
        idempotency_key: str | None = None,
        logical_request: dict[str, Any] | None = None,
        parent_prediction_id: str | None = None,
        rerun_reason: str | None = None,
        preprocessing_version: str | None = None,
        model_versions: dict[str, Any] | None = None,
    ) -> str:
        return await self._call(
            self._repository.create_prediction(
                request_id=request_id,
                owner_user_id=principal.user_id or principal.subject,
                source_type=source_type,
                client_correlation_id=client_correlation_id,
                idempotency_key=idempotency_key,
                logical_request=logical_request,
                parent_prediction_id=parent_prediction_id,
                rerun_reason=rerun_reason,
                preprocessing_version=preprocessing_version,
                model_versions=model_versions,
            )
        )

    async def create_uploaded(self, **kwargs) -> str:
        return await self.create_queued(**kwargs)

    async def mark_validating(self, request_id: str) -> None:
        await self._transition(request_id, PredictionStatus.validating)

    async def attach_validated_upload(
        self,
        request_id: str,
        upload_metadata: AudioUploadMetadata,
    ) -> None:
        await self._call(
            self._repository.attach_upload_metadata(request_id, upload_metadata)
        )

    async def mark_storing(self, request_id: str) -> None:
        await self._transition(request_id, PredictionStatus.storing)

    async def attach_storage(
        self,
        request_id: str,
        storage_metadata: AudioStorageMetadata,
        *,
        storage_required: bool = True,
    ) -> None:
        await self._call(
            self._repository.attach_cloudinary_asset(request_id, storage_metadata)
        )
        if storage_metadata.status == BranchStatus.failed and storage_required:
            await self.mark_failed(
                request_id,
                stage="storage",
                code="cloudinary_upload_failed",
            )
        if storage_metadata.status == BranchStatus.skipped and storage_required:
            await self.mark_failed(
                request_id,
                stage="storage",
                code="cloudinary_upload_skipped",
            )

    async def mark_processing(self, request_id: str) -> None:
        await self._transition(request_id, PredictionStatus.processing)

    async def save_result(
        self,
        request_id: str,
        response: VoicePredictionResponse,
        *,
        storage_metadata: AudioStorageMetadata,
        storage_required: bool = True,
    ) -> None:
        error_codes = _result_error_codes(response)
        status = (
            PredictionStatus.failed
            if (
                (storage_required and storage_metadata.status == BranchStatus.failed)
                or response.fusion.status == BranchStatus.failed
            )
            else PredictionStatus.completed
        )
        await self._call(
            self._repository.save_prediction_result(
                request_id,
                response,
                status=status,
                error_codes=error_codes,
            )
        )

    async def mark_failed(
        self,
        request_id: str,
        *,
        stage: str,
        code: str,
    ) -> None:
        await self._call(
            self._repository.update_prediction_status(
                request_id,
                PredictionStatus.failed,
                error_code=code,
                error_stage=stage,
            )
        )

    async def compensate_cloudinary_upload(
        self,
        *,
        storage: AudioStorage,
        storage_metadata: AudioStorageMetadata | None,
    ) -> None:
        if storage_metadata is None or not storage_metadata.public_id:
            return
        try:
            await storage.delete_audio(storage_metadata.public_id)
        except Exception:
            logger.exception("Cloudinary compensating cleanup failed.")

    async def record_storage_reconciliation_event(
        self,
        event: StorageReconciliationEvent,
    ) -> None:
        request_id = event.request_id or _request_id_from_public_id(event.public_id)
        if request_id is None:
            return
        recorder = getattr(
            self._repository,
            "record_storage_reconciliation_event",
            None,
        )
        if not callable(recorder):
            return
        try:
            await recorder(request_id=request_id, event=event)
        except Exception:
            logger.exception(
                "Storage reconciliation persistence failed.",
                extra={"request_id": request_id, "event_type": event.event_type},
            )

    async def _transition(
        self,
        request_id: str,
        status: PredictionStatus,
    ) -> None:
        await self._call(
            self._repository.update_prediction_status(request_id, status)
        )

    async def _call(self, awaitable):
        try:
            return await awaitable
        except Exception as error:
            raise PredictionPersistenceError(
                "Prediction persistence failed."
            ) from error


def sanitized_error_code(error: Exception) -> str:
    name = type(error).__name__
    code = []
    for index, character in enumerate(name):
        if character.isupper() and index > 0:
            code.append("_")
        code.append(character.lower())
    return "".join(code).removesuffix("_error")


def _result_error_codes(
    response: VoicePredictionResponse,
) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for branch in response.branches:
        if branch.status == BranchStatus.failed:
            errors.append(_error("model_branch", f"{branch.model_name}_failed"))
    if response.fusion.status == BranchStatus.failed:
        errors.append(_error("fusion", "fusion_failed"))
    return errors


def _error(stage: str, code: str) -> dict[str, str]:
    return {
        "stage": stage,
        "code": code,
    }


def _request_id_from_public_id(public_id: str | None) -> str | None:
    if not public_id:
        return None
    value = public_id.rsplit("/", 1)[-1].strip()
    return value or None
