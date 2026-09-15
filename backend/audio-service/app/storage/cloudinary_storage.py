from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from app.config.settings import Settings, settings
from app.core.timing import stage_timer
from app.schemas.common import BranchStatus
from app.schemas.prediction import AudioStorageMetadata
from app.storage.protocols import StorageReconciliationEvent

logger = logging.getLogger(__name__)


class CloudinaryAudioStorage:
    def __init__(
        self,
        app_settings: Settings = settings,
        *,
        reconciliation_recorder: Callable[
            [StorageReconciliationEvent],
            Awaitable[None],
        ]
        | None = None,
    ) -> None:
        self._settings = app_settings
        self._reconciliation_recorder = reconciliation_recorder
        self._reconciliation_backlog: list[StorageReconciliationEvent] = []
        self._reconciliation_tasks: set[asyncio.Task] = set()

    def set_reconciliation_recorder(
        self,
        recorder: Callable[[StorageReconciliationEvent], Awaitable[None]] | None,
    ) -> None:
        self._reconciliation_recorder = recorder

    async def upload_audio(
        self,
        source_path: Path,
        *,
        owner_user_id: str,
        audio_id: str,
        content_type: str | None = None,
    ) -> AudioStorageMetadata:
        if not self._is_configured():
            return _failed_storage_metadata("Audio storage is unavailable.")

        public_id = self._public_id(owner_user_id=owner_user_id, audio_id=audio_id)
        upload_ownership = {"prediction_owns_asset": True}
        upload_task = asyncio.create_task(
            asyncio.to_thread(
                self._upload_sync,
                source_path,
                public_id,
            )
        )
        upload_task.add_done_callback(
            lambda task: self._schedule_upload_completion_reconciliation(
                task=task,
                public_id=public_id,
                owner_user_id=owner_user_id,
                prediction_owns_asset=upload_ownership["prediction_owns_asset"],
            )
        )
        try:
            result = await asyncio.wait_for(
                asyncio.shield(upload_task),
                timeout=self._settings.cloudinary_upload_timeout_seconds,
            )
        except TimeoutError:
            upload_ownership["prediction_owns_asset"] = False
            logger.warning(
                "Cloudinary audio upload timed out with ambiguous outcome.",
                extra={"owner_user_id": owner_user_id, "audio_id": audio_id},
            )
            return _failed_storage_metadata(
                "Audio storage timed out; outcome is being reconciled.",
                public_id=public_id,
            )
        except Exception:
            logger.exception(
                "Cloudinary audio upload failed.",
                extra={"owner_user_id": owner_user_id, "audio_id": audio_id},
            )
            return _failed_storage_metadata("Audio storage failed.")

        return AudioStorageMetadata(
            status=BranchStatus.success,
            asset_id=_optional_string(result.get("asset_id")),
            public_id=_optional_string(result.get("public_id")),
            resource_type=_optional_string(result.get("resource_type")),
            version=_optional_int(result.get("version")),
            format=_optional_string(result.get("format")),
            bytes=_optional_int(result.get("bytes")),
            duration=_optional_float(result.get("duration")),
            created_at=_parse_cloudinary_datetime(result.get("created_at")),
            error=None,
        )

    async def delete_audio(self, public_id: str) -> None:
        if not self._is_configured():
            return None

        await asyncio.to_thread(self._delete_sync, public_id)

    async def reconcile_upload(
        self,
        *,
        public_id: str,
        owner_user_id: str,
        prediction_owns_asset: bool = False,
    ) -> StorageReconciliationEvent:
        """Reconcile a deterministic Cloudinary public ID after ambiguous upload.

        This function is safe to run from a later background job. It does not
        assume that an asyncio timeout cancelled the underlying SDK upload.
        """

        with stage_timer(
            "storage_reconciliation",
            logger_name=__name__,
        ):
            return await self._reconcile_upload_inner(
                public_id=public_id,
                owner_user_id=owner_user_id,
                prediction_owns_asset=prediction_owns_asset,
            )

    async def _reconcile_upload_inner(
        self,
        *,
        public_id: str,
        owner_user_id: str,
        prediction_owns_asset: bool,
    ) -> StorageReconciliationEvent:
        if not self._is_configured():
            return StorageReconciliationEvent(
                event_type="cloudinary_reconciliation_storage_unavailable",
                public_id=public_id,
                owner_user_id=owner_user_id,
                asset_exists=False,
                deleted=False,
                request_id=_request_id_from_public_id(public_id),
            )
        if not _public_id_belongs_to_owner(public_id, owner_user_id):
            return StorageReconciliationEvent(
                event_type="cloudinary_reconciliation_owner_mismatch",
                public_id=public_id,
                owner_user_id=owner_user_id,
                asset_exists=False,
                deleted=False,
                request_id=_request_id_from_public_id(public_id),
            )

        asset = await asyncio.to_thread(self._resource_sync, public_id)
        if asset is None:
            event = StorageReconciliationEvent(
                event_type="cloudinary_reconciliation_no_asset",
                public_id=public_id,
                owner_user_id=owner_user_id,
                asset_exists=False,
                deleted=False,
                request_id=_request_id_from_public_id(public_id),
            )
            await self._record_reconciliation_event(event)
            _log_reconciliation_event(event)
            return event

        if prediction_owns_asset:
            event = StorageReconciliationEvent(
                event_type="cloudinary_reconciliation_asset_owned",
                public_id=public_id,
                owner_user_id=owner_user_id,
                asset_exists=True,
                deleted=False,
                request_id=_request_id_from_public_id(public_id),
            )
            await self._record_reconciliation_event(event)
            _log_reconciliation_event(event)
            return event

        try:
            await self.delete_audio(public_id)
        except Exception:
            logger.exception(
                "Cloudinary reconciliation deletion failed.",
                extra={"owner_user_id": owner_user_id},
            )
            event = StorageReconciliationEvent(
                event_type="cloudinary_reconciliation_delete_failed",
                public_id=public_id,
                owner_user_id=owner_user_id,
                asset_exists=True,
                deleted=False,
                deletion_failed=True,
                retry_required=True,
                request_id=_request_id_from_public_id(public_id),
            )
            await self._record_reconciliation_event(event)
            _log_reconciliation_event(event)
            return event

        event = StorageReconciliationEvent(
            event_type="cloudinary_reconciliation_orphan_deleted",
            public_id=public_id,
            owner_user_id=owner_user_id,
            asset_exists=True,
            deleted=True,
            request_id=_request_id_from_public_id(public_id),
        )
        await self._record_reconciliation_event(event)
        _log_reconciliation_event(event)
        return event

    async def download_audio(
        self,
        public_id: str,
        destination_path: Path,
        *,
        owner_user_id: str,
        max_bytes: int,
    ) -> None:
        if not _public_id_belongs_to_owner(public_id, owner_user_id):
            raise PermissionError("Audio asset does not belong to this owner.")
        if not self._is_configured():
            raise RuntimeError("Audio storage is unavailable.")

        playback_url = await self.generate_signed_playback_url(
            public_id,
            owner_user_id=owner_user_id,
            expires_in_seconds=120,
        )
        await _download_signed_url(
            playback_url,
            destination_path,
            max_bytes=max_bytes,
            timeout_seconds=self._settings.cloudinary_upload_timeout_seconds,
        )

    async def generate_signed_playback_url(
        self,
        public_id: str,
        *,
        owner_user_id: str,
        expires_in_seconds: int = 300,
    ) -> str:
        if not _public_id_belongs_to_owner(public_id, owner_user_id):
            raise PermissionError("Audio asset does not belong to this owner.")
        if not self._is_configured():
            raise RuntimeError("Audio storage is unavailable.")

        expires_at = int(
            (datetime.now(UTC) + timedelta(seconds=expires_in_seconds)).timestamp()
        )
        return await asyncio.to_thread(
            self._signed_url_sync,
            public_id,
            expires_at,
        )

    async def health(self) -> dict[str, bool]:
        enabled = self._settings.cloudinary_storage_enabled
        return {
            "storage_enabled": enabled,
            "storage_available": enabled and self._is_configured(),
            "storage_reconciliation_backlog": len(self._reconciliation_backlog),
        }

    async def reconcile_pending_uploads(self) -> list[StorageReconciliationEvent]:
        events: list[StorageReconciliationEvent] = []
        pending = list(self._reconciliation_backlog)
        self._reconciliation_backlog.clear()
        for event in pending:
            events.append(
                await self.reconcile_upload(
                    public_id=event.public_id,
                    owner_user_id=event.owner_user_id,
                    prediction_owns_asset=False,
                )
            )
        return events

    def _upload_sync(self, source_path: Path, public_id: str) -> dict[str, Any]:
        cloudinary, uploader, _utils, _api = self._load_cloudinary_modules()
        self._configure(cloudinary)
        return uploader.upload(
            str(source_path),
            resource_type="video",
            type="authenticated",
            public_id=public_id,
            overwrite=False,
            timeout=self._settings.cloudinary_sdk_timeout_seconds,
        )

    def _delete_sync(self, public_id: str) -> None:
        cloudinary, uploader, _utils, _api = self._load_cloudinary_modules()
        self._configure(cloudinary)
        result = uploader.destroy(
            public_id,
            resource_type="video",
            type="authenticated",
            invalidate=True,
        )
        deletion_result = str((result or {}).get("result", "")).lower()
        if deletion_result not in {"ok", "not found", "not_found"}:
            raise RuntimeError("Cloudinary deletion was not confirmed.")

    def _resource_sync(self, public_id: str) -> dict[str, Any] | None:
        cloudinary, _uploader, _utils, api = self._load_cloudinary_modules()
        self._configure(cloudinary)
        try:
            return api.resource(
                public_id,
                resource_type="video",
                type="authenticated",
            )
        except Exception as error:
            if _is_cloudinary_not_found(error):
                return None
            logger.exception("Cloudinary reconciliation lookup failed.")
            return None

    def _signed_url_sync(self, public_id: str, expires_at: int) -> str:
        cloudinary, _uploader, utils, _api = self._load_cloudinary_modules()
        self._configure(cloudinary)
        url, _options = utils.cloudinary_url(
            public_id,
            resource_type="video",
            type="authenticated",
            secure=True,
            sign_url=True,
            expires_at=expires_at,
        )
        return url

    def _configure(self, cloudinary) -> None:
        cloudinary.config(
            cloud_name=self._settings.cloudinary_cloud_name,
            api_key=self._settings.cloudinary_api_key,
            api_secret=self._settings.cloudinary_api_secret,
            secure=True,
        )

    def _is_configured(self) -> bool:
        return (
            self._settings.cloudinary_storage_enabled
            and bool(self._settings.cloudinary_cloud_name.strip())
            and bool(self._settings.cloudinary_api_key.strip())
            and bool(self._settings.cloudinary_api_secret.strip())
        )

    def _public_id(self, *, owner_user_id: str, audio_id: str) -> str:
        safe_owner = _safe_public_id_part(owner_user_id)
        safe_audio_id = _safe_public_id_part(audio_id)
        folder = self._settings.cloudinary_audio_folder.strip().strip("/")
        return f"{folder}/{safe_owner}/{safe_audio_id}"

    @staticmethod
    def _load_cloudinary_modules():
        try:
            import cloudinary
            import cloudinary.api
            import cloudinary.uploader
            import cloudinary.utils
        except ImportError as error:
            raise RuntimeError("Cloudinary SDK is not installed.") from error
        return cloudinary, cloudinary.uploader, cloudinary.utils, cloudinary.api

    def _schedule_upload_completion_reconciliation(
        self,
        *,
        task: asyncio.Task,
        public_id: str,
        owner_user_id: str,
        prediction_owns_asset: bool,
    ) -> None:
        reconciliation_task = asyncio.create_task(
            self._reconcile_upload_completion(
                task=task,
                public_id=public_id,
                owner_user_id=owner_user_id,
                prediction_owns_asset=prediction_owns_asset,
            )
        )
        self._reconciliation_tasks.add(reconciliation_task)
        reconciliation_task.add_done_callback(self._reconciliation_tasks.discard)

    async def _reconcile_upload_completion(
        self,
        *,
        task: asyncio.Task,
        public_id: str,
        owner_user_id: str,
        prediction_owns_asset: bool,
    ) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Cloudinary upload worker finished with an error.")
            return

        event = await self.reconcile_upload(
            public_id=public_id,
            owner_user_id=owner_user_id,
            prediction_owns_asset=prediction_owns_asset and not task.cancelled(),
        )
        if event.deletion_failed or event.retry_required:
            self._record_reconciliation_retry(event)

    async def _record_reconciliation_event(
        self,
        event: StorageReconciliationEvent,
    ) -> None:
        if event.retry_required or event.deletion_failed:
            self._record_reconciliation_retry(event)
        if self._reconciliation_recorder is None:
            return
        try:
            await self._reconciliation_recorder(event)
        except Exception:
            logger.exception(
                "Cloudinary reconciliation event persistence failed.",
                extra={"event_type": event.event_type},
            )

    def _record_reconciliation_retry(self, event: StorageReconciliationEvent) -> None:
        if any(
            existing.public_id == event.public_id
            and existing.owner_user_id == event.owner_user_id
            for existing in self._reconciliation_backlog
        ):
            return
        self._reconciliation_backlog.append(event)


def _failed_storage_metadata(
    message: str,
    *,
    public_id: str | None = None,
) -> AudioStorageMetadata:
    return AudioStorageMetadata(
        status=BranchStatus.failed,
        public_id=public_id,
        error=message,
    )


def _log_late_upload_failure(task: asyncio.Task) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("Cloudinary upload worker finished with an error.")


def _log_reconciliation_event(event: StorageReconciliationEvent) -> None:
    logger.info(
        "Cloudinary upload reconciliation event recorded.",
        extra={
            "event_type": event.event_type,
            "owner_user_id": event.owner_user_id,
            "asset_exists": event.asset_exists,
            "deleted": event.deleted,
            "deletion_failed": event.deletion_failed,
        },
    )


def _is_cloudinary_not_found(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None) or getattr(error, "http_code", None)
    if status_code == 404:
        return True
    name = type(error).__name__.lower()
    return "notfound" in name or "not_found" in name


def _public_id_belongs_to_owner(public_id: str, owner_user_id: str) -> bool:
    safe_owner = _safe_public_id_part(owner_user_id)
    return f"/{safe_owner}/" in f"/{public_id}/"


def _request_id_from_public_id(public_id: str | None) -> str | None:
    if not public_id:
        return None
    value = public_id.rsplit("/", 1)[-1].strip()
    return value or None


def _safe_public_id_part(value: str) -> str:
    cleaned = "".join(
        character
        if character.isalnum() or character in {"_", "-"}
        else "_"
        for character in value.strip()
    )
    if not cleaned:
        raise ValueError("Cloudinary public ID component is empty.")
    return cleaned[:128]


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip()
    return parsed or None


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_cloudinary_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    parsed = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(parsed).astimezone(UTC)
    except ValueError:
        return None


async def _download_signed_url(
    url: str,
    destination_path: Path,
    *,
    max_bytes: int,
    timeout_seconds: int,
) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with destination_path.open("wb") as output_file:
                    async for chunk in response.aiter_bytes():
                        if not chunk:
                            continue
                        bytes_written += len(chunk)
                        if bytes_written > max_bytes:
                            raise ValueError("Downloaded audio exceeds size limit.")
                        output_file.write(chunk)
    except Exception:
        destination_path.unlink(missing_ok=True)
        raise

    if bytes_written == 0:
        destination_path.unlink(missing_ok=True)
        raise FileNotFoundError("Downloaded audio is empty.")
