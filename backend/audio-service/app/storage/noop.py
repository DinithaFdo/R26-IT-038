from pathlib import Path

from app.schemas.common import BranchStatus
from app.schemas.prediction import AudioStorageMetadata
from app.storage.protocols import StorageReconciliationEvent


class NoOpAudioStorage:
    def __init__(self, *, reason: str = "Audio storage is disabled.") -> None:
        self._reason = reason

    async def upload_audio(
        self,
        source_path: Path,
        *,
        owner_user_id: str,
        audio_id: str,
        content_type: str | None = None,
    ) -> AudioStorageMetadata:
        return AudioStorageMetadata(
            status=BranchStatus.skipped,
            error=self._reason,
        )

    async def delete_audio(self, public_id: str) -> None:
        return None

    async def reconcile_upload(
        self,
        *,
        public_id: str,
        owner_user_id: str,
        prediction_owns_asset: bool = False,
    ) -> StorageReconciliationEvent:
        return StorageReconciliationEvent(
            event_type="cloudinary_reconciliation_storage_disabled",
            public_id=public_id,
            owner_user_id=owner_user_id,
            asset_exists=False,
            deleted=False,
        )

    async def download_audio(
        self,
        public_id: str,
        destination_path: Path,
        *,
        owner_user_id: str,
        max_bytes: int,
    ) -> None:
        raise FileNotFoundError("Audio storage is disabled.")

    async def generate_signed_playback_url(
        self,
        public_id: str,
        *,
        owner_user_id: str,
        expires_in_seconds: int = 300,
    ) -> str:
        raise PermissionError("Audio storage is disabled.")

    async def health(self) -> dict[str, bool]:
        return {"storage_enabled": False, "storage_available": False}
