from pathlib import Path
from dataclasses import dataclass
from typing import Any, Protocol

from app.schemas.prediction import AudioStorageMetadata


@dataclass(frozen=True)
class StorageReconciliationEvent:
    event_type: str
    public_id: str
    owner_user_id: str
    asset_exists: bool
    deleted: bool
    deletion_failed: bool = False
    retry_required: bool = False
    request_id: str | None = None


class AudioStorage(Protocol):
    """Object-storage contract for persisted owner-scoped audio artifacts."""

    async def upload_audio(
        self,
        source_path: Path,
        *,
        owner_user_id: str,
        audio_id: str,
        content_type: str | None = None,
    ) -> AudioStorageMetadata:
        raise NotImplementedError

    async def delete_audio(self, public_id: str) -> None:
        raise NotImplementedError

    async def reconcile_upload(
        self,
        *,
        public_id: str,
        owner_user_id: str,
        prediction_owns_asset: bool = False,
    ) -> StorageReconciliationEvent:
        raise NotImplementedError

    async def download_audio(
        self,
        public_id: str,
        destination_path: Path,
        *,
        owner_user_id: str,
        max_bytes: int,
    ) -> None:
        raise NotImplementedError

    async def generate_signed_playback_url(
        self,
        public_id: str,
        *,
        owner_user_id: str,
        expires_in_seconds: int = 300,
    ) -> str:
        raise NotImplementedError

    async def health(self) -> dict[str, Any]:
        raise NotImplementedError
