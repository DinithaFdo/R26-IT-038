"""Durable source-audio replay for manually triggered Voice XAI runs.

The normal prediction path passes ``ProcessedAudio`` directly to Voice XAI.
Manual triggers and retries occur after that in-memory object has gone away,
so they must retrieve the owner-scoped source asset and preprocess it again.
This module deliberately does *not* rerun or persist a classifier prediction:
it only rebuilds the bounded, ephemeral audio input required by XAI services.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4

from app.config.settings import Settings, settings
from app.ingestion.audio import ProcessedAudio, preprocess_audio_file
from app.storage.protocols import AudioStorage


class XaiSourceAudioUnavailableError(RuntimeError):
    """Raised when a persisted prediction cannot safely be replayed for XAI."""


class StoredAudioReplayService:
    """Download and preprocess an owner's persisted source audio temporarily."""

    def __init__(
        self,
        *,
        storage: AudioStorage,
        app_settings: Settings = settings,
    ) -> None:
        self._storage = storage
        self._settings = app_settings

    async def load_processed_audio(
        self,
        *,
        prediction_document: dict,
        owner_user_id: str,
    ) -> ProcessedAudio:
        public_id = _cloudinary_public_id(prediction_document)
        if public_id is None:
            raise XaiSourceAudioUnavailableError(
                "Source audio is unavailable for this explanation."
            )

        extension = _source_extension(prediction_document, self._settings)
        destination = self._temporary_download_path(extension)
        try:
            await self._storage.download_audio(
                public_id,
                destination,
                owner_user_id=owner_user_id,
                max_bytes=self._settings.max_upload_size_mb * 1024 * 1024,
            )
            return await asyncio.to_thread(
                preprocess_audio_file,
                destination,
                extension,
                self._settings,
            )
        except XaiSourceAudioUnavailableError:
            raise
        except (FileNotFoundError, PermissionError) as error:
            raise XaiSourceAudioUnavailableError(
                "Source audio is unavailable for this explanation."
            ) from error
        except Exception as error:
            raise XaiSourceAudioUnavailableError(
                "Source audio could not be prepared for this explanation."
            ) from error
        finally:
            destination.unlink(missing_ok=True)

    def _temporary_download_path(self, extension: str) -> Path:
        upload_dir = self._settings.resolved_upload_dir.resolve()
        upload_dir.mkdir(parents=True, exist_ok=True)
        return upload_dir / f".xai-replay-{uuid4().hex}.{extension}"


def _cloudinary_public_id(document: dict) -> str | None:
    asset = document.get("cloudinary_asset") or {}
    public_id = asset.get("public_id")
    return public_id.strip() if isinstance(public_id, str) and public_id.strip() else None


def _source_extension(document: dict, app_settings: Settings) -> str:
    candidates = (
        document.get("original_extension"),
        (document.get("cloudinary_asset") or {}).get("format"),
        Path(str(document.get("original_filename") or "")).suffix.lstrip("."),
    )
    for value in candidates:
        if isinstance(value, str):
            extension = value.strip().lower().lstrip(".")
            if extension in app_settings.allowed_audio_extension_list:
                return extension
    raise XaiSourceAudioUnavailableError(
        "Source audio has no supported format for this explanation."
    )
