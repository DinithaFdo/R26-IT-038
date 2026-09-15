from pathlib import Path

import pytest

from app.config.settings import Settings
from app.voice_xai.replay import (
    StoredAudioReplayService,
    XaiSourceAudioUnavailableError,
)


class RecordingStorage:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path, str, int]] = []

    async def download_audio(
        self,
        public_id: str,
        destination_path: Path,
        *,
        owner_user_id: str,
        max_bytes: int,
    ) -> None:
        self.calls.append((public_id, destination_path, owner_user_id, max_bytes))
        destination_path.write_bytes(b"validated-audio")


@pytest.mark.anyio
async def test_replay_rebuilds_processed_audio_from_owner_scoped_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        upload_dir=str(tmp_path),
        max_upload_size_mb=3,
    )
    storage = RecordingStorage()
    expected_processed_audio = object()

    def preprocess(path: Path, extension: str, app_settings: Settings):
        assert path.exists()
        assert extension == "wav"
        assert app_settings is settings
        return expected_processed_audio

    monkeypatch.setattr("app.voice_xai.replay.preprocess_audio_file", preprocess)
    service = StoredAudioReplayService(storage=storage, app_settings=settings)

    result = await service.load_processed_audio(
        prediction_document={
            "cloudinary_asset": {"public_id": "voice/user-123/prediction-123"},
            "original_extension": "wav",
        },
        owner_user_id="user-123",
    )

    assert result is expected_processed_audio
    assert storage.calls == [
        (
            "voice/user-123/prediction-123",
            storage.calls[0][1],
            "user-123",
            3 * 1024 * 1024,
        )
    ]
    assert not storage.calls[0][1].exists()


@pytest.mark.anyio
async def test_replay_rejects_prediction_without_persisted_source_audio(
    tmp_path: Path,
) -> None:
    service = StoredAudioReplayService(
        storage=RecordingStorage(),
        app_settings=Settings(_env_file=None, upload_dir=str(tmp_path)),
    )

    with pytest.raises(XaiSourceAudioUnavailableError, match="Source audio is unavailable"):
        await service.load_processed_audio(
            prediction_document={"original_extension": "wav"},
            owner_user_id="user-123",
        )
