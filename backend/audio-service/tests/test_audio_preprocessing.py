import asyncio
from io import BytesIO
import shutil
import subprocess
import wave

import numpy as np
import pytest
from starlette.datastructures import Headers, UploadFile

from app.config.settings import Settings
from app.core.exceptions import (
    AudioDurationExceededError,
    AudioFileTooLargeError,
    AudioProcessingTimeoutError,
    AudioTooShortError,
    CorruptedAudioError,
    EmptyAudioFileError,
    SilentAudioError,
    UnsafeAudioFilenameError,
    UnsupportedAudioFormatError,
    UnusableAudioError,
)
from app.ingestion.audio import (
    NORMALIZED_TARGET_PEAK,
    preprocess_audio_file,
    preprocess_waveform,
    save_validated_audio_upload,
    segment_waveform,
)

AUDIO_TOOLS_AVAILABLE = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
requires_audio_tools = pytest.mark.skipif(
    not AUDIO_TOOLS_AVAILABLE,
    reason="FFmpeg and ffprobe are not available in the test environment.",
)


def make_upload_file(
    *,
    filename: str,
    content: bytes,
    content_type: str = "audio/wav",
) -> UploadFile:
    return UploadFile(
        file=BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": content_type}),
    )


def make_settings(tmp_path, **overrides) -> Settings:
    values = {
        "upload_dir": str(tmp_path / "uploads"),
        "max_upload_size_mb": 25,
        "max_audio_duration_seconds": 60,
        "min_audio_duration_seconds": 0.05,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_wav_bytes(
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    duration_seconds: float = 0.1,
    tone_hz: float | None = None,
) -> bytes:
    frame_count = int(sample_rate * duration_seconds)
    if tone_hz is None:
        samples = np.zeros((frame_count, channels), dtype=np.int16)
    else:
        time = np.arange(frame_count, dtype=np.float32) / sample_rate
        mono = 0.25 * np.sin(2 * np.pi * tone_hz * time)
        samples = np.repeat(mono[:, None], channels, axis=1)
        samples = np.asarray(samples * 32767, dtype=np.int16)

    buffer = BytesIO()
    with wave.open(buffer, "wb") as audio_file:
        audio_file.setnchannels(channels)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(samples.tobytes())
    return buffer.getvalue()


def save_upload(upload_file: UploadFile, settings: Settings):
    return asyncio.run(save_validated_audio_upload(upload_file, settings))


@requires_audio_tools
def test_valid_wav_upload_is_saved_with_uuid_filename(tmp_path) -> None:
    settings = make_settings(tmp_path)
    upload = make_upload_file(filename="sample.wav", content=make_wav_bytes())

    metadata = save_upload(upload, settings)

    assert metadata.original_filename == "sample.wav"
    assert metadata.sanitized_filename == "sample.wav"
    assert metadata.saved_filename.endswith(".wav")
    assert metadata.saved_filename != "sample.wav"
    assert metadata.saved_path.exists()
    assert metadata.saved_path.parent == settings.resolved_upload_dir
    assert metadata.file_size_bytes > 0
    assert metadata.sample_rate == 16000
    assert metadata.channels == 1
    assert metadata.duration_seconds <= settings.max_audio_duration_seconds


def test_empty_file_is_rejected_and_not_saved(tmp_path) -> None:
    settings = make_settings(tmp_path)
    upload = make_upload_file(filename="empty.wav", content=b"")

    with pytest.raises(EmptyAudioFileError):
        save_upload(upload, settings)

    assert list(settings.resolved_upload_dir.iterdir()) == []


def test_unsupported_extension_is_rejected(tmp_path) -> None:
    settings = make_settings(tmp_path)
    upload = make_upload_file(filename="sample.txt", content=make_wav_bytes())

    with pytest.raises(UnsupportedAudioFormatError):
        save_upload(upload, settings)

    assert not settings.resolved_upload_dir.exists()


@requires_audio_tools
def test_extension_spoofing_is_rejected(tmp_path) -> None:
    settings = make_settings(tmp_path)
    upload = make_upload_file(filename="spoof.mp3", content=make_wav_bytes())

    with pytest.raises(UnsupportedAudioFormatError):
        save_upload(upload, settings)

    assert list(settings.resolved_upload_dir.iterdir()) == []


def test_oversized_upload_is_rejected_and_temp_file_is_removed(tmp_path) -> None:
    settings = make_settings(tmp_path, max_upload_size_mb=0)
    upload = make_upload_file(filename="large.wav", content=make_wav_bytes())

    with pytest.raises(AudioFileTooLargeError):
        save_upload(upload, settings)

    assert list(settings.resolved_upload_dir.iterdir()) == []


@requires_audio_tools
def test_corrupted_audio_is_rejected_and_temp_file_is_removed(tmp_path) -> None:
    settings = make_settings(tmp_path)
    upload = make_upload_file(filename="corrupt.wav", content=b"not a wav file")

    with pytest.raises(CorruptedAudioError):
        save_upload(upload, settings)

    assert list(settings.resolved_upload_dir.iterdir()) == []


@requires_audio_tools
def test_audio_duration_limit_is_enforced_and_temp_file_is_removed(tmp_path) -> None:
    settings = make_settings(
        tmp_path,
        max_audio_duration_seconds=0.01,
        min_audio_duration_seconds=0.001,
    )
    upload = make_upload_file(filename="long.wav", content=make_wav_bytes())

    with pytest.raises(AudioDurationExceededError):
        save_upload(upload, settings)

    assert list(settings.resolved_upload_dir.iterdir()) == []


def test_unsafe_filename_is_rejected(tmp_path) -> None:
    settings = make_settings(tmp_path)
    upload = make_upload_file(filename="../sample.wav", content=make_wav_bytes())

    with pytest.raises(UnsafeAudioFilenameError):
        save_upload(upload, settings)

    assert not settings.resolved_upload_dir.exists()


@requires_audio_tools
def test_preprocess_mono_16khz_audio_file(tmp_path) -> None:
    audio_path = tmp_path / "tone.wav"
    audio_path.write_bytes(make_wav_bytes(tone_hz=440.0))

    processed = preprocess_audio_file(audio_path, "wav", make_settings(tmp_path))

    assert processed.waveform.dtype == np.float32
    assert processed.waveform.ndim == 1
    assert processed.sample_rate == 16000
    assert processed.original_sample_rate == 16000
    assert processed.was_resampled is False
    assert processed.was_converted_to_mono is False
    assert processed.peak_amplitude == pytest.approx(NORMALIZED_TARGET_PEAK)
    assert processed.rms_energy > 0


def test_preprocess_converts_stereo_to_mono(tmp_path) -> None:
    waveform = np.column_stack(
        [
            np.linspace(-0.5, 0.5, 160, dtype=np.float32),
            np.linspace(0.5, -0.5, 160, dtype=np.float32),
        ]
    )
    waveform[:, 1] *= 0.5

    processed = preprocess_waveform(
        waveform,
        sample_rate=16000,
        app_settings=make_settings(tmp_path, min_audio_duration_seconds=0.001),
    )

    assert processed.waveform.ndim == 1
    assert processed.was_converted_to_mono is True
    assert processed.was_resampled is False
    assert processed.sample_rate == 16000


def test_preprocess_resamples_to_target_sample_rate(tmp_path) -> None:
    waveform = np.sin(
        2 * np.pi * 220 * np.arange(800, dtype=np.float32) / 8000
    ).astype(np.float32)

    processed = preprocess_waveform(
        waveform,
        sample_rate=8000,
        target_sample_rate=16000,
        app_settings=make_settings(tmp_path, min_audio_duration_seconds=0.001),
    )

    assert processed.sample_rate == 16000
    assert processed.original_sample_rate == 8000
    assert processed.was_resampled is True
    assert processed.waveform.shape == (1600,)


def test_preprocess_rejects_silence() -> None:
    with pytest.raises(SilentAudioError):
        preprocess_waveform(np.zeros(16000, dtype=np.float32), sample_rate=16000)


def test_preprocess_rejects_invalid_numeric_waveform() -> None:
    waveform = np.array([np.nan, np.inf, -np.inf], dtype=np.float32)

    with pytest.raises(UnusableAudioError):
        preprocess_waveform(waveform, sample_rate=16000)


def test_preprocess_is_deterministic(tmp_path) -> None:
    waveform = np.sin(
        2 * np.pi * 440 * np.arange(1600, dtype=np.float32) / 16000
    ).astype(np.float32)

    test_settings = make_settings(tmp_path, min_audio_duration_seconds=0.001)
    first = preprocess_waveform(
        waveform,
        sample_rate=16000,
        app_settings=test_settings,
    )
    second = preprocess_waveform(
        waveform,
        sample_rate=16000,
        app_settings=test_settings,
    )

    np.testing.assert_array_equal(first.waveform, second.waveform)
    assert first.sample_rate == second.sample_rate
    assert first.original_sample_rate == second.original_sample_rate
    assert first.duration_seconds == second.duration_seconds
    assert first.was_resampled == second.was_resampled
    assert first.was_converted_to_mono == second.was_converted_to_mono
    assert first.peak_amplitude == second.peak_amplitude
    assert first.rms_energy == second.rms_energy


def test_preprocess_rejects_decoded_audio_below_minimum_duration(tmp_path) -> None:
    waveform = np.sin(
        2 * np.pi * 440 * np.arange(8000, dtype=np.float32) / 16000
    ).astype(np.float32)

    with pytest.raises(AudioTooShortError):
        preprocess_waveform(
            waveform,
            sample_rate=16000,
            app_settings=make_settings(tmp_path, min_audio_duration_seconds=1.0),
        )


def test_segment_waveform_right_pads_final_window() -> None:
    waveform = np.ones(10, dtype=np.float32)

    segments = segment_waveform(
        waveform,
        sample_rate=10,
        window_duration_seconds=0.6,
        overlap_seconds=0.2,
    )

    assert [segment.index for segment in segments] == [0, 1]
    assert [(segment.start_sample, segment.end_sample) for segment in segments] == [
        (0, 6),
        (4, 10),
    ]
    assert [segment.valid_sample_count for segment in segments] == [6, 6]
    assert [segment.padded_sample_count for segment in segments] == [0, 0]


def test_segment_waveform_pads_short_audio() -> None:
    segments = segment_waveform(
        np.ones(4, dtype=np.float32),
        sample_rate=10,
        window_duration_seconds=1.0,
        overlap_seconds=0.0,
    )

    assert len(segments) == 1
    assert segments[0].valid_sample_count == 4
    assert segments[0].padded_sample_count == 6
    assert segments[0].waveform.shape == (10,)


def test_segment_waveform_rejects_invalid_overlap() -> None:
    with pytest.raises(UnusableAudioError):
        segment_waveform(
            np.ones(10, dtype=np.float32),
            sample_rate=10,
            window_duration_seconds=1.0,
            overlap_seconds=1.0,
        )


def test_probe_timeout_removes_temporary_upload(
    tmp_path,
    monkeypatch,
) -> None:
    settings = make_settings(tmp_path)
    upload = make_upload_file(filename="timeout.wav", content=make_wav_bytes())

    monkeypatch.setattr(
        "app.ingestion.audio._resolve_executable",
        lambda _name: "/usr/bin/ffprobe",
    )

    def raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["ffprobe"], timeout=1)

    monkeypatch.setattr(
        "app.ingestion.audio.subprocess.run",
        raise_timeout,
    )

    with pytest.raises(AudioProcessingTimeoutError):
        save_upload(upload, settings)

    assert list(settings.resolved_upload_dir.iterdir()) == []


def test_cancelled_upload_removes_temporary_file(tmp_path) -> None:
    settings = make_settings(tmp_path)

    class CancellingUpload:
        filename = "cancelled.wav"
        content_type = "audio/wav"

        def __init__(self) -> None:
            self.read_count = 0

        async def read(self, _size):
            self.read_count += 1
            if self.read_count == 1:
                return b"partial upload"
            raise asyncio.CancelledError

        async def close(self):
            return None

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(save_validated_audio_upload(CancellingUpload(), settings))

    assert list(settings.resolved_upload_dir.iterdir()) == []
