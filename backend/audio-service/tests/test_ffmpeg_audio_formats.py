import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from app.core.exceptions import (
    AudioChannelCountExceededError,
    AudioDurationExceededError,
    AudioSampleRateExceededError,
    AudioStreamValidationError,
    CorruptedAudioError,
    DecodedAudioTooLargeError,
    UnsupportedAudioFormatError,
    VideoStreamNotAllowedError,
)
from app.ingestion.audio import (
    AudioInspectionResult,
    decode_audio_file,
    inspect_audio_file,
    preprocess_audio_file,
)
from tests.test_audio_preprocessing import (
    make_settings,
    make_upload_file,
    make_wav_bytes,
    save_upload,
)

FFMPEG = shutil.which("ffmpeg")
FFPROBE = shutil.which("ffprobe")
AUDIO_TOOLS_AVAILABLE = bool(FFMPEG and FFPROBE)
requires_audio_tools = pytest.mark.skipif(
    not AUDIO_TOOLS_AVAILABLE,
    reason="FFmpeg and ffprobe are not available in the test environment.",
)


@pytest.mark.parametrize(
    ("extension", "codec", "container"),
    [
        ("wav", "pcm_s16le", None),
        ("mp3", "libmp3lame", None),
        ("flac", "flac", None),
        ("m4a", "aac", None),
        ("m4a", "alac", None),
        ("aac", "aac", "adts"),
        ("opus", "libopus", "ogg"),
        ("ogg", "libopus", "ogg"),
        ("ogg", "libvorbis", "ogg"),
        ("webm", "libopus", "webm"),
    ],
)
@requires_audio_tools
def test_supported_audio_formats_are_probed_and_decoded(
    tmp_path,
    extension,
    codec,
    container,
) -> None:
    encoded_path = _encode_audio(
        tmp_path,
        extension=extension,
        codec=codec,
        container=container,
    )
    settings = make_settings(tmp_path)
    upload = make_upload_file(
        filename=f"sample.{extension}",
        content=encoded_path.read_bytes(),
        content_type="application/octet-stream",
    )

    metadata = save_upload(upload, settings)
    try:
        processed = preprocess_audio_file(
            metadata.saved_path,
            extension,
            settings,
            inspection=metadata.inspection,
        )

        assert metadata.original_extension == extension
        assert metadata.detected_container
        assert metadata.detected_format == metadata.detected_container
        assert metadata.detected_codec
        assert processed.waveform.dtype == np.float32
        assert processed.waveform.ndim == 1
        assert processed.sample_rate == 16000
        assert processed.duration_seconds > 0
    finally:
        metadata.saved_path.unlink(missing_ok=True)

    assert _upload_files(settings) == []


@requires_audio_tools
def test_webm_with_video_is_rejected_and_cleaned_up(tmp_path) -> None:
    media_path = _encode_webm_with_video(tmp_path, include_audio=True)
    settings = make_settings(tmp_path)
    upload = make_upload_file(
        filename="video.webm",
        content=media_path.read_bytes(),
        content_type="video/webm",
    )

    with pytest.raises(VideoStreamNotAllowedError):
        save_upload(upload, settings)

    assert _upload_files(settings) == []


@requires_audio_tools
def test_renamed_extension_is_rejected_and_cleaned_up(tmp_path) -> None:
    wav_path = _encode_audio(tmp_path, extension="wav", codec="pcm_s16le")
    settings = make_settings(tmp_path)
    upload = make_upload_file(
        filename="renamed.mp3",
        content=wav_path.read_bytes(),
        content_type="audio/mpeg",
    )

    with pytest.raises(UnsupportedAudioFormatError):
        save_upload(upload, settings)

    assert _upload_files(settings) == []


@requires_audio_tools
def test_corrupted_audio_is_rejected_and_cleaned_up(tmp_path) -> None:
    settings = make_settings(tmp_path)
    upload = make_upload_file(
        filename="corrupt.ogg",
        content=b"not an audio container",
        content_type="audio/ogg",
    )

    with pytest.raises(CorruptedAudioError):
        save_upload(upload, settings)

    assert _upload_files(settings) == []


def test_no_audio_stream_is_rejected(monkeypatch, tmp_path) -> None:
    payload = {
        "streams": [],
        "format": {"format_name": "wav", "duration": "1.0"},
    }
    _mock_ffprobe(monkeypatch, payload)
    path = tmp_path / "no-audio.wav"
    path.write_bytes(b"placeholder")

    with pytest.raises(AudioStreamValidationError):
        inspect_audio_file(path, "wav", make_settings(tmp_path))


def test_multiple_audio_streams_select_first_valid_audio_stream(monkeypatch, tmp_path) -> None:
    stream = {
        "index": 0,
        "codec_type": "audio",
        "codec_name": "pcm_s16le",
        "sample_rate": "16000",
        "channels": 1,
        "duration": "1.0",
    }
    payload = {
        "streams": [stream, {**stream, "index": 1}],
        "format": {"format_name": "wav", "duration": "1.0"},
    }
    _mock_ffprobe(monkeypatch, payload)
    path = tmp_path / "multiple.wav"
    path.write_bytes(b"placeholder")

    inspection = inspect_audio_file(path, "wav", make_settings(tmp_path))

    assert inspection.audio_stream_index == 0
    assert inspection.audio_stream_count == 2
    assert inspection.stream_count == 2
    assert inspection.has_video_stream is False


def test_video_stream_metadata_is_rejected(monkeypatch, tmp_path) -> None:
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "audio",
                "codec_name": "opus",
                "sample_rate": "48000",
                "channels": 2,
                "duration": "1.0",
            },
            {
                "index": 1,
                "codec_type": "video",
                "codec_name": "vp9",
            },
        ],
        "format": {"format_name": "matroska,webm", "duration": "1.0"},
    }
    _mock_ffprobe(monkeypatch, payload)
    path = tmp_path / "video.webm"
    path.write_bytes(b"placeholder")

    with pytest.raises(VideoStreamNotAllowedError):
        inspect_audio_file(path, "webm", make_settings(tmp_path))


def test_extension_container_mismatch_is_rejected(monkeypatch, tmp_path) -> None:
    payload = {
        "streams": [
            {
                "index": 0,
                "codec_type": "audio",
                "codec_name": "pcm_s16le",
                "sample_rate": "16000",
                "channels": 1,
                "duration": "1.0",
            }
        ],
        "format": {"format_name": "wav", "duration": "1.0"},
    }
    _mock_ffprobe(monkeypatch, payload)
    path = tmp_path / "renamed.mp3"
    path.write_bytes(b"placeholder")

    with pytest.raises(UnsupportedAudioFormatError):
        inspect_audio_file(path, "mp3", make_settings(tmp_path))


@requires_audio_tools
def test_exactly_180_seconds_is_accepted(tmp_path) -> None:
    settings = make_settings(tmp_path, max_audio_duration_seconds=180)
    upload = make_upload_file(
        filename="exact.wav",
        content=make_wav_bytes(duration_seconds=180),
    )

    metadata = save_upload(upload, settings)
    try:
        assert metadata.duration_seconds == pytest.approx(180.0)
    finally:
        metadata.saved_path.unlink(missing_ok=True)


@requires_audio_tools
def test_more_than_180_seconds_is_rejected_and_cleaned_up(tmp_path) -> None:
    settings = make_settings(tmp_path, max_audio_duration_seconds=180)
    upload = make_upload_file(
        filename="too-long.wav",
        content=make_wav_bytes(duration_seconds=181),
    )

    with pytest.raises(AudioDurationExceededError):
        save_upload(upload, settings)

    assert _upload_files(settings) == []


@requires_audio_tools
def test_excessive_channel_count_is_rejected_and_cleaned_up(tmp_path) -> None:
    settings = make_settings(tmp_path, max_audio_channels=2)
    upload = make_upload_file(
        filename="channels.wav",
        content=make_wav_bytes(channels=3, duration_seconds=0.1),
    )

    with pytest.raises(AudioChannelCountExceededError):
        save_upload(upload, settings)

    assert _upload_files(settings) == []


@requires_audio_tools
def test_excessive_sample_rate_is_rejected_and_cleaned_up(tmp_path) -> None:
    settings = make_settings(tmp_path, max_input_sample_rate=96000)
    upload = make_upload_file(
        filename="sample-rate.wav",
        content=make_wav_bytes(sample_rate=192000, duration_seconds=0.1),
    )

    with pytest.raises(AudioSampleRateExceededError):
        save_upload(upload, settings)

    assert _upload_files(settings) == []


def test_decoded_audio_memory_budget_is_enforced(monkeypatch, tmp_path) -> None:
    settings = make_settings(tmp_path, max_decoded_audio_size_mb=0)
    inspection = AudioInspectionResult(
        duration_seconds=1.0,
        sample_rate=16000,
        channels=1,
        detected_container="wav",
        detected_codec="pcm_s16le",
        audio_stream_index=0,
    )
    path = tmp_path / "decoded.wav"
    path.write_bytes(b"placeholder")

    monkeypatch.setattr(
        "app.ingestion.audio._resolve_executable",
        lambda _name: "/usr/bin/ffmpeg",
    )
    def write_decoded_output(*_args, **kwargs):
        kwargs["stdout"].write(np.array([0.1, 0.2], dtype="<f4").tobytes())
        return subprocess.CompletedProcess(args=["ffmpeg"], returncode=0)

    monkeypatch.setattr(
        "app.ingestion.audio.subprocess.run",
        write_decoded_output,
    )

    with pytest.raises(DecodedAudioTooLargeError):
        decode_audio_file(path, "wav", settings, inspection)


@pytest.mark.parametrize(
    ("extension", "codec", "container"),
    [
        ("webm", "libopus", "webm"),
        ("ogg", "libopus", "ogg"),
        ("m4a", "aac", None),
        ("opus", "libopus", "ogg"),
    ],
)
@requires_audio_tools
def test_browser_recording_formats_are_supported(
    tmp_path,
    extension,
    codec,
    container,
) -> None:
    encoded_path = _encode_audio(
        tmp_path,
        extension=extension,
        codec=codec,
        container=container,
    )
    settings = make_settings(tmp_path)
    upload = make_upload_file(
        filename=f"browser-recording.{extension}",
        content=encoded_path.read_bytes(),
        content_type="application/octet-stream",
    )

    metadata = save_upload(upload, settings)
    try:
        assert metadata.original_extension == extension
        assert metadata.detected_container
        assert metadata.detected_codec
    finally:
        metadata.saved_path.unlink(missing_ok=True)

    assert _upload_files(settings) == []


def _encode_audio(
    tmp_path: Path,
    *,
    extension: str,
    codec: str,
    container: str | None = None,
) -> Path:
    output_path = tmp_path / f"encoded-{codec}.{extension}"
    command = [
        str(FFMPEG),
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=0.25",
        "-ac",
        "2",
        "-c:a",
        codec,
    ]
    if container is not None:
        command.extend(["-f", container])
    command.append(str(output_path))
    _run_fixture_command(command, codec)
    return output_path


def _encode_webm_with_video(tmp_path: Path, *, include_audio: bool) -> Path:
    output_path = tmp_path / "video.webm"
    command = [
        str(FFMPEG),
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
    ]
    if include_audio:
        command.extend(
            [
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:sample_rate=48000:duration=0.25",
            ]
        )
    command.extend(
        [
            "-f",
            "lavfi",
            "-i",
            "color=black:size=32x32:duration=0.25",
            "-c:v",
            "libvpx-vp9",
        ]
    )
    if include_audio:
        command.extend(["-c:a", "libopus", "-shortest"])
    command.append(str(output_path))
    _run_fixture_command(command, "WebM video")
    return output_path


def _run_fixture_command(command: list[str], description: str) -> None:
    result = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        shell=False,
        timeout=30,
    )
    if result.returncode != 0:
        pytest.skip(f"Installed FFmpeg cannot generate the {description} fixture.")


def _mock_ffprobe(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr(
        "app.ingestion.audio._resolve_executable",
        lambda _name: "/usr/bin/ffprobe",
    )
    def write_probe_output(*_args, **kwargs):
        kwargs["stdout"].write(json.dumps(payload).encode("utf-8"))
        return subprocess.CompletedProcess(args=["ffprobe"], returncode=0)

    monkeypatch.setattr(
        "app.ingestion.audio.subprocess.run",
        write_probe_output,
    )


def _upload_files(settings) -> list[Path]:
    if not settings.resolved_upload_dir.exists():
        return []
    return [
        path
        for path in settings.resolved_upload_dir.rglob("*")
        if path.is_file()
    ]
