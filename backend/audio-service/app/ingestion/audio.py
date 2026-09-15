from dataclasses import dataclass, field, replace
from functools import lru_cache
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from uuid import uuid4

from fastapi import UploadFile
import numpy as np
from numpy.typing import NDArray

from app.config.settings import Settings, settings
from app.core.exceptions import (
    AudioChannelCountExceededError,
    AudioDurationExceededError,
    AudioFileTooLargeError,
    AudioProcessingTimeoutError,
    AudioProcessingUnavailableError,
    AudioSampleRateExceededError,
    AudioStreamValidationError,
    AudioTooShortError,
    CorruptedAudioError,
    DecodedAudioTooLargeError,
    EmptyAudioFileError,
    MissingAudioFilenameError,
    SilentAudioError,
    UnsafeAudioFilenameError,
    UnsupportedAudioFormatError,
    UnusableAudioError,
    VideoStreamNotAllowedError,
)
from app.core.timing import stage_timer

logger = logging.getLogger(__name__)

CHUNK_SIZE_BYTES = 1024 * 1024
MAX_FFPROBE_OUTPUT_BYTES = 1024 * 1024
MAX_TOOL_STDERR_BYTES = 16 * 1024
MAX_SANITIZED_TOOL_ERROR_LENGTH = 500
SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
SILENCE_PEAK_THRESHOLD = 1e-8
SILENCE_RMS_THRESHOLD = 1e-10
MIN_NORMALIZATION_PEAK = 1e-3
NORMALIZED_TARGET_PEAK = 0.95

SUPPORTED_CODEC_FAMILIES = frozenset(
    {"pcm", "flac", "mp3", "aac", "alac", "opus", "vorbis"}
)
EXTENSION_CONTAINER_RULES = {
    "wav": frozenset({"wav"}),
    "flac": frozenset({"flac"}),
    "mp3": frozenset({"mp3"}),
    "m4a": frozenset({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}),
    "aac": frozenset({"aac"}),
    "opus": frozenset({"ogg"}),
    "ogg": frozenset({"ogg"}),
    "webm": frozenset({"matroska", "webm"}),
}
EXTENSION_CODEC_RULES = {
    "wav": frozenset({"pcm"}),
    "flac": frozenset({"flac"}),
    "mp3": frozenset({"mp3"}),
    "m4a": frozenset({"aac", "alac"}),
    "aac": frozenset({"aac"}),
    "opus": frozenset({"opus"}),
    "ogg": frozenset({"opus", "vorbis"}),
    "webm": frozenset({"opus", "vorbis"}),
}

Float32Waveform = NDArray[np.float32]


@dataclass(frozen=True)
class AudioPayload:
    filename: str
    content_type: str | None
    data: bytes


def validate_audio_payload(payload: AudioPayload) -> None:
    if not payload.data:
        raise ValueError("Audio payload is empty.")


@dataclass(frozen=True)
class AudioInspectionResult:
    duration_seconds: float
    sample_rate: int
    channels: int
    detected_container: str
    detected_codec: str
    audio_stream_index: int
    stream_count: int = 0
    audio_stream_count: int = 0
    has_video_stream: bool = False
    frame_count: int | None = None
    duration_timestamp: int | None = None
    time_base: str | None = None
    bits_per_sample: int | None = None

    @property
    def detected_format(self) -> str:
        """Backward-compatible alias for the detected container metadata."""

        return self.detected_container


@dataclass(frozen=True)
class AudioUploadMetadata:
    original_filename: str
    sanitized_filename: str
    saved_filename: str
    saved_path: Path
    content_type: str | None
    file_size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int
    original_extension: str = ""
    detected_container: str = ""
    detected_codec: str = ""
    inspection: AudioInspectionResult | None = None

    @property
    def detected_format(self) -> str:
        """Backward-compatible alias for the detected container metadata."""

        return self.detected_container

    @property
    def size_bytes(self) -> int:
        """Schema-friendly alias for the uploaded compressed file size."""

        return self.file_size_bytes


@dataclass(frozen=True)
class DecodedAudio:
    waveform: Float32Waveform
    sample_rate: int
    original_sample_rate: int
    original_channels: int


@dataclass(frozen=True)
class AudioSegment:
    index: int
    waveform: Float32Waveform
    start_sample: int
    end_sample: int
    valid_sample_count: int
    padded_sample_count: int
    start_sec: float
    end_sec: float


@dataclass(frozen=True)
class ProcessedAudio:
    """Model-agnostic audio prepared once for all MULTI-SCOPE branches.

    FFmpeg decodes the validated stream directly to mono float32 PCM at the
    configured target sample rate. This shared layer performs only numeric
    cleanup, silence checks, and safe amplitude normalization. MFCC/LFCC, SSL
    processor inputs, and glottal features remain branch responsibilities.
    Training pipelines must use equivalent decoding and normalization settings
    before real model results are considered research-reproducible.
    """

    waveform: Float32Waveform
    sample_rate: int
    original_sample_rate: int
    original_channels: int
    duration_seconds: float
    was_resampled: bool
    was_converted_to_mono: bool
    normalisation_applied: bool
    peak_amplitude: float
    rms_energy: float
    unnormalised_waveform: Float32Waveform | None = None
    segments: list[AudioSegment] = field(default_factory=list)
    preprocessing_version: str = "audio-preprocessing-v2"
    minimum_duration_seconds: float = 1.0
    model_window_duration_seconds: float = 6.0
    model_window_overlap_seconds: float = 1.0
    trim_applied: bool = False

    def __post_init__(self) -> None:
        """Fill deterministic model windows for legacy/internal callers.

        The normal pipeline supplies segments explicitly. This fallback keeps
        existing dummy-model tests and any older internal fixtures compatible
        while preserving the invariant that processed audio has at least one
        shared branch input window.
        """

        if self.segments:
            return
        object.__setattr__(
            self,
            "segments",
            segment_waveform(
                self.waveform,
                sample_rate=self.sample_rate,
                window_duration_seconds=self.model_window_duration_seconds,
                overlap_seconds=self.model_window_overlap_seconds,
            ),
        )


def audio_tool_status(app_settings: Settings = settings) -> dict[str, bool]:
    return {
        "ffmpeg_available": _resolve_executable(app_settings.ffmpeg_binary) is not None,
        "ffprobe_available": (
            _resolve_executable(app_settings.ffprobe_binary) is not None
        ),
    }


def audio_tool_versions(app_settings: Settings = settings) -> dict[str, str | None]:
    return {
        "ffmpeg_version": _media_tool_version(app_settings.ffmpeg_binary),
        "ffprobe_version": _media_tool_version(app_settings.ffprobe_binary),
    }


async def save_validated_audio_upload(
    upload_file: UploadFile,
    app_settings: Settings = settings,
    validation_filename: str | None = None,
    display_filename: str | None = None,
) -> AudioUploadMetadata:
    original_filename = _validate_original_filename(
        validation_filename or upload_file.filename
    )
    sanitized_filename = sanitize_audio_filename(original_filename)
    extension = _get_allowed_extension(sanitized_filename, app_settings)
    metadata_filename = _metadata_filename(
        display_filename=display_filename,
        fallback_filename=original_filename,
        extension=extension,
    )
    upload_dir = _prepare_upload_dir(app_settings)
    temp_path: Path | None = None
    completed_metadata: AudioUploadMetadata | None = None

    try:
        temp_path, file_size_bytes = await _stream_upload_to_temp_file(
            upload_file=upload_file,
            upload_dir=upload_dir,
            max_upload_size_bytes=app_settings.max_upload_size_mb * 1024 * 1024,
        )
        inspection = inspect_audio_file(temp_path, extension, app_settings)

        saved_filename = f"{uuid4().hex}.{extension}"
        saved_path = _safe_upload_path(upload_dir, saved_filename)
        os.replace(temp_path, saved_path)
        temp_path = None

        completed_metadata = AudioUploadMetadata(
            original_filename=metadata_filename,
            sanitized_filename=metadata_filename,
            saved_filename=saved_filename,
            saved_path=saved_path,
            content_type=upload_file.content_type,
            file_size_bytes=file_size_bytes,
            duration_seconds=inspection.duration_seconds,
            sample_rate=inspection.sample_rate,
            channels=inspection.channels,
            original_extension=extension,
            detected_container=inspection.detected_container,
            detected_codec=inspection.detected_codec,
            inspection=inspection,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        try:
            await upload_file.close()
        except BaseException:
            if completed_metadata is not None:
                completed_metadata.saved_path.unlink(missing_ok=True)
            raise

    if completed_metadata is None:
        raise CorruptedAudioError("Audio upload validation did not complete.")
    return completed_metadata


def save_validated_audio_upload_blocking(
    upload_file: UploadFile,
    app_settings: Settings = settings,
    validation_filename: str | None = None,
    display_filename: str | None = None,
) -> AudioUploadMetadata:
    original_filename = _validate_original_filename(
        validation_filename or upload_file.filename
    )
    sanitized_filename = sanitize_audio_filename(original_filename)
    extension = _get_allowed_extension(sanitized_filename, app_settings)
    metadata_filename = _metadata_filename(
        display_filename=display_filename,
        fallback_filename=original_filename,
        extension=extension,
    )
    upload_dir = _prepare_upload_dir(app_settings)
    temp_path: Path | None = None
    completed_metadata: AudioUploadMetadata | None = None

    try:
        temp_path, file_size_bytes = _stream_upload_to_temp_file_blocking(
            upload_file=upload_file,
            upload_dir=upload_dir,
            max_upload_size_bytes=app_settings.max_upload_size_mb * 1024 * 1024,
        )
        inspection = inspect_audio_file(temp_path, extension, app_settings)

        saved_filename = f"{uuid4().hex}.{extension}"
        saved_path = _safe_upload_path(upload_dir, saved_filename)
        os.replace(temp_path, saved_path)
        temp_path = None

        completed_metadata = AudioUploadMetadata(
            original_filename=metadata_filename,
            sanitized_filename=metadata_filename,
            saved_filename=saved_filename,
            saved_path=saved_path,
            content_type=upload_file.content_type,
            file_size_bytes=file_size_bytes,
            duration_seconds=inspection.duration_seconds,
            sample_rate=inspection.sample_rate,
            channels=inspection.channels,
            original_extension=extension,
            detected_container=inspection.detected_container,
            detected_codec=inspection.detected_codec,
            inspection=inspection,
        )
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        try:
            upload_file.file.close()
        except BaseException:
            if completed_metadata is not None:
                completed_metadata.saved_path.unlink(missing_ok=True)
            raise

    if completed_metadata is None:
        raise CorruptedAudioError("Audio upload validation did not complete.")
    return completed_metadata


def save_validated_local_audio_file(
    source_path: Path,
    *,
    original_filename: str,
    content_type: str | None = None,
    app_settings: Settings = settings,
    display_filename: str | None = None,
) -> AudioUploadMetadata:
    original_filename = _validate_original_filename(original_filename)
    sanitized_filename = sanitize_audio_filename(original_filename)
    extension = _get_allowed_extension(sanitized_filename, app_settings)
    metadata_filename = _metadata_filename(
        display_filename=display_filename,
        fallback_filename=original_filename,
        extension=extension,
    )
    upload_dir = _prepare_upload_dir(app_settings)
    temp_path = source_path.resolve()
    completed_metadata: AudioUploadMetadata | None = None

    try:
        if not temp_path.is_file():
            raise CorruptedAudioError("Source audio is unavailable.")
        file_size_bytes = temp_path.stat().st_size
        if file_size_bytes == 0:
            raise EmptyAudioFileError("Audio file is empty.")
        if file_size_bytes > app_settings.max_upload_size_mb * 1024 * 1024:
            raise AudioFileTooLargeError(
                "Audio file exceeds the configured maximum size."
            )

        inspection = inspect_audio_file(temp_path, extension, app_settings)
        saved_filename = f"{uuid4().hex}.{extension}"
        saved_path = _safe_upload_path(upload_dir, saved_filename)
        os.replace(temp_path, saved_path)

        completed_metadata = AudioUploadMetadata(
            original_filename=metadata_filename,
            sanitized_filename=metadata_filename,
            saved_filename=saved_filename,
            saved_path=saved_path,
            content_type=content_type,
            file_size_bytes=file_size_bytes,
            duration_seconds=inspection.duration_seconds,
            sample_rate=inspection.sample_rate,
            channels=inspection.channels,
            original_extension=extension,
            detected_container=inspection.detected_container,
            detected_codec=inspection.detected_codec,
            inspection=inspection,
        )
    finally:
        if completed_metadata is None:
            temp_path.unlink(missing_ok=True)

    return completed_metadata


def sanitize_audio_filename(filename: str) -> str:
    if "/" in filename or "\\" in filename or "\x00" in filename:
        raise UnsafeAudioFilenameError(
            "Audio filename contains unsafe path characters."
        )
    if Path(filename).name != filename:
        raise UnsafeAudioFilenameError("Audio filename must not include a path.")

    sanitized = SAFE_FILENAME_PATTERN.sub("_", filename).strip("._ ")
    if not sanitized or "." not in sanitized:
        raise UnsafeAudioFilenameError(
            "Audio filename is not usable after sanitization."
        )
    return sanitized


def sanitize_audio_display_filename(
    filename: str,
    *,
    extension: str,
) -> str:
    if "/" in filename or "\\" in filename or "\x00" in filename:
        raise UnsafeAudioFilenameError(
            "Audio display filename contains unsafe path characters."
        )
    sanitized = SAFE_FILENAME_PATTERN.sub("_", filename).strip("._ ")
    if not sanitized:
        sanitized = "audio"
    suffix = f".{extension}"
    if Path(sanitized).suffix.lower() != suffix:
        sanitized = f"{Path(sanitized).stem or sanitized}{suffix}"
    return sanitized


def preprocess_audio_file(
    path: Path,
    extension: str | None = None,
    app_settings: Settings = settings,
    inspection: AudioInspectionResult | None = None,
) -> ProcessedAudio:
    """Decode a validated file once to the branch-shared waveform.

    ffprobe validation must precede decoding. FFmpeg emits bounded mono float32
    PCM at `TARGET_SAMPLE_RATE`; actual sample count is then checked against both
    duration and decoded-memory limits before model-neutral normalization.
    """

    resolved_extension = extension or path.suffix.lower().lstrip(".")
    if resolved_extension not in app_settings.allowed_audio_extension_list:
        raise UnsupportedAudioFormatError(
            f"Unsupported audio extension: {resolved_extension}"
        )

    resolved_inspection = inspection or inspect_audio_file(
        path,
        resolved_extension,
        app_settings,
    )
    decoded_audio = decode_audio_file(
        path,
        resolved_extension,
        app_settings,
        resolved_inspection,
    )
    processed = preprocess_waveform(
        decoded_audio.waveform,
        sample_rate=decoded_audio.sample_rate,
        target_sample_rate=app_settings.target_sample_rate,
        app_settings=app_settings,
    )
    return replace(
        processed,
        original_sample_rate=decoded_audio.original_sample_rate,
        original_channels=decoded_audio.original_channels,
        was_resampled=(
            decoded_audio.original_sample_rate != app_settings.target_sample_rate
        ),
        was_converted_to_mono=decoded_audio.original_channels > 1,
    )


def preprocess_waveform(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    target_sample_rate: int = 16000,
    app_settings: Settings = settings,
) -> ProcessedAudio:
    """Prepare an in-memory waveform without model-specific feature extraction.

    This function remains available for deterministic unit tests and internal
    callers. API uploads are already converted to mono and the target rate by
    FFmpeg before reaching this function.
    """

    if sample_rate <= 0 or target_sample_rate <= 0:
        raise UnusableAudioError("Sample rates must be positive.")

    original_sample_rate = sample_rate
    clean_waveform = _coerce_float32_waveform(waveform)
    original_channels = clean_waveform.shape[1] if clean_waveform.ndim == 2 else 1
    clean_waveform = _replace_invalid_values(clean_waveform)
    clean_waveform, was_converted_to_mono = _convert_to_mono(clean_waveform)
    clean_waveform = _resample_if_needed(
        clean_waveform,
        sample_rate=sample_rate,
        target_sample_rate=target_sample_rate,
    )
    clean_waveform = _replace_invalid_values(clean_waveform)
    clean_waveform = _reject_silent_or_unusable_audio(clean_waveform)
    unnormalised_waveform = np.ascontiguousarray(
        clean_waveform.copy(),
        dtype=np.float32,
    )
    clean_waveform, normalisation_applied = _normalize_amplitude_safely(clean_waveform)

    peak_amplitude = float(np.max(np.abs(clean_waveform)))
    rms_energy = float(np.sqrt(np.mean(np.square(clean_waveform, dtype=np.float32))))
    duration_seconds = float(clean_waveform.shape[0] / target_sample_rate)
    _validate_minimum_decoded_duration(duration_seconds, app_settings)
    segments = segment_waveform(
        clean_waveform,
        sample_rate=target_sample_rate,
        window_duration_seconds=app_settings.model_window_duration_seconds,
        overlap_seconds=app_settings.model_window_overlap_seconds,
    )

    return ProcessedAudio(
        waveform=np.ascontiguousarray(clean_waveform, dtype=np.float32),
        sample_rate=target_sample_rate,
        original_sample_rate=original_sample_rate,
        original_channels=original_channels,
        duration_seconds=duration_seconds,
        was_resampled=original_sample_rate != target_sample_rate,
        was_converted_to_mono=was_converted_to_mono,
        normalisation_applied=normalisation_applied,
        peak_amplitude=peak_amplitude,
        rms_energy=rms_energy,
        unnormalised_waveform=unnormalised_waveform,
        segments=segments,
        preprocessing_version=app_settings.preprocessing_version,
        minimum_duration_seconds=app_settings.min_audio_duration_seconds,
        model_window_duration_seconds=app_settings.model_window_duration_seconds,
        model_window_overlap_seconds=app_settings.model_window_overlap_seconds,
    )


def segment_waveform(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    window_duration_seconds: float,
    overlap_seconds: float,
) -> list[AudioSegment]:
    if sample_rate <= 0:
        raise UnusableAudioError("Sample rate must be positive.")
    if window_duration_seconds <= 0:
        raise UnusableAudioError("Model window duration must be positive.")
    if overlap_seconds < 0 or overlap_seconds >= window_duration_seconds:
        raise UnusableAudioError("Model window overlap must be smaller than the window.")

    clean_waveform = np.ascontiguousarray(_coerce_float32_waveform(waveform), dtype=np.float32)
    if clean_waveform.ndim != 1:
        clean_waveform, _converted = _convert_to_mono(clean_waveform)

    window_samples = max(int(round(window_duration_seconds * sample_rate)), 1)
    overlap_samples = int(round(overlap_seconds * sample_rate))
    step_samples = max(window_samples - overlap_samples, 1)
    total_samples = clean_waveform.shape[0]
    segments: list[AudioSegment] = []
    start_sample = 0

    while start_sample < total_samples:
        end_sample = min(start_sample + window_samples, total_samples)
        valid_sample_count = end_sample - start_sample
        padded_sample_count = window_samples - valid_sample_count
        segment = np.zeros(window_samples, dtype=np.float32)
        segment[:valid_sample_count] = clean_waveform[start_sample:end_sample]
        segments.append(
            AudioSegment(
                index=len(segments),
                waveform=np.ascontiguousarray(segment, dtype=np.float32),
                start_sample=start_sample,
                end_sample=end_sample,
                valid_sample_count=valid_sample_count,
                padded_sample_count=padded_sample_count,
                start_sec=start_sample / sample_rate,
                end_sec=end_sample / sample_rate,
            )
        )
        if end_sample >= total_samples:
            break
        start_sample += step_samples

    if not segments:
        raise UnusableAudioError("Decoded audio contains no usable segments.")
    return segments


def inspect_audio_file(
    path: Path,
    extension: str,
    app_settings: Settings = settings,
) -> AudioInspectionResult:
    ffprobe = _require_executable(
        app_settings.ffprobe_binary,
        display_name="ffprobe",
    )
    command = [
        ffprobe,
        # -nostdin is deliberately omitted here: it is an ffmpeg-only option
        # in some ffprobe builds (observed failing with "Option not found"
        # on ffprobe 8.1.2/Homebrew, which does not define it for ffprobe),
        # and is redundant anyway since _run_media_process always runs this
        # subprocess with stdin=subprocess.DEVNULL, so ffprobe can never
        # block reading interactive input regardless of this flag.
        "-hide_banner",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_entries",
        (
            "format=format_name,duration:"
            "stream=index,codec_type,codec_name,sample_rate,channels,duration,"
            "duration_ts,time_base,nb_frames,bits_per_sample,bits_per_raw_sample"
        ),
        "-show_format",
        "-show_streams",
        str(path),
    ]
    with stage_timer("audio_probe", logger_name=__name__):
        result = _run_media_process(
            command,
            timeout_seconds=app_settings.ffprobe_timeout_seconds,
            tool_name="ffprobe",
            source_path=path,
            stdout_limit_bytes=MAX_FFPROBE_OUTPUT_BYTES,
        )
    if result.returncode != 0:
        _log_tool_failure("ffprobe", result.stderr, path)
        raise CorruptedAudioError("Uploaded content cannot be inspected as audio.")
    if len(result.stdout) > MAX_FFPROBE_OUTPUT_BYTES:
        raise CorruptedAudioError("Audio stream metadata is excessively large.")

    try:
        payload = json.loads(result.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorruptedAudioError("Audio stream metadata is invalid.") from error

    streams = payload.get("streams")
    format_metadata = payload.get("format")
    if not isinstance(streams, list) or not isinstance(format_metadata, dict):
        raise CorruptedAudioError("Audio stream metadata is incomplete.")

    stream_count = len(streams)
    has_video_stream = any(stream.get("codec_type") == "video" for stream in streams)
    if has_video_stream:
        raise VideoStreamNotAllowedError(
            "Uploaded media must not contain a video stream."
        )

    audio_streams = [
        stream for stream in streams if stream.get("codec_type") == "audio"
    ]
    if not audio_streams:
        raise AudioStreamValidationError(
            "At least one usable audio stream is required."
        )

    detected_format = str(format_metadata.get("format_name") or "").lower()
    audio_stream = _select_valid_audio_stream(
        audio_streams,
        extension=extension,
        detected_container=detected_format,
    )
    detected_codec = str(audio_stream.get("codec_name") or "").lower()

    sample_rate = _positive_int(audio_stream.get("sample_rate"))
    channels = _positive_int(audio_stream.get("channels"))
    duration_seconds = _first_positive_float(
        audio_stream.get("duration"),
        format_metadata.get("duration"),
    )
    if sample_rate is None or channels is None or duration_seconds is None:
        raise CorruptedAudioError("Audio stream has invalid metadata.")
    if duration_seconds > app_settings.max_audio_duration_seconds:
        raise AudioDurationExceededError(
            "Audio duration exceeds the configured maximum."
        )
    if duration_seconds < app_settings.min_audio_duration_seconds:
        raise AudioTooShortError("Audio duration is shorter than the configured minimum.")
    if channels > app_settings.max_audio_channels:
        raise AudioChannelCountExceededError(
            "Audio channel count exceeds the configured maximum."
        )
    if sample_rate > app_settings.max_input_sample_rate:
        raise AudioSampleRateExceededError(
            "Audio sample rate exceeds the configured maximum."
        )

    stream_index = _non_negative_int(audio_stream.get("index"))
    if stream_index is None:
        raise CorruptedAudioError("Audio stream index is invalid.")

    return AudioInspectionResult(
        duration_seconds=duration_seconds,
        sample_rate=sample_rate,
        channels=channels,
        detected_container=detected_format,
        detected_codec=detected_codec,
        audio_stream_index=stream_index,
        stream_count=stream_count,
        audio_stream_count=len(audio_streams),
        has_video_stream=has_video_stream,
        frame_count=_non_negative_int(audio_stream.get("nb_frames")),
        duration_timestamp=_non_negative_int(audio_stream.get("duration_ts")),
        time_base=_optional_string(audio_stream.get("time_base")),
        bits_per_sample=(
            _non_negative_int(audio_stream.get("bits_per_raw_sample"))
            or _non_negative_int(audio_stream.get("bits_per_sample"))
        ),
    )


def decode_audio_file(
    path: Path,
    extension: str,
    app_settings: Settings = settings,
    inspection: AudioInspectionResult | None = None,
) -> DecodedAudio:
    resolved_inspection = inspection or inspect_audio_file(
        path,
        extension,
        app_settings,
    )
    ffmpeg = _require_executable(
        app_settings.ffmpeg_binary,
        display_name="ffmpeg",
    )

    bytes_per_sample = np.dtype("<f4").itemsize
    maximum_duration_samples = int(
        np.ceil(
        app_settings.max_audio_duration_seconds
        * app_settings.target_sample_rate
    )
    )
    maximum_duration_bytes = maximum_duration_samples * bytes_per_sample
    decoded_memory_budget = app_settings.max_decoded_audio_size_mb * 1024 * 1024
    output_limit = int(min(maximum_duration_bytes, decoded_memory_budget) + bytes_per_sample)

    command = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-v",
        "error",
        "-i",
        str(path),
        "-map",
        f"0:{resolved_inspection.audio_stream_index}",
        "-vn",
        "-sn",
        "-dn",
        "-threads",
        "1",
        "-ac",
        "1",
        "-ar",
        str(app_settings.target_sample_rate),
        "-acodec",
        "pcm_f32le",
        "-f",
        "f32le",
        "-t",
        str(app_settings.max_audio_duration_seconds + 1),
        "-fs",
        str(output_limit),
        "pipe:1",
    ]
    with stage_timer("audio_decode", logger_name=__name__):
        result = _run_media_process(
            command,
            timeout_seconds=app_settings.ffmpeg_timeout_seconds,
            tool_name="ffmpeg",
            source_path=path,
            stdout_limit_bytes=output_limit,
        )
    if result.returncode != 0:
        _log_tool_failure("ffmpeg", result.stderr, path)
        raise CorruptedAudioError("Audio file could not be decoded.")
    if not result.stdout:
        raise CorruptedAudioError("Decoded audio contains no samples.")
    if len(result.stdout) > decoded_memory_budget:
        raise DecodedAudioTooLargeError(
            "Decoded audio exceeds the configured memory budget."
        )
    if len(result.stdout) > maximum_duration_bytes:
        raise AudioDurationExceededError(
            "Decoded audio duration exceeds the configured maximum."
        )
    if len(result.stdout) % bytes_per_sample != 0:
        raise CorruptedAudioError("Decoded audio output is incomplete.")

    waveform = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)
    if waveform.size > maximum_duration_samples:
        raise AudioDurationExceededError(
            "Decoded audio duration exceeds the configured maximum."
        )

    actual_duration = waveform.size / app_settings.target_sample_rate
    if actual_duration > app_settings.max_audio_duration_seconds:
        raise AudioDurationExceededError(
            "Decoded audio duration exceeds the configured maximum."
        )
    _validate_minimum_decoded_duration(actual_duration, app_settings)

    return DecodedAudio(
        waveform=waveform,
        sample_rate=app_settings.target_sample_rate,
        original_sample_rate=resolved_inspection.sample_rate,
        original_channels=resolved_inspection.channels,
    )


def _validate_minimum_decoded_duration(
    duration_seconds: float,
    app_settings: Settings,
) -> None:
    if duration_seconds < app_settings.min_audio_duration_seconds:
        raise AudioTooShortError(
            "Audio duration is shorter than the configured minimum."
        )


def _coerce_float32_waveform(waveform: np.ndarray) -> Float32Waveform:
    coerced = np.asarray(waveform, dtype=np.float32)
    if coerced.size == 0:
        raise UnusableAudioError("Decoded audio contains no samples.")
    if coerced.ndim not in {1, 2}:
        raise UnusableAudioError("Decoded audio must be mono or multi-channel.")
    if not np.any(np.isfinite(coerced)):
        raise UnusableAudioError("Decoded audio contains no finite samples.")
    return coerced


def _convert_to_mono(waveform: Float32Waveform) -> tuple[Float32Waveform, bool]:
    if waveform.ndim == 1:
        return waveform, False
    if waveform.shape[0] == 0 or waveform.shape[1] == 0:
        raise UnusableAudioError("Decoded audio has an empty channel dimension.")
    return np.mean(waveform, axis=1, dtype=np.float32), True


def _replace_invalid_values(waveform: Float32Waveform) -> Float32Waveform:
    cleaned = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0)
    return np.asarray(cleaned, dtype=np.float32)


def _resample_if_needed(
    waveform: Float32Waveform,
    *,
    sample_rate: int,
    target_sample_rate: int,
) -> Float32Waveform:
    if sample_rate == target_sample_rate:
        return waveform

    source_sample_count = waveform.shape[0]
    duration_seconds = source_sample_count / sample_rate
    target_sample_count = int(round(duration_seconds * target_sample_rate))
    if target_sample_count <= 0:
        raise UnusableAudioError("Resampled audio would contain no samples.")

    source_positions = np.linspace(
        0.0,
        duration_seconds,
        num=source_sample_count,
        endpoint=False,
        dtype=np.float64,
    )
    target_positions = np.linspace(
        0.0,
        duration_seconds,
        num=target_sample_count,
        endpoint=False,
        dtype=np.float64,
    )
    resampled = np.interp(target_positions, source_positions, waveform)
    return np.asarray(resampled, dtype=np.float32)


def _reject_silent_or_unusable_audio(waveform: Float32Waveform) -> Float32Waveform:
    peak_amplitude = float(np.max(np.abs(waveform)))
    rms_energy = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float32))))
    if peak_amplitude <= SILENCE_PEAK_THRESHOLD or rms_energy <= SILENCE_RMS_THRESHOLD:
        raise SilentAudioError("Decoded audio is silent or too close to silence.")
    return waveform


def _normalize_amplitude_safely(
    waveform: Float32Waveform,
) -> tuple[Float32Waveform, bool]:
    peak_amplitude = float(np.max(np.abs(waveform)))
    if peak_amplitude < MIN_NORMALIZATION_PEAK:
        return waveform, False
    scale = NORMALIZED_TARGET_PEAK / peak_amplitude
    if np.isclose(scale, 1.0, rtol=1e-6, atol=1e-8):
        return waveform, False
    return np.asarray(waveform * scale, dtype=np.float32), True


def _validate_original_filename(filename: str | None) -> str:
    if filename is None or not filename.strip():
        raise MissingAudioFilenameError("Audio filename is required.")
    return filename.strip()


def _get_allowed_extension(filename: str, app_settings: Settings) -> str:
    extension = Path(filename).suffix.lower().lstrip(".")
    if extension not in app_settings.allowed_audio_extension_list:
        raise UnsupportedAudioFormatError(f"Unsupported audio extension: {extension}")
    return extension


def _metadata_filename(
    *,
    display_filename: str | None,
    fallback_filename: str,
    extension: str,
) -> str:
    if display_filename is None or not display_filename.strip():
        return sanitize_audio_filename(fallback_filename)
    return sanitize_audio_display_filename(
        display_filename.strip(),
        extension=extension,
    )


def _prepare_upload_dir(app_settings: Settings) -> Path:
    upload_dir = app_settings.resolved_upload_dir.resolve()
    backend_root = app_settings.backend_root.resolve()
    configured_upload_dir = Path(app_settings.upload_dir)
    if not configured_upload_dir.is_absolute() and not upload_dir.is_relative_to(
        backend_root
    ):
        raise UnsafeAudioFilenameError(
            "Configured upload directory escapes backend root."
        )
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


async def _stream_upload_to_temp_file(
    *,
    upload_file: UploadFile,
    upload_dir: Path,
    max_upload_size_bytes: int,
) -> tuple[Path, int]:
    file_size_bytes = 0
    keep_file = False
    temp_file = tempfile.NamedTemporaryFile(
        dir=upload_dir,
        prefix=".upload-",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(temp_file.name)

    try:
        with temp_file:
            while chunk := await upload_file.read(CHUNK_SIZE_BYTES):
                file_size_bytes += len(chunk)
                if file_size_bytes > max_upload_size_bytes:
                    raise AudioFileTooLargeError(
                        "Audio file exceeds the configured maximum size."
                    )
                temp_file.write(chunk)
        if file_size_bytes == 0:
            raise EmptyAudioFileError("Audio file is empty.")
        keep_file = True
        return temp_path, file_size_bytes
    finally:
        if not keep_file:
            temp_path.unlink(missing_ok=True)


def _stream_upload_to_temp_file_blocking(
    *,
    upload_file: UploadFile,
    upload_dir: Path,
    max_upload_size_bytes: int,
) -> tuple[Path, int]:
    file_size_bytes = 0
    keep_file = False
    temp_file = tempfile.NamedTemporaryFile(
        dir=upload_dir,
        prefix=".upload-",
        suffix=".tmp",
        delete=False,
    )
    temp_path = Path(temp_file.name)

    try:
        with temp_file:
            upload_file.file.seek(0)
            while chunk := upload_file.file.read(CHUNK_SIZE_BYTES):
                file_size_bytes += len(chunk)
                if file_size_bytes > max_upload_size_bytes:
                    raise AudioFileTooLargeError(
                        "Audio file exceeds the configured maximum size."
                    )
                temp_file.write(chunk)
        if file_size_bytes == 0:
            raise EmptyAudioFileError("Audio file is empty.")
        keep_file = True
        return temp_path, file_size_bytes
    finally:
        if not keep_file:
            temp_path.unlink(missing_ok=True)


def _safe_upload_path(upload_dir: Path, filename: str) -> Path:
    saved_path = (upload_dir / filename).resolve()
    if saved_path.parent != upload_dir.resolve():
        raise UnsafeAudioFilenameError(
            "Resolved upload path escapes upload directory."
        )
    return saved_path


def _validate_detected_format(
    extension: str,
    detected_format: str,
    detected_codec: str,
) -> None:
    format_names = {
        item.strip().lower() for item in detected_format.split(",") if item.strip()
    }
    allowed_containers = EXTENSION_CONTAINER_RULES.get(extension, frozenset())
    if not format_names.intersection(allowed_containers):
        raise UnsupportedAudioFormatError(
            "File extension does not match the detected audio container."
        )

    codec_family = _codec_family(detected_codec)
    if codec_family not in SUPPORTED_CODEC_FAMILIES:
        raise UnsupportedAudioFormatError("Detected audio codec is not supported.")
    if codec_family not in EXTENSION_CODEC_RULES.get(extension, frozenset()):
        raise UnsupportedAudioFormatError(
            "Detected audio codec is not valid for the file extension."
        )


def _select_valid_audio_stream(
    audio_streams: list[dict],
    *,
    extension: str,
    detected_container: str,
) -> dict:
    last_format_error: UnsupportedAudioFormatError | None = None
    last_metadata_error: CorruptedAudioError | None = None

    for audio_stream in audio_streams:
        detected_codec = str(audio_stream.get("codec_name") or "").lower()
        try:
            _validate_detected_format(extension, detected_container, detected_codec)
        except UnsupportedAudioFormatError as error:
            last_format_error = error
            continue

        if (
            _positive_int(audio_stream.get("sample_rate")) is None
            or _positive_int(audio_stream.get("channels")) is None
        ):
            last_metadata_error = CorruptedAudioError(
                "Audio stream has invalid metadata."
            )
            continue
        return audio_stream

    if last_format_error is not None:
        raise last_format_error
    if last_metadata_error is not None:
        raise last_metadata_error
    raise AudioStreamValidationError("At least one usable audio stream is required.")


def _codec_family(codec_name: str) -> str:
    if codec_name.startswith("pcm_"):
        return "pcm"
    return codec_name


def _resolve_executable(configured_name: str) -> str | None:
    if not configured_name.strip():
        return None
    return shutil.which(configured_name)


def _require_executable(configured_name: str, *, display_name: str) -> str:
    executable = _resolve_executable(configured_name)
    if executable is None:
        raise AudioProcessingUnavailableError(
            f"{display_name} is not available.",
        )
    return executable


@lru_cache(maxsize=8)
def _media_tool_version(configured_name: str) -> str | None:
    executable = _resolve_executable(configured_name)
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [executable, "-version"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            shell=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or result.stderr)[:1024]
    first_line = output.decode("utf-8", errors="replace").splitlines()
    if not first_line:
        return None
    return re.sub(r"\s+", " ", first_line[0]).strip()[:200] or None


def _run_media_process(
    command: list[str],
    *,
    timeout_seconds: int,
    tool_name: str,
    source_path: Path,
    stdout_limit_bytes: int,
) -> subprocess.CompletedProcess[bytes]:
    try:
        with (
            tempfile.TemporaryFile(dir=source_path.parent) as stdout_file,
            tempfile.TemporaryFile(dir=source_path.parent) as stderr_file,
        ):
            try:
                result = subprocess.run(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    check=False,
                    shell=False,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                stderr_file.seek(0)
                _log_tool_failure(
                    tool_name,
                    stderr_file.read(MAX_TOOL_STDERR_BYTES + 1),
                    source_path,
                )
                raise AudioProcessingTimeoutError(
                    "Audio processing exceeded the configured timeout."
                ) from error

            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout = stdout_file.read(stdout_limit_bytes + 1)
            stderr = stderr_file.read(MAX_TOOL_STDERR_BYTES + 1)
            return subprocess.CompletedProcess(
                args=result.args,
                returncode=result.returncode,
                stdout=stdout,
                stderr=stderr,
            )
    except AudioProcessingTimeoutError:
        raise
    except OSError as error:
        logger.exception("%s could not be started.", tool_name)
        raise AudioProcessingUnavailableError(
            "Required audio processing tooling is unavailable."
        ) from error


def _log_tool_failure(tool_name: str, stderr: bytes, source_path: Path) -> None:
    sanitized = _sanitize_tool_error(stderr, source_path)
    if sanitized:
        logger.warning(
            "%s reported an audio processing error: %s",
            tool_name,
            sanitized,
        )
    else:
        logger.warning("%s reported an audio processing error.", tool_name)


def _sanitize_tool_error(stderr: bytes, source_path: Path) -> str:
    text = stderr.decode("utf-8", errors="replace")
    text = text.replace(str(source_path), "<audio-file>")
    text = text.replace(str(source_path.resolve()), "<audio-file>")
    text = re.sub(r"(?<![A-Za-z0-9])/[^\s:'\"]+", "<path>", text)
    text = re.sub(r"\b[A-Za-z]:\\[^\s:'\"]+", "<path>", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_SANITIZED_TOOL_ERROR_LENGTH]


def _positive_int(value) -> int | None:
    parsed = _non_negative_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _non_negative_int(value) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _first_positive_float(*values) -> float | None:
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed > 0:
            return parsed
    return None


def _optional_string(value) -> str | None:
    if value is None:
        return None
    parsed = str(value).strip()
    return parsed or None
