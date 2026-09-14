"""Bounded mel-spectrogram payloads for the user-facing temporal evidence UI."""

from __future__ import annotations

import math

import numpy as np

from app.ingestion.audio import ProcessedAudio
from app.schemas.xai import TemporalExplanation
from app.voice_xai.temporal.contracts import TemporalVisualization


def mel_spectrogram_payload(
    audio: ProcessedAudio,
    temporal: TemporalExplanation,
    visualization: TemporalVisualization,
    *,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    max_frames: int,
) -> dict[str, object]:
    """Create a size-bounded temporal-attention display artifact.

    Attention density is aligned to the mel-frame timeline so the client can
    render a continuous colour-intensity overlay before final candidate spans.
    It remains temporal evidence, not frequency-specific attribution.
    """

    waveform = np.asarray(audio.waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError("Processed audio waveform is invalid for spectrogram display.")
    if audio.sample_rate <= 0:
        raise ValueError("Processed audio sample rate is invalid for spectrogram display.")

    starts = _frame_starts(
        sample_count=waveform.size,
        n_fft=n_fft,
        hop_length=hop_length,
        max_frames=max_frames,
    )
    frames = _windowed_frames(waveform, starts, n_fft)
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)) ** 2
    filters, frequencies_hz = _mel_filterbank(
        sample_rate=audio.sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
    )
    mel_power = filters @ power.T
    reference = max(float(np.max(mel_power)), 1e-12)
    log_mel_db = np.maximum(10.0 * np.log10(np.maximum(mel_power, 1e-12) / reference), -80.0)
    frame_times = np.minimum(
        (starts.astype(np.float64) + n_fft / 2.0) / audio.sample_rate,
        audio.duration_seconds,
    )
    attention_times = np.asarray(visualization.time_seconds, dtype=np.float64)
    attention_scores = np.asarray(visualization.attention_scores, dtype=np.float64)
    if (
        attention_times.ndim != 1
        or attention_scores.ndim != 1
        or attention_times.size == 0
        or attention_times.size != attention_scores.size
        or not np.isfinite(attention_times).all()
        or not np.isfinite(attention_scores).all()
    ):
        raise ValueError("Temporal attention visualization is invalid for spectrogram display.")
    frame_attention = np.interp(
        frame_times,
        attention_times,
        attention_scores,
        left=float(attention_scores[0]),
        right=float(attention_scores[-1]),
    ).astype(np.float32)

    visualization_regions = temporal.visualization_high_attention_regions
    if visualization_regions is None:
        # Compatibility with explanations persisted before the display-only
        # threshold was introduced.
        visualization_regions = temporal.high_attention_regions

    return {
        "title": "Log-mel spectrogram with attention-density heatmap overlay",
        "audio_representation": "classifier_preprocessed_mono_waveform",
        "matrix_layout": "log_mel_db[mel_bin][time_frame]",
        "sample_rate_hz": audio.sample_rate,
        "duration_seconds": float(audio.duration_seconds),
        "n_fft": n_fft,
        "n_mels": n_mels,
        "frame_times_seconds": frame_times.tolist(),
        "mel_frequencies_hz": frequencies_hz.tolist(),
        "log_mel_db": log_mel_db.astype(np.float32).tolist(),
        "attention_density": frame_attention.tolist(),
        "attention_score_definition": "xlsr_attention_rollout_density_uniform_baseline_1",
        "attention_overlay_instruction": (
            "Render the continuous attention-density values as a colour-intensity "
            "overlay across the full mel-frequency axis. This is temporal "
            "attention, not frequency-specific attribution."
        ),
        "high_attention_region_count": len(visualization_regions),
        "high_attention_regions": [
            region.model_dump(mode="json") for region in visualization_regions
        ],
        "attention_region_overlay_instruction": (
            "Render the per-clip visualization high-attention regions as "
            "label-neutral time-span markers. They are not ground-truth "
            "manipulated-audio labels."
        ),
    }


def _frame_starts(
    *, sample_count: int, n_fft: int, hop_length: int, max_frames: int
) -> np.ndarray:
    unbounded_count = max(1, math.ceil(sample_count / hop_length))
    frame_count = min(unbounded_count, max_frames)
    last_start = max(0, sample_count - n_fft)
    if frame_count == 1:
        return np.array([0], dtype=np.int64)
    return np.rint(np.linspace(0, last_start, frame_count)).astype(np.int64)


def _windowed_frames(
    waveform: np.ndarray, starts: np.ndarray, n_fft: int
) -> np.ndarray:
    frames = np.zeros((len(starts), n_fft), dtype=np.float32)
    for index, start in enumerate(starts):
        source = waveform[int(start) : int(start) + n_fft]
        frames[index, : source.size] = source
    return frames * np.hanning(n_fft).astype(np.float32)


def _mel_filterbank(
    *, sample_rate: int, n_fft: int, n_mels: int
) -> tuple[np.ndarray, np.ndarray]:
    frequency_bins = np.linspace(0.0, sample_rate / 2.0, n_fft // 2 + 1)
    mel_points = np.linspace(_hz_to_mel(0.0), _hz_to_mel(sample_rate / 2.0), n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    filters = np.zeros((n_mels, frequency_bins.size), dtype=np.float32)
    centers = hz_points[1:-1]
    for index in range(n_mels):
        left, center, right = hz_points[index : index + 3]
        rising = (frequency_bins - left) / max(center - left, 1e-12)
        falling = (right - frequency_bins) / max(right - center, 1e-12)
        filters[index] = np.maximum(0.0, np.minimum(rising, falling))
    return filters, centers.astype(np.float32)


def _hz_to_mel(value: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(value) / 700.0)


def _mel_to_hz(value: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (value / 2595.0) - 1.0)
