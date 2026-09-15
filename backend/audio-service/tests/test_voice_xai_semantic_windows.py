import numpy as np
import pytest

from app.ingestion.audio import ProcessedAudio
from app.voice_xai.semantic.features import FEATURE_NAMES
from app.voice_xai.semantic.windows import (
    SemanticWindowConfig,
    SemanticWindowError,
    extract_semantic_feature_windows,
)


def test_semantic_feature_windows_use_deterministic_overlap_and_exact_timestamps() -> None:
    pytest.importorskip("librosa")
    pytest.importorskip("parselmouth")
    audio = _processed_audio(2.2)

    windows = extract_semantic_feature_windows(
        audio,
        config=SemanticWindowConfig(duration_seconds=1.0, overlap_seconds=0.5),
    )

    assert [(window.start_seconds, window.end_seconds) for window in windows] == [
        (0.0, 1.0),
        (0.5, 1.5),
        (1.0, 2.0),
        (1.2, 2.2),
    ]
    assert all(window.features.feature_names == FEATURE_NAMES for window in windows)
    assert all(window.features.values.shape == (len(FEATURE_NAMES),) for window in windows)
    assert all(
        window.features.metadata["semantic_window_start_seconds"]
        == window.start_seconds
        for window in windows
    )
    assert windows[-1].end_sample == len(audio.waveform)


def test_semantic_feature_windows_do_not_pad_short_clips() -> None:
    pytest.importorskip("librosa")
    pytest.importorskip("parselmouth")
    audio = _processed_audio(0.6)

    windows = extract_semantic_feature_windows(
        audio,
        config=SemanticWindowConfig(duration_seconds=1.0, overlap_seconds=0.5),
    )

    assert len(windows) == 1
    assert windows[0].start_seconds == 0.0
    assert windows[0].end_seconds == pytest.approx(0.6)
    assert windows[0].features.duration_seconds == pytest.approx(0.6)


def test_semantic_feature_windows_reject_clips_shorter_than_an_acoustic_frame() -> None:
    with pytest.raises(SemanticWindowError, match="too short"):
        extract_semantic_feature_windows(_processed_audio(0.01))


@pytest.mark.parametrize(
    ("duration_seconds", "overlap_seconds"),
    [(0.0, 0.0), (1.0, 1.0), (float("inf"), 0.5)],
)
def test_semantic_window_config_rejects_invalid_values(
    duration_seconds: float,
    overlap_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        SemanticWindowConfig(
            duration_seconds=duration_seconds,
            overlap_seconds=overlap_seconds,
        )


def _processed_audio(seconds: float) -> ProcessedAudio:
    sample_rate = 16_000
    sample_count = int(sample_rate * seconds)
    times = np.arange(sample_count, dtype=np.float32) / sample_rate
    waveform = (0.5 * np.sin(2 * np.pi * 180 * times)).astype(np.float32)
    return ProcessedAudio(
        waveform=waveform,
        sample_rate=sample_rate,
        original_sample_rate=sample_rate,
        original_channels=1,
        duration_seconds=seconds,
        was_resampled=False,
        was_converted_to_mono=False,
        normalisation_applied=True,
        peak_amplitude=0.5,
        rms_energy=float(np.sqrt(np.mean(waveform**2))),
    )
