import numpy as np
import pytest

from app.voice_xai.semantic.features import (
    FEATURE_NAMES,
    AcousticFeatureExtractor,
    FeatureExtractionError,
)


def test_acoustic_feature_extractor_returns_semantic_notebook_contract() -> None:
    pytest.importorskip("librosa")
    pytest.importorskip("parselmouth")
    sample_rate = 16_000
    seconds = 1.0
    times = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    waveform = (0.5 * np.sin(2 * np.pi * 160 * times)).astype(np.float32)

    result = AcousticFeatureExtractor().extract_waveform(
        waveform, sample_rate=sample_rate
    )

    assert result.feature_names == FEATURE_NAMES
    assert result.values.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(result.values).all()
    assert set(result.as_feature_mapping()) == set(FEATURE_NAMES)
    assert result.metadata["feature_count"] == len(FEATURE_NAMES)


def test_acoustic_feature_extractor_resamples_like_the_training_notebook() -> None:
    pytest.importorskip("librosa")
    pytest.importorskip("parselmouth")
    pytest.importorskip("resampy")
    times = np.arange(8_000, dtype=np.float32) / 8_000
    waveform = (0.5 * np.sin(2 * np.pi * 160 * times)).astype(np.float32)

    result = AcousticFeatureExtractor().extract_waveform(waveform, sample_rate=8_000)

    assert result.sample_rate == 16_000


def test_acoustic_feature_extractor_rejects_non_finite_waveforms() -> None:
    waveform = np.ones(800, dtype=np.float32)
    waveform[10] = np.nan

    with pytest.raises(FeatureExtractionError, match="non-finite"):
        AcousticFeatureExtractor().extract_waveform(waveform, sample_rate=16_000)


def test_acoustic_feature_result_is_json_serializable() -> None:
    pytest.importorskip("librosa")
    pytest.importorskip("parselmouth")
    times = np.arange(16_000, dtype=np.float32) / 16_000
    waveform = (0.5 * np.sin(2 * np.pi * 180 * times)).astype(np.float32)

    result = AcousticFeatureExtractor().extract_waveform(waveform, sample_rate=16_000)

    payload = result.as_json_dict()
    assert payload["extractor_version"]
    assert payload["features"]["lfcc_20"] == pytest.approx(
        result.as_feature_mapping()["lfcc_20"]
    )
