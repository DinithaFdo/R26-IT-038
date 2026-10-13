import numpy as np

from app.ingestion.audio import ProcessedAudio
from app.schemas.xai import ComponentStatus, TemporalExplanation, TemporalRegion
from app.voice_xai.temporal.contracts import TemporalVisualization
from app.voice_xai.temporal.spectrogram import mel_spectrogram_payload


def test_mel_spectrogram_payload_aligns_density_and_visualization_regions() -> None:
    sample_rate = 16_000
    audio = ProcessedAudio(
        waveform=np.sin(np.arange(sample_rate * 3, dtype=np.float32) * 0.05),
        sample_rate=sample_rate,
        original_sample_rate=sample_rate,
        original_channels=1,
        duration_seconds=3.0,
        was_resampled=False,
        was_converted_to_mono=False,
        normalisation_applied=True,
        peak_amplitude=0.95,
        rms_energy=0.5,
    )
    temporal = TemporalExplanation(
        status=ComponentStatus.completed,
        method_version="fixture-v1",
        attention_threshold=0.5,
        high_attention_regions=[
            TemporalRegion(
                region_id=1,
                start_seconds=0.5,
                end_seconds=1.25,
                attention_score=1.2,
            )
        ],
        visualization_high_attention_regions=[
            TemporalRegion(
                region_id=1,
                start_seconds=0.75,
                end_seconds=1.0,
                attention_score=1.2,
            )
        ],
    )
    visualization = TemporalVisualization(
        time_seconds=np.array([0.0, 0.5, 1.25, 3.0], dtype=np.float32),
        attention_scores=np.array([0.8, 1.2, 1.2, 0.9], dtype=np.float32),
        threshold=1.1,
        high_attention_mask=np.array([False, True, True, False]),
        visualization_quantile=0.90,
    )

    payload = mel_spectrogram_payload(
        audio,
        temporal,
        visualization,
        n_mels=16,
        n_fft=128,
        hop_length=16,
        max_frames=32,
    )

    assert payload["matrix_layout"] == "log_mel_db[mel_bin][time_frame]"
    assert payload["high_attention_region_count"] == 1
    assert payload["high_attention_regions"] == [
        region.model_dump(mode="json")
        for region in temporal.visualization_high_attention_regions
    ]
    assert payload["attention_score_definition"] == (
        "xlsr_attention_rollout_density_uniform_baseline_1"
    )
    np.testing.assert_allclose(
        payload["attention_density"],
        np.interp(
            payload["frame_times_seconds"],
            visualization.time_seconds,
            visualization.attention_scores,
        ),
        rtol=1e-6,
    )
    assert len(payload["attention_density"]) == len(payload["frame_times_seconds"])
    assert len(payload["log_mel_db"]) == 16
    assert len(payload["frame_times_seconds"]) <= 32
    assert all(
        len(row) == len(payload["frame_times_seconds"])
        for row in payload["log_mel_db"]
    )
