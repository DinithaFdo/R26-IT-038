import numpy as np

from app.ingestion.audio import ProcessedAudio
from app.schemas.xai import ComponentStatus, TemporalExplanation, TemporalRegion
from app.voice_xai.temporal.spectrogram import mel_spectrogram_payload


def test_mel_spectrogram_payload_is_bounded_and_carries_candidate_overlays() -> None:
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
        regions=[
            TemporalRegion(
                region_id=1,
                start_seconds=0.5,
                end_seconds=1.25,
                attention_score=1.2,
                local_spoof_probability=0.8,
                local_bonafide_probability=0.2,
            )
        ],
    )

    payload = mel_spectrogram_payload(
        audio,
        temporal,
        n_mels=16,
        n_fft=128,
        hop_length=16,
        max_frames=32,
    )

    assert payload["matrix_layout"] == "log_mel_db[mel_bin][time_frame]"
    assert payload["regions_label"] == "candidate_spoof_evidence_regions"
    assert payload["candidate_region_count"] == 1
    assert payload["candidate_regions"][0]["local_spoof_probability"] == 0.8
    assert len(payload["log_mel_db"]) == 16
    assert len(payload["frame_times_seconds"]) <= 32
    assert all(len(row) == len(payload["frame_times_seconds"]) for row in payload["log_mel_db"])
