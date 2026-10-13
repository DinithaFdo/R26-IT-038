"""Hermetic coverage for pure-p90 original-pass temporal attention."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from app.config.settings import Settings
from app.schemas.common import BranchStatus, ModelMode, PredictionLabel, SourceType
from app.schemas.prediction import (
    AudioMetadata, BranchPrediction, FusionResult, ProbabilityScores,
    VoicePredictionResponse,
)
from app.voice_xai.integration.contracts import ClassifierInferenceBundle
from app.voice_xai.temporal.contracts import (
    TemporalAttentionError, TemporalAttentionWindowInput,
)
from app.voice_xai.temporal.spectrogram import mel_spectrogram_payload
from app.voice_xai.temporal.threshold import pure_p90_threshold
from app.voice_xai.temporal.xlsr_attention_service import XLSRTemporalAttentionService


def test_p90_works_without_artifacts_and_matches_all_outputs(tmp_path) -> None:
    scores = np.linspace(0.1, 0.8, 100, dtype=np.float32)
    inference = _density_inference(scores)
    prediction_before = inference.prediction.model_dump(mode="json")
    result = _service(tmp_path).analyze(inference)
    explanation, visualization = result.explanation, result.visualization
    expected = float(np.float32(np.quantile(scores, 0.90, method="linear")))

    assert list(tmp_path.iterdir()) == []
    assert explanation.attention_threshold == expected
    assert explanation.attention_threshold < 1.0  # No density floor.
    np.testing.assert_allclose(visualization.attention_scores, scores)
    np.testing.assert_array_equal(visualization.high_attention_mask, scores >= expected)
    assert explanation.threshold_percentile == 90.0
    assert explanation.visualization_threshold_percentile == 90.0
    assert explanation.visualization_attention_threshold == expected
    assert visualization.threshold == expected
    assert visualization.visualization_quantile == 0.90
    assert explanation.high_attention_regions
    assert explanation.high_attention_regions == (
        explanation.visualization_high_attention_regions
    )
    assert explanation.high_attention_region_count == (
        explanation.visualization_high_attention_region_count
    )
    assert explanation.high_attention_combined_duration_seconds == (
        explanation.visualization_high_attention_combined_duration_seconds
    )
    assert explanation.development_placeholder is False
    assert explanation.research_eligible is False
    assert explanation.method_version.endswith("p90-v5")
    assert set(explanation.high_attention_regions[0].model_dump()) == {
        "region_id", "start_seconds", "end_seconds", "attention_score",
    }
    artifact = visualization.as_json_dict()
    assert artifact["threshold"] == expected
    assert artifact["threshold_role"] == "per_clip_percentile"
    assert "evaluation_threshold" not in artifact
    assert "evaluation_high_attention_mask" not in artifact
    json.dumps(artifact, allow_nan=False)
    assert inference.prediction.model_dump(mode="json") == prediction_before


def test_p90_uses_linear_quantile_and_float32_rounding() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.7], dtype=np.float32)
    assert pure_p90_threshold(scores) == float(np.float32(0.58))
    assert pure_p90_threshold(scores) != float(scores[-1])  # Not the higher method.
    assert pure_p90_threshold(scores) == float(np.float32(pure_p90_threshold(scores)))


@pytest.mark.parametrize("value", [0.0, 0.4, 1.0, 2.0])
def test_flat_attention_selects_all_tied_bins_without_floor(tmp_path, value) -> None:
    result = _service(tmp_path).analyze(
        _density_inference(np.full(100, value, dtype=np.float32))
    )
    assert result.visualization.threshold == float(np.float32(value))
    assert result.visualization.high_attention_mask.all()
    assert result.explanation.high_attention_region_count == 1
    assert "coverage other than 10%" in result.explanation.warning


def test_cutoff_is_per_clip_and_scale_does_not_change_selection(tmp_path) -> None:
    scores = np.linspace(0.1, 0.8, 100, dtype=np.float32)
    low = _service(tmp_path).analyze(_density_inference(scores))
    high = _service(tmp_path).analyze(_density_inference(scores * 10))
    assert low.visualization.threshold < 1.0 < high.visualization.threshold
    assert high.visualization.threshold == pytest.approx(low.visualization.threshold * 10)
    np.testing.assert_array_equal(
        low.visualization.high_attention_mask, high.visualization.high_attention_mask
    )


def test_p90_is_applied_after_overlapping_window_aggregation(tmp_path) -> None:
    scores = np.linspace(0.1, 1.0, 100, dtype=np.float32)
    first = _density_inference(scores).temporal_evidence[0]
    second = replace(first, attention_density=scores[::-1].copy())
    result = _service(tmp_path).analyze(
        _inference("overlap", temporal_evidence=(first, second))
    )
    assert result.visualization.threshold == pure_p90_threshold(
        result.visualization.attention_scores
    )
    assert result.visualization.threshold < pure_p90_threshold(scores)
    np.testing.assert_allclose(result.visualization.attention_scores, 0.55, atol=1e-7)


def test_highlights_do_not_depend_on_classifier_label_or_probability(tmp_path) -> None:
    scores = np.linspace(0.1, 1.2, 100, dtype=np.float32)
    inference = _density_inference(scores)
    service = _service(tmp_path)
    spoof = service.analyze(inference)
    probabilities = ProbabilityScores(bonafide=0.99, spoof=0.01)
    prediction = inference.prediction.model_copy(deep=True)
    prediction.branches[0].prediction = PredictionLabel.bonafide
    prediction.branches[0].probabilities = probabilities
    prediction.fusion.prediction = PredictionLabel.bonafide
    prediction.fusion.probabilities = probabilities
    window = replace(inference.temporal_evidence[0], spoof_probability=0.01)
    bonafide = service.analyze(
        replace(inference, prediction=prediction, temporal_evidence=(window,))
    )
    assert spoof.explanation.high_attention_regions == (
        bonafide.explanation.high_attention_regions
    )
    assert spoof.visualization.threshold == bonafide.visualization.threshold
    np.testing.assert_array_equal(
        spoof.visualization.high_attention_mask,
        bonafide.visualization.high_attention_mask,
    )


def test_region_postprocessing_merges_close_p90_spans(tmp_path) -> None:
    scores = np.full(40, 0.2, dtype=np.float32)
    scores[5:8] = scores[10:13] = 1.2
    result = _service(tmp_path).analyze(_density_inference(scores))
    assert np.count_nonzero(result.visualization.high_attention_mask) == 6
    assert result.explanation.high_attention_region_count == 1
    region = result.explanation.high_attention_regions[0]
    assert region.start_seconds == pytest.approx(0.10)
    assert region.end_seconds == pytest.approx(0.26)


def test_region_postprocessing_keeps_distant_p90_spans_separate(tmp_path) -> None:
    scores = np.full(40, 0.2, dtype=np.float32)
    scores[5:8] = scores[30:33] = 1.2
    result = _service(tmp_path, merge_gap=0.01).analyze(_density_inference(scores))
    assert result.explanation.high_attention_region_count == 2


def test_minimum_duration_can_remove_p90_selected_bins(tmp_path) -> None:
    result = _service(tmp_path, minimum=0.20).analyze(
        _density_inference(np.linspace(0.1, 0.8, 50, dtype=np.float32))
    )
    assert result.visualization.high_attention_mask.any()
    assert result.explanation.high_attention_regions == []
    assert result.explanation.high_attention_region_count == 0


def test_temporal_regions_remain_within_original_evidence_span(tmp_path) -> None:
    result = _service(tmp_path).analyze(
        _inference("cropped-span", temporal_evidence=(
            _window(0.0, 6.0, [0.5, 2.5, 5.5], [0.2, 1.6, 0.2]),
        ))
    )
    assert result.visualization.time_seconds[-1] < 6.0
    assert len(result.explanation.high_attention_regions) == 1
    region = result.explanation.high_attention_regions[0]
    assert 0.0 <= region.start_seconds < region.end_seconds <= 6.0


def test_spectrogram_uses_the_same_p90_regions(tmp_path) -> None:
    from app.ingestion.audio import ProcessedAudio

    result = _service(tmp_path).analyze(
        _density_inference(np.linspace(0.1, 0.8, 100, dtype=np.float32))
    )
    audio = ProcessedAudio(
        waveform=np.sin(np.arange(32000, dtype=np.float32) * 0.05),
        sample_rate=16000, original_sample_rate=16000, original_channels=1,
        duration_seconds=2.0, was_resampled=False, was_converted_to_mono=False,
        normalisation_applied=False, peak_amplitude=1.0, rms_energy=0.7,
    )
    payload = mel_spectrogram_payload(
        audio, result.explanation, result.visualization,
        n_mels=16, n_fft=128, hop_length=160, max_frames=32,
    )
    assert payload["high_attention_regions"] == [
        region.model_dump(mode="json") for region in result.explanation.high_attention_regions
    ]


def test_source_crop_has_no_zero_scores_or_highlights_outside_evidence(tmp_path) -> None:
    scores = np.linspace(0.1, 0.8, 300, dtype=np.float32)
    window = _window(
        7.0, 13.0, np.arange(0.01, 6.0, 0.02, dtype=np.float32), scores
    )
    inference = _inference("source-time", temporal_evidence=(window,))
    prediction = inference.prediction.model_copy(deep=True)
    prediction.audio.duration_seconds = 20.0
    inference = replace(inference, prediction=prediction)
    result = _service(tmp_path).analyze(inference)
    plot = result.visualization
    outside = (plot.time_seconds < 7.0) | (plot.time_seconds >= 13.0)
    inside = ~outside
    assert plot.duration_seconds == 20.0
    assert plot.time_seconds[-1] > 19.9
    assert plot.analyzed_spans == ((7.0, 13.0),)
    assert np.isnan(plot.attention_scores[outside]).all()
    assert not plot.high_attention_mask[outside].any()
    assert plot.threshold == pure_p90_threshold(plot.attention_scores[inside])
    assert plot.threshold < 1.0
    assert result.explanation.high_attention_regions
    assert all(
        7.0 <= region.start_seconds < region.end_seconds <= 13.0
        for region in result.explanation.high_attention_regions
    )
    artifact = plot.as_json_dict()
    assert artifact["time_reference"] == "source_audio"
    assert all(
        score is None for score, absent in zip(artifact["attention_scores"], outside)
        if absent
    )
    json.dumps(artifact, allow_nan=False)


def test_unknown_gap_is_not_interpolated_or_merged_into_a_region(tmp_path) -> None:
    result = _service(tmp_path).analyze(
        _inference("gap", temporal_evidence=(
            _window(0.0, 0.2, [0.01, 0.19], [1.0, 1.0]),
            _window(0.24, 0.44, [0.01, 0.19], [1.0, 1.0]),
        ))
    )
    plot = result.visualization
    assert result.explanation.high_attention_region_count == 2
    assert plot.analyzed_spans == ((0.0, 0.2), (0.24, 0.44))
    gap = (plot.time_seconds >= 0.2) & (plot.time_seconds < 0.24)
    assert np.isnan(plot.attention_scores[gap]).all()
    assert not plot.high_attention_mask[gap].any()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_mutated_nonfinite_evidence_is_rejected(tmp_path, value) -> None:
    inference = _density_inference(np.linspace(0.1, 0.8, 100, dtype=np.float32))
    inference.temporal_evidence[0].attention_density[0] = value
    with pytest.raises(TemporalAttentionError) as error:
        _service(tmp_path).analyze(inference)
    assert error.value.code == "xai_temporal_window_invalid"


@pytest.mark.parametrize("scores", [[], [np.nan], [np.inf], [[0.1, 0.2]]])
def test_p90_rejects_invalid_timelines(scores) -> None:
    with pytest.raises(TemporalAttentionError) as error:
        pure_p90_threshold(np.array(scores, dtype=np.float32))
    assert error.value.code == "xai_temporal_density_invalid"


def test_real_temporal_service_requires_original_forward_evidence(tmp_path) -> None:
    with pytest.raises(TemporalAttentionError) as error:
        _service(tmp_path).analyze(_inference("missing-evidence"))
    assert error.value.code == "xai_attention_evidence_missing"


def test_temporal_evidence_contract_rejects_invalid_window() -> None:
    with pytest.raises(TemporalAttentionError) as error:
        _window(0.0, 2.0, [0.75, 0.25], [1.2, 1.4])
    assert error.value.code == "xai_temporal_window_invalid"


def _density_inference(scores):
    duration = len(scores) * 0.02
    times = np.arange(0.01, duration, 0.02, dtype=np.float32)
    return _inference("p90-request", temporal_evidence=(
        _window(0.0, duration, times, scores),
    ))


def _inference(
    request_id: str,
    *,
    temporal_evidence: tuple[TemporalAttentionWindowInput, ...] | None = None,
) -> ClassifierInferenceBundle:
    probabilities = ProbabilityScores(bonafide=0.2, spoof=0.8)
    prediction = VoicePredictionResponse(
        request_id=request_id,
        audio=AudioMetadata(
            original_filename="sample.wav",
            content_type="audio/wav",
            file_size_bytes=32000,
            duration_seconds=2.0,
            sample_rate=16000,
            channels=1,
        ),
        branches=[
            BranchPrediction(
                model_name="ssl_wavlm_xlsr",
                display_name="SSL sequence",
                status=BranchStatus.success,
                mode=ModelMode.real,
                prediction=PredictionLabel.spoof,
                confidence=0.8,
                probabilities=probabilities,
                processing_time_ms=1.0,
            )
        ],
        fusion=FusionResult(
            status=BranchStatus.success,
            prediction=PredictionLabel.spoof,
            confidence=0.8,
            probabilities=probabilities,
            method="weighted_average",
            contains_dummy_branches=False,
            eligible_for_research_evaluation=False,
        ),
        total_processing_time_ms=1.0,
    )
    return ClassifierInferenceBundle(
        prediction_id="prediction-123",
        request_id=request_id,
        owner_user_id="user-123",
        source_type=SourceType.dashboard_upload,
        prediction=prediction,
        temporal_evidence=temporal_evidence,
    )


def _window(
    start: float,
    end: float,
    token_times: list[float],
    density: list[float],
    *,
    spoof_probability: float = 0.8,
) -> TemporalAttentionWindowInput:
    return TemporalAttentionWindowInput(
        start_seconds=start,
        end_seconds=end,
        token_times_seconds=np.array(token_times),
        attention_density=np.array(density),
        spoof_probability=spoof_probability,
    )


def _service(tmp_path, *, minimum=0.04, merge_gap=0.1):
    return XLSRTemporalAttentionService(
        app_settings=Settings(
            _env_file=None, model_root_dir=str(tmp_path / "models"),
            xai_temporal_minimum_region_duration_seconds=minimum,
            xai_temporal_merge_gap_seconds=merge_gap,
        )
    )
