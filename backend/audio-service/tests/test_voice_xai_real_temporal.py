"""Hermetic coverage for the locked XLS-R temporal localisation path."""

from __future__ import annotations

import json

import numpy as np
import pytest

from app.config.settings import Settings
from app.schemas.common import BranchStatus, ModelMode, PredictionLabel, SourceType
from app.schemas.prediction import (
    AudioMetadata,
    BranchPrediction,
    FusionResult,
    ProbabilityScores,
    VoicePredictionResponse,
)
from app.voice_xai.contracts import ClassifierInferenceBundle
from app.voice_xai.temporal.calibration import (
    TemporalCalibrationError,
    load_fixed_density_threshold,
)
from app.voice_xai.temporal.contracts import (
    TemporalAttentionError,
    TemporalAttentionWindowInput,
)
from app.voice_xai.temporal.xlsr_attention_service import XLSRTemporalAttentionService


def _calibration(
    path,
    *,
    threshold: float = 1.1,
    smoothing_seconds: float = 0.08,
    minimum_region_duration_seconds: float = 0.04,
    merge_gap_seconds: float = 0.1,
) -> None:
    path.write_text(
        json.dumps(
            {
                "mode": "global_density",
                "value": threshold,
                "status": "CALIBRATED_ON_PARTIALSPOOF_V1_2_DEV",
                "pipeline": {
                    "target_sample_rate": 16000,
                    "max_window_seconds": 6.0,
                    "window_stride_seconds": 3.0,
                    "global_hop_seconds": 0.02,
                    "smoothing_seconds": smoothing_seconds,
                    "minimum_region_duration_seconds": minimum_region_duration_seconds,
                    "merge_gap_seconds": merge_gap_seconds,
                    "local_probability_aggregation": "cosine_weighted_overlap_mean",
                },
                "model_artifact": {"sha256": "expected-hash"},
            }
        ),
        encoding="utf-8",
    )


def test_locked_calibration_rejects_pipeline_drift(tmp_path) -> None:
    path = tmp_path / "threshold.json"
    _calibration(path)
    with pytest.raises(TemporalCalibrationError, match="target_sample_rate"):
        load_fixed_density_threshold(path, expected={"target_sample_rate": 8000.0})


def test_locked_calibration_rejects_model_hash_drift(tmp_path) -> None:
    path = tmp_path / "threshold.json"
    _calibration(path)

    with pytest.raises(TemporalCalibrationError, match="loaded temporal model"):
        load_fixed_density_threshold(
            path,
            expected={"target_sample_rate": 16000.0},
            expected_model_sha256="different-hash",
        )


def test_locked_calibration_accepts_matching_model_hash(tmp_path) -> None:
    path = tmp_path / "threshold.json"
    _calibration(path)

    threshold, _ = load_fixed_density_threshold(
        path,
        expected={"target_sample_rate": 16000.0},
        expected_model_sha256="expected-hash",
    )

    assert threshold == pytest.approx(1.1)


def test_real_temporal_service_uses_fixed_density_and_spoof_context(tmp_path) -> None:
    path = tmp_path / "threshold.json"
    _calibration(path)
    app_settings = Settings(
        xai_temporal_threshold_config_path=str(path),
        model_root_dir=str(tmp_path / "models"),
    )
    inference = _inference(
        "real-temporal-request",
        temporal_evidence=(
            TemporalAttentionWindowInput(
                start_seconds=0.0,
                end_seconds=2.0,
                token_times_seconds=np.array([0.25, 0.75, 1.25, 1.75]),
                attention_density=np.array([0.8, 1.2, 1.4, 0.9]),
                spoof_probability=0.8,
            ),
        ),
    )
    service = XLSRTemporalAttentionService(app_settings=app_settings)

    result = service.analyze(inference)

    assert result.development_placeholder is False
    assert result.explanation.attention_threshold == pytest.approx(1.1)
    assert result.explanation.regions
    assert result.explanation.regions_label == "candidate_spoof_evidence_regions"
    assert result.explanation.candidate_region_count == len(result.explanation.regions)
    assert result.explanation.regions[0].local_spoof_probability == pytest.approx(0.8)
    assert result.explanation.regions[0].local_bonafide_probability == pytest.approx(0.2)
    assert result.visualization.title.startswith("XLS-R original-forward")


def test_temporal_peak_below_threshold_has_zero_regions(tmp_path) -> None:
    path = tmp_path / "threshold.json"
    _calibration(path, threshold=1.5, smoothing_seconds=0.0)
    result = _service(tmp_path, path, smoothing_ms=0.0).analyze(
        _inference(
            "below-threshold",
            temporal_evidence=(
                _window(0.0, 1.0, [0.2, 0.4, 0.6, 0.8], [0.4, 0.8, 1.2, 0.7]),
            ),
        )
    )

    assert result.explanation.attention_score_peak < result.explanation.attention_threshold
    assert result.explanation.threshold_crossing_count == 0
    assert result.explanation.regions == []


def test_temporal_one_peak_above_threshold_has_region(tmp_path) -> None:
    path = tmp_path / "threshold.json"
    _calibration(path, threshold=1.0, smoothing_seconds=0.0)
    result = _service(tmp_path, path, smoothing_ms=0.0).analyze(
        _inference(
            "one-peak",
            temporal_evidence=(
                _window(0.0, 1.0, [0.2, 0.4, 0.6, 0.8], [0.2, 1.4, 0.3, 0.2]),
            ),
        )
    )

    assert result.explanation.attention_score_peak >= result.explanation.attention_threshold
    assert result.explanation.threshold_crossing_count > 0
    assert len(result.explanation.regions) == 1


def test_temporal_contiguous_points_merge_into_one_interval(tmp_path) -> None:
    path = tmp_path / "threshold.json"
    _calibration(path, threshold=1.0, smoothing_seconds=0.0)
    result = _service(tmp_path, path, smoothing_ms=0.0).analyze(
        _inference(
            "contiguous",
            temporal_evidence=(
                _window(0.0, 1.0, [0.2, 0.4, 0.6, 0.8], [0.2, 1.3, 1.4, 0.2]),
            ),
        )
    )

    assert result.explanation.raw_region_count == 1
    assert len(result.explanation.regions) == 1
    assert result.explanation.regions[0].duration_seconds >= 0.04


def test_temporal_separated_peaks_create_multiple_regions(tmp_path) -> None:
    path = tmp_path / "threshold.json"
    _calibration(path, threshold=1.0, smoothing_seconds=0.0, merge_gap_seconds=0.01)
    result = _service(tmp_path, path, smoothing_ms=0.0, merge_gap=0.01).analyze(
        _inference(
            "separated",
            temporal_evidence=(
                _window(0.0, 1.0, [0.2, 0.4, 0.6, 0.8], [1.4, 0.2, 0.3, 1.5]),
            ),
        )
    )

    assert result.explanation.raw_region_count == 2
    assert len(result.explanation.regions) == 2


def test_temporal_minimum_duration_filters_short_crossing(tmp_path) -> None:
    path = tmp_path / "threshold.json"
    _calibration(
        path,
        threshold=1.0,
        smoothing_seconds=0.0,
        minimum_region_duration_seconds=0.08,
    )
    result = _service(tmp_path, path, smoothing_ms=0.0, minimum=0.08).analyze(
        _inference(
            "short-crossing",
            temporal_evidence=(
                _window(0.0, 0.2, [0.01, 0.03, 0.05, 0.07], [0.2, 1.4, 0.2, 0.2]),
            ),
        )
    )

    assert result.explanation.threshold_crossing_count > 0
    assert result.explanation.raw_region_count > 0
    assert result.explanation.regions == []
    assert result.explanation.post_filter_region_count == 0


def test_temporal_invalid_nan_or_inf_values_are_rejected(tmp_path) -> None:
    path = tmp_path / "threshold.json"
    _calibration(path, threshold=1.0, smoothing_seconds=0.0)
    service = _service(tmp_path, path, smoothing_ms=0.0)

    with pytest.raises(TemporalAttentionError) as error:
        service.analyze(
            _inference(
                "nan-density",
                temporal_evidence=(
                    _window(0.0, 1.0, [0.2, 0.4], [0.8, np.nan]),
                ),
            )
        )

    assert error.value.code == "xai_temporal_window_invalid"


def test_temporal_cropped_input_duration_maps_regions_to_classifier_span(tmp_path) -> None:
    path = tmp_path / "threshold.json"
    _calibration(path, threshold=1.0, smoothing_seconds=0.0)
    result = _service(tmp_path, path, smoothing_ms=0.0).analyze(
        _inference(
            "cropped-span",
            temporal_evidence=(
                _window(0.0, 6.0, [0.5, 2.5, 5.5], [0.2, 1.6, 0.2]),
            ),
        )
    )

    assert result.visualization.time_seconds[-1] < 6.0
    assert len(result.explanation.regions) == 1
    region = result.explanation.regions[0]
    assert 0.0 <= region.start_seconds < region.end_seconds <= 6.0


def test_real_temporal_service_uses_configured_smoothing(tmp_path) -> None:
    unsmoothed_path = tmp_path / "threshold-unsmoothed.json"
    smoothed_path = tmp_path / "threshold-smoothed.json"
    _calibration(unsmoothed_path, threshold=1.05, smoothing_seconds=0.0)
    _calibration(smoothed_path, threshold=1.05, smoothing_seconds=0.08)
    inference = _inference(
        "smoothing-request",
        temporal_evidence=(
            TemporalAttentionWindowInput(
                start_seconds=0.0,
                end_seconds=2.0,
                token_times_seconds=np.array([0.25, 0.75, 1.25, 1.75]),
                attention_density=np.array([0.8, 1.8, 0.8, 0.8]),
                spoof_probability=0.8,
            ),
        ),
    )
    unsmoothed = XLSRTemporalAttentionService(
        app_settings=Settings(
            xai_temporal_threshold_config_path=str(unsmoothed_path),
            model_root_dir=str(tmp_path / "models"),
            xai_temporal_smoothing_ms=0.0,
        )
    ).analyze(inference)
    smoothed = XLSRTemporalAttentionService(
        app_settings=Settings(
            xai_temporal_threshold_config_path=str(smoothed_path),
            model_root_dir=str(tmp_path / "models"),
            xai_temporal_smoothing_ms=80.0,
        )
    ).analyze(inference)

    assert smoothed.explanation.attention_score_peak < (
        unsmoothed.explanation.attention_score_peak
    )


def test_real_temporal_service_requires_original_forward_evidence(tmp_path) -> None:
    path = tmp_path / "threshold.json"
    _calibration(path)
    app_settings = Settings(
        xai_temporal_threshold_config_path=str(path),
        model_root_dir=str(tmp_path / "models"),
    )
    inference = _inference("missing-temporal-evidence")
    service = XLSRTemporalAttentionService(app_settings=app_settings)

    with pytest.raises(TemporalAttentionError) as error:
        service.analyze(inference)

    assert error.value.code == "xai_attention_evidence_missing"


def test_temporal_evidence_contract_rejects_invalid_window() -> None:
    with pytest.raises(TemporalAttentionError) as error:
        TemporalAttentionWindowInput(
            start_seconds=0.0,
            end_seconds=2.0,
            token_times_seconds=np.array([0.75, 0.25]),
            attention_density=np.array([1.2, 1.4]),
            spoof_probability=0.8,
        )

    assert error.value.code == "xai_temporal_window_invalid"


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


def _service(
    tmp_path,
    calibration_path,
    *,
    smoothing_ms: float,
    minimum: float = 0.04,
    merge_gap: float = 0.1,
) -> XLSRTemporalAttentionService:
    return XLSRTemporalAttentionService(
        app_settings=Settings(
            xai_temporal_threshold_config_path=str(calibration_path),
            model_root_dir=str(tmp_path / "models"),
            xai_temporal_smoothing_ms=smoothing_ms,
            xai_temporal_minimum_region_duration_seconds=minimum,
            xai_temporal_merge_gap_seconds=merge_gap,
        )
    )
