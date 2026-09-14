"""Real XLS-R attention-rollout localization using the locked calibration."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from app.config.settings import Settings
from app.schemas.xai import ComponentStatus, TemporalExplanation, TemporalRegion
from app.voice_xai.contracts import ClassifierInferenceBundle
from app.voice_xai.temporal.calibration import load_fixed_density_threshold
from app.voice_xai.temporal.contracts import (
    TemporalAnalysisResult,
    TemporalAttentionError,
    TemporalVisualization,
)

class XLSRTemporalAttentionService:
    """Aggregate reduced evidence from the original XLS-R prediction pass."""

    def __init__(self, *, app_settings: Settings) -> None:
        self._settings = app_settings
        expected = {
            "target_sample_rate": float(app_settings.target_sample_rate),
            "max_window_seconds": 6.0,
            "window_stride_seconds": app_settings.xai_temporal_window_stride_seconds,
            "global_hop_seconds": app_settings.xai_temporal_global_hop_seconds,
            "minimum_region_duration_seconds": (
                app_settings.xai_temporal_minimum_region_duration_seconds
            ),
            "merge_gap_seconds": app_settings.xai_temporal_merge_gap_seconds,
        }
        self._threshold, _ = load_fixed_density_threshold(
            app_settings.resolve_xai_artifact_path(
                app_settings.xai_temporal_threshold_config_path
            ),
            expected=expected,
        )

    def analyze(self, inference: ClassifierInferenceBundle) -> TemporalAnalysisResult:
        if not inference.temporal_evidence:
            raise TemporalAttentionError(
                "Original-pass XLS-R temporal evidence is unavailable.",
                code="xai_attention_evidence_missing",
            )
        raw_windows = [window.as_mapping() for window in inference.temporal_evidence]
        duration = max(
            _finite_positive(
                window["end_seconds"],
                name="attention window end",
                code="xai_temporal_window_invalid",
            )
            for window in raw_windows
        )
        hop = self._settings.xai_temporal_global_hop_seconds
        times = np.arange(hop / 2.0, duration, hop, dtype=np.float32)
        if not times.size:
            times = np.array([duration / 2.0], dtype=np.float32)
        density_sum = np.zeros_like(times, dtype=np.float64)
        density_weights = np.zeros_like(times, dtype=np.float64)
        for index, raw_window in enumerate(raw_windows):
            window = _validated_window(raw_window, index=index, duration=duration)
            start = window["start_seconds"]
            end = window["end_seconds"]
            inside = (times >= start) & (times <= end)
            if not inside.any():
                continue
            local_times = window["token_times_seconds"]
            local_density = window["attention_density"]
            interpolated = np.interp(
                times[inside] - start, local_times, local_density
            )
            if not np.isfinite(interpolated).all():
                raise TemporalAttentionError(
                    "Interpolated XLS-R attention density is invalid.",
                    code="xai_temporal_density_invalid",
                )
            u = np.clip(
                (times[inside] - start) / max(end - start, 1e-12), 0.0, 1.0
            )
            weights = np.maximum(0.5 - 0.5 * np.cos(2.0 * np.pi * u), 0.1)
            density_sum[inside] += interpolated * weights
            density_weights[inside] += weights
        if not density_weights.any():
            raise TemporalAttentionError(
                "XLS-R attention windows do not cover the processed audio.",
                code="xai_temporal_windows_uncovered",
            )
        density = (density_sum / np.maximum(density_weights, 1e-12)).astype(np.float32)
        if not np.isfinite(density).all():
            raise TemporalAttentionError(
                "Aggregated XLS-R temporal evidence is invalid.",
                code="xai_temporal_density_invalid",
            )
        # Fixed PartialSpoof DEV-calibrated evaluation mask. It remains
        # independent of the classifier label because attention is label-neutral.
        evaluation_high_attention_mask = density >= self._threshold
        evaluation_high_attention_regions = _regions(
            times,
            density,
            evaluation_high_attention_mask,
            duration,
            self._settings.xai_temporal_minimum_region_duration_seconds,
            self._settings.xai_temporal_merge_gap_seconds,
        )
        visualization_quantile = self._settings.xai_temporal_visualization_quantile
        visualization_threshold = _visualization_threshold(
            density,
            quantile=visualization_quantile,
            minimum_density=(
                self._settings.xai_temporal_visualization_minimum_density
            ),
        )
        visualization_high_attention_mask = density >= visualization_threshold
        visualization_high_attention_regions = _regions(
            times,
            density,
            visualization_high_attention_mask,
            duration,
            self._settings.xai_temporal_minimum_region_duration_seconds,
            self._settings.xai_temporal_merge_gap_seconds,
        )
        explanation = TemporalExplanation(
            status=ComponentStatus.completed,
            method_version="xlsr-mamba-windowed-attention-rollout-v3",
            model_version="xlsr-mamba-asvspoof2019-best",
            development_placeholder=False,
            research_eligible=False,
            attention_score_peak=float(density.max()),
            threshold_percentile=None,
            attention_threshold=self._threshold,
            high_attention_region_count=len(evaluation_high_attention_regions),
            high_attention_regions=evaluation_high_attention_regions,
            high_attention_combined_duration_seconds=sum(
                r.end_seconds - r.start_seconds
                for r in evaluation_high_attention_regions
            ),
            visualization_threshold_percentile=100.0 * visualization_quantile,
            visualization_attention_threshold=visualization_threshold,
            visualization_high_attention_region_count=len(
                visualization_high_attention_regions
            ),
            visualization_high_attention_regions=visualization_high_attention_regions,
            visualization_high_attention_combined_duration_seconds=sum(
                r.end_seconds - r.start_seconds
                for r in visualization_high_attention_regions
            ),
            warning=(
                "Evaluation regions use the fixed PartialSpoof DEV-calibrated "
                "threshold. Visualization regions use a separate per-clip top-"
                f"{100.0 * (1.0 - visualization_quantile):g}% attention rule. "
                "Neither region type is a ground-truth fake-audio percentage."
            ),
        )
        return TemporalAnalysisResult(
            explanation=explanation,
            visualization=TemporalVisualization(
                time_seconds=times,
                attention_scores=density,
                threshold=visualization_threshold,
                high_attention_mask=visualization_high_attention_mask,
                evaluation_threshold=self._threshold,
                evaluation_high_attention_mask=evaluation_high_attention_mask,
                visualization_quantile=visualization_quantile,
                title=(
                    "XLS-R 6-second windowed attention rollout density "
                    "(per-clip visualization threshold)"
                ),
            ),
            development_placeholder=False,
            research_eligible=False,
            warnings=(),
        )


def _visualization_threshold(
    density: np.ndarray,
    *,
    quantile: float,
    minimum_density: float,
) -> float:
    """Return the per-clip display threshold without altering evaluation."""

    finite_density = density[np.isfinite(density)]
    if finite_density.size == 0:
        raise TemporalAttentionError(
            "Aggregated XLS-R temporal evidence is invalid.",
            code="xai_temporal_density_invalid",
        )
    percentile_value = float(np.quantile(finite_density, quantile))
    return float(max(minimum_density, percentile_value))


def _regions(
    times: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    duration: float,
    minimum: float,
    merge_gap: float,
) -> list[TemporalRegion]:
    dt = float(np.median(np.diff(times))) if times.size > 1 else duration
    spans: list[list[float]] = []
    start: int | None = None
    for index, selected in enumerate(mask):
        if selected and start is None:
            start = index
        if start is not None and (not selected or index == len(mask) - 1):
            end = index if selected else index - 1
            spans.append(
                [
                    max(0.0, float(times[start] - dt / 2)),
                    min(duration, float(times[end] + dt / 2)),
                ]
            )
            start = None
    merged: list[list[float]] = []
    for span in spans:
        if merged and span[0] - merged[-1][1] <= merge_gap:
            merged[-1][1] = span[1]
        else:
            merged.append(span)
    result = []
    for start, end in merged:
        if end - start < minimum:
            continue
        selected = (times >= start) & (times <= end)
        result.append(
            TemporalRegion(
                region_id=len(result) + 1,
                start_seconds=start,
                end_seconds=end,
                attention_score=float(scores[selected].max()),
            )
        )
    return result


def _finite_positive(value: Any, *, name: str, code: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TemporalAttentionError(f"{name} is invalid.", code=code) from error
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise TemporalAttentionError(f"{name} is invalid.", code=code)
    return parsed


def _validated_window(
    raw_window: Any, *, index: int, duration: float
) -> dict[str, Any]:
    """Validate the reduced model-boundary contract before interpolation."""

    if not isinstance(raw_window, Mapping):
        raise TemporalAttentionError(
            f"Attention window {index} is invalid.", code="xai_temporal_window_invalid"
        )
    required = {
        "start_seconds",
        "end_seconds",
        "token_times_seconds",
        "attention_density",
    }
    if not required.issubset(raw_window):
        raise TemporalAttentionError(
            f"Attention window {index} is incomplete.", code="xai_temporal_window_invalid"
        )
    start = _finite_nonnegative(raw_window["start_seconds"], name="window start")
    end = _finite_positive(
        raw_window["end_seconds"],
        name="window end",
        code="xai_temporal_window_invalid",
    )
    if end <= start or end > duration + 1e-6:
        raise TemporalAttentionError(
            f"Attention window {index} has invalid bounds.",
            code="xai_temporal_window_invalid",
        )
    try:
        token_times = np.asarray(
            raw_window["token_times_seconds"], dtype=np.float32
        ).reshape(-1)
        density = np.asarray(
            raw_window["attention_density"], dtype=np.float32
        ).reshape(-1)
    except (TypeError, ValueError, OverflowError) as error:
        raise TemporalAttentionError(
            f"Attention window {index} has invalid values.",
            code="xai_temporal_window_invalid",
        ) from error
    if (
        token_times.size == 0
        or token_times.size != density.size
        or not np.isfinite(token_times).all()
        or not np.isfinite(density).all()
    ):
        raise TemporalAttentionError(
            f"Attention window {index} has invalid values.",
            code="xai_temporal_window_invalid",
        )
    window_duration = end - start
    if (
        np.any(np.diff(token_times) <= 0.0)
        or token_times[0] < 0.0
        or token_times[-1] > window_duration + 1e-6
    ):
        raise TemporalAttentionError(
            f"Attention window {index} has invalid token timing.",
            code="xai_temporal_window_invalid",
        )
    return {
        "start_seconds": start,
        "end_seconds": end,
        "token_times_seconds": token_times,
        "attention_density": density,
    }


def _finite_nonnegative(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TemporalAttentionError(
            f"{name} is invalid.", code="xai_temporal_window_invalid"
        ) from error
    if not np.isfinite(parsed) or parsed < 0.0:
        raise TemporalAttentionError(
            f"{name} is invalid.", code="xai_temporal_window_invalid"
        )
    return parsed
