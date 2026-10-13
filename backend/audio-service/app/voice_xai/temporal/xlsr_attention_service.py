"""Real XLS-R attention-rollout localization using pure per-clip p90."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from app.config.settings import Settings
from app.schemas.xai import ComponentStatus, TemporalExplanation, TemporalRegion
from app.voice_xai.integration.contracts import ClassifierInferenceBundle
from app.voice_xai.temporal.contracts import (
    TemporalAnalysisResult,
    TemporalAttentionError,
    TemporalVisualization,
)
from app.voice_xai.temporal.threshold import (
    TEMPORAL_METHOD_VERSION,
    TEMPORAL_THRESHOLD_PERCENTILE,
    TEMPORAL_THRESHOLD_QUANTILE,
    pure_p90_threshold,
)


class XLSRTemporalAttentionService:
    """Aggregate reduced evidence from the original XLS-R prediction pass."""

    def __init__(self, *, app_settings: Settings) -> None:
        self._settings = app_settings

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
        duration = max(
            duration,
            (
                inference.processed_audio.duration_seconds
                if inference.processed_audio is not None
                else inference.prediction.audio.duration_seconds
            ),
        )
        windows = [
            _validated_window(window, index=index, duration=duration)
            for index, window in enumerate(raw_windows)
        ]
        analyzed_spans = _analyzed_spans(windows)
        hop = self._settings.xai_temporal_global_hop_seconds
        times = np.arange(hop / 2.0, duration, hop, dtype=np.float32)
        if not times.size:
            times = np.array([duration / 2.0], dtype=np.float32)
        density_sum = np.zeros_like(times, dtype=np.float64)
        density_weights = np.zeros_like(times, dtype=np.float64)
        for window in windows:
            start = window["start_seconds"]
            end = window["end_seconds"]
            inside = (times >= start) & (times < end)
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
        covered = density_weights > 0.0
        density = np.full(times.shape, np.nan, dtype=np.float32)
        density[covered] = (density_sum[covered] / density_weights[covered]).astype(np.float32)
        if not np.isfinite(density[covered]).all():
            raise TemporalAttentionError(
                "Aggregated XLS-R temporal evidence is invalid.",
                code="xai_temporal_density_invalid",
            )
        # One label-neutral rule for both the explanation and its visualization.
        threshold = pure_p90_threshold(density[covered])
        high_attention_mask = covered & (density >= threshold)
        high_attention_regions = []
        # Process each evidence span separately: merging must not fill unknown gaps.
        for start, end in analyzed_spans:
            span_mask = high_attention_mask & (times >= start) & (times < end)
            regions = _regions(
                times, density, span_mask, duration,
                self._settings.xai_temporal_minimum_region_duration_seconds,
                self._settings.xai_temporal_merge_gap_seconds,
                span_start=start, span_end=end,
            )
            for region in regions:
                high_attention_regions.append(
                    region.model_copy(update={"region_id": len(high_attention_regions) + 1})
                )
        combined_duration = sum(
            r.end_seconds - r.start_seconds for r in high_attention_regions
        )
        explanation = TemporalExplanation(
            status=ComponentStatus.completed,
            method_version=TEMPORAL_METHOD_VERSION,
            model_version="xlsr-mamba-asvspoof2019-best",
            development_placeholder=False,
            research_eligible=False,
            attention_score_peak=float(density[covered].max()),
            threshold_percentile=TEMPORAL_THRESHOLD_PERCENTILE,
            attention_threshold=threshold,
            high_attention_region_count=len(high_attention_regions),
            high_attention_regions=high_attention_regions,
            high_attention_combined_duration_seconds=combined_duration,
            # Retain the existing UI fields as aliases of the same p90 result.
            visualization_threshold_percentile=TEMPORAL_THRESHOLD_PERCENTILE,
            visualization_attention_threshold=threshold,
            visualization_high_attention_region_count=len(high_attention_regions),
            visualization_high_attention_regions=high_attention_regions,
            visualization_high_attention_combined_duration_seconds=combined_duration,
            warning=(
                "Regions use the pure per-clip 90th percentile of attention density. "
                "Ties and region postprocessing can produce coverage other than 10%. "
                "These regions are not a ground-truth fake-audio percentage."
            ),
        )
        return TemporalAnalysisResult(
            explanation=explanation,
            visualization=TemporalVisualization(
                time_seconds=times,
                attention_scores=density,
                threshold=threshold,
                high_attention_mask=high_attention_mask,
                visualization_quantile=TEMPORAL_THRESHOLD_QUANTILE,
                analyzed_spans=analyzed_spans,
                duration_seconds=duration,
                title="XLS-R original-forward attention rollout density (pure p90)",
            ),
            development_placeholder=False,
            research_eligible=False,
            warnings=(),
        )


def _regions(
    times: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    duration: float,
    minimum: float,
    merge_gap: float,
    *,
    span_start: float = 0.0,
    span_end: float | None = None,
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
                    max(span_start, float(times[start] - dt / 2)),
                    min(duration if span_end is None else span_end, float(times[end] + dt / 2)),
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
        selected = (times >= start) & (times <= end) & np.isfinite(scores)
        result.append(
            TemporalRegion(
                region_id=len(result) + 1,
                start_seconds=start,
                end_seconds=end,
                attention_score=float(scores[selected].max()),
            )
        )
    return result


def _analyzed_spans(windows: list[dict[str, Any]]) -> tuple[tuple[float, float], ...]:
    """Union actual evidence bounds, retaining gaps with no model evidence."""

    merged: list[list[float]] = []
    for start, end in sorted(
        (window["start_seconds"], window["end_seconds"]) for window in windows
    ):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


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
