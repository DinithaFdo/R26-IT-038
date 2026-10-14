"""Real XLS-R attention-rollout localization using pure per-clip p90."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from app.config.settings import Settings
from app.schemas.common import BranchStatus, ModelMode
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
        sample_rate = (
            inference.processed_audio.sample_rate
            if inference.processed_audio is not None
            else (
                inference.prediction.provenance.preprocessing.target_sample_rate
                if inference.prediction.provenance is not None
                else self._settings.target_sample_rate
            )
        )
        sample_count = (
            int(inference.processed_audio.waveform.size)
            if inference.processed_audio is not None
            else int(round(duration * sample_rate))
        )
        if sample_rate <= 0 or sample_count <= 0:
            raise TemporalAttentionError("Temporal audio sample boundaries are invalid.")
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
        final_region_mask = np.zeros(times.shape, dtype=bool)
        enriched_regions = []
        sample_intervals = []
        for region in high_attention_regions:
            start, end = region.start_seconds, region.end_seconds
            selected = covered & (times >= start) & (times <= end)
            final_region_mask |= selected
            start_sample = max(0, min(sample_count, int(round(start * sample_rate))))
            end_sample = max(start_sample, min(sample_count, int(round(end * sample_rate))))
            sample_intervals.append((start_sample, end_sample))
            enriched_regions.append(region.model_copy(update={
                "duration_seconds": end - start,
                "percentage_of_audio": 100.0 * (end - start) / duration,
                "mean_attention_density": float(np.mean(density[selected])),
                "start_sample": start_sample,
                "end_sample_exclusive": end_sample,
            }))
        high_attention_regions = enriched_regions
        # Count the union in sample coordinates, not selected-bin counts or
        # unrounded seconds. Merged gaps count; unknown evidence gaps do not.
        highlighted_samples = _sample_union_length(sample_intervals)
        highlighted_duration = highlighted_samples / sample_rate
        coverage_percentage = 100.0 * highlighted_samples / sample_count
        raw_bin_percentage = 100.0 * np.count_nonzero(high_attention_mask) / np.count_nonzero(covered)
        flat_attention_map = bool(float(np.ptp(density[covered])) == 0.0)
        research_blockers = _temporal_research_blockers(inference)
        research_eligible = not research_blockers
        explanation = TemporalExplanation(
            status=ComponentStatus.completed,
            method_version=TEMPORAL_METHOD_VERSION,
            model_version=_temporal_model_version(inference),
            development_placeholder=False,
            research_eligible=research_eligible,
            research_blockers=research_blockers,
            sample_rate_hz=sample_rate,
            audio_sample_count=sample_count,
            highlighted_samples=highlighted_samples,
            highlighted_duration_seconds=highlighted_duration,
            highlighted_coverage_percentage=coverage_percentage,
            raw_high_attention_bin_percentage=raw_bin_percentage,
            flat_attention_map=flat_attention_map,
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
                final_region_mask=final_region_mask,
                regions=tuple(region.model_dump(mode="json") for region in high_attention_regions),
                sample_rate_hz=sample_rate,
                audio_sample_count=sample_count,
                highlighted_samples=highlighted_samples,
                highlighted_duration_seconds=highlighted_duration,
                highlighted_coverage_percentage=coverage_percentage,
                raw_high_attention_bin_percentage=raw_bin_percentage,
                flat_attention_map=flat_attention_map,
                title="XLS-R original-forward attention rollout density (pure p90)",
            ),
            development_placeholder=False,
            research_eligible=research_eligible,
            warnings=(),
        )


def _sample_union_length(intervals: list[tuple[int, int]]) -> int:
    total = 0
    previous_end = 0
    for start, end in sorted(intervals):
        total += max(0, end - max(start, previous_end))
        previous_end = max(previous_end, end)
    return total


def _temporal_model_version(inference: ClassifierInferenceBundle) -> str | None:
    # Describe the checkpoint that produced the evidence, not a notebook model.
    provenance = inference.prediction.provenance
    if provenance is None:
        return None
    return next((
        model.model_version for model in provenance.models
        if model.model_name in {"ssl_wavlm_xlsr", "ssl_sequence"}
    ), None)


def _temporal_research_blockers(inference: ClassifierInferenceBundle) -> list[str]:
    """Use original prediction provenance, including on a persisted retry.

    Temporal eligibility concerns the SSL evidence-producing branch, independently
    of other fusion branches and semantic XAI. Configuration changes cannot
    retrospectively attest an unverified prediction.
    """
    branches = [
        branch for branch in inference.prediction.branches
        if branch.model_name in {"ssl_wavlm_xlsr", "ssl_sequence"}
    ]
    if len(branches) != 1:
        return ["ssl_branch_provenance_missing"]
    branch = branches[0]
    blockers = []
    if branch.mode != ModelMode.real:
        blockers.append("ssl_branch_not_real")
    if branch.status != BranchStatus.success:
        blockers.append("ssl_branch_not_successful")
    for field in ("preprocessing_verified", "class_mapping_verified", "research_result"):
        if branch.metadata.get(field) is not True:
            blockers.append(f"ssl_{field}_not_verified")
    return blockers


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
