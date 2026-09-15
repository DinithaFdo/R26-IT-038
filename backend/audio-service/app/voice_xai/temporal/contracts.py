"""Private result contracts for real XLS-R temporal localisation."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.schemas.xai import TemporalExplanation


@dataclass(frozen=True)
class TemporalAttentionWindowInput:
    """Reduced XLS-R evidence created during the classifier forward pass.

    This is deliberately the only temporal representation allowed to cross
    from the model worker to Voice XAI.  It contains a compact per-token
    density, never raw per-head attention matrices.
    """

    start_seconds: float
    end_seconds: float
    token_times_seconds: NDArray[np.float32]
    attention_density: NDArray[np.float32]
    # Kept in the handoff contract for compatibility with the current
    # SSL-sequence model worker. The restored v3 attention algorithm is
    # intentionally label-neutral and does not use this value to select
    # high-attention regions.
    spoof_probability: float = 0.5

    def __post_init__(self) -> None:
        try:
            start = float(self.start_seconds)
            end = float(self.end_seconds)
            probability = float(self.spoof_probability)
            times = np.asarray(self.token_times_seconds, dtype=np.float32).reshape(-1)
            density = np.asarray(self.attention_density, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError, OverflowError) as error:
            raise TemporalAttentionError(
                "Original-pass temporal evidence is invalid.",
                code="xai_temporal_window_invalid",
            ) from error
        if (
            not np.isfinite(start)
            or not np.isfinite(end)
            or end <= start
            or not np.isfinite(probability)
            or not 0.0 <= probability <= 1.0
            or times.size == 0
            or times.size != density.size
            or not np.isfinite(times).all()
            or not np.isfinite(density).all()
            or np.any(np.diff(times) <= 0.0)
            or times[0] < 0.0
            or times[-1] > (end - start) + 1e-6
        ):
            raise TemporalAttentionError(
                "Original-pass temporal evidence is invalid.",
                code="xai_temporal_window_invalid",
            )

    def as_mapping(self) -> dict[str, Any]:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "token_times_seconds": self.token_times_seconds,
            "attention_density": self.attention_density,
            "spoof_probability": self.spoof_probability,
        }


class TemporalAttentionError(ValueError):
    """Raised when temporal attention evidence cannot be safely persisted.

    The code is a small, safe-to-return diagnostic category. The chained cause
    is retained for server logs only and never sent through the API.
    """

    def __init__(self, message: str, *, code: str = "temporal_analysis_invalid") -> None:
        super().__init__(message)
        self.code = (
            code
            if isinstance(code, str) and re.fullmatch(r"[a-z0-9_-]{1,80}", code)
            else "temporal_analysis_invalid"
        )


@dataclass(frozen=True)
class TemporalVisualization:
    """Private dashboard data; raw attention tensors never leave the model layer."""

    time_seconds: NDArray[np.float32]
    attention_scores: NDArray[np.float32]
    threshold: float
    high_attention_mask: NDArray[np.bool_]
    title: str = "XLS-R attention rollout density"
    evaluation_threshold: float | None = None
    evaluation_high_attention_mask: NDArray[np.bool_] | None = None
    visualization_quantile: float | None = None

    def as_json_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "time_seconds": self.time_seconds.tolist(),
            "attention_scores": self.attention_scores.tolist(),
            "threshold": self.threshold,
            "high_attention_mask": self.high_attention_mask.astype(int).tolist(),
            "evaluation_threshold": self.evaluation_threshold,
            "evaluation_high_attention_mask": (
                self.evaluation_high_attention_mask.astype(int).tolist()
                if self.evaluation_high_attention_mask is not None
                else None
            ),
            "visualization_quantile": self.visualization_quantile,
            "threshold_role": (
                "per_clip_visualization"
                if self.visualization_quantile is not None
                else "unspecified"
            ),
        }


@dataclass(frozen=True)
class TemporalAnalysisResult:
    """Internal real-temporal result; public APIs receive only its explanation."""

    explanation: TemporalExplanation
    visualization: TemporalVisualization
    development_placeholder: bool
    research_eligible: bool
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.explanation.development_placeholder != self.development_placeholder:
            raise TemporalAttentionError(
                "Temporal result and explanation placeholder flags differ."
            )
        if self.explanation.research_eligible != self.research_eligible:
            raise TemporalAttentionError(
                "Temporal result and explanation eligibility flags differ."
            )
