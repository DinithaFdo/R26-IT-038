"""The single operational threshold rule for temporal attention."""

from __future__ import annotations

import numpy as np

from app.voice_xai.temporal.contracts import TemporalAttentionError

TEMPORAL_THRESHOLD_MODE = "per_clip_percentile"
TEMPORAL_THRESHOLD_PERCENTILE = 90.0
TEMPORAL_THRESHOLD_QUANTILE = TEMPORAL_THRESHOLD_PERCENTILE / 100.0
TEMPORAL_PERCENTILE_METHOD = "linear_float32_comparison"
TEMPORAL_METHOD_VERSION = "xlsr-original-forward-attention-rollout-source-time-p90-v6"


def pure_p90_threshold(density: np.ndarray) -> float:
    """Match the notebook's linear p90 and float32 ``>=`` comparison.

    No density floor, probability gate, or external threshold artifact is used.
    Ties (including a flat map) are deliberately retained by the caller's >=.
    """

    scores = np.asarray(density, dtype=np.float32)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise TemporalAttentionError(
            "Aggregated XLS-R temporal evidence is invalid.",
            code="xai_temporal_density_invalid",
        )
    return float(
        np.float32(np.quantile(scores, TEMPORAL_THRESHOLD_QUANTILE, method="linear"))
    )
