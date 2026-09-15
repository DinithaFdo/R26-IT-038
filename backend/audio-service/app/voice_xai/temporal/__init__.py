"""Real XLS-R attention-rollout localisation services."""

from typing import Any

from app.voice_xai.temporal.contracts import TemporalAnalysisResult

__all__ = ["XLSRTemporalAttentionService", "TemporalAnalysisResult"]


def __getattr__(name: str) -> Any:
    """Avoid importing the service while the classifier handoff loads types."""

    if name == "XLSRTemporalAttentionService":
        from app.voice_xai.temporal.xlsr_attention_service import (
            XLSRTemporalAttentionService,
        )

        return XLSRTemporalAttentionService
    raise AttributeError(name)
