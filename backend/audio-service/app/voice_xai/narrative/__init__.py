"""Optional, validated natural-language rendering of Voice XAI evidence.

This package deliberately consumes only persisted/validated XAI structures.
It never receives audio, tensors, user identifiers, or classifier controls.
"""

from app.voice_xai.narrative.service import (
    AlibabaQwenNarrativeService,
    NarrativeGenerationError,
)

__all__ = ["AlibabaQwenNarrativeService", "NarrativeGenerationError"]
