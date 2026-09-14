"""Bounded private serialization for compact original-pass XLS-R evidence."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from app.voice_xai.temporal.contracts import (
    TemporalAttentionError,
    TemporalAttentionWindowInput,
)

TEMPORAL_EVIDENCE_ARTIFACT_CONTENT_TYPE = (
    "application/vnd.voice-xai.temporal-evidence+json"
)
TEMPORAL_EVIDENCE_ARTIFACT_VERSION = "xlsr-temporal-evidence-v2"
MAX_TEMPORAL_EVIDENCE_ARTIFACT_BYTES = 128 * 1024
MAX_TEMPORAL_EVIDENCE_WINDOWS = 64
MAX_TEMPORAL_EVIDENCE_TOKENS_PER_WINDOW = 4_096


def serialize_temporal_evidence(
    evidence: tuple[TemporalAttentionWindowInput, ...],
) -> bytes:
    """Serialize only reduced per-token density; raw attention is never accepted."""

    windows = _validated_windows(evidence)
    payload = {
        "version": TEMPORAL_EVIDENCE_ARTIFACT_VERSION,
        "windows": [
            {
                "start_seconds": float(window.start_seconds),
                "end_seconds": float(window.end_seconds),
                "token_times_seconds": window.token_times_seconds.astype(
                    np.float32, copy=False
                ).tolist(),
                "attention_density": window.attention_density.astype(
                    np.float32, copy=False
                ).tolist(),
                "spoof_probability": float(window.spoof_probability),
            }
            for window in windows
        ],
    }
    try:
        content = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TemporalAttentionError(
            "Original-pass temporal evidence could not be persisted.",
            code="xai_temporal_evidence_invalid",
        ) from error
    if len(content) > MAX_TEMPORAL_EVIDENCE_ARTIFACT_BYTES:
        raise TemporalAttentionError(
            "Original-pass temporal evidence exceeds its storage limit.",
            code="xai_temporal_evidence_too_large",
        )
    return content


def deserialize_temporal_evidence(
    content: bytes,
) -> tuple[TemporalAttentionWindowInput, ...]:
    """Load and validate a compact timeline artifact for a manual retry."""

    if (
        not isinstance(content, bytes)
        or not content
        or len(content) > MAX_TEMPORAL_EVIDENCE_ARTIFACT_BYTES
    ):
        raise TemporalAttentionError(
            "Stored original-pass temporal evidence is invalid.",
            code="xai_temporal_evidence_invalid",
        )
    try:
        payload: Any = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TemporalAttentionError(
            "Stored original-pass temporal evidence is invalid.",
            code="xai_temporal_evidence_invalid",
        ) from error
    if (
        not isinstance(payload, dict)
        or set(payload) != {"version", "windows"}
        or payload.get("version") != TEMPORAL_EVIDENCE_ARTIFACT_VERSION
        or not isinstance(payload.get("windows"), list)
    ):
        raise TemporalAttentionError(
            "Stored original-pass temporal evidence is invalid.",
            code="xai_temporal_evidence_invalid",
        )
    try:
        return _validated_windows(
            tuple(_window_from_payload(raw) for raw in payload["windows"])
        )
    except TemporalAttentionError:
        raise
    except (TypeError, ValueError, OverflowError) as error:
        raise TemporalAttentionError(
            "Stored original-pass temporal evidence is invalid.",
            code="xai_temporal_evidence_invalid",
        ) from error


def _validated_windows(
    evidence: tuple[TemporalAttentionWindowInput, ...],
) -> tuple[TemporalAttentionWindowInput, ...]:
    if not evidence or len(evidence) > MAX_TEMPORAL_EVIDENCE_WINDOWS:
        raise TemporalAttentionError(
            "Original-pass temporal evidence has an invalid window count.",
            code="xai_temporal_evidence_invalid",
        )
    validated: list[TemporalAttentionWindowInput] = []
    for window in evidence:
        if not isinstance(window, TemporalAttentionWindowInput):
            raise TemporalAttentionError(
                "Original-pass temporal evidence is invalid.",
                code="xai_temporal_evidence_invalid",
            )
        if window.token_times_seconds.size > MAX_TEMPORAL_EVIDENCE_TOKENS_PER_WINDOW:
            raise TemporalAttentionError(
                "Original-pass temporal evidence exceeds its storage limit.",
                code="xai_temporal_evidence_too_large",
            )
        # Numpy arrays remain mutable even though the surrounding dataclass is
        # frozen, so validate their current state again before persistence.
        validated.append(
            TemporalAttentionWindowInput(
                start_seconds=window.start_seconds,
                end_seconds=window.end_seconds,
                token_times_seconds=window.token_times_seconds,
                attention_density=window.attention_density,
                spoof_probability=window.spoof_probability,
            )
        )
    return tuple(validated)


def _window_from_payload(raw: Any) -> TemporalAttentionWindowInput:
    if not isinstance(raw, dict) or not {
        "start_seconds",
        "end_seconds",
        "token_times_seconds",
        "attention_density",
    }.issubset(raw):
        raise TemporalAttentionError(
            "Stored original-pass temporal evidence is invalid.",
            code="xai_temporal_evidence_invalid",
        )
    return TemporalAttentionWindowInput(
        start_seconds=raw["start_seconds"],
        end_seconds=raw["end_seconds"],
        token_times_seconds=np.asarray(raw["token_times_seconds"], dtype=np.float32),
        attention_density=np.asarray(raw["attention_density"], dtype=np.float32),
        # Version 2 artifacts written before this compatibility restoration
        # did not retain the worker's probability context.
        spoof_probability=raw.get("spoof_probability", 0.5),
    )
