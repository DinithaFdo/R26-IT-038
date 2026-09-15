"""Load versioned, offline-validation metrics for Voice XAI."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.schemas.xai import MetricEvidence, MetricStatus

SURROGATE_FIDELITY_ARTIFACT_VERSION = "voice-xai-surrogate-fidelity-r2-v1"
SURROGATE_FIDELITY_ARTIFACT_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "surrogate_fidelity_r2_v1.json"
)


class OfflineMetricArtifactError(RuntimeError):
    """Raised when a versioned offline metric artifact is invalid."""


class OfflineMetricSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifact_name: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_column: str = Field(min_length=1)
    dataset_value: str = Field(min_length=1)
    metric_column: str = Field(min_length=1)


class SurrogateFidelityArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    artifact_version: Literal["voice-xai-surrogate-fidelity-r2-v1"]
    metric_name: Literal["surrogate_fidelity_r2"]
    value: float
    scope: Literal["offline_validation"]
    dataset_version: str = Field(min_length=1)
    sample_count: int = Field(gt=0)
    semantic_model_version: str = Field(min_length=1)
    source: OfflineMetricSource


@lru_cache(maxsize=4)
def load_surrogate_fidelity_artifact(
    path: Path = SURROGATE_FIDELITY_ARTIFACT_PATH,
) -> SurrogateFidelityArtifact:
    """Read and strictly validate the immutable surrogate-fidelity artifact."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        artifact = SurrogateFidelityArtifact.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValidationError) as error:
        raise OfflineMetricArtifactError(
            "The surrogate-fidelity validation artifact is unavailable or invalid."
        ) from error
    if not math.isfinite(artifact.value):
        raise OfflineMetricArtifactError(
            "The surrogate-fidelity validation value must be finite."
        )
    return artifact


def surrogate_fidelity_for_model(
    semantic_model_version: str | None,
) -> MetricEvidence:
    """Return fidelity only when the explanation uses the validated model."""

    try:
        artifact = load_surrogate_fidelity_artifact()
    except OfflineMetricArtifactError:
        return MetricEvidence(
            status=MetricStatus.not_computed,
            scope="offline_validation",
            reason="The versioned surrogate-fidelity artifact is unavailable.",
        )
    if semantic_model_version != artifact.semantic_model_version:
        return MetricEvidence(
            status=MetricStatus.not_applicable,
            scope="offline_validation",
            reason=(
                "The semantic explanation does not use the model validated by "
                "the surrogate-fidelity artifact."
            ),
            dataset_version=artifact.dataset_version,
        )
    return MetricEvidence(
        status=MetricStatus.available,
        value=artifact.value,
        scope=artifact.scope,
        dataset_version=artifact.dataset_version,
    )


__all__ = [
    "OfflineMetricArtifactError",
    "SURROGATE_FIDELITY_ARTIFACT_PATH",
    "SURROGATE_FIDELITY_ARTIFACT_VERSION",
    "SurrogateFidelityArtifact",
    "load_surrogate_fidelity_artifact",
    "surrogate_fidelity_for_model",
]
