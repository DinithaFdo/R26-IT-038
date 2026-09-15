"""Stable data contracts shared by all Phase 1 capture components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from app.voice_xai.semantic.features import FeatureExtractionResult

CaptureCategory = Literal["attention", "embedding", "feature_map"]
Normalization = Literal["none", "minmax", "zscore"]


class ExtractionInterfaceError(RuntimeError):
    """Raised for an invalid capture plan or unavailable PyTorch runtime."""


@dataclass(frozen=True)
class CaptureTarget:
    """A selected output from a PyTorch submodule in one real model branch."""

    branch_name: str
    layer_index: int
    module_path: str
    category: CaptureCategory
    output_path: tuple[int | str, ...] = ()
    normalization: Normalization = "minmax"
    max_elements: int = 2_000_000

    def __post_init__(self) -> None:
        if not self.branch_name or not self.module_path:
            raise ValueError("branch_name and module_path are required.")
        if self.layer_index < 0 or self.max_elements < 1:
            raise ValueError("layer_index must be non-negative and max_elements positive.")


@dataclass(frozen=True)
class CapturedRepresentation:
    branch_name: str
    layer_index: int
    category: CaptureCategory
    values: NDArray[np.float32] | None
    normalization: Normalization
    status: Literal["captured", "skipped"]
    reason: str | None = None

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "normalization": self.normalization,
            "status": self.status,
            "reason": self.reason,
            "shape": list(self.values.shape) if self.values is not None else None,
            "values": self.values.tolist() if self.values is not None else None,
        }


@dataclass(frozen=True)
class ExtractionBundle:
    """Phase 1 result indexed as ``branches[branch_name][layer_index]``."""

    request_id: str
    branches: dict[str, dict[int, CapturedRepresentation]]
    acoustic_features: FeatureExtractionResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_json_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "branches": {
                branch: {
                    str(layer): capture.as_json_dict()
                    for layer, capture in layers.items()
                }
                for branch, layers in self.branches.items()
            },
            "acoustic_features": (
                self.acoustic_features.as_json_dict()
                if self.acoustic_features is not None
                else None
            ),
            "metadata": self.metadata,
        }
