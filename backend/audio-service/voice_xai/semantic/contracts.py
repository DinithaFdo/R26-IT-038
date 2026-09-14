"""Private contracts for mock semantic explanation generation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from app.schemas.xai import SemanticExplanation, SemanticTargetType
from app.voice_xai.semantic.features import FEATURE_NAMES
from app.voice_xai.semantic.windows import SemanticWindow


class SemanticExplanationError(ValueError):
    """Raised when semantic evidence violates the frozen feature contract."""


@dataclass(frozen=True)
class SemanticExplanationConfig:
    method_version: str = "mock-xgboost-shap-v4"
    model_version: str = "fixture-acoustic-evidence-model-v1"
    feature_schema_version: str = "xgboost-surrogate-v4-148"
    output_space: str = "raw_margin"
    decision_threshold: float | None = None
    top_k: int = 10
    target_type: SemanticTargetType = SemanticTargetType.independent_acoustic_evidence_model
    contribution_warning: str | None = (
        "Development placeholder with deterministic fixture SHAP values. "
        "Whole-clip feature contributions do not localize evidence in time."
    )
    reference_summaries: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.method_version.strip() or not self.model_version.strip():
            raise SemanticExplanationError("Semantic version strings are required.")
        if self.output_space not in {"raw_margin", "log_odds", "probability"}:
            raise SemanticExplanationError("Unsupported SHAP output space.")
        if (
            self.decision_threshold is not None
            and not 0.0 < self.decision_threshold < 1.0
        ):
            raise SemanticExplanationError(
                "Semantic decision threshold must be strictly between zero and one."
            )
        if not 1 <= self.top_k <= len(FEATURE_NAMES):
            raise SemanticExplanationError("top_k must be within the feature schema size.")


@dataclass(frozen=True)
class SemanticEvidence:
    """Private one-clip feature values and corresponding SHAP-style values."""

    feature_names: tuple[str, ...]
    feature_values: NDArray[np.float32]
    shap_values: NDArray[np.float32]
    base_value: float
    predicted_value: float
    extractor_version: str
    source: str
    imputed_feature_count: int = 0

    def __post_init__(self) -> None:
        if self.feature_names != FEATURE_NAMES:
            raise SemanticExplanationError(
                "Semantic evidence must use the frozen v4 feature order."
            )
        values = np.asarray(self.feature_values)
        shap_values = np.asarray(self.shap_values)
        expected_shape = (len(FEATURE_NAMES),)
        if values.shape != expected_shape or shap_values.shape != expected_shape:
            raise SemanticExplanationError(
                "Feature values and SHAP values must match the frozen v4 schema."
            )
        if not np.isfinite(values).all() or not np.isfinite(shap_values).all():
            raise SemanticExplanationError("Semantic evidence must be finite.")
        if not self.extractor_version.strip() or not self.source.strip():
            raise SemanticExplanationError("Semantic evidence provenance is required.")
        if self.imputed_feature_count < 0 or self.imputed_feature_count > len(FEATURE_NAMES):
            raise SemanticExplanationError("Semantic imputation count is invalid.")


@dataclass(frozen=True)
class SemanticAnalysisResult:
    explanation: SemanticExplanation
    development_placeholder: bool = True
    research_eligible: bool = False
    warnings: tuple[str, ...] = (
        "Development placeholder generated from deterministic fixture SHAP values.",
        "Whole-clip semantic values have no timestamp-level localization.",
    )

    def __post_init__(self) -> None:
        if self.development_placeholder and self.research_eligible:
            raise SemanticExplanationError(
                "Development placeholder semantic results are not research eligible."
            )


@dataclass(frozen=True)
class SemanticWindowAnalysisResult:
    """Timestamped semantic evidence retained separately from its API summary."""

    explanation: SemanticExplanation
    windows: tuple[SemanticWindow, ...]
    development_placeholder: bool = True
    research_eligible: bool = False

    def __post_init__(self) -> None:
        if not self.windows:
            raise SemanticExplanationError("Windowed semantic analysis requires windows.")
        if self.development_placeholder and self.research_eligible:
            raise SemanticExplanationError(
                "Development placeholder semantic results are not research eligible."
            )
