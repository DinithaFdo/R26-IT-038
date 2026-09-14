"""Convert ordered SHAP-style evidence to the public semantic schema."""

from __future__ import annotations

import math

import numpy as np

from app.schemas.xai import (
    ComponentStatus,
    SemanticExplanation,
    SemanticClipSummary,
    SemanticFeatureContribution,
    SemanticEvidenceWindow,
    SemanticTargetType,
    ShapDirection,
)
from app.schemas.common import PredictionLabel
from app.voice_xai.semantic.contracts import (
    SemanticEvidence,
    SemanticExplanationConfig,
)
from app.voice_xai.semantic.feature_labels import label_for_feature
from app.voice_xai.semantic.windows import SemanticWindow


def build_semantic_explanation(
    evidence: SemanticEvidence,
    config: SemanticExplanationConfig,
) -> SemanticExplanation:
    """Rank the notebook's sample-level contributions by absolute SHAP value."""

    contributions = build_semantic_contributions(evidence, config)
    return SemanticExplanation(
        status=ComponentStatus.completed,
        target_type=config.target_type,
        model_version=config.model_version,
        feature_schema_version=config.feature_schema_version,
        extractor_version=evidence.extractor_version,
        output_space=config.output_space,
        base_value=evidence.base_value,
        predicted_value=evidence.predicted_value,
        feature_importance=contributions,
        warning=_warning_with_imputation(
            config.contribution_warning,
            evidence.imputed_feature_count,
        ),
    )


def _warning_with_imputation(
    warning: str | None,
    imputed_feature_count: int,
) -> str | None:
    if not imputed_feature_count:
        return warning
    detail = (
        f"{imputed_feature_count} unavailable feature value(s) were replaced with "
        "training median values before XGBoost and SHAP."
    )
    return f"{warning} {detail}" if warning else detail


def build_semantic_contributions(
    evidence: SemanticEvidence,
    config: SemanticExplanationConfig,
    *,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
) -> list[SemanticFeatureContribution]:
    """Return one window's strongest SHAP contributions with optional timing."""

    if (start_seconds is None) != (end_seconds is None):
        raise ValueError("Semantic contribution timestamps must be supplied together.")
    order = np.argsort(-np.abs(evidence.shap_values), kind="stable")[: config.top_k]
    contributions: list[SemanticFeatureContribution] = []
    for rank, index in enumerate(order, start=1):
        feature_name = evidence.feature_names[int(index)]
        label = label_for_feature(feature_name)
        shap_value = float(evidence.shap_values[index])
        contributions.append(
            SemanticFeatureContribution(
                rank=rank,
                feature_name=feature_name,
                display_name=label.display_name,
                value=float(evidence.feature_values[index]),
                unit=label.unit,
                reference_summary=config.reference_summaries.get(
                    feature_name,
                    "Reference distribution is unavailable in mock mode.",
                ),
                shap_value=shap_value,
                direction=(
                    ShapDirection.toward_spoof
                    if shap_value > 0.0
                    else ShapDirection.toward_bonafide
                ),
                start_seconds=start_seconds,
                end_seconds=end_seconds,
            )
        )
    return contributions


def build_windowed_semantic_explanation(
    windows: tuple[SemanticWindow, ...],
    config: SemanticExplanationConfig,
    *,
    extractor_version: str,
) -> SemanticExplanation:
    """Expose the strongest timestamped evidence without averaging raw margins.

    Per-window raw margins have different baselines and must not be summed or
    averaged into a fabricated clip-level model score.  The complete window
    set is retained by the service result; this public summary remains bounded
    by the normal ``top_k`` contract.
    """

    if not windows:
        raise ValueError("At least one semantic window is required.")
    candidates = [
        contribution
        for window in windows
        for contribution in window.contributions
    ]
    ranked = sorted(
        candidates,
        key=lambda item: (
            -abs(item.shap_value),
            item.start_seconds if item.start_seconds is not None else -1.0,
            item.feature_name,
        ),
    )[: config.top_k]
    top_contributions = [
        contribution.model_copy(update={"rank": rank})
        for rank, contribution in enumerate(ranked, start=1)
    ]
    return SemanticExplanation(
        status=ComponentStatus.completed,
        target_type=config.target_type,
        model_version=config.model_version,
        feature_schema_version=config.feature_schema_version,
        extractor_version=extractor_version,
        output_space=config.output_space,
        base_value=None,
        predicted_value=None,
        clip_summary=_build_clip_summary(windows, config),
        feature_importance=top_contributions,
        windows=[
            SemanticEvidenceWindow(
                start_seconds=window.start_seconds,
                end_seconds=window.end_seconds,
                feature_importance=list(window.contributions),
            )
            for window in windows
        ],
        warning=_windowed_warning(config),
    )


def _build_clip_summary(
    windows: tuple[SemanticWindow, ...],
    config: SemanticExplanationConfig,
) -> SemanticClipSummary | None:
    """Aggregate window model probabilities without counting overlap twice.

    The XGBoost binary-logistic score is converted to P(spoof) per window.
    Each score represents the time nearest to that window's centre, so the
    midpoint boundaries between centres provide a deterministic coverage
    weight. This avoids giving overlapped audio twice the influence of the
    clip edges while retaining the model's threshold selected during training.
    """

    if config.decision_threshold is None:
        return None
    if any(
        later.start_seconds <= earlier.start_seconds
        or later.start_seconds > earlier.end_seconds
        for earlier, later in zip(windows, windows[1:])
    ):
        raise ValueError(
            "Semantic windows must be ordered and cover the clip continuously."
        )
    centres = [
        (window.start_seconds + window.end_seconds) / 2.0 for window in windows
    ]
    boundaries = [windows[0].start_seconds]
    boundaries.extend(
        (left + right) / 2.0 for left, right in zip(centres, centres[1:])
    )
    boundaries.append(windows[-1].end_seconds)
    weights = np.diff(np.asarray(boundaries, dtype=np.float64))
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise ValueError("Semantic window coverage weights are invalid.")
    probabilities = np.asarray(
        [window.spoof_probability for window in windows], dtype=np.float64
    )
    spoof_probability = float(np.average(probabilities, weights=weights))
    if not math.isfinite(spoof_probability) or not 0.0 <= spoof_probability <= 1.0:
        raise ValueError("Semantic clip spoof probability is invalid.")
    return SemanticClipSummary(
        spoof_probability=spoof_probability,
        bonafide_probability=1.0 - spoof_probability,
        decision_threshold=config.decision_threshold,
        predicted_label=(
            PredictionLabel.spoof
            if spoof_probability >= config.decision_threshold
            else PredictionLabel.bonafide
        ),
        window_count=len(windows),
    )


def _windowed_warning(config: SemanticExplanationConfig) -> str:
    base_warning = (
        "Sliding-window SHAP contributions are timestamped. Per-window raw "
        "margins are not aggregated into a clip-level model score."
    )
    if config.decision_threshold is None:
        return base_warning
    return (
        f"{base_warning} clip_summary is a coverage-weighted mean of per-window "
        "XGBoost P(spoof), compared with this model's validation-selected "
        "decision threshold; it is auxiliary semantic evidence and does not "
        "change the primary classifier decision."
    )
