from __future__ import annotations

import pytest

from app.schemas.xai import (
    ComponentStatus,
    SemanticEvidenceWindow,
    SemanticExplanation,
    SemanticFeatureContribution,
    SemanticTargetType,
    ShapDirection,
    TemporalExplanation,
    TemporalRegion,
)
from app.voice_xai.evaluation.consistency import quality_for_analysis


def _contribution(rank: int, start: float, end: float) -> SemanticFeatureContribution:
    return SemanticFeatureContribution(
        rank=rank,
        feature_name=f"feature_{rank}",
        display_name=f"Feature {rank}",
        value=1.0,
        shap_value=0.1 * rank,
        direction=ShapDirection.toward_spoof,
        start_seconds=start,
        end_seconds=end,
    )


def _semantic(
    contributions: list[SemanticFeatureContribution],
    *,
    model_version: str = "test",
) -> SemanticExplanation:
    return SemanticExplanation(
        status=ComponentStatus.completed,
        target_type=SemanticTargetType.classifier_surrogate,
        model_version=model_version,
        feature_schema_version="test",
        extractor_version="test",
        output_space="probability",
        feature_importance=contributions,
        # Deliberately broad: IoU must ignore the sliding analysis grid.
        windows=[SemanticEvidenceWindow(start_seconds=0.0, end_seconds=10.0)],
    )


def _temporal() -> TemporalExplanation:
    return TemporalExplanation(
        status=ComponentStatus.completed,
        method_version="test",
        high_attention_regions=[
            TemporalRegion(
                region_id=1,
                start_seconds=0.0,
                end_seconds=1.0,
                attention_score=1.0,
            )
        ],
    )


def test_iou_uses_only_the_three_top_ranked_shap_windows() -> None:
    quality = quality_for_analysis(
        _temporal(),
        _semantic([
            _contribution(3, 4.0, 5.0),
            _contribution(1, 0.0, 1.0),
            _contribution(2, 2.0, 3.0),
            _contribution(4, 0.0, 10.0),
        ]),
    )

    assert quality.temporal_semantic_iou is not None
    assert quality.temporal_semantic_iou.value == pytest.approx(1 / 3)
    assert quality.surrogate_fidelity_r2 is not None
    assert quality.surrogate_fidelity_r2.status.value == "not_applicable"


def test_iou_is_not_computed_without_three_timestamped_top_shap_windows() -> None:
    quality = quality_for_analysis(_temporal(), _semantic([_contribution(1, 0.0, 1.0)]))

    assert quality.temporal_semantic_iou is not None
    assert quality.temporal_semantic_iou.status.value == "not_computed"


def test_quality_attaches_fidelity_for_the_validated_surrogate() -> None:
    quality = quality_for_analysis(
        _temporal(),
        _semantic(
            [
                _contribution(1, 0.0, 1.0),
                _contribution(2, 2.0, 3.0),
                _contribution(3, 4.0, 5.0),
            ],
            model_version="xgboost-surrogate-v4-2026-08-19",
        ),
    )

    assert quality.surrogate_fidelity_r2 is not None
    assert quality.surrogate_fidelity_r2.status.value == "available"
    assert quality.surrogate_fidelity_r2.value == pytest.approx(0.6342774342328266)
    assert quality.surrogate_fidelity_r2.scope == "offline_validation"
