import pytest

from app.schemas.common import BranchStatus, ModelMode, PredictionLabel
from app.schemas.prediction import BranchPrediction, ProbabilityScores
from app.utils.fusion import DUMMY_FUSION_WARNING, FusionEngine


def make_branch(
    model_name: str,
    spoof_probability: float,
    *,
    mode: ModelMode = ModelMode.real,
    status: BranchStatus = BranchStatus.success,
) -> BranchPrediction:
    if status != BranchStatus.success:
        return BranchPrediction(
            model_name=model_name,
            display_name=model_name,
            status=status,
            mode=mode,
            prediction=None,
            confidence=None,
            probabilities=None,
            processing_time_ms=1.0,
            error="branch unavailable",
        )

    bonafide_probability = 1.0 - spoof_probability
    prediction = (
        PredictionLabel.spoof
        if spoof_probability >= bonafide_probability
        else PredictionLabel.bonafide
    )
    confidence = max(spoof_probability, bonafide_probability)
    return BranchPrediction(
        model_name=model_name,
        display_name=model_name,
        status=status,
        mode=mode,
        prediction=prediction,
        confidence=confidence,
        probabilities=ProbabilityScores(
            bonafide=bonafide_probability,
            spoof=spoof_probability,
        ),
        processing_time_ms=1.0,
    )


def default_branches(*, mode: ModelMode = ModelMode.real) -> list[BranchPrediction]:
    return [
        make_branch("cnn_acoustic", 0.2, mode=mode),
        make_branch("aasist", 0.4, mode=mode),
        make_branch("ssl_wavlm_xlsr", 0.6, mode=mode),
        make_branch("glottal_features", 0.8, mode=mode),
    ]


def test_equal_weighted_fusion_uses_development_defaults() -> None:
    result = FusionEngine(method="weighted_average").fuse(default_branches())

    assert result.status == BranchStatus.success
    assert result.method == "weighted_average"
    assert result.probabilities is not None
    assert result.probabilities.spoof == pytest.approx(0.5)
    assert result.probabilities.bonafide == pytest.approx(0.5)
    assert result.prediction == PredictionLabel.spoof
    assert result.branch_weights == {
        "cnn_acoustic": pytest.approx(0.25),
        "aasist": pytest.approx(0.25),
        "ssl_wavlm_xlsr": pytest.approx(0.25),
        "glottal_features": pytest.approx(0.25),
    }


def test_custom_weights_are_normalized_across_successful_branches() -> None:
    engine = FusionEngine(
        branch_weights={
            "cnn": 2.0,
            "aasist": 1.0,
            "ssl": 1.0,
            "glottal": 0.0,
        }
    )

    result = engine.fuse(default_branches())

    assert result.probabilities is not None
    assert result.branch_weights == {
        "cnn_acoustic": pytest.approx(0.5),
        "aasist": pytest.approx(0.25),
        "ssl_wavlm_xlsr": pytest.approx(0.25),
        "glottal_features": pytest.approx(0.0),
    }
    assert result.probabilities.spoof == pytest.approx(0.35)
    assert result.prediction == PredictionLabel.bonafide


def test_majority_vote_handles_branch_disagreement() -> None:
    branches = [
        make_branch("cnn_acoustic", 0.8),
        make_branch("aasist", 0.7),
        make_branch("ssl_wavlm_xlsr", 0.3),
    ]

    result = FusionEngine(method="majority_vote").fuse(branches)

    assert result.prediction == PredictionLabel.spoof
    assert result.probabilities is not None
    assert result.probabilities.spoof == pytest.approx(2 / 3)


def test_failed_branch_is_excluded_and_weights_are_renormalized() -> None:
    branches = [
        make_branch("cnn_acoustic", 0.2),
        make_branch("aasist", 0.4),
        make_branch("ssl_wavlm_xlsr", 0.0, status=BranchStatus.failed),
        make_branch("glottal_features", 0.8),
    ]

    result = FusionEngine(method="weighted_average").fuse(branches)

    assert result.probabilities is not None
    assert result.branch_weights == {
        "cnn_acoustic": pytest.approx(1 / 3),
        "aasist": pytest.approx(1 / 3),
        "glottal_features": pytest.approx(1 / 3),
    }
    assert "ssl_wavlm_xlsr" not in result.branch_weights
    assert result.probabilities.spoof == pytest.approx((0.2 + 0.4 + 0.8) / 3)


def test_fewer_than_two_successful_branches_returns_failed_fusion() -> None:
    branches = [
        make_branch("cnn_acoustic", 0.2),
        make_branch("aasist", 0.0, status=BranchStatus.skipped),
    ]

    result = FusionEngine().fuse(branches)

    assert result.status == BranchStatus.failed
    assert result.prediction is None
    assert result.probabilities is None
    assert result.warning == "At least two successful branches are required for fusion."


def test_dummy_branch_makes_fusion_not_research_eligible() -> None:
    result = FusionEngine().fuse(default_branches(mode=ModelMode.dummy))

    assert result.contains_dummy_branches is True
    assert result.eligible_for_research_evaluation is False
    assert result.warning == DUMMY_FUSION_WARNING


@pytest.mark.parametrize(
    "weights",
    [
        {"cnn": -0.1, "aasist": 0.5},
        {"unknown": 1.0},
        {},
    ],
)
def test_invalid_weights_are_rejected(weights) -> None:
    with pytest.raises(ValueError):
        FusionEngine(branch_weights=weights)


def test_fusion_is_deterministic() -> None:
    engine = FusionEngine(method="simple_average")
    branches = default_branches()

    first = engine.fuse(branches)
    second = engine.fuse(branches)

    assert first == second


def test_fusion_rejects_non_finite_weight() -> None:
    with pytest.raises(ValueError):
        FusionEngine(branch_weights={"lfcc_cnn_tcn": float("nan")})


def test_weighted_fusion_without_equal_fallback_fails_on_zero_effective_weights() -> None:
    engine = FusionEngine(
        branch_weights={
            "lfcc_cnn_tcn": 0.0,
            "aasist": 0.0,
            "ssl_sequence": 0.0,
            "glottal": 0.0,
        },
        allow_equal_weight_fallback=False,
    )

    result = engine.fuse(default_branches())

    assert result.status == BranchStatus.failed
    assert result.warning == "No positive fusion weights are available."


def test_required_fusion_branch_missing_fails_safely() -> None:
    engine = FusionEngine(required_branches=("ssl_sequence",))

    result = engine.fuse(
        [
            make_branch("cnn_acoustic", 0.3),
            make_branch("aasist", 0.4),
        ]
    )

    assert result.status == BranchStatus.failed
    assert result.excluded_branches["ssl_sequence"] == "required_branch_missing"
