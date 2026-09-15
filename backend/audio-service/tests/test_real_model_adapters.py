"""Real adapters, registry modes, readiness, and current-stage fusion."""

from __future__ import annotations

import pytest

from app.models.factory import ModelFactory
from app.models.registry import ModelRegistry
from app.schemas.common import BranchStatus, ModelMode, PredictionLabel
from app.utils.fusion import FusionEngine
from tests.real_model_helpers import (
    processed_audio,
    real_model_settings,
    requires_aasist_checkpoint,
    requires_cnn_checkpoint,
    requires_torch,
    synthetic_speech,
)

real_models = pytest.mark.usefixtures()


# --------------------------------------------------------------------------
# Mode selection -- these need neither torch nor a checkpoint.
# --------------------------------------------------------------------------


def test_disabled_branches_never_load_and_report_skipped() -> None:
    settings = real_model_settings(cnn_model_mode="dummy", aasist_model_mode="dummy")
    registry = ModelRegistry(app_settings=settings)

    disabled = [m for m in registry.models if m.mode == ModelMode.disabled]
    assert {m.branch_name for m in disabled} == {"ssl_sequence", "glottal"}

    for model in disabled:
        prediction = model.predict_safe(processed_audio())
        assert prediction.status == BranchStatus.skipped
        assert prediction.mode == ModelMode.disabled
        assert prediction.probabilities is None
        assert prediction.metadata["error_code"] == "branch_disabled"
        assert model.is_loaded is False


def test_disabled_branch_load_is_a_no_op() -> None:
    settings = real_model_settings()
    model = ModelFactory(settings).create("ssl_sequence")

    model.load()
    model.unload()

    assert model.is_loaded is False
    assert model.health()["mode"] == "disabled"
    # Disabled is a configured state, not an outage waiting to be fixed.
    assert model.health()["ready"] is True
    assert model.health()["research_ready"] is False


def test_dummy_mode_remains_available_for_development() -> None:
    settings = real_model_settings(cnn_model_mode="dummy", aasist_model_mode="dummy")
    model = ModelFactory(settings).create("lfcc_cnn_tcn")

    prediction = model.predict_safe(processed_audio())

    assert model.mode == ModelMode.dummy
    assert prediction.status == BranchStatus.success
    assert prediction.metadata["research_result"] is False


def test_a_missing_checkpoint_fails_the_branch_without_leaking_the_path() -> None:
    settings = real_model_settings(cnn_model_path="does-not-exist.pth")
    model = ModelFactory(settings).create("lfcc_cnn_tcn")

    prediction = model.predict_safe(processed_audio())

    assert prediction.status == BranchStatus.failed
    assert prediction.probabilities is None
    assert "does-not-exist" not in (prediction.error or "")
    assert "/" not in (prediction.error or "")


def test_checkpoint_paths_cannot_escape_the_model_root() -> None:
    settings = real_model_settings(cnn_model_path="../../../etc/passwd")
    config = ModelFactory(settings).branch_config("lfcc_cnn_tcn")

    assert config.checkpoint.valid is False
    assert config.checkpoint.error_code in {
        "checkpoint_path_escape",
        "checkpoint_unsupported_extension",
    }


def test_branch_health_never_exposes_a_filesystem_path() -> None:
    settings = real_model_settings()
    registry = ModelRegistry(app_settings=settings)

    for branch in registry.health():
        serialised = repr(branch)
        assert "/models/" not in serialised
        assert ".pth" not in serialised


# --------------------------------------------------------------------------
# Real inference.
# --------------------------------------------------------------------------


@requires_torch
@requires_cnn_checkpoint
def test_real_cnn_branch_produces_a_valid_probability_pair() -> None:
    settings = real_model_settings()
    model = ModelFactory(settings).create("lfcc_cnn_tcn")

    prediction = model.predict_safe(processed_audio())

    assert prediction.status == BranchStatus.success
    assert prediction.mode == ModelMode.real
    assert prediction.probabilities is not None
    assert 0.0 <= prediction.probabilities.spoof <= 1.0
    assert prediction.probabilities.spoof + prediction.probabilities.bonafide == pytest.approx(1.0, abs=1e-3)
    assert prediction.metadata["architecture"] == "cnn-v2-lfcc-delta-vgg-4block-v1"
    assert prediction.metadata["device"] == "cpu"
    assert prediction.metadata["parameter_count"] == 616_162


@requires_torch
@requires_aasist_checkpoint
def test_real_aasist_branch_produces_a_valid_probability_pair() -> None:
    settings = real_model_settings()
    model = ModelFactory(settings).create("aasist")

    prediction = model.predict_safe(processed_audio())

    assert prediction.status == BranchStatus.success
    assert prediction.mode == ModelMode.real
    assert prediction.probabilities is not None
    assert prediction.probabilities.spoof + prediction.probabilities.bonafide == pytest.approx(1.0, abs=1e-3)
    # Production deploys AASIST-Light V2 (finalized baseline), not the V1
    # reconstruction these constants described before checkpoint resolution
    # was repaired; this test previously always skipped, so the drift went
    # unnoticed. See test_aasist_light_v2_checkpoint_integration.py for the
    # dedicated V2 strict-load contract test.
    assert prediction.metadata["architecture"] == "aasist-light-v2-finalized-baseline"
    assert prediction.metadata["parameter_count"] == 641_795


@requires_torch
@requires_cnn_checkpoint
def test_the_same_audio_gives_the_same_prediction_every_time() -> None:
    settings = real_model_settings()
    model = ModelFactory(settings).create("lfcc_cnn_tcn")
    audio = processed_audio()

    first = model.predict_safe(audio)
    second = model.predict_safe(audio)

    assert first.probabilities.spoof == second.probabilities.spoof


@requires_torch
@requires_cnn_checkpoint
def test_the_checkpoint_is_loaded_once_and_reused() -> None:
    settings = real_model_settings()
    model = ModelFactory(settings).create("lfcc_cnn_tcn")

    model.predict_safe(processed_audio())
    loaded_module = model._runtime_model.module
    model.predict_safe(processed_audio())

    assert model._runtime_model.module is loaded_module


@requires_torch
@requires_cnn_checkpoint
def test_class_order_selects_the_opposite_logit() -> None:
    """The single most dangerous setting: the wrong value inverts everything."""

    forward = ModelFactory(real_model_settings(cnn_class_order="bonafide_spoof")).create(
        "lfcc_cnn_tcn"
    )
    reversed_order = ModelFactory(
        real_model_settings(cnn_class_order="spoof_bonafide")
    ).create("lfcc_cnn_tcn")
    audio = processed_audio()

    forward_spoof = forward.predict_safe(audio).probabilities.spoof
    reversed_spoof = reversed_order.predict_safe(audio).probabilities.spoof

    assert forward_spoof == pytest.approx(1.0 - reversed_spoof, abs=1e-6)


@requires_torch
@requires_cnn_checkpoint
def test_spoof_probability_drives_the_public_label() -> None:
    settings = real_model_settings()
    model = ModelFactory(settings).create("lfcc_cnn_tcn")

    prediction = model.predict_safe(processed_audio())
    probabilities = prediction.probabilities

    expected = (
        PredictionLabel.spoof
        if probabilities.spoof >= probabilities.bonafide
        else PredictionLabel.bonafide
    )
    assert prediction.prediction == expected
    assert prediction.confidence == pytest.approx(
        max(probabilities.spoof, probabilities.bonafide)
    )


@requires_torch
@requires_cnn_checkpoint
def test_non_finite_audio_is_rejected_rather_than_fed_to_the_model() -> None:
    import numpy as np

    settings = real_model_settings()
    model = ModelFactory(settings).create("lfcc_cnn_tcn")

    corrupted = synthetic_speech()
    corrupted[100] = np.nan

    prediction = model.predict_safe(processed_audio(corrupted))

    assert prediction.status == BranchStatus.failed
    assert prediction.metadata["error_code"] == "model_input_invalid"


@requires_torch
@requires_cnn_checkpoint
def test_unverified_real_branches_are_not_research_results() -> None:
    settings = real_model_settings()
    model = ModelFactory(settings).create("lfcc_cnn_tcn")

    prediction = model.predict_safe(processed_audio())

    assert prediction.metadata["research_result"] is False
    assert prediction.metadata["preprocessing_verified"] is False
    assert prediction.metadata["class_mapping_verified"] is False
    assert "not been verified" in prediction.metadata["warning"].lower()
    assert model.health()["research_ready"] is False


@requires_torch
@requires_cnn_checkpoint
def test_attesting_both_flags_makes_a_branch_research_ready() -> None:
    settings = real_model_settings(
        cnn_preprocessing_verified=True, cnn_class_mapping_verified=True
    )
    model = ModelFactory(settings).create("lfcc_cnn_tcn")

    prediction = model.predict_safe(processed_audio())

    assert prediction.metadata["research_result"] is True
    assert model.health()["research_ready"] is True


@requires_torch
@requires_cnn_checkpoint
def test_attesting_only_one_flag_is_not_enough() -> None:
    settings = real_model_settings(cnn_preprocessing_verified=True)
    model = ModelFactory(settings).create("lfcc_cnn_tcn")

    assert model.predict_safe(processed_audio()).metadata["research_result"] is False


# --------------------------------------------------------------------------
# Readiness.
# --------------------------------------------------------------------------


@requires_torch
@requires_cnn_checkpoint
@requires_aasist_checkpoint
def test_readiness_distinguishes_real_from_disabled_branches() -> None:
    settings = real_model_settings(model_load_strategy="startup")
    registry = ModelRegistry(app_settings=settings)
    registry.load_startup_models()

    readiness = registry.readiness()

    assert readiness["mode_summary"]["real"] == ["aasist", "lfcc_cnn_tcn"]
    assert readiness["mode_summary"]["disabled"] == ["glottal", "ssl_sequence"]
    assert readiness["ready"] is True
    # Two of four branches, both unattested -- not a research-ready system.
    assert readiness["research_ready"] is False
    assert readiness["unverified_real_branches"] == ["aasist", "lfcc_cnn_tcn"]


def test_requiring_a_disabled_branch_makes_the_system_not_ready() -> None:
    settings = real_model_settings(
        cnn_model_mode="dummy",
        aasist_model_mode="dummy",
        required_model_branches="lfcc_cnn_tcn,aasist,ssl_sequence,glottal",
    )
    readiness = ModelRegistry(app_settings=settings).readiness()

    assert readiness["ready"] is False


# --------------------------------------------------------------------------
# Current-stage fusion.
# --------------------------------------------------------------------------


@requires_torch
@requires_cnn_checkpoint
@requires_aasist_checkpoint
def test_two_real_branches_fuse_with_renormalised_weights() -> None:
    settings = real_model_settings()
    registry = ModelRegistry(app_settings=settings)
    audio = processed_audio()

    branches = [model.predict_safe(audio) for model in registry.models]
    result = FusionEngine.from_settings(settings).fuse(branches)

    assert result.status == BranchStatus.success
    assert sorted(result.contributing_branches) == ["aasist", "cnn_acoustic"]
    # Configured 0.25 each, renormalised across the two that contributed.
    assert result.branch_weights == pytest.approx({"cnn_acoustic": 0.5, "aasist": 0.5})
    assert sum(result.branch_weights.values()) == pytest.approx(1.0)
    assert result.excluded_branches == {
        "ssl_wavlm_xlsr": "branch_disabled",
        "glottal_features": "branch_disabled",
    }


@requires_torch
@requires_cnn_checkpoint
@requires_aasist_checkpoint
def test_a_two_branch_result_is_never_research_eligible() -> None:
    settings = real_model_settings()
    registry = ModelRegistry(app_settings=settings)
    audio = processed_audio()

    result = FusionEngine.from_settings(settings).fuse(
        [model.predict_safe(audio) for model in registry.models]
    )

    assert result.eligible_for_research_evaluation is False
    assert result.system_stage == "partial"
    assert "incomplete_branch_set" in result.research_blockers


@requires_torch
@requires_cnn_checkpoint
@requires_aasist_checkpoint
def test_attested_branches_still_blocked_by_the_incomplete_branch_set() -> None:
    """Verifying CNN and AASIST must not promote a 2-of-4 system to research."""

    settings = real_model_settings(
        cnn_preprocessing_verified=True,
        cnn_class_mapping_verified=True,
        aasist_preprocessing_verified=True,
        aasist_class_mapping_verified=True,
    )
    registry = ModelRegistry(app_settings=settings)
    audio = processed_audio()

    result = FusionEngine.from_settings(settings).fuse(
        [model.predict_safe(audio) for model in registry.models]
    )

    assert result.research_blockers == ["incomplete_branch_set"]
    assert result.eligible_for_research_evaluation is False


def test_fusion_preserves_spoof_direction() -> None:
    from app.schemas.prediction import BranchPrediction, ProbabilityScores

    def branch(name: str, spoof: float) -> BranchPrediction:
        return BranchPrediction(
            model_name=name,
            display_name=name,
            status=BranchStatus.success,
            mode=ModelMode.real,
            prediction=(
                PredictionLabel.spoof if spoof >= 0.5 else PredictionLabel.bonafide
            ),
            confidence=max(spoof, 1 - spoof),
            probabilities=ProbabilityScores(bonafide=1 - spoof, spoof=spoof),
            processing_time_ms=1.0,
            metadata={"research_result": True},
        )

    engine = FusionEngine(minimum_successful_branches=2)

    high = engine.fuse([branch("cnn_acoustic", 0.9), branch("aasist", 0.8)])
    low = engine.fuse([branch("cnn_acoustic", 0.1), branch("aasist", 0.2)])

    assert high.probabilities.spoof > 0.5
    assert high.prediction == PredictionLabel.spoof
    assert low.probabilities.spoof < 0.5
    assert low.prediction == PredictionLabel.bonafide


def test_a_single_available_branch_fails_the_minimum_branch_policy() -> None:
    from app.schemas.prediction import BranchPrediction, ProbabilityScores

    only_branch = BranchPrediction(
        model_name="cnn_acoustic",
        display_name="LFCC CNN/TCN",
        status=BranchStatus.success,
        mode=ModelMode.real,
        prediction=PredictionLabel.spoof,
        confidence=0.9,
        probabilities=ProbabilityScores(bonafide=0.1, spoof=0.9),
        processing_time_ms=1.0,
        metadata={"research_result": True},
    )

    result = FusionEngine(minimum_successful_branches=2).fuse([only_branch])

    assert result.status == BranchStatus.failed
    assert result.prediction is None
    assert result.eligible_for_research_evaluation is False


def test_a_failed_branch_is_excluded_and_the_rest_still_fuse() -> None:
    from app.schemas.prediction import BranchPrediction, ProbabilityScores

    def success(name: str, spoof: float) -> BranchPrediction:
        return BranchPrediction(
            model_name=name,
            display_name=name,
            status=BranchStatus.success,
            mode=ModelMode.real,
            prediction=PredictionLabel.spoof if spoof >= 0.5 else PredictionLabel.bonafide,
            confidence=max(spoof, 1 - spoof),
            probabilities=ProbabilityScores(bonafide=1 - spoof, spoof=spoof),
            processing_time_ms=1.0,
            metadata={"research_result": True},
        )

    failed = BranchPrediction(
        model_name="ssl_wavlm_xlsr",
        display_name="SSL Sequence",
        status=BranchStatus.failed,
        mode=ModelMode.real,
        processing_time_ms=1.0,
        error="Model inference failed.",
        metadata={"error_code": "model_inference_failed"},
    )

    result = FusionEngine(minimum_successful_branches=2).fuse(
        [success("cnn_acoustic", 0.8), success("aasist", 0.6), failed]
    )

    assert result.status == BranchStatus.success
    assert "ssl_wavlm_xlsr" in result.excluded_branches
    assert sum(result.branch_weights.values()) == pytest.approx(1.0)


def test_a_dummy_contribution_blocks_research_eligibility() -> None:
    from app.schemas.prediction import BranchPrediction, ProbabilityScores

    def branch(name: str, mode: ModelMode, research: bool) -> BranchPrediction:
        return BranchPrediction(
            model_name=name,
            display_name=name,
            status=BranchStatus.success,
            mode=mode,
            prediction=PredictionLabel.spoof,
            confidence=0.8,
            probabilities=ProbabilityScores(bonafide=0.2, spoof=0.8),
            processing_time_ms=1.0,
            metadata={"research_result": research},
        )

    result = FusionEngine(minimum_successful_branches=2).fuse(
        [
            branch("cnn_acoustic", ModelMode.real, True),
            branch("aasist", ModelMode.dummy, False),
        ]
    )

    assert result.contains_dummy_branches is True
    assert result.eligible_for_research_evaluation is False
    assert "dummy_branch_contributed" in result.research_blockers


def test_a_complete_attested_branch_set_is_research_eligible() -> None:
    """The only configuration that may claim research validity."""

    from app.schemas.prediction import BranchPrediction, ProbabilityScores

    def branch(name: str) -> BranchPrediction:
        return BranchPrediction(
            model_name=name,
            display_name=name,
            status=BranchStatus.success,
            mode=ModelMode.real,
            prediction=PredictionLabel.spoof,
            confidence=0.8,
            probabilities=ProbabilityScores(bonafide=0.2, spoof=0.8),
            processing_time_ms=1.0,
            metadata={"research_result": True},
        )

    result = FusionEngine(minimum_successful_branches=2).fuse(
        [
            branch("cnn_acoustic"),
            branch("aasist"),
            branch("ssl_wavlm_xlsr"),
            branch("glottal_features"),
        ]
    )

    assert result.research_blockers == []
    assert result.eligible_for_research_evaluation is True
    assert result.system_stage == "full"
    assert result.warning is None
