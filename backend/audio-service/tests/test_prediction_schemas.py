from datetime import datetime

import pytest
from pydantic import ValidationError

from app.schemas.common import BranchStatus, ModelMode, PredictionLabel
from app.schemas.prediction import (
    AudioMetadata,
    BranchPrediction,
    FusionResult,
    ProbabilityScores,
    VoicePredictionResponse,
)


def make_audio_metadata() -> AudioMetadata:
    return AudioMetadata(
        original_filename="sample.wav",
        content_type="audio/wav",
        file_size_bytes=1024,
        duration_seconds=1.5,
        sample_rate=16000,
        channels=1,
    )


def make_successful_branch(mode: ModelMode = ModelMode.real) -> BranchPrediction:
    return BranchPrediction(
        model_name="cnn_acoustic",
        display_name="CNN Acoustic",
        status=BranchStatus.success,
        mode=mode,
        prediction=PredictionLabel.spoof,
        confidence=0.75,
        probabilities=ProbabilityScores(bonafide=0.25, spoof=0.75),
        processing_time_ms=12.5,
    )


def make_fusion_result(
    *,
    contains_dummy_branches: bool = False,
    eligible_for_research_evaluation: bool = True,
) -> FusionResult:
    return FusionResult(
        status=BranchStatus.success,
        prediction=PredictionLabel.spoof,
        confidence=0.75,
        probabilities=ProbabilityScores(bonafide=0.25, spoof=0.75),
        method="weighted_average",
        branch_weights={"cnn_acoustic": 1.0},
        contains_dummy_branches=contains_dummy_branches,
        eligible_for_research_evaluation=eligible_for_research_evaluation,
    )


def test_probability_scores_accept_valid_distribution() -> None:
    scores = ProbabilityScores(bonafide=0.4, spoof=0.6)

    assert scores.bonafide == 0.4
    assert scores.spoof == 0.6


def test_probability_scores_reject_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        ProbabilityScores(bonafide=-0.1, spoof=1.1)


def test_probability_scores_reject_values_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValidationError, match="approximately sum to 1"):
        ProbabilityScores(bonafide=0.2, spoof=0.5)


def test_branch_prediction_requires_failed_branch_without_prediction() -> None:
    with pytest.raises(ValidationError, match="failed branch"):
        BranchPrediction(
            model_name="cnn_acoustic",
            display_name="CNN Acoustic",
            status=BranchStatus.failed,
            mode=ModelMode.real,
            prediction=PredictionLabel.spoof,
            confidence=None,
            probabilities=None,
            processing_time_ms=3.0,
            error="model unavailable",
        )


def test_branch_prediction_confidence_matches_winning_probability() -> None:
    with pytest.raises(ValidationError, match="Confidence must match"):
        BranchPrediction(
            model_name="cnn_acoustic",
            display_name="CNN Acoustic",
            status=BranchStatus.success,
            mode=ModelMode.real,
            prediction=PredictionLabel.spoof,
            confidence=0.51,
            probabilities=ProbabilityScores(bonafide=0.25, spoof=0.75),
            processing_time_ms=3.0,
        )


def test_branch_prediction_uses_safe_metadata_default() -> None:
    first = make_successful_branch()
    second = make_successful_branch()

    first.metadata["dataset"] = "local-dev"

    assert second.metadata == {}


def test_dummy_branch_is_identifiable_through_mode() -> None:
    branch = make_successful_branch(mode=ModelMode.dummy)

    assert branch.mode == ModelMode.dummy
    assert branch.model_dump(mode="json")["mode"] == "dummy"


def test_audio_metadata_validates_non_negative_and_positive_values() -> None:
    with pytest.raises(ValidationError):
        AudioMetadata(
            original_filename="sample.wav",
            content_type="audio/wav",
            file_size_bytes=-1,
            duration_seconds=-0.1,
            sample_rate=0,
            channels=0,
        )


def test_audio_metadata_exposes_new_and_legacy_size_and_container_names() -> None:
    metadata = AudioMetadata(
        original_filename="sample.wav",
        content_type="audio/wav",
        detected_container="wav",
        detected_codec="pcm_s16le",
        file_size_bytes=1024,
        duration_seconds=1.5,
        sample_rate=16000,
        channels=1,
    )

    assert metadata.size_bytes == 1024
    assert metadata.detected_format == "wav"


def test_fusion_with_dummy_branches_is_not_research_eligible() -> None:
    with pytest.raises(ValidationError, match="not eligible"):
        make_fusion_result(
            contains_dummy_branches=True,
            eligible_for_research_evaluation=True,
        )


def test_voice_prediction_response_rejects_dummy_branch_without_fusion_flag() -> None:
    with pytest.raises(ValidationError, match="must identify"):
        VoicePredictionResponse(
            request_id="request-123",
            audio=make_audio_metadata(),
            branches=[make_successful_branch(mode=ModelMode.dummy)],
            fusion=make_fusion_result(
                contains_dummy_branches=False,
                eligible_for_research_evaluation=True,
            ),
            total_processing_time_ms=15.0,
        )


def test_voice_prediction_response_accepts_consistent_payload() -> None:
    response = VoicePredictionResponse(
        request_id="request-123",
        audio=make_audio_metadata(),
        branches=[make_successful_branch(mode=ModelMode.dummy)],
        fusion=make_fusion_result(
            contains_dummy_branches=True,
            eligible_for_research_evaluation=False,
        ),
        total_processing_time_ms=15.0,
    )

    assert response.fusion.eligible_for_research_evaluation is False
    assert isinstance(response.created_at, datetime)
