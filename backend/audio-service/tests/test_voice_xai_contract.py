import pytest

from app.models.runtime import CANONICAL_BRANCH_ORDER, PUBLIC_MODEL_NAMES
from app.schemas.common import (
    BranchStatus,
    ModelMode,
    PredictionLabel,
    SourceType,
)
from app.schemas.prediction import (
    AudioMetadata,
    BranchPrediction,
    FusionResult,
    ProbabilityScores,
    VoicePredictionResponse,
)
from app.voice_xai.contracts import (
    CANONICAL_XAI_BRANCHES,
    CLASSIFIER_XAI_CONTRACT_VERSION,
    PUBLIC_MODEL_NAME_BY_XAI_BRANCH,
    ClassifierInferenceBundle,
)
from app.voice_xai.capture.contracts import ExtractionBundle


def test_classifier_xai_contract_freezes_branch_names_and_aliases() -> None:
    assert CANONICAL_XAI_BRANCHES == CANONICAL_BRANCH_ORDER
    assert CANONICAL_XAI_BRANCHES == (
        "lfcc_cnn_tcn",
        "aasist",
        "ssl_sequence",
        "glottal",
    )
    assert PUBLIC_MODEL_NAME_BY_XAI_BRANCH == PUBLIC_MODEL_NAMES
    assert PUBLIC_MODEL_NAME_BY_XAI_BRANCH == {
        "lfcc_cnn_tcn": "cnn_acoustic",
        "aasist": "aasist",
        "ssl_sequence": "ssl_wavlm_xlsr",
        "glottal": "glottal_features",
    }


def test_classifier_inference_bundle_accepts_matching_contracts() -> None:
    prediction = _prediction("request-123")
    extraction = ExtractionBundle(
        request_id="request-123",
        branches={},
    )

    bundle = ClassifierInferenceBundle(
        prediction_id="prediction-123",
        request_id="request-123",
        owner_user_id="user-123",
        source_type=SourceType.dashboard_upload,
        prediction=prediction,
        extraction=extraction,
    )

    assert bundle.contract_version == CLASSIFIER_XAI_CONTRACT_VERSION
    assert bundle.prediction.fusion.probabilities.spoof == 0.8


def test_classifier_inference_bundle_rejects_mismatched_request_ids() -> None:
    prediction = _prediction("request-123")

    with pytest.raises(ValueError, match="prediction request_id"):
        ClassifierInferenceBundle(
            prediction_id="prediction-123",
            request_id="request-other",
            owner_user_id="user-123",
            source_type=SourceType.dashboard_upload,
            prediction=prediction,
        )

    extraction = ExtractionBundle(request_id="request-other", branches={})
    with pytest.raises(ValueError, match="Extraction bundle request_id"):
        ClassifierInferenceBundle(
            prediction_id="prediction-123",
            request_id="request-123",
            owner_user_id="user-123",
            source_type=SourceType.dashboard_upload,
            prediction=prediction,
            extraction=extraction,
        )


def _prediction(request_id: str) -> VoicePredictionResponse:
    branch = BranchPrediction(
        model_name="cnn_acoustic",
        display_name="CNN Acoustic",
        status=BranchStatus.success,
        mode=ModelMode.dummy,
        prediction=PredictionLabel.spoof,
        confidence=0.8,
        probabilities=ProbabilityScores(bonafide=0.2, spoof=0.8),
        processing_time_ms=1.0,
    )
    fusion = FusionResult(
        status=BranchStatus.success,
        prediction=PredictionLabel.spoof,
        confidence=0.8,
        probabilities=ProbabilityScores(bonafide=0.2, spoof=0.8),
        method="weighted_average",
        branch_weights={"cnn_acoustic": 1.0},
        contains_dummy_branches=True,
        eligible_for_research_evaluation=False,
    )
    return VoicePredictionResponse(
        request_id=request_id,
        audio=AudioMetadata(
            original_filename="sample.wav",
            content_type="audio/wav",
            file_size_bytes=100,
            duration_seconds=1.0,
            sample_rate=16000,
            channels=1,
        ),
        branches=[branch],
        fusion=fusion,
        total_processing_time_ms=2.0,
    )
