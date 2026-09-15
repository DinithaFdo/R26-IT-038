import numpy as np
import pytest

from app.schemas.common import BranchStatus, ModelMode, PredictionLabel, SourceType
from app.schemas.prediction import (
    AudioMetadata,
    BranchPrediction,
    FusionResult,
    ProbabilityScores,
    VoicePredictionResponse,
)
from app.voice_xai.contracts import ClassifierInferenceBundle
from app.voice_xai.semantic.features import (
    FEATURE_EXTRACTION_VERSION,
    FEATURE_NAMES,
    FeatureExtractionResult,
)
from app.voice_xai.capture.contracts import ExtractionBundle
from app.voice_xai.semantic.artifact_loader import (
    SemanticArtifactCompatibilityError,
    SemanticArtifactManifest,
)
from app.voice_xai.semantic.service import MockSemanticExplanationService


def test_mock_semantic_service_is_deterministic_and_explicitly_non_research() -> None:
    service = MockSemanticExplanationService()

    first = service.analyze(_inference("request-123"))
    second = service.analyze(_inference("request-123"))

    assert first.development_placeholder is True
    assert first.research_eligible is False
    assert first.explanation.status.value == "completed"
    assert first.explanation.target_type.value == "independent_acoustic_evidence_model"
    assert first.explanation.output_space == "raw_margin"
    assert len(first.explanation.feature_importance) == 10
    assert [item.rank for item in first.explanation.feature_importance] == list(range(1, 11))
    assert first.explanation.model_dump(mode="json") == second.explanation.model_dump(
        mode="json"
    )


def test_mock_semantic_service_uses_phase1_feature_values_when_available() -> None:
    values = np.arange(len(FEATURE_NAMES), dtype=np.float32)
    inference = _inference("request-with-features", feature_values=values)

    result = MockSemanticExplanationService().analyze(inference)

    returned = {item.feature_name: item.value for item in result.explanation.feature_importance}
    assert set(returned).issubset(set(FEATURE_NAMES))
    for feature_name, value in returned.items():
        assert value == float(values[FEATURE_NAMES.index(feature_name)])
    assert "timestamp-level localization" in result.warnings[1]


def test_semantic_manifest_rejects_feature_order_drift() -> None:
    with pytest.raises(SemanticArtifactCompatibilityError, match="runtime order"):
        SemanticArtifactManifest(
            model_version="model-v1",
            feature_schema_version="xgboost-surrogate-v4-148",
            feature_names=tuple(reversed(FEATURE_NAMES)),
            extractor_version=FEATURE_EXTRACTION_VERSION,
            output_space="raw_margin",
            feature_count=len(FEATURE_NAMES),
            model_sha256="0" * 64,
            feature_columns_sha256="0" * 64,
            decision_threshold_sha256="0" * 64,
            training_metadata_sha256="0" * 64,
            imputer_sha256="0" * 64,
            decision_threshold=0.5,
            training_dataset_version="test",
        )


def _inference(
    request_id: str,
    feature_values: np.ndarray | None = None,
) -> ClassifierInferenceBundle:
    probabilities = ProbabilityScores(bonafide=0.2, spoof=0.8)
    prediction = VoicePredictionResponse(
        request_id=request_id,
        audio=AudioMetadata(
            original_filename="sample.wav",
            content_type="audio/wav",
            file_size_bytes=32000,
            duration_seconds=2.0,
            sample_rate=16000,
            channels=1,
        ),
        branches=[
            BranchPrediction(
                model_name="glottal_features",
                display_name="Glottal features",
                status=BranchStatus.success,
                mode=ModelMode.dummy,
                prediction=PredictionLabel.spoof,
                confidence=0.8,
                probabilities=probabilities,
                processing_time_ms=1.0,
            )
        ],
        fusion=FusionResult(
            status=BranchStatus.success,
            prediction=PredictionLabel.spoof,
            confidence=0.8,
            probabilities=probabilities,
            method="weighted_average",
            contains_dummy_branches=True,
            eligible_for_research_evaluation=False,
        ),
        total_processing_time_ms=1.0,
    )
    extraction = None
    if feature_values is not None:
        extraction = ExtractionBundle(
            request_id=request_id,
            branches={},
            acoustic_features=FeatureExtractionResult(
                feature_names=FEATURE_NAMES,
                values=feature_values,
                sample_rate=16000,
                duration_seconds=2.0,
                extractor_version=FEATURE_EXTRACTION_VERSION,
                metadata={},
            ),
        )
    return ClassifierInferenceBundle(
        prediction_id="prediction-123",
        request_id=request_id,
        owner_user_id="user-123",
        source_type=SourceType.dashboard_upload,
        prediction=prediction,
        extraction=extraction,
    )
