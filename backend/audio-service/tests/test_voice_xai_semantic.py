import pytest

from app.voice_xai.semantic.features import FEATURE_NAMES, FEATURE_EXTRACTION_VERSION
from app.voice_xai.semantic.artifact_loader import SemanticArtifactManifest, SemanticArtifactCompatibilityError


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
