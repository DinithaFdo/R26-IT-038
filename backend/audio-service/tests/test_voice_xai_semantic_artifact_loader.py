from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.voice_xai.semantic.artifact_loader import (
    SemanticArtifactCompatibilityError,
    SemanticArtifactManifest,
    _build_xgboost_tree_explainer,
    _load_feature_columns,
    _load_imputer,
    _sha256,
    _validate_model_file,
    load_production_semantic_artifacts,
)
from app.voice_xai.semantic.features import FEATURE_EXTRACTION_VERSION, FEATURE_NAMES


ROOT = Path(__file__).parents[1] / "app" / "voice_xai" / "semantic"
MANIFEST_PATH = ROOT / "manifests" / "xgboost-surrogate-v4.json"
ARTIFACTS = ROOT / "artifacts"


def test_checked_in_v4_manifest_and_artifacts_freeze_the_supplied_contract() -> None:
    manifest = SemanticArtifactManifest.from_file(MANIFEST_PATH)

    assert manifest.feature_count == 148
    assert manifest.feature_count == len(FEATURE_NAMES)
    assert manifest.extractor_version == FEATURE_EXTRACTION_VERSION
    assert manifest.decision_threshold == pytest.approx(0.32371800847220844)
    assert manifest.model_sha256 == hashlib.sha256(
        (ARTIFACTS / "xgboost_surrogate_v4.json").read_bytes()
    ).hexdigest()
    assert manifest.imputer_sha256 == hashlib.sha256(
        (ARTIFACTS / "imputer_v4.joblib").read_bytes()
    ).hexdigest()
    assert manifest.feature_columns_sha256 == _sha256(ARTIFACTS / "feature_columns_v4.json")
    assert manifest.decision_threshold_sha256 == _sha256(ARTIFACTS / "threshold_v4.json")
    assert manifest.training_metadata_sha256 == _sha256(ARTIFACTS / "training_metadata_v4.json")
    assert _load_feature_columns(ARTIFACTS / "feature_columns_v4.json", manifest) == FEATURE_NAMES


def test_json_artifact_hash_is_stable_across_windows_line_endings(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_bytes(b'{\r\n  "feature": "value"\r\n}\r\n')

    assert _sha256(path) == hashlib.sha256(
        b'{\n  "feature": "value"\n}\n'
    ).hexdigest()


def test_loader_rejects_feature_columns_with_the_right_count_but_wrong_order(tmp_path: Path) -> None:
    columns_path = tmp_path / "columns.json"
    columns_path.write_text(json.dumps(list(reversed(FEATURE_NAMES))), encoding="utf-8")
    manifest = replace(
        SemanticArtifactManifest.from_file(MANIFEST_PATH),
        feature_columns_sha256=hashlib.sha256(columns_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(SemanticArtifactCompatibilityError, match="runtime extractor order"):
        _load_feature_columns(columns_path, manifest)


def test_model_validation_uses_count_when_xgboost_json_has_no_feature_names(tmp_path: Path) -> None:
    model_path = tmp_path / "model.json"
    model_path.write_text(
        json.dumps(
            {
                "version": [3, 3, 0],
                "learner": {
                    "objective": {"name": "binary:logistic"},
                    "learner_model_param": {"num_feature": str(len(FEATURE_NAMES))},
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = replace(
        SemanticArtifactManifest.from_file(MANIFEST_PATH),
        model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
    )

    _validate_model_file(model_path, manifest)


def test_loader_restores_newer_sklearn_runtime_state_for_pinned_imputer() -> None:
    manifest = SemanticArtifactManifest.from_file(MANIFEST_PATH)

    imputer = _load_imputer(ARTIFACTS / "imputer_v4.joblib", manifest)
    transformed = imputer.transform([[float("nan")] * len(FEATURE_NAMES)])

    assert hasattr(imputer, "_fill_dtype")
    assert transformed.shape == (1, len(FEATURE_NAMES))
    assert transformed.tolist()[0] == pytest.approx(imputer.statistics_.tolist())


def test_xgboost_tree_explainer_does_not_probe_transformers_backends() -> None:
    original_detector = object()
    state = {"detector_during_init": None}

    def build_tree_explainer(_booster, *, data, model_output):
        state["detector_during_init"] = fake_explainer_module.is_transformers_lm
        assert data is None
        assert model_output == "raw"
        return "tree-explainer"

    fake_explainer_module = SimpleNamespace(is_transformers_lm=original_detector)
    fake_shap = SimpleNamespace(
        TreeExplainer=build_tree_explainer,
        explainers=SimpleNamespace(_explainer=fake_explainer_module),
    )

    result = _build_xgboost_tree_explainer(fake_shap, object(), None)

    assert result == "tree-explainer"
    assert state["detector_during_init"](object()) is False
    assert fake_explainer_module.is_transformers_lm is original_detector
