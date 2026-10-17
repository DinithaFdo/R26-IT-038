"""Manual model replacement, startup validation, and cache provenance tests."""
import json
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from app.config.settings import Settings
from app.voice_xai.semantic.artifact_loader import (
    SemanticArtifactCompatibilityError, load_production_semantic_artifacts, model_fingerprint,
)
from app.voice_xai.semantic.feature_extraction import FeatureExtractionResult
from app.voice_xai.semantic.schema import FEATURE_NAMES, FEATURE_EXTRACTION_VERSION
from app.voice_xai.semantic.service import ProductionSemanticExplanationService
from app.voice_xai.workflow.orchestrator import _configuration_hash
from tests.semantic_support import write_model_export


@pytest.fixture
def model_dir(tmp_path):
    path = tmp_path / "model"
    write_model_export(path)
    return path


def feature_input(service):
    return SimpleNamespace(acoustic_features=FeatureExtractionResult(
        FEATURE_NAMES, service.artifacts.imputer.statistics_.copy(),
        16000, 12.0, FEATURE_EXTRACTION_VERSION, {},
    ))


def test_fixed_directory_loads_without_a_manifest_or_import_command(model_dir):
    assert len(list(model_dir.iterdir())) == 4
    service = ProductionSemanticExplanationService.load_once(model_dir=model_dir)
    result = service.analyze(feature_input(service)).explanation
    assert len(result.feature_importance) == 165
    assert result.model_version.startswith("fullclip-test-v2-")
    assert result.clip_summary is not None
    assert len(list(model_dir.iterdir())) == 4


def test_manually_replaced_model_loads_after_restart(model_dir):
    old_service = ProductionSemanticExplanationService.load_once(model_dir=model_dir)
    first = old_service.artifacts.model_version
    settings = Settings(_env_file=None, xai_semantic_model_dir=str(model_dir))
    first_hash = _configuration_hash(settings)
    rng = np.random.default_rng(12)
    inputs = pd.DataFrame(rng.normal(size=(24, 165)), columns=FEATURE_NAMES)
    model = xgb.XGBRegressor(n_estimators=4, max_depth=2, objective="reg:squarederror", n_jobs=1)
    model.fit(inputs, np.linspace(0, 1, 24))
    model.save_model(model_dir / "surrogate_xgboost.json")
    metadata = json.loads((model_dir / "training_metadata.json").read_text())
    metadata["experiment_name"] = "retrained-run"
    metadata["primary_test_fidelity"]["r2"] = 0.93
    (model_dir / "training_metadata.json").write_text(json.dumps(metadata))
    assert _configuration_hash(settings) != first_hash
    with pytest.raises(SemanticArtifactCompatibilityError, match="Restart"):
        old_service.analyze(feature_input(old_service))
    new_service = ProductionSemanticExplanationService.load_once(model_dir=model_dir)
    result = new_service.analyze(feature_input(new_service)).explanation
    assert result.model_version != first
    assert result.fidelity_r2.value == 0.93
    assert len(list(model_dir.iterdir())) == 4


def test_model_bytes_change_identity_without_metadata_edits(model_dir):
    first = load_production_semantic_artifacts(model_dir).model_version
    metadata = (model_dir / "training_metadata.json").read_bytes()
    inputs = pd.DataFrame(np.zeros((4, 165)), columns=FEATURE_NAMES)
    model = xgb.XGBRegressor(n_estimators=1, n_jobs=1)
    model.fit(inputs, np.array([0.1, 0.2, 0.3, 0.4]))
    model.save_model(model_dir / "surrogate_xgboost.json")
    assert load_production_semantic_artifacts(model_dir).model_version != first
    assert (model_dir / "training_metadata.json").read_bytes() == metadata


def test_invalid_feature_order_is_rejected_at_startup(model_dir):
    (model_dir / "feature_columns.json").write_text(json.dumps(list(reversed(FEATURE_NAMES))))
    with pytest.raises(SemanticArtifactCompatibilityError, match="feature order"):
        load_production_semantic_artifacts(model_dir)


@pytest.mark.parametrize("metric", [None, {}, {"r2": float("nan"), "n": 12}])
def test_unavailable_fidelity_does_not_block_inference(model_dir, metric):
    metadata = json.loads((model_dir / "training_metadata.json").read_text())
    metadata["primary_test_fidelity"] = metric
    (model_dir / "training_metadata.json").write_text(json.dumps(metadata))
    service = ProductionSemanticExplanationService.load_once(model_dir=model_dir)
    result = service.analyze(feature_input(service)).explanation
    assert result.fidelity_r2.status.value == "not_computed"
    assert result.clip_summary is not None


@pytest.mark.parametrize("filename", ["surrogate_xgboost.json", "imputer.joblib", "feature_columns.json", "training_metadata.json"])
def test_missing_required_file_is_rejected(model_dir, filename):
    (model_dir / filename).unlink()
    with pytest.raises(SemanticArtifactCompatibilityError, match="Missing required model file"):
        load_production_semantic_artifacts(model_dir)


def test_partial_replacement_cannot_use_an_old_in_memory_model(model_dir):
    service = ProductionSemanticExplanationService.load_once(model_dir=model_dir)
    (model_dir / "imputer.joblib").unlink()
    with pytest.raises(SemanticArtifactCompatibilityError, match="Restart"):
        service.analyze(feature_input(service))


def test_binary_classifier_export_is_rejected(model_dir):
    model = json.loads((model_dir / "surrogate_xgboost.json").read_text())
    model["learner"]["objective"]["name"] = "binary:logistic"
    (model_dir / "surrogate_xgboost.json").write_text(json.dumps(model))
    with pytest.raises(SemanticArtifactCompatibilityError, match="Expected a 165-feature XGBRegressor"):
        load_production_semantic_artifacts(model_dir)


@pytest.mark.parametrize("field", ["extractor_version", "feature_schema_version", "preprocessing"])
def test_missing_fullclip_contract_is_rejected(model_dir, field):
    metadata = json.loads((model_dir / "training_metadata.json").read_text())
    metadata.pop(field)
    (model_dir / "training_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(SemanticArtifactCompatibilityError, match="Re-extract training features"):
        load_production_semantic_artifacts(model_dir)


@pytest.mark.parametrize("setting,value", [("crop_seconds", 6.0), ("disvoice_input", "original_audio_pcm16")])
def test_old_preprocessing_is_rejected(model_dir, setting, value):
    metadata = json.loads((model_dir / "training_metadata.json").read_text())
    metadata["preprocessing"][setting] = value
    (model_dir / "training_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(SemanticArtifactCompatibilityError, match="Re-extract training features"):
        load_production_semantic_artifacts(model_dir)


def test_wrong_imputer_is_rejected(model_dir):
    imputer = joblib.load(model_dir / "imputer.joblib")
    imputer.statistics_ = imputer.statistics_[:-1]
    joblib.dump(imputer, model_dir / "imputer.joblib")
    with pytest.raises(SemanticArtifactCompatibilityError, match="165-feature median imputer"):
        load_production_semantic_artifacts(model_dir)


def test_corrupted_model_json_is_rejected(model_dir):
    (model_dir / "surrogate_xgboost.json").write_text("{invalid")
    with pytest.raises(SemanticArtifactCompatibilityError, match="Cannot read surrogate_xgboost.json"):
        load_production_semantic_artifacts(model_dir)


def test_configuration_cache_includes_extraction_version(monkeypatch):
    settings = Settings(_env_file=None)
    first = _configuration_hash(settings)
    monkeypatch.setattr("app.voice_xai.semantic.schema.FEATURE_EXTRACTION_VERSION", "different-extractor")
    assert _configuration_hash(settings) != first


@pytest.mark.parametrize("filename", ["notes.txt", "unused_model.joblib", "unrelated.json"])
def test_fingerprint_ignores_unrelated_files(model_dir, filename):
    first = model_fingerprint(model_dir)
    (model_dir / filename).write_text("Unrelated content")
    assert model_fingerprint(model_dir) == first
