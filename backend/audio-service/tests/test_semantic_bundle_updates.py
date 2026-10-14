"""A retrained model activates without changing source or a hardcoded R²."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from app.config.settings import Settings
from app.voice_xai.semantic.artifact_loader import SemanticArtifactCompatibilityError,resolve_bundle,load_production_semantic_artifacts
from app.voice_xai.semantic.import_bundle import import_bundle
from app.voice_xai.semantic.features import FeatureExtractionResult
from app.voice_xai.semantic.schema import FEATURE_NAMES,FEATURE_EXTRACTION_VERSION
from app.voice_xai.semantic.service import ProductionSemanticExplanationService
from app.voice_xai.workflow.orchestrator import _configuration_hash
from tests.semantic_support import write_model_export

@pytest.fixture
def source(tmp_path):
    path=tmp_path/"export"
    write_model_export(path)
    return path

def test_retrained_model_and_new_r2_replace_as_one_bundle(source,tmp_path):
    destination=tmp_path/"models"
    first=import_bundle(source,destination)
    old_service=ProductionSemanticExplanationService.load_once(bundle_dir=destination)
    settings=Settings(_env_file=None,xai_semantic_bundle_dir=str(destination))
    first_hash=_configuration_hash(settings)
    rng=np.random.default_rng(12)
    X=pd.DataFrame(rng.normal(size=(24,165)),columns=FEATURE_NAMES)
    model=xgb.XGBRegressor(n_estimators=4,max_depth=2,objective="reg:squarederror",n_jobs=1)
    model.fit(X,np.linspace(0,1,24))
    model.save_model(source/"surrogate_xgboost.json")
    metadata=json.loads((source/"training_metadata.json").read_text())
    metadata["experiment_name"]="retrained-run"
    metadata["primary_test_fidelity"]["r2"]=0.93
    (source/"training_metadata.json").write_text(json.dumps(metadata))
    second=import_bundle(source,destination)
    assert first!=second
    assert _configuration_hash(settings)!=first_hash
    with pytest.raises(SemanticArtifactCompatibilityError,match="Restart"):
        old_service.analyze(SimpleNamespace())
    new=ProductionSemanticExplanationService.load_once(bundle_dir=destination)
    features=FeatureExtractionResult(FEATURE_NAMES,new.artifacts.imputer.statistics_.copy(),16000,6,FEATURE_EXTRACTION_VERSION,{})
    result=new.analyze(SimpleNamespace(acoustic_features=features)).explanation
    assert result.model_version==second
    assert result.fidelity_r2.value==0.93
    assert first.startswith("fullclip-test-v2-")
    assert len(list((destination/"bundles").iterdir()))==2

def test_invalid_update_preserves_active_model(source,tmp_path):
    destination=tmp_path/"models"
    import_bundle(source,destination)
    pointer=(destination/"active.json").read_bytes()
    (source/"feature_columns.json").write_text(json.dumps(list(reversed(FEATURE_NAMES))))
    with pytest.raises(SemanticArtifactCompatibilityError,match="feature order"):
        import_bundle(source,destination)
    assert (destination/"active.json").read_bytes()==pointer
    assert load_production_semantic_artifacts(destination).model_version

@pytest.mark.parametrize("null_metric", [False, True, "nan"])
def test_import_without_fidelity_still_runs(source,tmp_path,null_metric):
    metadata=json.loads((source/"training_metadata.json").read_text())
    metadata.pop("primary_test_fidelity",None)
    if null_metric:
        metadata["primary_test_fidelity"] = None
    if null_metric == "nan":
        metadata["primary_test_fidelity"] = {"r2": float("nan"), "n": 12}
    (source/"training_metadata.json").write_text(json.dumps(metadata))
    destination=tmp_path/"models"
    import_bundle(source,destination)
    service=ProductionSemanticExplanationService.load_once(bundle_dir=destination)
    features=FeatureExtractionResult(FEATURE_NAMES,service.artifacts.imputer.statistics_.copy(),16000,6,FEATURE_EXTRACTION_VERSION,{})
    result=service.analyze(SimpleNamespace(acoustic_features=features)).explanation
    assert result.fidelity_r2.status.value=="not_computed"
    assert result.clip_summary is not None

def test_tampering_is_rejected_before_loading(source,tmp_path):
    destination=tmp_path/"models"
    import_bundle(source,destination)
    (resolve_bundle(destination)/"training_metadata.json").write_text("{}")
    with pytest.raises(SemanticArtifactCompatibilityError,match="hash mismatch"):
        load_production_semantic_artifacts(destination)

def test_partial_export_does_not_replace_current(source,tmp_path):
    destination=tmp_path/"models"
    import_bundle(source,destination)
    pointer=(destination/"active.json").read_bytes()
    (source/"imputer.joblib").unlink()
    with pytest.raises(SemanticArtifactCompatibilityError,match="Missing required model export"):
        import_bundle(source,destination)
    assert (destination/"active.json").read_bytes()==pointer

def test_binary_classifier_export_cannot_activate(source,tmp_path):
    destination=tmp_path/"models"
    import_bundle(source,destination)
    pointer=(destination/"active.json").read_bytes()
    model=json.loads((source/"surrogate_xgboost.json").read_text())
    model["learner"]["objective"]["name"]="binary:logistic"
    (source/"surrogate_xgboost.json").write_text(json.dumps(model))
    with pytest.raises(SemanticArtifactCompatibilityError,match="Expected a 165-feature XGBRegressor"):
        import_bundle(source,destination)
    assert (destination/"active.json").read_bytes()==pointer


@pytest.mark.parametrize("field", ["extractor_version", "feature_schema_version", "preprocessing"])
def test_missing_fullclip_contract_cannot_relabel_old_exports(source, tmp_path, field):
    destination = tmp_path / "models"
    import_bundle(source, destination)
    pointer = (destination / "active.json").read_bytes()
    metadata = json.loads((source / "training_metadata.json").read_text())
    metadata.pop(field)
    (source / "training_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(SemanticArtifactCompatibilityError, match="Re-extract training features"):
        import_bundle(source, destination)
    assert (destination / "active.json").read_bytes() == pointer


def test_existing_v1_bundle_requires_retraining(source, tmp_path):
    destination = tmp_path / "models"
    import_bundle(source, destination)
    directory = resolve_bundle(destination)
    manifest = json.loads((directory / "bundle_manifest.json").read_text())
    manifest["bundle_version"] = "notebook-surrogate-bundle-v1"
    manifest["extractor_version"] = "selected20-aligned-audio-v1"
    (directory / "bundle_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SemanticArtifactCompatibilityError, match="retrained v2 bundle"):
        load_production_semantic_artifacts(directory)


@pytest.mark.parametrize("setting,value", [("crop_seconds", 6.0), ("disvoice_input", "original_audio_pcm16")])
def test_wrong_preprocessing_cannot_activate(source, tmp_path, setting, value):
    destination = tmp_path / "models"
    import_bundle(source, destination)
    pointer = (destination / "active.json").read_bytes()
    metadata = json.loads((source / "training_metadata.json").read_text())
    metadata["preprocessing"][setting] = value
    (source / "training_metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(SemanticArtifactCompatibilityError, match="Re-extract training features"):
        import_bundle(source, destination)
    assert (destination / "active.json").read_bytes() == pointer


def test_configuration_cache_includes_extraction_version(monkeypatch):
    settings = Settings(_env_file=None)
    first = _configuration_hash(settings)
    monkeypatch.setattr("app.voice_xai.semantic.schema.FEATURE_EXTRACTION_VERSION", "different-extractor")
    assert _configuration_hash(settings) != first
