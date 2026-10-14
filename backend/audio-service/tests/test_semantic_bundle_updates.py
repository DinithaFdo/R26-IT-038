"""A retrained model activates without changing source or a hardcoded R²."""
from pathlib import Path
import json
import shutil
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

ROOT=Path(__file__).parents[1]/"app/voice_xai/semantic/artifacts"

@pytest.fixture
def source(tmp_path):
    path=tmp_path/"export"
    shutil.copytree(resolve_bundle(ROOT),path)
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
    assert first.startswith("asv2019_selected20_aligned_reg50_v1-")
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
        metadata["primary_test_fidelity"] = {"r2": float("nan"), "n": 3991}
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
