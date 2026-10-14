"""Reference inference outputs and original-audio handoff regression tests."""
import ast
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from app.config.settings import Settings
from app.ingestion import audio as ingestion
from app.voice_xai.semantic.artifact_loader import load_production_semantic_artifacts
from app.voice_xai.semantic.features import AcousticFeatureExtractor, FeatureExtractionResult
from app.voice_xai.semantic.schema import FEATURE_NAMES, FEATURE_EXTRACTION_VERSION, SELECTED20
from app.voice_xai.semantic.service import ProductionSemanticExplanationService

ROOT=Path(__file__).parents[1]/"app/voice_xai/semantic/artifacts"
FIXTURES=Path(__file__).parent/"fixtures/semantic_reference"

@pytest.fixture(scope="module")
def service():
    return ProductionSemanticExplanationService.load_once(bundle_dir=ROOT)

def feature_row(values):
    return FeatureExtractionResult(FEATURE_NAMES,np.asarray(values,dtype=np.float64),16000,6.0,FEATURE_EXTRACTION_VERSION,{})

def test_imputer_reproduces_all_ten_reference_vectors(service):
    raw=pd.read_csv(FIXTURES/"raw_features_165.csv").set_index("analysis_id")
    expected=pd.read_csv(FIXTURES/"imputed_features_165.csv").set_index("analysis_id")
    actual=service.artifacts.imputer.transform(raw[list(FEATURE_NAMES)])
    np.testing.assert_array_equal(actual,expected[list(FEATURE_NAMES)])

@pytest.mark.parametrize("suffix",["","_clip08"])
def test_reference_prediction_and_all_165_shap_values_match(service,suffix):
    expected=json.loads((FIXTURES/f"prediction{suffix}.json").read_text())
    raw=pd.read_csv(FIXTURES/"raw_features_165.csv").set_index("analysis_id")
    result=service.analyze(SimpleNamespace(acoustic_features=feature_row(raw.loc[expected["analysis_id"],list(FEATURE_NAMES)]),processed_audio=None)).explanation
    assert result.predicted_value==expected["raw_xgb_score"]
    assert result.base_value==expected["shap_base_value"]
    assert result.clip_summary.spoof_probability==expected["spoof_probability_clipped"]
    assert result.target_type.value=="classifier_surrogate"
    assert result.output_space=="raw_regression"
    assert result.clip_summary.aggregation_method=="whole_clip_surrogate"
    assert "windows" not in result.model_dump()
    reference=pd.read_csv(FIXTURES/f"all_feature_shap_values{suffix}.csv").set_index("feature")
    assert len(result.feature_importance)==165
    for item in result.feature_importance:
        row=reference.loc[item.feature_name]
        assert item.value==row.feature_value_after_imputation
        assert item.shap_value==pytest.approx(row.shap_value,abs=1e-7)
        assert item.direction.value==row.direction
    assert result.base_value+sum(item.shap_value for item in result.feature_importance)==pytest.approx(result.predicted_value,abs=1e-4)

def test_missing_input_features_use_training_medians(service):
    values=service.artifacts.imputer.statistics_.copy()
    values[:2]=np.nan
    evidence=service._provider.extract_features(feature_row(values))
    assert evidence.imputed_feature_count==2
    np.testing.assert_array_equal(evidence.feature_values[:2],service.artifacts.imputer.statistics_[:2])

def test_original_source_is_retained_after_upload_cleanup(monkeypatch,tmp_path):
    path=tmp_path/"source.wav"
    path.write_bytes(b"validated-source-audio")
    decoded=ingestion.DecodedAudio(np.sin(np.arange(32000,dtype=np.float32)*0.1),16000,16000,1)
    monkeypatch.setattr(ingestion,"decode_audio_file",lambda *args:decoded)
    audio=ingestion.preprocess_audio_file(path,app_settings=Settings(_env_file=None,xai_enabled=True,xai_mode="real"),inspection=object())
    path.unlink()
    assert audio.source_audio_bytes==b"validated-source-audio"
    assert audio.source_audio_extension==".wav"

def test_extractor_preserves_base_overlap_and_original_audio(monkeypatch):
    seen=[]
    def load(path):
        seen.append(Path(path).read_bytes())
        return np.ones(32000),16000
    recipe=SimpleNamespace(load_audio_path=load,extract_acoustic_features=lambda *args:dict(zip(FEATURE_NAMES[:148],range(148))),extract_selected20_features_exact=lambda path:{name:999 for name in SELECTED20})
    monkeypatch.setitem(sys.modules,"app.voice_xai.semantic.recipe",recipe)
    result=AcousticFeatureExtractor().extract_processed_audio(SimpleNamespace(source_audio_bytes=b"original-upload",source_audio_extension=".wav"))
    assert seen==[b"original-upload"]
    np.testing.assert_array_equal(result.values[:148],np.arange(148))
    np.testing.assert_array_equal(result.values[148:],np.full(17,999))
    assert result.values.dtype==np.float64

def test_no_normalised_waveform_fallback():
    with pytest.raises(ValueError,match="original source audio"):
        AcousticFeatureExtractor().extract_processed_audio(SimpleNamespace(source_audio_bytes=None))
