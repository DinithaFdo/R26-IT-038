"""Source-aligned window extraction, SHAP localization and coverage regression tests."""

from dataclasses import replace
import json
from time import monotonic
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from app.config.settings import Settings
from app.schemas.xai import ComponentStatus, SemanticWindowExplanation, TemporalExplanation, TemporalRegion
from app.voice_xai.evaluation.consistency import quality_for_analysis
from app.voice_xai.evaluation.intervals import union_spans, intersect_spans, subtract_spans, span_duration
from app.voice_xai.semantic import feature_extraction as extraction
from app.voice_xai.semantic.artifact_loader import load_production_semantic_artifacts, SemanticArtifactCompatibilityError
from app.voice_xai.semantic.contracts import SemanticWindowOptions, WINDOW_EXTRACTION_VERSION, WINDOW_PREPROCESSING_CONTRACT, SemanticEvidence
from app.voice_xai.semantic.explainer import build_window_explanation, build_semantic_localization
from app.voice_xai.semantic.schema import FEATURE_NAMES, FEATURE_EXTRACTION_VERSION
from app.voice_xai.semantic.service import ProductionSemanticExplanationService, XgboostShapEvidenceProvider
from app.voice_xai.workflow.orchestrator import _configuration_hash
from tests.semantic_support import write_model_export


@pytest.fixture(scope='module')
def artifacts(tmp_path_factory):
    directory = tmp_path_factory.mktemp('window-model')
    write_model_export(directory)
    return load_production_semantic_artifacts(directory)


def prepared(start=0, seconds=6):
    sr = 16000
    return extraction.PreparedSemanticAudio(np.ones(round(seconds * sr), dtype=np.float32),
                                            sr, round((start + seconds) * sr), round(start * sr),
                                            round((start + seconds) * sr))


def window(start, end, activity=1.0, status='completed', index=1):
    bounds = extraction.SemanticWindow(round(start * 16000), round(end * 16000))
    if status != 'completed':
        return SemanticWindowExplanation(window_id=index, start_seconds=start, end_seconds=end,
                                         start_sample=bounds.start_sample, end_sample_exclusive=bounds.end_sample,
                                         status='not_available', reason='fixture_failure')
    shap = np.zeros(165)
    shap[0] = activity
    return build_window_explanation(bounds, SemanticEvidence(np.ones(165), shap, 0.2, 0.2 + activity),
                                    window_id=index, sample_rate=16000, processing_seconds=0.0)


@pytest.mark.parametrize('size,hop,count', [(1, .5, 11), (2, 1, 5), (3, 1.5, 3), (4, 2, 2)])
def test_configuration_candidates_use_identical_temporal_coverage(size, hop, count):
    plan = extraction.build_semantic_windows(prepared(3), [(3, 9)], window_seconds=size,
                                              hop_seconds=hop, max_windows=32)
    assert len(plan) == count
    assert plan[0].start_sample == 3 * 16000
    assert plan[-1].end_sample == 9 * 16000
    assert all(3 * 16000 <= item.start_sample < item.end_sample <= 9 * 16000 for item in plan)


def test_window_plan_does_not_bridge_gaps_or_pad_short_spans():
    plan = extraction.build_semantic_windows(prepared(seconds=10), [(1, 2.3), (4, 6.5)],
                                              window_seconds=2, hop_seconds=1, max_windows=32)
    assert [(item.start_sample / 16000, item.end_sample / 16000) for item in plan] == [(1, 2.3), (4, 6), (4.5, 6.5)]


def test_window_plan_respects_trim_bounds_and_limit():
    audio = prepared(start=1, seconds=5)
    plan = extraction.build_semantic_windows(audio, [(0, 7)], window_seconds=2, hop_seconds=1, max_windows=4)
    assert plan[0].start_sample == 16000 and plan[-1].end_sample == 6 * 16000
    with pytest.raises(ValueError, match='limit exceeded'):
        extraction.build_semantic_windows(audio, [(0, 7)], window_seconds=1, hop_seconds=.1, max_windows=4)
    assert extraction.build_semantic_windows(audio, [(0, .1)], window_seconds=2, hop_seconds=1, max_windows=4) == ()


def test_preprocessing_records_offsets_without_changing_fullclip_values():
    sr = 16000
    signal = np.concatenate([np.zeros(sr), np.sin(2 * np.pi * 180 * np.arange(3 * sr) / sr), np.zeros(sr)])
    audio = extraction.prepare_audio_with_timing(signal, sr)
    y, rate = extraction.prepare_audio(signal, sr)
    np.testing.assert_array_equal(audio.waveform, y)
    assert rate == audio.sample_rate == sr
    assert audio.source_sample_count == len(signal)
    assert 0 < audio.trim_start_sample < sr
    assert audio.trim_end_sample - audio.trim_start_sample == len(y)


def test_window_extraction_does_not_trim_or_normalize_again(monkeypatch):
    audio = prepared(start=1, seconds=4)
    audio.waveform[:32000] = .25
    seen = []
    def mapping(y, sr):
        seen.append(y.copy())
        return dict.fromkeys(FEATURE_NAMES, .5)
    monkeypatch.setattr(extraction, 'extract_feature_mapping', mapping)
    result = extraction.AcousticFeatureExtractor().extract_prepared_window(audio, extraction.SemanticWindow(16000, 48000))
    assert np.all(seen[0] == .25)
    assert result.extractor_version == WINDOW_EXTRACTION_VERSION
    assert result.duration_seconds == 2
    assert result.metadata['window_trim'] is False and result.metadata['window_normalization'] is False
    with pytest.raises(ValueError, match='outside cleaned'):
        extraction.AcousticFeatureExtractor().extract_prepared_window(audio, extraction.SemanticWindow(0, 32000))


def test_fullclip_and_window_model_contracts_remain_distinct(artifacts):
    provider = XgboostShapEvidenceProvider(artifacts)
    values = artifacts.imputer.statistics_.copy()
    full = extraction.FeatureExtractionResult(FEATURE_NAMES, values, 16000, 2, FEATURE_EXTRACTION_VERSION, {})
    measured = replace(full, extractor_version=WINDOW_EXTRACTION_VERSION,
                       metadata={**WINDOW_PREPROCESSING_CONTRACT, 'analysis_scope': 'window'})
    with pytest.raises(ValueError, match='semantic model contract'):
        provider.extract_features(measured)
    with pytest.raises(ValueError, match='window contract'):
        provider.extract_window_features(full)
    assert provider.extract_window_features(measured).predicted_value == provider.extract_features(full).predicted_value


def test_failed_window_support_is_excluded_even_when_successful_windows_overlap(artifacts):
    result = build_semantic_localization([window(0, 2), window(1, 3, status='failed', index=2), window(2, 4, index=3)],
                                         [(0, 4)], options=SemanticWindowOptions(), artifacts=artifacts)
    assert result.status == 'partial'
    assert [(span.start_seconds, span.end_seconds) for span in result.valid_spans] == [(0, 1), (3, 4)]
    assert result.coverage_percentage == 50
    assert [(region.start_seconds, region.end_seconds) for region in result.regions] == [(0, 1), (3, 4)]


def test_activity_uses_absolute_shap_and_duration_weighted_threshold(artifacts):
    # A long low-activity segment must carry more percentile mass than a short high segment.
    result = build_semantic_localization([window(0, 9, activity=1), window(9, 10, activity=9, index=2)],
                                         [(0, 10)], options=SemanticWindowOptions(), artifacts=artifacts)
    assert result.activity_threshold == 1
    assert result.model_validation_status == 'experimental'
    assert result.window_fidelity_r2.status.value == 'not_computed'
    assert result.window_seconds == 2
    assert result.regions[0].end_seconds == 10


def test_iou_uses_common_valid_coverage_and_union_lengths(artifacts):
    localization = build_semantic_localization([window(0, 2), window(4, 6, index=2)], [(0, 6)],
                                               options=SemanticWindowOptions(), artifacts=artifacts)
    temporal = TemporalExplanation(status=ComponentStatus.completed, method_version='fixture',
                                    high_attention_regions=[TemporalRegion(region_id=1, start_seconds=1, end_seconds=5, attention_score=2)])
    semantic = SimpleNamespace(localization=localization, fidelity_r2=None)
    result = quality_for_analysis(temporal, semantic, temporal_spans=[(0, 6)])
    assert result.temporal_semantic_iou.value == .5
    assert '4.000000s' in result.temporal_semantic_iou.reason
    assert 'experimental' in result.temporal_semantic_iou.reason


def test_empty_union_and_missing_coverage_do_not_fabricate_iou(artifacts):
    localization = build_semantic_localization([window(0, 2, activity=0)], [(0, 2)],
                                               options=SemanticWindowOptions(), artifacts=artifacts)
    temporal = TemporalExplanation(status=ComponentStatus.completed, method_version='fixture')
    semantic = SimpleNamespace(localization=localization, fidelity_r2=None)
    assert quality_for_analysis(temporal, semantic).temporal_semantic_iou.status.value == 'not_computed'
    assert quality_for_analysis(temporal, semantic, temporal_spans=[(3, 4)]).temporal_semantic_iou.value is None


def test_source_is_decoded_once_and_windows_keep_fullclip_prediction(monkeypatch, tmp_path, artifacts):
    sr = 16000
    signal = np.sin(2 * np.pi * 180 * np.arange(12 * sr) / sr)
    path = tmp_path / 'source.wav'
    sf.write(path, signal, sr)
    inference = SimpleNamespace(acoustic_features=None, processed_audio=SimpleNamespace(
        source_audio_bytes=path.read_bytes(), source_audio_extension='.wav'))
    mapping = dict(zip(FEATURE_NAMES, artifacts.imputer.statistics_))
    monkeypatch.setattr(extraction, 'extract_feature_mapping', lambda *args: mapping.copy())
    calls = []
    read = sf.read
    def read_once(*args, **kwargs):
        calls.append(args[0])
        return read(*args, **kwargs)
    monkeypatch.setattr(sf, 'read', read_once)
    service = ProductionSemanticExplanationService(artifacts)
    result = service.analyze(inference, analyzed_spans=[(3, 9)]).explanation
    assert len(calls) == 1
    assert len(result.localization.windows) == 5
    assert result.localization.coverage_percentage == 100
    assert result.localization.model_validation_status == 'experimental'
    assert result.research_eligible is False
    baseline = ProductionSemanticExplanationService(artifacts, window_options=SemanticWindowOptions(enabled=False)).analyze(inference).explanation
    assert result.predicted_value == baseline.predicted_value
    assert result.feature_importance == baseline.feature_importance
    # Optional defaults retain compatibility with older serialized explanations.
    payload = result.model_dump(mode='json')
    assert json.loads(json.dumps(payload))['localization']['time_reference'] == 'source_audio'


def test_expired_window_budget_preserves_fullclip_and_excludes_unprocessed_windows(monkeypatch, artifacts):
    extractor = extraction.AcousticFeatureExtractor()
    monkeypatch.setattr(extractor, 'prepare_processed_audio', lambda _: prepared())
    monkeypatch.setattr(extraction, 'extract_feature_mapping', lambda *args: dict(zip(FEATURE_NAMES, artifacts.imputer.statistics_)))
    inference = SimpleNamespace(acoustic_features=None, processed_audio=SimpleNamespace(source_audio_bytes=b'fixture'))
    result = ProductionSemanticExplanationService(artifacts, feature_extractor=extractor).analyze(
        inference, analyzed_spans=[(0, 6)], deadline=monotonic() - 1).explanation
    assert result.predicted_value is not None
    assert result.localization.status == 'not_available'
    assert result.localization.valid_spans == []
    assert all(item.reason == 'window_budget_exhausted' for item in result.localization.windows)


def test_one_window_failure_keeps_other_evidence_and_fullclip_result(monkeypatch, artifacts):
    extractor = extraction.AcousticFeatureExtractor()
    monkeypatch.setattr(extractor, 'prepare_processed_audio', lambda _: prepared())
    count = 0
    def mapping(y, sr):
        nonlocal count
        if len(y) == 32000:
            count += 1
            if count == 3:
                raise ValueError('private extraction details')
        return dict(zip(FEATURE_NAMES, artifacts.imputer.statistics_))
    monkeypatch.setattr(extraction, 'extract_feature_mapping', mapping)
    inference = SimpleNamespace(acoustic_features=None, processed_audio=SimpleNamespace(source_audio_bytes=b'fixture'))
    result = ProductionSemanticExplanationService(artifacts, feature_extractor=extractor).analyze(
        inference, analyzed_spans=[(0, 6)]).explanation
    assert result.predicted_value is not None
    assert result.localization.status == 'partial'
    assert result.localization.windows[2].reason == 'window_extraction_or_inference_failed'
    assert [(item.start_seconds, item.end_seconds) for item in result.localization.valid_spans] == [(0, 2), (4, 6)]


def test_precomputed_features_do_not_fabricate_windows_without_original_audio(artifacts):
    features = extraction.FeatureExtractionResult(FEATURE_NAMES, artifacts.imputer.statistics_.copy(),
                                                  16000, 6, FEATURE_EXTRACTION_VERSION, {})
    result = ProductionSemanticExplanationService(artifacts).analyze(
        SimpleNamespace(acoustic_features=features), analyzed_spans=[(0, 6)]).explanation
    assert result.predicted_value is not None
    assert result.localization.status == 'not_available'
    assert result.localization.windows == []


def test_exact_full_cleaned_window_reuses_measured_features(monkeypatch, artifacts):
    extractor = extraction.AcousticFeatureExtractor()
    monkeypatch.setattr(extractor, 'prepare_processed_audio', lambda _: prepared(start=1, seconds=1))
    measured = []
    def mapping(y, sr):
        measured.append(y)
        return dict(zip(FEATURE_NAMES, artifacts.imputer.statistics_))
    monkeypatch.setattr(extraction, 'extract_feature_mapping', mapping)
    inference = SimpleNamespace(acoustic_features=None, processed_audio=SimpleNamespace(source_audio_bytes=b'fixture'))
    result = ProductionSemanticExplanationService(artifacts, feature_extractor=extractor).analyze(
        inference, analyzed_spans=[(0, 3)]).explanation
    assert len(measured) == 1
    assert result.localization.windows[0].start_seconds == 1
    assert result.localization.windows[0].end_seconds == 2
    assert result.localization.windows[0].feature_importance == result.feature_importance


@pytest.mark.parametrize('contract', [None, [], {'version': 'incorrect'}])
def test_malformed_window_validation_does_not_claim_validity(tmp_path, contract):
    write_model_export(tmp_path)
    path = tmp_path / 'training_metadata.json'
    metadata = json.loads(path.read_text())
    metadata['window_validation'] = {'contract': contract}
    path.write_text(json.dumps(metadata))
    with pytest.raises(SemanticArtifactCompatibilityError):
        load_production_semantic_artifacts(tmp_path)


def test_window_contract_metadata_is_distinct_from_clip_fidelity(tmp_path):
    write_model_export(tmp_path)
    path = tmp_path / 'training_metadata.json'
    metadata = json.loads(path.read_text())
    metadata['window_validation'] = {'contract': SemanticWindowOptions().model_contract(),
                                     'split_unit': 'recording', 'target': 'teacher_window_spoof_probability',
                                     'n_recordings': 10, 'n_windows': 50, 'r2': .8, 'mae': .1, 'rmse': .15}
    path.write_text(json.dumps(metadata))
    artifacts = load_production_semantic_artifacts(tmp_path)
    assert artifacts.window_validation(SemanticWindowOptions())[0] is True
    assert artifacts.window_validation(SemanticWindowOptions(window_seconds=3))[0] is False
    metadata['window_validation']['split_unit'] = 'random_windows'
    path.write_text(json.dumps(metadata))
    with pytest.raises(SemanticArtifactCompatibilityError, match='Window validation'):
        load_production_semantic_artifacts(tmp_path)


@pytest.mark.parametrize('field,value', [('xai_semantic_window_seconds', 3), ('xai_semantic_hop_seconds', .5),
                                         ('xai_semantic_windows_enabled', False), ('xai_semantic_activity_percentile', 80)])
def test_configuration_hash_changes_with_window_behavior(field, value):
    assert _configuration_hash(Settings(_env_file=None)) != _configuration_hash(Settings(_env_file=None, **{field: value}))


def test_settings_reject_a_hop_larger_than_the_window():
    with pytest.raises(ValueError, match='SEMANTIC_HOP_SECONDS'):
        Settings(_env_file=None, xai_semantic_window_seconds=1, xai_semantic_hop_seconds=2)


def test_interval_operations_never_double_count_overlaps():
    assert union_spans([(0, 2), (1, 3), (5, 6)]) == ((0.0, 3.0), (5.0, 6.0))
    assert intersect_spans([(0, 3)], [(2, 4)]) == ((2.0, 3.0),)
    assert subtract_spans([(0, 6)], [(1, 2), (4, 5)]) == ((0.0, 1.0), (2.0, 4.0), (5.0, 6.0))
    assert span_duration([(0, 2), (1, 3)]) == 3
