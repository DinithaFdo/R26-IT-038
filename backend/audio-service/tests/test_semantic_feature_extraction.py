"""Production unified extraction, validation and missing-value behavior."""

import numpy as np
import pytest
import soundfile as sf

from app.voice_xai.semantic import feature_extraction as extraction
from app.voice_xai.semantic.artifact_loader import load_production_semantic_artifacts
from app.voice_xai.semantic.schema import FEATURE_NAMES
from app.voice_xai.semantic.service import XgboostShapEvidenceProvider
from tests.semantic_support import write_model_export


@pytest.fixture
def voice_waveform():
    time = np.arange(12800) / 16000
    return (0.3 * np.sin(2 * np.pi * 180 * time)).astype(np.float32)


@pytest.fixture
def glottal_measurements(monkeypatch):
    values = dict.fromkeys(FEATURE_NAMES, 123.0)
    monkeypatch.setattr(
        extraction, 'extract_complete_glottal_features_exact', lambda *args: values.copy()
    )
    return values


def test_unified_features_preserve_shared_acoustic_values(voice_waveform, glottal_measurements):
    waveform, rate = extraction.prepare_audio(voice_waveform, 16000)
    result = extraction.extract_feature_mapping(waveform, rate)
    assert tuple(result) == FEATURE_NAMES
    assert len(result) == 165
    for name in ('spectral_centroid_mean', 'spectral_centroid_std', 'f0_std'):
        assert np.isfinite(result[name])
        assert result[name] != glottal_measurements[name]
    assert result['jitter_local'] == glottal_measurements['jitter_local']


def test_glottal_writes_only_its_model_inputs(voice_waveform, glottal_measurements):
    output = dict(zip(FEATURE_NAMES, np.arange(len(FEATURE_NAMES), dtype=float)))
    # An unavailable acoustic value must remain missing for the fitted imputer.
    output['f0_std'] = np.nan
    expected = output.copy()
    for name in FEATURE_NAMES:
        if name.startswith('glottal_') or name in {'shimmer_local', 'hnr_std', 'f0_min', 'jitter_local'}:
            expected[name] = 123.0
    # Measurements unused by this helper must not add validation requirements.
    glottal_measurements['f0_mean'] = np.nan
    glottal_measurements['spectral_tilt'] = np.nan
    extraction._write_glottal(output, voice_waveform, 16000)
    assert tuple(output) == FEATURE_NAMES
    np.testing.assert_array_equal(list(output.values()), list(expected.values()))


@pytest.mark.parametrize('name', [
    'spectral_centroid_mean', 'spectral_centroid_std', 'f0_std',
    'jitter_local', 'glottal_global_avg_avg_h1h2', 'glottal_global_kurtosis_std_naq',
])
@pytest.mark.parametrize('problem', ['missing', 'nan', 'inf'])
def test_required_glottal_measurements_keep_exact_errors(
    voice_waveform, glottal_measurements, name, problem,
):
    if problem == 'missing':
        del glottal_measurements[name]
        error_type = KeyError
        message = f'Missing selected Glottal feature: {name}'
    else:
        glottal_measurements[name] = float(problem)
        error_type = ValueError
        message = f'Non-finite selected Glottal feature: {name}={problem}'
    with pytest.raises(error_type) as error:
        extraction._write_glottal({}, voice_waveform, 16000)
    assert error.value.args == (message,)


@pytest.mark.parametrize('first_problem', ['missing', 'nan'])
def test_glottal_validation_keeps_first_error_order(
    voice_waveform, glottal_measurements, first_problem,
):
    del glottal_measurements['shimmer_local']
    if first_problem == 'missing':
        del glottal_measurements['spectral_centroid_mean']
        error_type = KeyError
        message = 'Missing selected Glottal feature: spectral_centroid_mean'
    else:
        glottal_measurements['spectral_centroid_mean'] = np.nan
        error_type = ValueError
        message = 'Non-finite selected Glottal feature: spectral_centroid_mean=nan'
    with pytest.raises(error_type) as error:
        extraction._write_glottal({}, voice_waveform, 16000)
    assert error.value.args == (message,)


@pytest.mark.parametrize('problem', ['missing', 'nan', 'inf', 'disvoice_failure'])
def test_glottal_validation_including_shared_values(
    monkeypatch, voice_waveform, glottal_measurements, problem,
):
    if problem == 'missing':
        del glottal_measurements['f0_std']
    elif problem == 'disvoice_failure':
        def failure(*args):
            raise RuntimeError('DisVoice extraction failed')
        monkeypatch.setattr(extraction, 'extract_complete_glottal_features_exact', failure)
    else:
        glottal_measurements['f0_std'] = float(problem)
    waveform, rate = extraction.prepare_audio(voice_waveform, 16000)
    expected_type = {'missing': KeyError, 'disvoice_failure': RuntimeError}.get(problem, ValueError)
    with pytest.raises(expected_type) as error:
        extraction.extract_feature_mapping(waveform, rate)
    assert ('DisVoice extraction failed' if problem == 'disvoice_failure' else 'f0_std') in str(error.value)


@pytest.mark.parametrize('waveform,rate', [
    (np.ones(160), 8000), (np.ones((16, 2)), 16000),
    (np.array([np.nan]), 16000), (np.array([np.inf]), 16000),
])
def test_invalid_prepared_waveform_rejections(waveform, rate):
    with pytest.raises(ValueError, match='finite, prepared mono waveform'):
        extraction.extract_feature_mapping(waveform, rate)


def test_missing_pitch_uses_fitted_imputation_and_additive_shap(
    monkeypatch, tmp_path, voice_waveform, glottal_measurements,
):
    import librosa

    def missing_pitch(*args, **kwargs):
        raise ValueError('Pitch unavailable')

    monkeypatch.setattr(librosa, 'pyin', missing_pitch)
    path = tmp_path / 'voice.wav'
    sf.write(path, voice_waveform, 16000)
    features = extraction.AcousticFeatureExtractor().extract_file(path)
    pitch_index = FEATURE_NAMES.index('f0_std')
    assert np.isnan(features.values[pitch_index])
    model_dir = tmp_path / 'model'
    write_model_export(model_dir)
    provider = XgboostShapEvidenceProvider(load_production_semantic_artifacts(model_dir))
    result = provider.extract_features(features)
    assert result.imputed_feature_count >= 5
    assert np.isfinite(result.feature_values).all()
    assert result.feature_values[pitch_index] == provider.artifacts.imputer.statistics_[pitch_index]
    assert result.base_value + result.shap_values.sum() == pytest.approx(result.predicted_value, abs=1e-4)


@pytest.mark.parametrize('case,message', [
    ('silence', 'Audio is silent after preprocessing'),
    ('constant', 'Audio is silent after preprocessing'),
    ('too_short', 'audio too short after preprocessing'),
    ('empty', 'Expected non-empty audio'),
    ('corrupt', 'Format not recognised'),
])
def test_invalid_source_audio_rejected_through_public_api(tmp_path, voice_waveform, case, message):
    path = tmp_path / f'{case}.wav'
    if case == 'corrupt':
        path.write_bytes(b'not-an-audio-file')
    else:
        signal = {
            'silence': np.zeros(16000),
            'constant': np.ones(16000),
            'too_short': voice_waveform[:800],
            'empty': np.array([], dtype=np.float32),
        }[case]
        sf.write(path, signal, 16000)
    error_type = sf.LibsndfileError if case == 'corrupt' else ValueError
    with pytest.raises(error_type, match=message):
        extraction.AcousticFeatureExtractor().extract_file(path)
