"""Parity at peak boundaries and isolated DisVoice dependency binding."""

import sys

import numpy as np
import pytest

from app.voice_xai.semantic.glottal_dsp import (
    _clone, peakdetect, private_glottal_extractor, reference_peakdetect,
)
from app.voice_xai.semantic import feature_extraction as reference


# Compare positions and amplitudes, including first-hit removal and missing data.
@pytest.mark.parametrize('lookahead', [1, 2, 16, 200, 900])
@pytest.mark.parametrize('delta', [0, 0.1, 2.0])
def test_compiled_peaks_match_reference_on_adversarial_signals(lookahead, delta):
    rng = np.random.default_rng(45)
    sequences = [
        np.array([], dtype=float), np.ones(1), np.zeros(30),
        np.arange(60, dtype=float), -np.arange(60, dtype=float),
        np.tile(np.array([0., 2., 2., 0., -2., -2., 0.]), 50),
        rng.normal(size=800), np.sin(np.arange(800) / 8),
        np.array([0., 1., np.nan, -1., 0., np.inf, 0., -np.inf] * 50),
    ]
    for sequence in sequences:
        expected = reference_peakdetect()(sequence, lookahead=lookahead, delta=delta)
        actual = peakdetect(sequence, lookahead=lookahead, delta=delta)
        assert actual == expected


@pytest.mark.parametrize('dtype', [np.float32, np.int64, np.float64])
def test_original_x_coordinates_and_numeric_dtypes_are_preserved(dtype):
    y = np.tile(np.array([0, 3, 0, -2, 0], dtype=dtype), 30)
    x = np.arange(len(y)) * 0.25 + 11
    expected = reference_peakdetect()(y, x_axis=x, lookahead=2, delta=0.1)
    assert peakdetect(y, x_axis=x, lookahead=2, delta=0.1) == expected


@pytest.mark.parametrize('kwargs', [
    {'lookahead': 0}, {'lookahead': 1.5}, {'delta': -1},
    {'delta': np.nan}, {'x_axis': [1]},
])
def test_validation_errors_match_reference(kwargs):
    y = np.array([0., 1., 0., -1.] * 20)
    with pytest.raises(Exception) as expected:
        reference_peakdetect()(y, **kwargs)
    with pytest.raises(type(expected.value)):
        peakdetect(y, **kwargs)


def test_private_extractor_never_rebinds_original_module_functions():
    base = reference.load_glottal_class()
    file_globals = base.extract_features_file.__globals__
    original_vq = file_globals['get_vq_params']
    harmonic = original_vq.__globals__['compute_h1h2_hrf_frame']
    closure = dict(zip(harmonic.__code__.co_freevars, harmonic.__closure__, strict=True))
    original_peak = closure['original_peakdetect'].cell_contents
    original_module_peak = sys.modules['peakdetect'].peakdetect
    candidate = private_glottal_extractor()
    assert isinstance(candidate, base)
    assert base.extract_features_file.__globals__['get_vq_params'] is original_vq
    assert original_vq.__globals__['compute_h1h2_hrf_frame'] is harmonic
    assert closure['original_peakdetect'].cell_contents is original_peak
    assert sys.modules['peakdetect'].peakdetect is original_module_peak
    private_vq = candidate.extract_features_file.__func__.__globals__['get_vq_params']
    private_harmonic = private_vq.__globals__['compute_h1h2_hrf_frame']
    captures = dict(zip(private_harmonic.__code__.co_freevars, private_harmonic.__closure__, strict=True))
    assert captures['original_peakdetect'].cell_contents is peakdetect
    private_se = candidate.extract_glottal_signal.__func__.__globals__['se_vq_varf0']
    intervals = private_se.__globals__['get_MBS_GCI_intervals']
    assert intervals.__globals__['peakdetect'] is peakdetect


def test_closure_mismatch_fails_explicitly_instead_of_silently_using_reference():
    with pytest.raises(RuntimeError, match='closure'):
        _clone(reference.praat_voice_features, closure_override={'missing': 1})


def test_selected_glottal_validation_is_retained(monkeypatch):
    measured = dict.fromkeys(reference._GLOTTAL_WRITE_POLICY, 1.0)
    measured['hnr_std'] = np.nan
    monkeypatch.setattr(reference, 'extract_glottal_voice_quality', lambda *args: measured)
    monkeypatch.setattr(reference, 'extract_glottal_spectral_features', lambda *args: {})
    # Bind a private exact-disvoice function that returns finite test measurements.
    finite_disvoice = lambda *args: np.ones(36)
    complete = _clone(reference.extract_complete_glottal_features_exact,
                      globals_override={'extract_disvoice_glottal_exact': finite_disvoice})
    writer = _clone(reference._write_glottal,
                    globals_override={'extract_complete_glottal_features_exact': complete})
    with pytest.raises(ValueError, match='Non-finite selected Glottal feature: hnr_std'):
        writer({}, np.zeros(16000), 16000)
