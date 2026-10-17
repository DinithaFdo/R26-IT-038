"""Exact transition matrices, unchanged GCI paths and warning/error fallbacks."""

import warnings

import numpy as np
import pytest

from app.voice_xai.semantic import glottal_dsp as candidate
from app.voice_xai.semantic.glottal_dsp import _clone


# Keep candidates inside the waveform unless a test explicitly checks boundaries.
def inputs():
    waveform = np.random.default_rng(813).normal(size=2000)
    candidates = np.array([[150 + n * 110 + c * 3 for c in range(5)] for n in range(12)], float)
    return candidates, waveform


@pytest.mark.parametrize('length', [2, 3, 80, 101, 200])
@pytest.mark.parametrize('scale', [1., 1e140, 1e-140])
def test_batched_correlations_match_every_reference_matrix_exactly(length, scale):
    candidates, waveform = inputs()
    waveform *= scale
    for frame in range(1, len(candidates)):
        expected = candidate.reference_cost_matrix()(candidates, waveform, 0.7, 5, length, frame)
        actual = candidate.batched_cost_matrix(candidates, waveform, 0.7, 5, length, frame)
        np.testing.assert_array_equal(actual, expected)


def test_regular_matrix_uses_one_scipy_call_instead_of_25(monkeypatch):
    candidates, waveform = inputs()
    calls = []
    original = candidate.pearsonr

    def observed(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(candidate, 'pearsonr', observed)
    candidate.batched_cost_matrix(candidates, waveform, 1., 5, 100, 2)
    assert calls == [{'axis': -1}]


@pytest.mark.parametrize('case', [
    'negative_start', 'right_boundary', 'empty_previous', 'nan_first',
    'nan_inside', 'infinite', 'constant', 'nearly_constant', 'float32',
    'one_sample', 'no_candidates',
])
def test_unusual_pulses_preserve_values_warnings_and_failures(case):
    candidates, waveform = inputs()
    count, length, frame = 5, 100, 2
    if case == 'negative_start':
        candidates[frame, 0] = 2
    elif case == 'right_boundary':
        candidates[frame, 0] = len(waveform) - 2
    elif case == 'empty_previous':
        candidates[frame-1, 0] = len(waveform) + 200
    elif case == 'nan_first':
        waveform[int(candidates[frame, 0] - length / 2)] = np.nan
    elif case == 'nan_inside':
        waveform[320] = np.nan
    elif case == 'infinite':
        waveform[320] = np.inf
    elif case == 'constant':
        waveform[:] = 1.
    elif case == 'nearly_constant':
        waveform = 1 + waveform * 1e-15
    elif case == 'float32':
        waveform = waveform.astype(np.float32)
    elif case == 'one_sample':
        length = 1
    else:
        candidates, count = candidates[:, :0], 0

    def observe(function):
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            try:
                result, error = function(candidates, waveform, 1., count, length, frame), None
            except Exception as exception:
                result, error = None, (type(exception), str(exception))
        return result, error, [(item.category, str(item.message)) for item in recorded]

    expected, expected_error, expected_warnings = observe(candidate.reference_cost_matrix())
    actual, error, actual_warnings = observe(candidate.batched_cost_matrix)
    assert error == expected_error
    assert actual_warnings == expected_warnings
    if expected_error is None:
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('ties', [False, True])
def test_entire_dynamic_program_selects_identical_gci_including_ties(ties):
    candidates, waveform = inputs()
    reference = candidate.reference_cost_matrix().__globals__['RESON_dyProg_mat']
    optimized = _clone(reference, globals_override={'get_costm_matrix': candidate.batched_cost_matrix})
    if ties:
        candidates[:] = candidates[:, :1]
    amplitudes = np.random.default_rng(7).uniform(size=candidates.shape)
    arguments = (amplitudes, candidates, 160., waveform, 16000, 1., 0.3)
    np.testing.assert_array_equal(optimized(*arguments, plots=False), reference(*arguments, plots=False))


def test_private_dynamic_program_does_not_rebind_original_or_peak_only_graph():
    original = candidate.reference_cost_matrix()
    peak_only = candidate.private_glottal_extractor()
    peak_se = peak_only.extract_glottal_signal.__func__.__globals__['se_vq_varf0']
    peak_dynamic = peak_se.__globals__['RESON_dyProg_mat']
    private = candidate.private_batched_glottal_extractor()
    private_se = private.extract_glottal_signal.__func__.__globals__['se_vq_varf0']
    private_dynamic = private_se.__globals__['RESON_dyProg_mat']
    assert private_dynamic is not peak_dynamic
    assert private_dynamic.__globals__['get_costm_matrix'] is candidate.batched_cost_matrix
    assert peak_dynamic.__globals__['get_costm_matrix'] is original
    assert original.__globals__['get_costm_matrix'] is original
