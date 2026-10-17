"""Semantic-only CPU acceleration for the pinned DisVoice 0.1.10 recipe.

Clone the required globals/closures without rebinding the classifier's modules.
Preserve peak comparisons, Pearson arithmetic, dynamic-programming ties and
reference fallbacks. Imported lazily by feature_extraction on first Glottal use.
"""
# Peak-state-machine adaptation: Copyright (C) 2016 Sixten Bergman, WTFPL v2.
# Its peakdetect algorithm derives from Eli Billauer's public-domain peakdet.

from functools import lru_cache
import importlib.metadata
import operator
from types import FunctionType
import warnings

import numpy as np
from numba import njit
from numba.core.errors import NumbaError
from scipy.stats import pearsonr

from . import feature_extraction as reference


# Permit read-only package installations to compile in memory without a disk cache.
def _compile_peaks(function):
    try:
        return njit(cache=True, fastmath=False)(function)
    except RuntimeError as error:
        if 'no locator available' not in str(error):
            raise
        return njit(cache=False, fastmath=False)(function)


# Compile the unchanged peak comparisons without fastmath or parallel execution.
@_compile_peaks
def _peak_indices(values, lookahead, delta):
    length = len(values)
    maxima = np.empty(length, dtype=np.int64)
    minima = np.empty(length, dtype=np.int64)
    max_count = min_count = 0
    first_kind = 0
    minimum, maximum = np.inf, -np.inf
    min_position = max_position = 0
    for index in range(max(0, length - lookahead)):
        value = values[index]
        if value > maximum:
            maximum, max_position = value, index
        if value < minimum:
            minimum, min_position = value, index
        if value < maximum - delta and maximum != np.inf:
            ahead_maximum = -np.inf
            has_nan = False
            for position in range(index, index + lookahead):
                ahead = values[position]
                if np.isnan(ahead):
                    has_nan = True
                if ahead > ahead_maximum:
                    ahead_maximum = ahead
            if not has_nan and ahead_maximum < maximum:
                maxima[max_count] = max_position
                max_count += 1
                if first_kind == 0:
                    first_kind = 1
                maximum, minimum = np.inf, np.inf
                if index + lookahead >= length:
                    break
                continue
        if value > minimum + delta and minimum != -np.inf:
            ahead_minimum = np.inf
            has_nan = False
            for position in range(index, index + lookahead):
                ahead = values[position]
                if np.isnan(ahead):
                    has_nan = True
                if ahead < ahead_minimum:
                    ahead_minimum = ahead
            if not has_nan and ahead_minimum > minimum:
                minima[min_count] = min_position
                min_count += 1
                if first_kind == 0:
                    first_kind = 2
                minimum, maximum = -np.inf, -np.inf
                if index + lookahead >= length:
                    break
    return (maxima[1 if first_kind == 1 else 0:max_count],
            minima[1 if first_kind == 2 else 0:min_count])


# Load the installed reference for validation and unsupported input fallbacks.
@lru_cache(maxsize=1)
def reference_peakdetect():
    import sys
    reference.load_glottal_class()
    return sys.modules['peakdetect'].peakdetect


# Preserve validation, amplitudes and source coordinates around compiled float64 peaks.
def peakdetect(y_axis, x_axis=None, lookahead=200, delta=0):
    original = reference_peakdetect()
    x, y = original.__globals__['_datacheck_peakdetect'](x_axis, y_axis)
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError('delta must be a positive number')
    lookahead = operator.index(lookahead)
    if (y.ndim != 1 or y.dtype != np.dtype('float64')
            or not isinstance(delta, (int, float, np.integer, np.floating))
            or isinstance(delta, np.longdouble)):
        return original(y_axis, x_axis=x_axis, lookahead=lookahead, delta=delta)
    try:
        maxima, minima = _peak_indices(np.ascontiguousarray(y), lookahead, float(delta))
    except (OSError, NumbaError):
        # Cache/JIT infrastructure cannot turn valid reference input into a failure.
        return original(y_axis, x_axis=x_axis, lookahead=lookahead, delta=delta)
    return [[[x[position], y[position]] for position in positions]
            for positions in (maxima, minima)]


# Construct a private closure cell without changing the original captures.
def _cell(value):
    return (lambda: value).__closure__[0]


# Clone one function with isolated dependency bindings and explicit closure validation.
def _clone(function, *, globals_override=None, closure_override=None):
    namespace = dict(function.__globals__)
    namespace.update(globals_override or {})
    overrides = closure_override or {}
    names = function.__code__.co_freevars
    if set(overrides) - set(names):
        raise RuntimeError('Installed DisVoice compatibility closure does not match the semantic DSP contract.')
    closure = (tuple(_cell(overrides[name]) if name in overrides else cell
                     for name, cell in zip(names, function.__closure__, strict=True))
               if function.__closure__ is not None else None)
    result = FunctionType(function.__code__, namespace, function.__name__,
                          function.__defaults__, closure)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


# Bind compiled peaks into private GCI and harmonic measurement functions.
@lru_cache(maxsize=1)
def private_glottal_extractor():
    if importlib.metadata.version('disvoice') != '0.1.10':
        raise RuntimeError('Semantic DSP requires the pinned DisVoice 0.1.10.')
    base = reference.load_glottal_class()
    namespace = base.extract_features_file.__globals__
    get_vq = namespace['get_vq_params']
    harmonic = get_vq.__globals__['compute_h1h2_hrf_frame']
    if 'original_peakdetect' not in harmonic.__code__.co_freevars:
        raise RuntimeError('Expected the production NumPy-2 harmonic compatibility closure.')
    fast_harmonic = _clone(harmonic, closure_override={'original_peakdetect': peakdetect})
    fast_vq = _clone(get_vq, globals_override={'compute_h1h2_hrf_frame': fast_harmonic})
    se_vq = base.extract_glottal_signal.__globals__['se_vq_varf0']
    intervals = se_vq.__globals__['get_MBS_GCI_intervals']
    fast_intervals = _clone(intervals, globals_override={'peakdetect': peakdetect})
    fast_se_vq = _clone(se_vq, globals_override={'get_MBS_GCI_intervals': fast_intervals})
    private_class = type('SemanticGlottal', (base,), {
        'extract_glottal_signal': _clone(base.extract_glottal_signal,
                                       globals_override={'se_vq_varf0': fast_se_vq}),
        'extract_features_file': _clone(base.extract_features_file,
                                      globals_override={'get_vq_params': fast_vq}),
    })
    return private_class()


# Read the installed transition-cost reference after compatibility setup.
@lru_cache(maxsize=1)
def reference_cost_matrix():
    base = reference.load_glottal_class()
    se_vq = base.extract_glottal_signal.__globals__['se_vq_varf0']
    return se_vq.__globals__['RESON_dyProg_mat'].__globals__['get_costm_matrix']


# Batch matching Pearson coefficients; replay unusual cases through the reference.
def batched_cost_matrix(GCI_N, x, trans_wgt, ncands, pulseLen, n):
    original = reference_cost_matrix()
    if (not isinstance(x, np.ndarray) or x.ndim != 1 or x.dtype != np.float64
            or not isinstance(GCI_N, np.ndarray) or GCI_N.ndim != 2
            or not isinstance(ncands, (int, np.integer)) or ncands < 1
            or not isinstance(n, (int, np.integer)) or not 1 <= n < len(GCI_N)
            or GCI_N.shape[1] != ncands
            or not isinstance(trans_wgt, (int, float, np.integer, np.floating))
            or isinstance(trans_wgt, np.longdouble) or not np.isfinite(trans_wgt)
            or not np.all(np.isfinite(GCI_N[n-1:n+1]))):
        return original(GCI_N, x, trans_wgt, ncands, pulseLen, n)
    current = [x[int(GCI_N[n, c] - pulseLen / 2):
                 min(int(GCI_N[n, c] + pulseLen / 2), len(x))]
               for c in range(ncands)]
    previous = [x[max(1, int(GCI_N[n-1, p] - pulseLen / 2)):
                  min(len(x), int(GCI_N[n-1, p] + pulseLen / 2))]
                for p in range(ncands)]
    lengths = {len(pulse) for pulse in current + previous}
    if len(lengths) != 1 or min(lengths) < 2:
        return original(GCI_N, x, trans_wgt, ncands, pulseLen, n)
    cur, prev = np.stack(current), np.stack(previous)
    if not (np.all(np.isfinite(cur)) and np.all(np.isfinite(prev))):
        return original(GCI_N, x, trans_wgt, ncands, pulseLen, n)
    # Replay warnings through the original loop, preserving category/count/order.
    try:
        with warnings.catch_warnings(record=True) as observed:
            correlations = pearsonr(cur[:, None, :], prev[None, :, :], axis=-1).statistic
        if observed:
            return original(GCI_N, x, trans_wgt, ncands, pulseLen, n)
    except Warning:
        return original(GCI_N, x, trans_wgt, ncands, pulseLen, n)
    return (1 - np.abs(correlations.T)) * trans_wgt


# Bind batched transition costs into the already private peak-detection graph.
@lru_cache(maxsize=1)
def private_batched_glottal_extractor():
    base = private_glottal_extractor()
    signal = base.extract_glottal_signal.__func__
    se_vq = signal.__globals__['se_vq_varf0']
    dynamic = _clone(se_vq.__globals__['RESON_dyProg_mat'],
                     globals_override={'get_costm_matrix': batched_cost_matrix})
    fast_se = _clone(se_vq, globals_override={'RESON_dyProg_mat': dynamic})
    private_class = type('SemanticBatchedGlottal', (type(base),), {
        'extract_glottal_signal': _clone(signal, globals_override={'se_vq_varf0': fast_se}),
    })
    return private_class()


