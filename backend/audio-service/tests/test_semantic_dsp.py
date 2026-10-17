"""Production CPU acceleration, lazy dependencies and infrastructure fallbacks."""

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
from numba.core.errors import NumbaError

from app.voice_xai.semantic import feature_extraction as extraction
from app.voice_xai.semantic import glottal_dsp as dsp
from app.voice_xai.semantic.schema import FEATURE_NAMES


# Exercise each public entry point, not just the benchmark's former subclass.
@pytest.mark.parametrize('entry', ['file', 'prepared', 'window'])
def test_public_production_paths_share_one_matching_spectrum(monkeypatch, tmp_path, entry):
    import librosa
    samples = np.arange(8000) / 16000
    waveform = np.sin(2 * np.pi * 180 * samples).astype(np.float32)
    prepared = extraction.PreparedSemanticAudio(waveform, 16000, 32000, 16000, 24000)
    measured = dict.fromkeys(extraction._GLOTTAL_WRITE_POLICY, 1.)
    monkeypatch.setattr(extraction, 'extract_complete_glottal_features_exact', lambda *args: measured)
    monkeypatch.setattr(extraction, 'praat_voice_features', lambda *args: {})
    monkeypatch.setattr(extraction, '_write_pitch', lambda *args: None)
    monkeypatch.setattr(extraction, '_write_cpp_and_pauses', lambda *args: None)
    original, matching = librosa.stft, []

    def observed(*args, **kwargs):
        if kwargs.get('n_fft') == 512 and kwargs.get('win_length') == 400:
            matching.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(librosa, 'stft', observed)
    extractor = extraction.AcousticFeatureExtractor()
    if entry == 'file':
        path = tmp_path / 'voice.wav'
        sf.write(path, waveform, 16000)
        result = extractor.extract_file(path)
    elif entry == 'prepared':
        result = extractor.extract_prepared_audio(prepared)
    else:
        result = extractor.extract_prepared_window(prepared, extraction.SemanticWindow(18000, 22000))
        assert result.metadata['start_sample'] == 18000
        assert result.metadata['end_sample_exclusive'] == 22000
    assert len(matching) == 1
    assert result.feature_names == FEATURE_NAMES and result.values.shape == (165,)
    assert np.isfinite(result.values[FEATURE_NAMES.index('lfcc_1')])
    assert result.metadata['analysis_scope'] == ('window' if entry == 'window' else 'whole_clip')


# A public production extraction must choose the promoted graph and cache its instance.
def test_production_factory_uses_batched_glottal_graph(monkeypatch):
    calls, instance = [], object()

    def factory():
        calls.append(True)
        return instance

    monkeypatch.setattr(dsp, 'private_batched_glottal_extractor', factory)
    extraction._get_glottal_extractor.cache_clear()
    try:
        assert extraction._get_glottal_extractor() is instance
        assert extraction._get_glottal_extractor() is instance
        assert calls == [True]
    finally:
        extraction._get_glottal_extractor.cache_clear()


# Disk-cache and compiler infrastructure failures preserve valid reference output.
@pytest.mark.parametrize('error', [OSError('cache is not writable'), NumbaError('compiler unavailable')])
def test_peak_jit_failure_uses_reference_values_and_coordinates(monkeypatch, error):
    waveform = np.sin(np.arange(400, dtype=np.float64) / 5)
    coordinates = np.arange(400) * .25 + 7
    expected = dsp.reference_peakdetect()(waveform, x_axis=coordinates, lookahead=3, delta=.1)

    def unavailable(*args):
        raise error

    monkeypatch.setattr(dsp, '_peak_indices', unavailable)
    assert dsp.peakdetect(waveform, x_axis=coordinates, lookahead=3, delta=.1) == expected


# Read-only package installs still compile the same function without persistent cache.
def test_no_cache_locator_compiles_in_memory(monkeypatch):
    calls = []

    def njit(**options):
        calls.append(options)

        def decorate(function):
            if options['cache']:
                raise RuntimeError('cannot cache function: no locator available for file')
            return function

        return decorate

    function = lambda values: values
    monkeypatch.setattr(dsp, 'njit', njit)
    assert dsp._compile_peaks(function) is function
    assert calls == [{'cache': True, 'fastmath': False}, {'cache': False, 'fastmath': False}]


# Unexpected decorator failures must stay visible instead of being misreported as cache issues.
def test_unexpected_compiler_setup_error_propagates(monkeypatch):
    def njit(**options):
        raise RuntimeError('unexpected setup failure')

    monkeypatch.setattr(dsp, 'njit', njit)
    with pytest.raises(RuntimeError, match='unexpected setup failure'):
        dsp._compile_peaks(lambda values: values)


# Importing the service must not require XAI DSP or benchmark packages at startup.
def test_service_import_keeps_dsp_dependencies_lazy():
    code = (
        'import sys; import app.voice_xai.semantic.service; '
        "assert not any(name in sys.modules for name in ['librosa', 'disvoice', 'numba', 'parselmouth']); "
        "assert not any(name.startswith('app.voice_xai.benchmarks') for name in sys.modules)"
    )
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parents[1],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
