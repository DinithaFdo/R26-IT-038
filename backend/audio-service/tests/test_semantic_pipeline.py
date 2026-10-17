"""Full-clip preprocessing, shared feature inputs, and real inference tests."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from app.config.settings import Settings
from app.ingestion import audio as ingestion
from app.voice_xai.semantic import feature_extraction as extraction
from app.voice_xai.semantic.feature_extraction import AcousticFeatureExtractor, FeatureExtractionResult
from app.voice_xai.semantic.schema import FEATURE_NAMES, FEATURE_EXTRACTION_VERSION
from app.voice_xai.semantic.service import ProductionSemanticExplanationService
from tests.semantic_support import write_model_export


def feature_row(values):
    return FeatureExtractionResult(FEATURE_NAMES, np.asarray(values, dtype=np.float64),
                                   16000, 12.0, FEATURE_EXTRACTION_VERSION, {})


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("fullclip-model")
    inputs, model, imputer = write_model_export(root / "source")
    service = ProductionSemanticExplanationService.load_once(model_dir=root / "source")
    return service, inputs, model, imputer


def test_prediction_shap_and_imputation_use_fitted_model(trained):
    service, inputs, model, imputer = trained
    raw = inputs.iloc[0].to_numpy()
    result = service.analyze(SimpleNamespace(acoustic_features=feature_row(raw))).explanation
    expected = imputer.transform(inputs.iloc[:1])
    assert result.predicted_value == float(model.predict(expected)[0])
    assert result.clip_summary.spoof_probability == float(np.clip(result.predicted_value, 0, 1))
    assert len(result.feature_importance) == 165
    actual = {item.feature_name: item.value for item in result.feature_importance}
    np.testing.assert_array_equal([actual[name] for name in FEATURE_NAMES], expected[0])
    assert result.base_value + sum(item.shap_value for item in result.feature_importance) == pytest.approx(result.predicted_value, abs=1e-4)
    assert result.extractor_version == FEATURE_EXTRACTION_VERSION
    assert "windows" not in result.model_dump()


def test_old_precomputed_vectors_cannot_be_used(trained):
    service, *_ = trained
    features = feature_row(np.ones(165))
    object.__setattr__(features, "extractor_version", "selected20-aligned-audio-v1")
    with pytest.raises(ValueError, match="semantic model contract"):
        service.analyze(SimpleNamespace(acoustic_features=features))


def test_preprocessing_preserves_long_audio_and_trims_only_boundaries():
    sr = 16000
    active = 0.2 * np.sin(2 * np.pi * 180 * np.arange(12 * sr) / sr)
    source = np.concatenate([np.zeros(sr), active, np.zeros(sr)])
    cleaned, actual_sr = extraction.prepare_audio(np.column_stack([source, source]), sr)
    assert actual_sr == sr
    assert 12 * sr <= len(cleaned) < len(source)
    assert cleaned.dtype == np.float32
    assert np.max(np.abs(cleaned)) == pytest.approx(1)
    assert abs(np.mean(cleaned)) < 1e-5


def test_resampling_and_numeric_cleanup_keep_complete_clip():
    sr = 48000
    signal = np.sin(2 * np.pi * 180 * np.arange(8 * sr) / sr).astype(np.float32)
    signal[0] = np.nan
    cleaned, rate = extraction.prepare_audio(signal, sr)
    assert rate == 16000
    assert len(cleaned) == 8 * rate
    assert np.isfinite(cleaned).all()


@pytest.mark.parametrize("signal", [np.zeros(16000), np.ones(16000), np.array([]), np.ones(100)])
def test_empty_silent_and_too_short_inputs_are_rejected(signal):
    with pytest.raises(ValueError):
        extraction.prepare_audio(signal, 16000)


def test_every_feature_group_receives_same_waveform_once(monkeypatch, tmp_path):
    path = tmp_path / "long.wav"
    sr = 16000
    sf.write(path, np.sin(2 * np.pi * 180 * np.arange(12 * sr) / sr), sr)
    decoded = []
    prepared = []
    seen = []
    original_read, original_prepare = sf.read, extraction.prepare_audio
    def read(*args, **kwargs):
        decoded.append(args[0])
        return original_read(*args, **kwargs)
    def prepare(*args):
        output = original_prepare(*args)
        prepared.append(output[0])
        return output
    def base(y, rate):
        seen.append(y)
        return dict(zip(FEATURE_NAMES[:148], range(148)))
    def voice(y, rate):
        seen.append(y)
        return dict.fromkeys(('shimmer_local', 'hnr_std', 'f0_std', 'f0_min', 'jitter_local'), 999.0)
    def spectral(y, rate):
        seen.append(y)
        return dict.fromkeys(('spectral_centroid_mean', 'spectral_centroid_std'), 999.0)
    def disvoice(y, rate):
        seen.append(y)
        return np.arange(36, dtype=float)
    def write_group(out, y, rate):
        seen.append(y)
    def energy(out, y):
        seen.append(y)
        return np.ones(10)
    def cpp_and_pauses(out, y, rate, rms):
        seen.append(y)
        assert len(rms) == 10
    monkeypatch.setattr(sf, "read", read)
    monkeypatch.setattr(extraction, "prepare_audio", prepare)
    monkeypatch.setattr(extraction, "praat_voice_features", base)
    for name in ("write_shared_spectra", "_write_pitch"):
        monkeypatch.setattr(extraction, name, write_group)
    monkeypatch.setattr(extraction, "_write_energy", energy)
    monkeypatch.setattr(extraction, "_write_cpp_and_pauses", cpp_and_pauses)
    monkeypatch.setattr(extraction, "extract_glottal_voice_quality", voice)
    monkeypatch.setattr(extraction, "extract_glottal_spectral_features", spectral)
    monkeypatch.setattr(extraction, "extract_disvoice_glottal_exact", disvoice)
    result = AcousticFeatureExtractor().extract_file(path)
    assert len(decoded) == len(prepared) == 1
    assert len(seen) == 8 and all(y is prepared[0] for y in seen)
    assert len(prepared[0]) == 12 * sr
    assert result.duration_seconds == 12
    assert result.metadata["acoustic_crop_seconds"] is None
    np.testing.assert_array_equal(result.values[:148], np.arange(148))
    assert len(result.values) == 165


def test_disvoice_uses_shared_waveform_and_removes_temporary_file(monkeypatch):
    waveform = np.linspace(-1, 1, 200000, dtype=np.float32)
    paths = []
    def extract(path, **kwargs):
        paths.append(Path(path))
        encoded, sr = sf.read(path)
        assert sr == 16000 and len(encoded) == len(waveform)
        np.testing.assert_allclose(encoded, waveform, atol=1 / 32768)
        return np.arange(36, dtype=float)
    monkeypatch.setattr(extraction, "_get_glottal_extractor", lambda: SimpleNamespace(extract_features_file=extract))
    assert len(extraction.extract_disvoice_glottal_exact(waveform)) == 36
    assert not paths[0].exists()


def test_disvoice_failure_removes_temporary_file(monkeypatch):
    paths = []
    def extract(path, **kwargs):
        paths.append(Path(path))
        raise RuntimeError("Extraction failed")
    monkeypatch.setattr(extraction, "_get_glottal_extractor", lambda: SimpleNamespace(extract_features_file=extract))
    with pytest.raises(RuntimeError):
        extraction.extract_disvoice_glottal_exact(np.ones(16000))
    assert not paths[0].exists()


def test_cpp_uses_frames_beyond_first_ten_seconds():
    sr = 16000
    signal = np.sin(2 * np.pi * 180 * np.arange(12 * sr) / sr)
    values = extraction.compute_framewise_cpp(signal, sr)
    assert len(values) == 1197


def test_unified_summary_includes_audio_outside_old_center_crop(monkeypatch):
    sr = 16000
    t = np.arange(12 * sr) / sr
    source = np.sin(2 * np.pi * 180 * t)
    source[:3 * sr] *= 0.1
    source[9 * sr:] *= 0.1
    cleaned, sr = extraction.prepare_audio(source, sr)
    monkeypatch.setattr(extraction, 'extract_complete_glottal_features_exact',
                        lambda *args: dict.fromkeys(FEATURE_NAMES, 123.0))
    features = extraction.extract_feature_mapping(cleaned, sr)
    assert len(features) == 165
    assert features["rms_mean"] < 0.5


def test_original_source_is_retained_after_upload_cleanup(monkeypatch, tmp_path):
    path = tmp_path / "source.wav"
    path.write_bytes(b"validated-source-audio")
    decoded = ingestion.DecodedAudio(np.sin(np.arange(32000, dtype=np.float32) * 0.1), 16000, 16000, 1)
    monkeypatch.setattr(ingestion, "decode_audio_file", lambda *args: decoded)
    audio = ingestion.preprocess_audio_file(path, app_settings=Settings(_env_file=None, xai_enabled=True, xai_mode="real"), inspection=object())
    path.unlink()
    assert audio.source_audio_bytes == b"validated-source-audio"
    assert audio.source_audio_extension == ".wav"


def test_no_normalised_waveform_fallback():
    with pytest.raises(ValueError, match="original source audio"):
        AcousticFeatureExtractor().extract_processed_audio(SimpleNamespace(source_audio_bytes=None))


def test_service_extracts_original_audio_through_unified_public_api(monkeypatch, tmp_path, trained):
    service, inputs, *_ = trained
    path = tmp_path / 'voice.wav'
    signal = np.sin(2 * np.pi * 180 * np.arange(16000) / 16000)
    sf.write(path, signal, 16000)
    source_bytes = path.read_bytes()
    path.unlink()
    seen = []
    expected = dict(zip(FEATURE_NAMES, inputs.iloc[0].to_numpy()))
    def unified(waveform, sample_rate):
        assert sample_rate == 16000 and len(waveform) == 16000
        assert np.max(np.abs(waveform)) == pytest.approx(1)
        seen.append(waveform)
        return expected
    monkeypatch.setattr(extraction, 'extract_feature_mapping', unified)
    audio = SimpleNamespace(source_audio_bytes=source_bytes, source_audio_extension='.wav')
    result = service.analyze(SimpleNamespace(acoustic_features=None, processed_audio=audio)).explanation
    assert len(seen) == 1
    precomputed = service.analyze(SimpleNamespace(acoustic_features=feature_row(list(expected.values())))).explanation
    assert result.predicted_value == precomputed.predicted_value
    assert result.feature_importance == precomputed.feature_importance


def test_disvoice_loader_and_instance_remain_cached(monkeypatch, tmp_path):
    from app.models.real import glottal_compat
    from app.voice_xai.semantic import glottal_dsp
    events = []
    class Glottal:
        def __init__(self):
            events.append("instance")
    module = SimpleNamespace(Glottal=Glottal)
    spec = SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: events.append("execute")))
    def find_spec(name):
        assert name == "disvoice"
        return SimpleNamespace(origin=str(tmp_path / "disvoice" / "__init__.py"))
    def module_spec(name, path):
        assert name == "_semantic_disvoice_glottal"
        assert path == tmp_path / "disvoice" / "glottal" / "glottal.py"
        return spec
    monkeypatch.setattr(extraction, "importlib", SimpleNamespace(util=SimpleNamespace(
        find_spec=find_spec, spec_from_file_location=module_spec,
        module_from_spec=lambda _: module,
    )))
    monkeypatch.setattr(glottal_compat, "apply_disvoice_compatibility_patches", lambda: events.append("compatibility"))
    monkeypatch.setattr(glottal_compat, "patch_disvoice_gci_numpy2_bug", lambda: events.append("gci"))
    extraction.load_glottal_class.cache_clear()
    extraction._get_glottal_extractor.cache_clear()
    monkeypatch.setattr(glottal_dsp, "private_batched_glottal_extractor", lambda: extraction.load_glottal_class()())
    try:
        first = extraction._get_glottal_extractor()
        assert extraction._get_glottal_extractor() is first
        assert extraction.load_glottal_class() is Glottal
        assert events == ["compatibility", "execute", "gci", "instance"]
    finally:
        extraction._get_glottal_extractor.cache_clear()
        extraction.load_glottal_class.cache_clear()
