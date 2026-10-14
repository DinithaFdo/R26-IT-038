"""Portable extraction parity and preservation of training-table metadata."""
from hashlib import sha256
import importlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from app.voice_xai.semantic import dataset, recipe
from app.voice_xai.semantic.features import AcousticFeatureExtractor
from app.voice_xai.semantic.schema import FEATURE_NAMES, FEATURE_EXTRACTION_VERSION
from scripts.export_semantic_extractor import export_extractor


def test_portable_package_matches_backend_features(monkeypatch, tmp_path):
    archive = export_extractor(tmp_path / "extractor.zip")
    with ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read("semantic_audio/export_manifest.json"))
        for name, expected in manifest["source_hashes"].items():
            assert sha256(zipped.read(f"semantic_audio/{name}")).hexdigest() == expected
        zipped.extractall(tmp_path / "training")
    monkeypatch.syspath_prepend(str(tmp_path / "training"))
    portable = importlib.import_module("semantic_audio")
    portable_recipe = importlib.import_module("semantic_audio.recipe")
    assert portable.FEATURE_EXTRACTION_VERSION == FEATURE_EXTRACTION_VERSION
    assert portable.FEATURE_NAMES == FEATURE_NAMES
    # Exercise real acoustic/Praat/spectral calculations in both distributions.
    # The separate DisVoice encoding tests cover its file-only interface.
    monkeypatch.setattr(recipe, "extract_disvoice_glottal_exact", lambda y, sr: np.arange(36, dtype=float))
    monkeypatch.setattr(portable_recipe, "extract_disvoice_glottal_exact", lambda y, sr: np.arange(36, dtype=float))
    sr = 16000
    signal = np.sin(2 * np.pi * 180 * np.arange(2 * sr) / sr)
    path = tmp_path / "source.wav"
    sf.write(path, signal, sr)
    trained = portable.extract_surrogate_165_features(path)
    deployed = AcousticFeatureExtractor().extract_file(path)
    np.testing.assert_array_equal(trained.to_numpy()[0], deployed.values)


def test_rebuild_replaces_features_without_changing_targets_or_rows(monkeypatch, tmp_path):
    paths = [tmp_path / "second.wav", tmp_path / "first.wav"]
    for path in paths:
        path.write_bytes(b"audio")
    frame = pd.DataFrame({
        "sample_key": ["b", "a"], "audio_path": list(map(str, paths)),
        "teacher_fused_spoof_probability": [0.8, 0.1],
        "label_int": [1, 0], "teacher_fingerprint": ["teacher", "teacher"],
        "duration_s": [13.0, 8.0], FEATURE_NAMES[0]: [-999, -999],
    }, index=[9, 4])
    original = frame.copy(deep=True)
    def extract(path):
        value = 20 if path.name == "second.wav" else 10
        return pd.DataFrame([[value] * 165], columns=FEATURE_NAMES)
    monkeypatch.setattr(dataset, "extract_surrogate_165_features", extract)
    rebuilt = dataset.rebuild_feature_frame(frame)
    metadata = [name for name in frame.columns if name not in FEATURE_NAMES]
    pd.testing.assert_frame_equal(rebuilt[metadata], original[metadata])
    pd.testing.assert_frame_equal(frame, original)
    assert rebuilt[FEATURE_NAMES[0]].tolist() == [20, 10]
    assert tuple(rebuilt[list(FEATURE_NAMES)].columns) == FEATURE_NAMES


def test_rebuild_stops_when_original_audio_is_missing(tmp_path):
    frame = pd.DataFrame({"audio_path": [str(tmp_path / "missing.wav")]})
    with pytest.raises(FileNotFoundError, match="Original audio is unavailable"):
        dataset.rebuild_feature_frame(frame)


def test_rebuild_never_keeps_old_features_when_extraction_fails(monkeypatch, tmp_path):
    path = tmp_path / "source.wav"
    path.write_bytes(b"audio")
    frame = pd.DataFrame({"audio_path": [str(path)], FEATURE_NAMES[0]: [123]})
    def extract(path):
        raise RuntimeError("Audio failed")
    monkeypatch.setattr(dataset, "extract_surrogate_165_features", extract)
    with pytest.raises(ValueError, match="row 0"):
        dataset.rebuild_feature_frame(frame)
    assert frame[FEATURE_NAMES[0]].iloc[0] == 123
