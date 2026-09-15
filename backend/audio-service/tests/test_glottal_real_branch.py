"""Real Glottal / voice-source branch integration tests.

Covers: selected-feature manifest loading/ordering/validation, DisVoice name
normalisation and the 36-value contract, the selected-20 feature-vector
contract, the real joblib pipeline's own contract (predict_proba,
classes_ == [0, 1], n_features_in_ == 20), the real adapter's success and
failure paths (mocked extraction, per the task contract), Glottal appearing
in VoiceService branch results, and -- the most safety-critical property --
that a real, successful Glottal branch never enters the primary
CNN+AASIST+SSL fusion decision, wired through the actual factory/loader/
predictor code path rather than only a synthetic `BranchPrediction` (see
`tests/test_final_detector_integration.py` for the existing synthetic-only
guard this complements).

The trained artifacts (`model_artifacts/glottal/*.joblib`/`*.json`) ship in
this repository, so most tests here run unconditionally, unlike the
gitignored CNN/AASIST/SSL checkpoints. DisVoice/parselmouth/librosa are an
optional "glottal" extra; extraction-heavy real-pipeline tests are guarded
with `requires_glottal_dependencies` and skip cleanly when it is absent.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.config.settings import Settings
from app.core.exceptions import ModelLoadError
from app.models.factory import ModelFactory
from app.models.preprocessing import glottal_v1
from app.models.real.glottal_inference import (
    LoadedGlottalBranch,
    build_glottal_loader,
    build_glottal_predictor,
)
from app.schemas.common import BranchStatus
from app.schemas.prediction import BranchPrediction
from app.services.voice_service import VoiceService
from app.utils.fusion import FROZEN_RESEARCH_THRESHOLD, FusionEngine
from tests.real_model_helpers import processed_audio

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_ROOT.parent
_MODEL_ARTIFACTS_ROOT = _REPO_ROOT / "model_artifacts"
_GLOTTAL_MODEL_PATH = _MODEL_ARTIFACTS_ROOT / "glottal" / "glottal_logreg_selected20_v1.joblib"
_GLOTTAL_MANIFEST_PATH = (
    _MODEL_ARTIFACTS_ROOT / "glottal" / "glottal_selected_features_v1.json"
)


def _glottal_dependencies_available() -> bool:
    """Presence check only -- deliberately does NOT `import disvoice`.

    DisVoice's own package `__init__.py` eagerly imports its `glottal`
    submodule, which does `from scipy.integrate import cumtrapz` at import
    time (removed in modern SciPy); that only succeeds after
    `app.models.real.glottal_compat.apply_disvoice_compatibility_patches()`
    has run. `importlib.util.find_spec` answers "is the package installed"
    without executing any of that top-level code, which is the question this
    skip condition actually needs answered.
    """

    import importlib.util

    return all(
        importlib.util.find_spec(module) is not None
        for module in ("disvoice", "joblib", "librosa", "parselmouth", "sklearn")
    )


requires_glottal_artifacts = pytest.mark.skipif(
    not (_GLOTTAL_MODEL_PATH.is_file() and _GLOTTAL_MANIFEST_PATH.is_file()),
    reason=f"Glottal artifacts not found under {_MODEL_ARTIFACTS_ROOT / 'glottal'}.",
)
requires_glottal_dependencies = pytest.mark.skipif(
    not _glottal_dependencies_available(),
    reason="DisVoice/parselmouth/librosa/scikit-learn are not installed (optional 'glottal' extra).",
)


def glottal_real_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "cnn_model_mode": "disabled",
        "aasist_model_mode": "disabled",
        "ssl_model_mode": "disabled",
        "glottal_model_mode": "real",
        "glottal_model_path": "glottal/glottal_logreg_selected20_v1.joblib",
        "glottal_selected_features_path": "glottal/glottal_selected_features_v1.json",
        "model_root_dir": "../model_artifacts",
        "model_device_policy": "cpu",
        "model_default_device": "cpu",
        "model_load_strategy": "lazy",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


class FakePipeline:
    """A minimal stand-in for the joblib sklearn Pipeline's public contract."""

    def __init__(
        self,
        *,
        spoof_probability: float = 0.5,
        classes: tuple[int, ...] = (0, 1),
        n_features_in: int = 20,
    ) -> None:
        self.classes_ = np.array(classes)
        self.n_features_in_ = n_features_in
        self._spoof_probability = spoof_probability
        self.last_input: np.ndarray | None = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.last_input = X
        return np.array([[1.0 - self._spoof_probability, self._spoof_probability]])


def _fake_loaded_branch(
    *,
    spoof_probability: float = 0.5,
    selected_features: list[str] | None = None,
) -> LoadedGlottalBranch:
    features = selected_features or [f"feature_{i}" for i in range(20)]
    pipeline = FakePipeline(spoof_probability=spoof_probability, n_features_in=len(features))
    return LoadedGlottalBranch(
        pipeline=pipeline,
        selected_features=tuple(features),
        manifest_metadata={"version": "v1-fake"},
        spoof_class_index=1,
        n_features_in=len(features),
        manifest_hash_short="fakehash123",
        load_time_ms=1.0,
        verification={
            "preprocessing_verified": False,
            "class_mapping_verified": False,
            "verified": False,
        },
    )


# ---------------------------------------------------------------------------
# 1-3: selected-feature manifest contract
# ---------------------------------------------------------------------------


def test_manifest_loads_exactly_20_features() -> None:
    manifest = json.loads(_GLOTTAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert len(manifest["selected_features"]) == 20
    assert manifest.get("selected_feature_count") == 20


def test_manifest_preserves_json_order() -> None:
    manifest = json.loads(_GLOTTAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["selected_features"][0] == "spectral_centroid_mean"
    assert manifest["selected_features"][-1] == "glottal_global_kurtosis_std_naq"
    assert manifest["selected_features"] == [
        "spectral_centroid_mean",
        "shimmer_local",
        "spectral_centroid_std",
        "hnr_std",
        "f0_std",
        "f0_min",
        "jitter_local",
        "glottal_global_avg_avg_h1h2",
        "glottal_global_avg_avg_naq",
        "glottal_global_avg_std_h1h2",
        "glottal_global_skewness_std_hrf",
        "glottal_global_avg_std_qoq",
        "glottal_global_std_avg_naq",
        "glottal_global_kurtosis_avg_hrf",
        "glottal_global_kurtosis_std_h1h2",
        "glottal_global_skewness_avg_qoq",
        "glottal_global_skewness_std_qoq",
        "glottal_global_skewness_avg_h1h2",
        "glottal_global_kurtosis_avg_naq",
        "glottal_global_kurtosis_std_naq",
    ]


def test_loader_rejects_manifest_with_duplicate_feature(tmp_path) -> None:
    from app.models.real.glottal_inference import _load_and_validate_manifest

    bad = tmp_path / "manifest.json"
    bad.write_text(
        json.dumps(
            {
                "selected_features": ["a"] * 19 + ["a"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ModelLoadError) as excinfo:
        _load_and_validate_manifest(bad)
    assert excinfo.value.error_code == "glottal_manifest_duplicate_feature"


def test_loader_rejects_manifest_with_wrong_feature_count(tmp_path) -> None:
    from app.models.real.glottal_inference import _load_and_validate_manifest

    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"selected_features": ["a", "b", "c"]}), encoding="utf-8")
    with pytest.raises(ModelLoadError) as excinfo:
        _load_and_validate_manifest(bad)
    assert excinfo.value.error_code == "glottal_manifest_feature_count_invalid"


def test_loader_rejects_manifest_missing_selected_features(tmp_path) -> None:
    from app.models.real.glottal_inference import _load_and_validate_manifest

    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"version": "v1"}), encoding="utf-8")
    with pytest.raises(ModelLoadError) as excinfo:
        _load_and_validate_manifest(bad)
    assert excinfo.value.error_code == "glottal_manifest_invalid"


# ---------------------------------------------------------------------------
# 4-6: DisVoice normalisation + 36-value contract
# ---------------------------------------------------------------------------


def test_normalize_disvoice_name_examples() -> None:
    assert glottal_v1.normalize_disvoice_name("global avg avg H1H2") == "glottal_global_avg_avg_h1h2"
    assert (
        glottal_v1.normalize_disvoice_name("global skewness std HRF")
        == "glottal_global_skewness_std_hrf"
    )
    assert glottal_v1.normalize_disvoice_name("  global   std   var GCI ") == "glottal_global_std_var_gci"


def test_disvoice_raw_names_has_36_entries_and_matches_manifest() -> None:
    manifest = json.loads(_GLOTTAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert len(glottal_v1.DISVOICE_RAW_NAMES) == 36
    normalized = {glottal_v1.normalize_disvoice_name(name) for name in glottal_v1.DISVOICE_RAW_NAMES}
    glottal_selected = {
        name for name in manifest["selected_features"] if name.startswith("glottal_")
    }
    assert glottal_selected <= normalized


def test_disvoice_extraction_rejects_wrong_value_count(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_extractor = _FakeDisvoiceExtractor(values=np.zeros(35))
    monkeypatch.setattr(glottal_v1, "_get_disvoice_extractor", lambda: fake_extractor)

    with pytest.raises(glottal_v1.GlottalFeatureExtractionError) as excinfo:
        glottal_v1.extract_disvoice_glottal_features(np.zeros(16000, dtype=np.float32))
    assert excinfo.value.error_code == "glottal_disvoice_feature_count_invalid"


def test_disvoice_extraction_rejects_non_finite_output(monkeypatch: pytest.MonkeyPatch) -> None:
    values = np.zeros(36)
    values[3] = np.nan
    fake_extractor = _FakeDisvoiceExtractor(values=values)
    monkeypatch.setattr(glottal_v1, "_get_disvoice_extractor", lambda: fake_extractor)

    with pytest.raises(glottal_v1.GlottalFeatureExtractionError) as excinfo:
        glottal_v1.extract_disvoice_glottal_features(np.zeros(16000, dtype=np.float32))
    assert excinfo.value.error_code == "glottal_disvoice_non_finite"


def test_disvoice_extraction_normalizes_names_in_exact_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = np.arange(36, dtype=np.float64)
    fake_extractor = _FakeDisvoiceExtractor(values=values)
    monkeypatch.setattr(glottal_v1, "_get_disvoice_extractor", lambda: fake_extractor)

    result = glottal_v1.extract_disvoice_glottal_features(np.zeros(16000, dtype=np.float32))
    assert result["glottal_global_avg_var_gci"] == 0.0
    assert result["glottal_global_kurtosis_std_hrf"] == 35.0
    assert len(result) == 36


def test_feature_extraction_logs_each_profiled_stage(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(glottal_v1, "extract_voice_quality_features", lambda _y: {"voice": 1.0})
    monkeypatch.setattr(glottal_v1, "extract_spectral_features", lambda _y: {"spectral": 2.0})
    monkeypatch.setattr(glottal_v1, "extract_disvoice_glottal_features", lambda _y: {"disvoice": 3.0})

    with caplog.at_level(logging.INFO, logger=glottal_v1.__name__):
        features = glottal_v1.extract_all_glottal_features(np.zeros(16000, dtype=np.float32))

    completed_stages = [
        record.feature_stage
        for record in caplog.records
        if record.getMessage() == "glottal_feature_extraction_stage_completed"
    ]
    assert features == {"voice": 1.0, "spectral": 2.0, "disvoice": 3.0}
    assert completed_stages == ["voice_quality", "spectral", "disvoice_iaif"]
    assert all(
        record.duration_ms >= 0.0
        for record in caplog.records
        if record.getMessage() == "glottal_feature_extraction_stage_completed"
    )
    assert any(
        record.getMessage() == "glottal_feature_extraction_completed"
        for record in caplog.records
    )


class _FakeDisvoiceExtractor:
    def __init__(self, values: np.ndarray) -> None:
        self._values = values

    def extract_features_file(self, path: str, **kwargs: object) -> np.ndarray:
        return self._values


# ---------------------------------------------------------------------------
# 7-8: selected-20 feature vector contract
# ---------------------------------------------------------------------------


def test_build_selected_feature_vector_shape() -> None:
    selected = [f"f{i}" for i in range(20)]
    all_features = {name: float(i) for i, name in enumerate(selected)}
    vector = glottal_v1.build_selected_feature_vector(all_features, selected)
    assert vector.shape == (1, 20)
    assert vector.dtype == np.float64
    assert list(vector[0]) == [float(i) for i in range(20)]


def test_build_selected_feature_vector_rejects_non_finite() -> None:
    selected = [f"f{i}" for i in range(20)]
    all_features = {name: float(i) for i, name in enumerate(selected)}
    all_features["f5"] = float("nan")
    with pytest.raises(glottal_v1.GlottalFeatureExtractionError) as excinfo:
        glottal_v1.build_selected_feature_vector(all_features, selected)
    assert excinfo.value.error_code == "glottal_selected_feature_non_finite"


def test_build_selected_feature_vector_rejects_missing_feature() -> None:
    selected = [f"f{i}" for i in range(20)]
    all_features = {name: float(i) for i, name in enumerate(selected[:19])}
    with pytest.raises(glottal_v1.GlottalFeatureExtractionError) as excinfo:
        glottal_v1.build_selected_feature_vector(all_features, selected)
    assert excinfo.value.error_code == "glottal_selected_feature_missing"


@requires_glottal_dependencies
def test_clean_glottal_waveform_rejects_empty_after_trim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import librosa

    monkeypatch.setattr(
        librosa.effects, "trim", lambda y, top_db=None: (np.array([], dtype=np.float32), None)
    )
    with pytest.raises(glottal_v1.GlottalFeatureExtractionError) as excinfo:
        glottal_v1.clean_glottal_waveform(np.ones(16000, dtype=np.float32), 16000)
    assert excinfo.value.error_code == "glottal_waveform_empty_after_trim"


def test_clean_glottal_waveform_rejects_zero_length_input() -> None:
    with pytest.raises(glottal_v1.GlottalFeatureExtractionError) as excinfo:
        glottal_v1.clean_glottal_waveform(np.zeros(0, dtype=np.float32), 16000)
    assert excinfo.value.error_code == "glottal_waveform_empty"


def test_clean_glottal_waveform_rejects_wrong_sample_rate() -> None:
    with pytest.raises(glottal_v1.GlottalFeatureExtractionError) as excinfo:
        glottal_v1.clean_glottal_waveform(np.ones(8000, dtype=np.float32), 8000)
    assert excinfo.value.error_code == "glottal_sample_rate_invalid"


# ---------------------------------------------------------------------------
# 9-12: the real joblib pipeline's own contract
# ---------------------------------------------------------------------------


@requires_glottal_artifacts
@requires_glottal_dependencies
def test_real_pipeline_loads_and_satisfies_its_contract() -> None:
    import joblib

    pipeline = joblib.load(_GLOTTAL_MODEL_PATH)
    assert hasattr(pipeline, "predict_proba")
    assert tuple(int(c) for c in pipeline.classes_) == (0, 1)
    assert int(pipeline.n_features_in_) == 20


@requires_glottal_artifacts
@requires_glottal_dependencies
def test_predict_proba_spoof_index_is_class_1() -> None:
    import joblib

    pipeline = joblib.load(_GLOTTAL_MODEL_PATH)
    spoof_index = list(pipeline.classes_).index(1)
    assert spoof_index == 1

    rng = np.random.RandomState(0)
    X = rng.normal(size=(1, 20))
    probabilities = pipeline.predict_proba(X)[0]
    assert probabilities.shape == (2,)
    assert math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-9)
    assert probabilities[spoof_index] == probabilities[1]


@requires_glottal_artifacts
@requires_glottal_dependencies
def test_glottal_loader_validates_real_artifact() -> None:
    settings = glottal_real_settings()
    factory = ModelFactory(settings)
    config = factory.branch_config("glottal")
    loader = build_glottal_loader(settings)

    branch = loader(config)
    assert branch.n_features_in == 20
    assert len(branch.selected_features) == 20
    assert branch.spoof_class_index == 1


@requires_glottal_dependencies
def test_loader_rejects_pipeline_without_predict_proba() -> None:
    import tempfile

    import joblib

    from app.models.real.glottal_inference import _load_and_validate_pipeline

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as handle:
        path = Path(handle.name)
    try:
        # A plain builtin instance has no `predict_proba` -- picklable (unlike
        # a class defined inside this test function) without needing a
        # dedicated module-level fixture class.
        joblib.dump(object(), path)
        with pytest.raises(ModelLoadError) as excinfo:
            _load_and_validate_pipeline(path, expected_n_features=20)
        assert excinfo.value.error_code == "glottal_artifact_invalid"
    finally:
        path.unlink(missing_ok=True)


@requires_glottal_dependencies
def test_loader_rejects_pipeline_with_wrong_classes() -> None:
    import tempfile

    import joblib

    from app.models.real.glottal_inference import _load_and_validate_pipeline

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as handle:
        path = Path(handle.name)
    try:
        joblib.dump(FakePipeline(classes=(0, 1, 2)), path)
        with pytest.raises(ModelLoadError) as excinfo:
            _load_and_validate_pipeline(path, expected_n_features=20)
        assert excinfo.value.error_code == "glottal_artifact_invalid"
    finally:
        path.unlink(missing_ok=True)


@requires_glottal_dependencies
def test_loader_rejects_pipeline_with_wrong_n_features() -> None:
    import tempfile

    import joblib

    from app.models.real.glottal_inference import _load_and_validate_pipeline

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as handle:
        path = Path(handle.name)
    try:
        joblib.dump(FakePipeline(n_features_in=15), path)
        with pytest.raises(ModelLoadError) as excinfo:
            _load_and_validate_pipeline(path, expected_n_features=20)
        assert excinfo.value.error_code == "glottal_artifact_invalid"
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 13-14: real adapter success/failure with a mocked extractor and model
# ---------------------------------------------------------------------------


@requires_glottal_dependencies
def test_glottal_adapter_succeeds_with_mocked_extractor_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.models.real.base import RealModelAdapter

    settings = glottal_real_settings()
    config = ModelFactory(settings).branch_config("glottal")
    fake_branch = _fake_loaded_branch(spoof_probability=0.83)

    monkeypatch.setattr(
        glottal_v1,
        "extract_all_glottal_features",
        lambda y: {name: float(i) for i, name in enumerate(fake_branch.selected_features)},
    )

    adapter = RealModelAdapter(
        config,
        loader=lambda _config: fake_branch,
        predictor=build_glottal_predictor(),
    )
    prediction = adapter.predict_safe(processed_audio())

    assert prediction.status == BranchStatus.success
    assert prediction.probabilities is not None
    assert prediction.probabilities.spoof == pytest.approx(0.83, abs=1e-9)
    assert prediction.metadata["selected_feature_count"] == 20
    assert len(prediction.metadata["selected_features"]) == 20
    assert prediction.metadata["research_result"] is False
    assert prediction.metadata["auxiliary_branch"] is True


@requires_glottal_dependencies
def test_glottal_extraction_failure_produces_typed_branch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.models.real.base import RealModelAdapter

    settings = glottal_real_settings()
    config = ModelFactory(settings).branch_config("glottal")
    fake_branch = _fake_loaded_branch()

    def _raise(_waveform: np.ndarray) -> dict[str, float]:
        raise glottal_v1.GlottalFeatureExtractionError(
            "synthetic failure",
            public_message="Audio could not be analysed.",
            error_code="glottal_waveform_empty_after_trim",
        )

    monkeypatch.setattr(glottal_v1, "extract_all_glottal_features", _raise)

    adapter = RealModelAdapter(
        config,
        loader=lambda _config: fake_branch,
        predictor=build_glottal_predictor(),
    )
    prediction = adapter.predict_safe(processed_audio())

    assert prediction.status == BranchStatus.failed
    assert prediction.prediction is None
    assert prediction.probabilities is None
    assert prediction.metadata["error_code"] == "glottal_waveform_empty_after_trim"
    assert prediction.metadata["research_result"] is False


@requires_glottal_dependencies
def test_glottal_predict_proba_failure_produces_typed_branch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.models.real.base import RealModelAdapter

    settings = glottal_real_settings()
    config = ModelFactory(settings).branch_config("glottal")
    fake_branch = _fake_loaded_branch()

    def _raising_predict_proba(_X: np.ndarray) -> np.ndarray:
        raise RuntimeError("model is broken")

    fake_branch.pipeline.predict_proba = _raising_predict_proba
    monkeypatch.setattr(
        glottal_v1,
        "extract_all_glottal_features",
        lambda y: {name: float(i) for i, name in enumerate(fake_branch.selected_features)},
    )

    adapter = RealModelAdapter(
        config,
        loader=lambda _config: fake_branch,
        predictor=build_glottal_predictor(),
    )
    prediction = adapter.predict_safe(processed_audio())

    assert prediction.status == BranchStatus.failed
    assert prediction.metadata["error_code"] == "glottal_predict_failed"


# ---------------------------------------------------------------------------
# 15: Glottal appears in VoiceService branch results
# ---------------------------------------------------------------------------


@requires_glottal_dependencies
def test_glottal_appears_in_voice_service_branch_results(tmp_path: Path) -> None:
    from app.models.registry import ModelRegistry

    settings = Settings(
        _env_file=None,
        cnn_model_mode="disabled",
        aasist_model_mode="disabled",
        ssl_model_mode="disabled",
        glottal_model_mode="real",
        glottal_model_path="glottal/glottal_logreg_selected20_v1.joblib",
        glottal_selected_features_path="glottal/glottal_selected_features_v1.json",
        required_model_branches="glottal",
        model_root_dir="../model_artifacts",
        model_device_policy="cpu",
        model_default_device="cpu",
        model_load_strategy="lazy",
        fusion_min_successful_branches=1,
    )
    fake_branch = _fake_loaded_branch(spoof_probability=0.4)
    registry = ModelRegistry(
        models=ModelFactory(
            settings,
            real_loaders={"glottal": lambda _config: fake_branch},
            real_predictors={"glottal": build_glottal_predictor()},
        ).create_all(),
        app_settings=settings,
    )
    service = VoiceService(
        model_registry=registry,
        preprocess_fn=lambda _upload: processed_audio(),
        app_settings=settings,
    )

    with _patched_extract_all_glottal_features(fake_branch):
        response = service.predict_from_validated_upload(
            _upload_metadata(tmp_path),
            cleanup_upload=False,
        )

    glottal_branches = [b for b in response.branches if b.model_name == "glottal_features"]
    assert len(glottal_branches) == 1
    assert glottal_branches[0].status == BranchStatus.success
    assert glottal_branches[0].probabilities is not None


# ---------------------------------------------------------------------------
# 16-18: Glottal never enters the primary CNN+AASIST+SSL fusion decision
# ---------------------------------------------------------------------------


@requires_glottal_dependencies
def test_real_glottal_excluded_from_primary_fusion_even_when_successful() -> None:
    """Wires the ACTUAL factory/loader/predictor code path for Glottal (not
    only a synthetic BranchPrediction, see
    tests/test_final_detector_integration.py for that complementary guard)
    with an extreme spoof score, alongside real-mode CNN/AASIST/SSL fakes,
    and proves: (a) the fused score is exactly the 3-way mean, (b) Glottal is
    absent from contributing_branches, (c) the frozen threshold is unchanged,
    (d) research eligibility is unaffected by Glottal's presence.
    """

    # All four branches set to "real" with their default (real, on-disk)
    # checkpoint paths, so `RealModelAdapter._load_impl`'s checkpoint-validity
    # pre-check passes for all of them -- but every loader/predictor below is
    # injected, so no checkpoint's actual contents are ever read. Using the
    # default `model_root_dir` also means the frozen fusion contract
    # (model_artifacts/fusion/final_detector_v1.json) resolves and loads
    # normally, which is the whole point of this test.
    settings = Settings(
        _env_file=None,
        cnn_model_mode="real",
        aasist_model_mode="real",
        ssl_model_mode="real",
        glottal_model_mode="real",
        glottal_model_path="glottal/glottal_logreg_selected20_v1.joblib",
        glottal_selected_features_path="glottal/glottal_selected_features_v1.json",
    )
    fixed_scores = {"lfcc_cnn_tcn": 0.20, "aasist": 0.30, "ssl_sequence": 0.40}
    glottal_extreme_spoof = 0.999

    fake_glottal_branch = _fake_loaded_branch(spoof_probability=glottal_extreme_spoof)

    def _fake_real_predictor(spoof_probability: float):
        def predictor(_processed_audio, _runtime_model, config):
            from app.models.real.base import real_prediction_from_spoof_probability

            return real_prediction_from_spoof_probability(
                config=config, spoof_probability=spoof_probability
            )

        return predictor

    factory = ModelFactory(
        settings,
        real_loaders={
            **{branch: (lambda _config: object()) for branch in fixed_scores},
            "glottal": lambda _config: fake_glottal_branch,
        },
        real_predictors={
            **{
                branch: _fake_real_predictor(score)
                for branch, score in fixed_scores.items()
            },
            "glottal": build_glottal_predictor(),
        },
    )

    predictions: list[BranchPrediction] = [
        factory.create(branch).predict_safe(processed_audio()) for branch in fixed_scores
    ]
    with _patched_extract_all_glottal_features(fake_glottal_branch):
        glottal_prediction = factory.create("glottal").predict_safe(processed_audio())
    assert glottal_prediction.status == BranchStatus.success
    assert glottal_prediction.probabilities is not None
    assert glottal_prediction.probabilities.spoof == pytest.approx(
        glottal_extreme_spoof, abs=1e-6
    )
    predictions.append(glottal_prediction)

    for branch, prediction in zip(fixed_scores, predictions[:3], strict=True):
        assert prediction.status == BranchStatus.success, (branch, prediction.error)

    engine = FusionEngine.from_settings(settings)
    assert engine.contract_version is not None  # the frozen contract is active
    result = engine.fuse(predictions)

    manual_mean = sum(fixed_scores.values()) / 3
    assert result.status == BranchStatus.success
    assert result.probabilities is not None
    # If Glottal (spoof=0.999) had leaked into a 4-way mean, the result would
    # be pulled far above the true 3-way mean -- this is a strong, sensitive
    # regression signal, not just presence/absence in a list.
    assert result.probabilities.spoof == pytest.approx(manual_mean, abs=1e-9)
    assert "glottal_features" not in result.contributing_branches
    assert "glottal" not in result.contributing_branches

    # 17: the frozen threshold is unchanged.
    assert result.decision_threshold == pytest.approx(FROZEN_RESEARCH_THRESHOLD, abs=1e-12)
    assert engine.spoof_threshold == pytest.approx(FROZEN_RESEARCH_THRESHOLD, abs=1e-12)

    # 18: research eligibility behaves exactly as it does without Glottal at
    # all -- all three primary branches present, real, and attested.
    assert result.eligible_for_research_evaluation is True
    assert result.research_blockers == []


def _upload_metadata(tmp_path: Path):
    from app.ingestion.audio import AudioInspectionResult, AudioUploadMetadata

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"not-real-audio-bytes")
    return AudioUploadMetadata(
        original_filename="sample.wav",
        sanitized_filename="sample.wav",
        saved_filename="sample.wav",
        saved_path=audio_path,
        content_type="audio/wav",
        file_size_bytes=audio_path.stat().st_size,
        duration_seconds=3.0,
        sample_rate=16000,
        channels=1,
        original_extension="wav",
        detected_container="wav",
        detected_codec="pcm_s16le",
        inspection=AudioInspectionResult(
            duration_seconds=3.0,
            sample_rate=16000,
            channels=1,
            detected_container="wav",
            detected_codec="pcm_s16le",
            audio_stream_index=0,
        ),
    )


class _patched_extract_all_glottal_features:
    """Context manager: monkeypatch glottal_v1.extract_all_glottal_features
    for the duration of a real predictor call, without a `monkeypatch`
    fixture (needed inside helpers called from other test bodies)."""

    def __init__(self, fake_branch: LoadedGlottalBranch) -> None:
        self._fake_branch = fake_branch
        self._original: Any = None

    def __enter__(self) -> None:
        self._original = glottal_v1.extract_all_glottal_features
        glottal_v1.extract_all_glottal_features = (
            lambda y: {
                name: float(i) for i, name in enumerate(self._fake_branch.selected_features)
            }
        )

    def __exit__(self, *exc_info: object) -> None:
        glottal_v1.extract_all_glottal_features = self._original
