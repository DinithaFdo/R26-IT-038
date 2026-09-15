from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pytest

from app.ingestion.audio import AudioUploadMetadata, inspect_audio_file, preprocess_audio_file
from app.models.base import BaseVoiceModel
from app.models.factory import ModelFactory
from app.models.registry import ModelRegistry
from app.models.runtime import REAL_ARCHITECTURES
from app.schemas.common import BranchStatus, ModelMode, PredictionLabel
from app.schemas.prediction import BranchPrediction, ProbabilityScores
from app.services.voice_service import VoiceService
from app.utils.fusion import DEFAULT_DEVELOPMENT_WEIGHTS, FusionEngine
from scripts.validate_aasist_light_v2_inference import (
    validate_audio_file,
    validation_settings,
)
from tests.real_model_helpers import CHECKPOINT_ROOT, requires_torch

pytestmark = requires_torch

AASIST_V2_CHECKPOINT = CHECKPOINT_ROOT / "aasist" / "aasist_light_v2_best.pt"
PHASE3_AUDIO = (
    Path(__file__).resolve().parents[2]
    / "frontend"
    / "e2e"
    / "fixtures"
    / "sample-voice.wav"
)
requires_aasist_v2_assets = pytest.mark.skipif(
    not AASIST_V2_CHECKPOINT.is_file() or not PHASE3_AUDIO.is_file(),
    reason="AASIST-Light V2 checkpoint or Phase 3 audio fixture is unavailable.",
)


class FixedBranchModel(BaseVoiceModel):
    def __init__(
        self,
        model_name: str,
        spoof_probability: float,
        *,
        mode: ModelMode = ModelMode.dummy,
        should_fail: bool = False,
    ) -> None:
        super().__init__()
        self._name = model_name
        self._spoof_probability = spoof_probability
        self._mode = mode
        self._should_fail = should_fail

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._name

    @property
    def mode(self) -> ModelMode:
        return self._mode

    @property
    def is_loaded(self) -> bool:
        return True

    def predict(self, processed_audio) -> BranchPrediction:
        if self._should_fail:
            raise RuntimeError("deterministic branch failure")
        return branch_prediction(
            self._name,
            self._spoof_probability,
            mode=self._mode,
        )


def branch_prediction(
    model_name: str,
    spoof_probability: float,
    *,
    mode: ModelMode = ModelMode.real,
    status: BranchStatus = BranchStatus.success,
) -> BranchPrediction:
    if status != BranchStatus.success:
        return BranchPrediction(
            model_name=model_name,
            display_name=model_name,
            status=status,
            mode=mode,
            prediction=None,
            confidence=None,
            probabilities=None,
            processing_time_ms=1.0,
            error="branch failed",
            metadata={"error_code": "branch_failed", "research_result": False},
        )
    bonafide_probability = 1.0 - spoof_probability
    prediction = (
        PredictionLabel.spoof
        if spoof_probability >= bonafide_probability
        else PredictionLabel.bonafide
    )
    return BranchPrediction(
        model_name=model_name,
        display_name=model_name,
        status=status,
        mode=mode,
        prediction=prediction,
        confidence=max(spoof_probability, bonafide_probability),
        probabilities=ProbabilityScores(
            bonafide=bonafide_probability,
            spoof=spoof_probability,
        ),
        processing_time_ms=1.0,
        metadata={"research_result": False},
    )


def default_phase5_settings(**overrides):
    values = {
        "model_root_dir": str(CHECKPOINT_ROOT),
        "aasist_model_mode": "real",
        "cnn_model_mode": "dummy",
        "ssl_model_mode": "dummy",
        "glottal_model_mode": "dummy",
        "required_model_branches": "aasist",
        "fusion_min_successful_branches": 1,
        "model_device_policy": "cpu",
        "model_default_device": "cpu",
        "model_load_strategy": "lazy",
    }
    values.update(overrides)
    return validation_settings(**values)


def test_fusion_branch_key_remains_aasist() -> None:
    result = FusionEngine(minimum_successful_branches=1).fuse(
        [branch_prediction("aasist", 0.8)]
    )

    assert result.contributing_branches == ["aasist"]
    assert result.branch_weights == {"aasist": pytest.approx(1.0)}


@requires_aasist_v2_assets
def test_aasist_fusion_input_equals_spoof_probability_not_bonafide() -> None:
    settings = default_phase5_settings()
    aasist = _real_aasist_branch_prediction(settings)
    result = FusionEngine(minimum_successful_branches=1).fuse([aasist])

    assert result.probabilities is not None
    assert result.probabilities.spoof == pytest.approx(
        aasist.probabilities.spoof,
        abs=1e-8,
    )
    assert result.probabilities.spoof != pytest.approx(aasist.probabilities.bonafide)
    assert result.probabilities.spoof == pytest.approx(0.9999964237213135, abs=1e-8)


def test_aasist_v2_internal_version_does_not_change_public_fusion_key() -> None:
    assert REAL_ARCHITECTURES["aasist"] == "aasist-light-v2-finalized-baseline"
    result = FusionEngine(minimum_successful_branches=1).fuse(
        [branch_prediction("aasist", 0.7)]
    )

    assert "aasist-light-v2-finalized-baseline" not in result.branch_weights
    assert "aasist" in result.branch_weights


def test_current_aasist_fusion_weight_is_reused() -> None:
    settings = default_phase5_settings()
    engine = FusionEngine.from_settings(settings)

    assert engine.branch_weights["aasist"] == pytest.approx(0.25)


def test_existing_fusion_weights_remain_unchanged() -> None:
    settings = default_phase5_settings()

    assert settings.fusion_weight_map == {
        "lfcc_cnn_tcn": pytest.approx(0.25),
        "aasist": pytest.approx(0.25),
        "ssl_sequence": pytest.approx(0.25),
        "glottal": pytest.approx(0.25),
    }
    assert DEFAULT_DEVELOPMENT_WEIGHTS == {
        "cnn": 0.25,
        "aasist": 0.25,
        "ssl": 0.25,
        "glottal": 0.25,
    }


def test_fusion_output_matches_manual_weighted_calculation() -> None:
    branches = [
        branch_prediction("cnn_acoustic", 0.10),
        branch_prediction("aasist", 0.80),
        branch_prediction("ssl_wavlm_xlsr", 0.30),
        branch_prediction("glottal_features", 0.70),
    ]
    result = FusionEngine().fuse(branches)
    expected = (0.10 + 0.80 + 0.30 + 0.70) / 4

    assert result.probabilities is not None
    assert result.probabilities.spoof == pytest.approx(expected, abs=1e-12)


def test_high_aasist_spoof_probability_increases_fused_spoof_contribution() -> None:
    engine = FusionEngine(minimum_successful_branches=2)
    low = engine.fuse(
        [branch_prediction("cnn_acoustic", 0.2), branch_prediction("aasist", 0.1)]
    )
    high = engine.fuse(
        [branch_prediction("cnn_acoustic", 0.2), branch_prediction("aasist", 0.9)]
    )

    assert low.probabilities is not None
    assert high.probabilities is not None
    assert high.probabilities.spoof > low.probabilities.spoof


def test_low_aasist_spoof_probability_lowers_fused_spoof_contribution() -> None:
    engine = FusionEngine(minimum_successful_branches=2)
    high = engine.fuse(
        [branch_prediction("ssl_wavlm_xlsr", 0.6), branch_prediction("aasist", 0.8)]
    )
    low = engine.fuse(
        [branch_prediction("ssl_wavlm_xlsr", 0.6), branch_prediction("aasist", 0.2)]
    )

    assert low.probabilities is not None
    assert high.probabilities is not None
    assert low.probabilities.spoof < high.probabilities.spoof


def test_aasist_score_is_not_thresholded_before_fusion() -> None:
    result = FusionEngine(minimum_successful_branches=1).fuse(
        [branch_prediction("aasist", 0.51)]
    )

    assert result.probabilities is not None
    assert result.probabilities.spoof == pytest.approx(0.51)


def test_aasist_raw_logits_are_not_passed_directly_into_fusion() -> None:
    result = FusionEngine(minimum_successful_branches=1).fuse(
        [branch_prediction("aasist", 0.9999964237213135)]
    )

    assert result.probabilities is not None
    assert result.probabilities.spoof == pytest.approx(0.9999964237213135)
    assert result.probabilities.spoof != pytest.approx(5.85805082321167)


def test_all_branch_success_path_works() -> None:
    result = FusionEngine().fuse(
        [
            branch_prediction("cnn_acoustic", 0.1),
            branch_prediction("aasist", 0.8),
            branch_prediction("ssl_wavlm_xlsr", 0.3),
            branch_prediction("glottal_features", 0.7),
        ]
    )

    assert result.status == BranchStatus.success
    assert result.contributing_branches == [
        "cnn_acoustic",
        "aasist",
        "ssl_wavlm_xlsr",
        "glottal_features",
    ]


def test_aasist_success_other_branch_failure_keeps_existing_fallback() -> None:
    result = FusionEngine().fuse(
        [
            branch_prediction("cnn_acoustic", 0.2, status=BranchStatus.failed),
            branch_prediction("aasist", 0.8),
            branch_prediction("ssl_wavlm_xlsr", 0.4),
        ]
    )

    assert result.status == BranchStatus.success
    assert result.probabilities is not None
    assert result.probabilities.spoof == pytest.approx((0.8 + 0.4) / 2)
    assert result.excluded_branches == {"cnn_acoustic": "branch_failed"}


def test_aasist_failure_other_branches_success_keeps_existing_fallback() -> None:
    result = FusionEngine().fuse(
        [
            branch_prediction("cnn_acoustic", 0.2),
            branch_prediction("aasist", 0.0, status=BranchStatus.failed),
            branch_prediction("ssl_wavlm_xlsr", 0.6),
        ]
    )

    assert result.status == BranchStatus.success
    assert result.probabilities is not None
    assert result.probabilities.spoof == pytest.approx((0.2 + 0.6) / 2)
    assert result.excluded_branches == {"aasist": "branch_failed"}


def test_no_valid_branches_behavior_remains_controlled() -> None:
    result = FusionEngine().fuse(
        [
            branch_prediction("cnn_acoustic", 0.0, status=BranchStatus.failed),
            branch_prediction("aasist", 0.0, status=BranchStatus.failed),
        ]
    )

    assert result.status == BranchStatus.failed
    assert result.probabilities is None
    assert result.warning == "At least two successful branches are required for fusion."


@requires_aasist_v2_assets
def test_real_mode_resolves_canonical_aasist_to_v2() -> None:
    settings = default_phase5_settings()
    model = ModelFactory(settings).create("aasist")
    model.load()

    assert model.branch_name == "aasist"
    assert model._runtime_model.architecture_version == REAL_ARCHITECTURES["aasist"]


def test_dummy_mode_remains_unchanged() -> None:
    model = ModelFactory(default_phase5_settings(aasist_model_mode="dummy")).create(
        "aasist"
    )

    assert model.branch_name == "aasist"
    assert model.mode == ModelMode.dummy
    assert model.model_name == "aasist"


def test_model_registry_and_factory_remain_compatible() -> None:
    settings = default_phase5_settings()
    registry = ModelRegistry(app_settings=settings)

    assert [model.branch_name for model in registry.models] == [
        "lfcc_cnn_tcn",
        "aasist",
        "ssl_sequence",
        "glottal",
    ]
    assert ModelFactory(settings).branch_config("aasist").branch_name == "aasist"


def test_fusion_api_public_schema_remains_unchanged() -> None:
    from app.main import create_app

    schema = create_app(app_settings=default_phase5_settings()).openapi()

    assert "/api/v1/predictions" in schema["paths"]
    assert not any("aasist-light-v2" in path for path in schema["paths"])
    assert "FusionResult" in schema["components"]["schemas"]


def test_cnn_is_now_v2() -> None:
    """Was `test_cnn_behavior_unchanged` (AASIST-V2 integration regression
    guard). CNN intentionally moved to CNN-V2 in the final-detector
    integration -- this now pins the deliberate new value instead."""

    assert REAL_ARCHITECTURES["lfcc_cnn_tcn"] == "cnn-v2-lfcc-delta-vgg-4block-v1"


def test_ssl_behavior_unchanged() -> None:
    assert (
        REAL_ARCHITECTURES["ssl_sequence"]
        == "xlsr-mamba-sequence-frozen-backbone-pure-pytorch-scan-v1"
    )


def test_glottal_is_now_real() -> None:
    """Was `test_glottal_behavior_unchanged`. Glottal intentionally moved from
    its placeholder contract string to a trained real pipeline (logistic
    regression over 20 selected DisVoice/parselmouth/spectral features) in
    the Glottal branch integration -- pins the deliberate new value instead.
    """

    assert REAL_ARCHITECTURES["glottal"] == "glottal-logreg-selected20-v1"


@requires_aasist_v2_assets
def test_phase3_standalone_inference_remains_deterministic() -> None:
    settings = default_phase5_settings()
    model = ModelFactory(settings).create("aasist")

    first = validate_audio_file(PHASE3_AUDIO, app_settings=settings, model=model)
    second = validate_audio_file(PHASE3_AUDIO, app_settings=settings, model=model)

    assert abs(first["spoof_probability"] - second["spoof_probability"]) < 1e-8


@requires_aasist_v2_assets
def test_real_aasist_participates_in_voice_service_fusion_path(tmp_path: Path) -> None:
    settings = default_phase5_settings()
    aasist_model = ModelFactory(settings).create("aasist")
    registry = ModelRegistry(
        models=[
            FixedBranchModel("cnn_acoustic", 0.2),
            aasist_model,
            FixedBranchModel("ssl_wavlm_xlsr", 0.4),
            FixedBranchModel("glottal_features", 0.6),
        ]
    )
    service = VoiceService(
        model_registry=registry,
        fusion_engine=FusionEngine.from_settings(settings),
        preprocess_fn=lambda metadata: preprocess_audio_file(
            metadata.saved_path,
            extension="wav",
            app_settings=settings,
            inspection=metadata.inspection,
        ),
        app_settings=settings,
    )
    upload_metadata = _phase3_upload_metadata(tmp_path)
    start = perf_counter()

    response = service.predict_from_validated_upload(
        upload_metadata,
        cleanup_upload=False,
    )
    elapsed_ms = (perf_counter() - start) * 1000
    aasist_prediction = next(
        branch for branch in response.branches if branch.model_name == "aasist"
    )
    expected = (0.2 + aasist_prediction.probabilities.spoof + 0.4 + 0.6) / 4

    assert response.fusion.status == BranchStatus.success
    assert response.fusion.probabilities is not None
    assert aasist_prediction.mode == ModelMode.real
    assert aasist_prediction.metadata["architecture"] == REAL_ARCHITECTURES["aasist"]
    assert aasist_prediction.probabilities.spoof == pytest.approx(
        0.9999964237213135,
        abs=1e-8,
    )
    assert response.fusion.branch_weights["aasist"] == pytest.approx(0.25)
    assert response.fusion.probabilities.spoof == pytest.approx(expected, abs=1e-8)
    assert response.total_processing_time_ms <= elapsed_ms + 20


@requires_aasist_v2_assets
def test_aasist_v2_readiness_feeds_model_registry() -> None:
    settings = default_phase5_settings(required_model_branches="aasist")
    registry = ModelRegistry(models=[ModelFactory(settings).create("aasist")], app_settings=settings)
    registry.models[0].load()

    readiness = registry.readiness()

    assert readiness["ready"] is True
    assert readiness["mode_summary"]["real"] == ["aasist"]
    assert readiness["unverified_real_branches"] == ["aasist"]


def _real_aasist_branch_prediction(settings) -> BranchPrediction:
    model = ModelFactory(settings).create("aasist")
    result = validate_audio_file(PHASE3_AUDIO, app_settings=settings, model=model)
    return branch_prediction(
        "aasist",
        result["spoof_probability"],
        mode=ModelMode.real,
    )


def _phase3_upload_metadata(tmp_path: Path) -> AudioUploadMetadata:
    saved_path = tmp_path / "sample-voice.wav"
    saved_path.write_bytes(PHASE3_AUDIO.read_bytes())
    inspection = inspect_audio_file(saved_path, "wav", default_phase5_settings())
    return AudioUploadMetadata(
        original_filename="sample-voice.wav",
        sanitized_filename="sample-voice.wav",
        saved_filename="sample-voice.wav",
        saved_path=saved_path,
        content_type="audio/wav",
        file_size_bytes=saved_path.stat().st_size,
        duration_seconds=inspection.duration_seconds,
        sample_rate=inspection.sample_rate,
        channels=inspection.channels,
        original_extension="wav",
        detected_container=inspection.detected_container,
        detected_codec=inspection.detected_codec,
        inspection=inspection,
    )
