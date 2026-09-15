from io import BytesIO
import json
from pathlib import Path
import wave

import numpy as np
import pytest

from app.core.exceptions import (
    AudioProcessingTimeoutError,
    CorruptedAudioError,
    DecodedAudioTooLargeError,
    ModelInferenceError,
    UnusableAudioError,
)
from app.config.settings import Settings
from app.api.dependencies import get_voice_service
from app.models.base import BaseVoiceModel
from app.ingestion.audio import AudioUploadMetadata, ProcessedAudio
from app.schemas.common import BranchStatus, ModelMode, PredictionLabel
from app.schemas.prediction import (
    AudioStorageMetadata,
    BranchPrediction,
    ProbabilityScores,
)
from app.services.voice_service import ModelRegistry, VoiceService
from app.voice_xai.temporal.contracts import TemporalAttentionWindowInput


class FakeVoiceModel(BaseVoiceModel):
    def __init__(
        self,
        *,
        model_name: str,
        spoof_probability: float = 0.6,
        should_fail: bool = False,
    ) -> None:
        self._model_name = model_name
        self._spoof_probability = spoof_probability
        self._should_fail = should_fail
        self._is_loaded = True
        self.seen_audio_ids: list[int] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def display_name(self) -> str:
        return self._model_name

    @property
    def mode(self) -> ModelMode:
        return ModelMode.dummy

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        self._is_loaded = True

    def predict(self, processed_audio) -> BranchPrediction:
        self.seen_audio_ids.append(id(processed_audio))
        if self._should_fail:
            raise ModelInferenceError(
                "internal branch failure",
                public_message="Branch failed.",
            )

        bonafide_probability = 1.0 - self._spoof_probability
        prediction = (
            PredictionLabel.spoof
            if self._spoof_probability >= bonafide_probability
            else PredictionLabel.bonafide
        )
        return BranchPrediction(
            model_name=self.model_name,
            display_name=self.display_name,
            status=BranchStatus.success,
            mode=self.mode,
            prediction=prediction,
            confidence=max(self._spoof_probability, bonafide_probability),
            probabilities=ProbabilityScores(
                bonafide=bonafide_probability,
                spoof=self._spoof_probability,
            ),
            processing_time_ms=0.0,
            metadata={
                "development_placeholder": True,
                "research_result": False,
            },
        )


class HookableRealCnnModel(BaseVoiceModel):
    """Small real-mode stand-in that exercises the production CNN hook plan."""

    def __init__(self) -> None:
        super().__init__()
        torch = pytest.importorskip("torch")
        from app.models.architectures.cnn_v2 import build_cnn_v2_net

        self._torch = torch
        self._module = build_cnn_v2_net()
        self._module.eval()

    @property
    def model_name(self) -> str:
        return "cnn_acoustic"

    @property
    def display_name(self) -> str:
        return "CNN Acoustic Features"

    @property
    def mode(self) -> ModelMode:
        return ModelMode.real

    def _load_impl(self) -> None:
        return None

    def xai_capture_root(self):
        self.load()
        return self._module

    def predict(self, processed_audio) -> BranchPrediction:
        with self._torch.inference_mode():
            self._module(self._torch.zeros(1, 1, 120, 401))
        return BranchPrediction(
            model_name=self.model_name,
            display_name=self.display_name,
            status=BranchStatus.success,
            mode=self.mode,
            prediction=PredictionLabel.bonafide,
            confidence=0.8,
            probabilities=ProbabilityScores(bonafide=0.8, spoof=0.2),
            processing_time_ms=1.0,
            metadata={"research_result": False},
        )


class OriginalPassTemporalSslModel(BaseVoiceModel):
    """Real-mode stand-in proving temporal evidence shares the prediction pass."""

    def __init__(self) -> None:
        super().__init__()
        self.predict_calls = 0
        self.temporal_prediction_calls = 0

    @property
    def model_name(self) -> str:
        return "ssl_wavlm_xlsr"

    @property
    def display_name(self) -> str:
        return "SSL sequence"

    @property
    def mode(self) -> ModelMode:
        return ModelMode.real

    def _load_impl(self) -> None:
        return None

    def predict(self, processed_audio) -> BranchPrediction:
        self.predict_calls += 1
        return self._prediction()

    def predict_safe_with_temporal_evidence(self, processed_audio):
        self.temporal_prediction_calls += 1
        return (
            self._prediction(),
            (
                TemporalAttentionWindowInput(
                    start_seconds=0.0,
                    end_seconds=1.0,
                    token_times_seconds=np.array([0.25, 0.75], dtype=np.float32),
                    attention_density=np.array([0.8, 1.2], dtype=np.float32),
                    spoof_probability=0.8,
                ),
            ),
        )

    def _prediction(self) -> BranchPrediction:
        return BranchPrediction(
            model_name=self.model_name,
            display_name=self.display_name,
            status=BranchStatus.success,
            mode=self.mode,
            prediction=PredictionLabel.spoof,
            confidence=0.8,
            probabilities=ProbabilityScores(bonafide=0.2, spoof=0.8),
            processing_time_ms=1.0,
            metadata={"research_result": False},
        )


def make_registry(*, failing_models: set[str] | None = None) -> ModelRegistry:
    failing_models = failing_models or set()
    model_names = ["cnn_acoustic", "aasist", "ssl_wavlm_xlsr", "glottal_features"]
    models = [
        FakeVoiceModel(
            model_name=model_name,
            spoof_probability=0.2 + index * 0.2,
            should_fail=model_name in failing_models,
        )
        for index, model_name in enumerate(model_names)
    ]
    return ModelRegistry(models=models)


def make_upload_metadata(tmp_path: Path) -> AudioUploadMetadata:
    saved_path = tmp_path / "validated.wav"
    wav_bytes = make_wav_bytes()
    saved_path.write_bytes(wav_bytes)
    return AudioUploadMetadata(
        original_filename="sample.wav",
        sanitized_filename="sample.wav",
        saved_filename="validated.wav",
        saved_path=saved_path,
        content_type="audio/wav",
        file_size_bytes=len(wav_bytes),
        duration_seconds=0.1,
        sample_rate=16000,
        channels=1,
    )


def make_wav_bytes() -> bytes:
    sample_rate = 16000
    frame_count = int(sample_rate * 0.1)
    time = np.arange(frame_count, dtype=np.float32) / sample_rate
    waveform = (0.25 * np.sin(2 * np.pi * 440 * time) * 32767).astype(np.int16)
    buffer = BytesIO()
    with wave.open(buffer, "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(waveform.tobytes())
    return buffer.getvalue()


def make_processed_audio() -> ProcessedAudio:
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = (0.25 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    return ProcessedAudio(
        waveform=waveform,
        sample_rate=sample_rate,
        original_sample_rate=sample_rate,
        original_channels=1,
        duration_seconds=1.0,
        was_resampled=False,
        was_converted_to_mono=False,
        normalisation_applied=False,
        peak_amplitude=float(np.max(np.abs(waveform))),
        rms_energy=float(np.sqrt(np.mean(np.square(waveform, dtype=np.float32)))),
    )


def make_stereo_48khz_processed_audio() -> ProcessedAudio:
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = (0.95 * np.sin(2 * np.pi * 440 * time)).astype(np.float32)
    return ProcessedAudio(
        waveform=waveform,
        sample_rate=sample_rate,
        original_sample_rate=48000,
        original_channels=2,
        duration_seconds=1.0,
        was_resampled=True,
        was_converted_to_mono=True,
        normalisation_applied=True,
        peak_amplitude=float(np.max(np.abs(waveform))),
        rms_energy=float(np.sqrt(np.mean(np.square(waveform, dtype=np.float32)))),
    )


def make_service(**kwargs) -> VoiceService:
    return VoiceService(
        preprocess_fn=lambda _metadata: make_processed_audio(),
        **kwargs,
    )


def test_dependency_provider_reuses_model_registry() -> None:
    get_voice_service.cache_clear()
    try:
        first_service = get_voice_service()
        second_service = get_voice_service()
    finally:
        get_voice_service.cache_clear()

    assert first_service is second_service
    assert first_service._model_registry is second_service._model_registry


def test_service_all_dummy_branches_succeed(tmp_path) -> None:
    service = make_service()
    upload_metadata = make_upload_metadata(tmp_path)

    response = service.predict_from_validated_upload(upload_metadata)

    assert len(response.branches) == 4
    assert all(branch.status == BranchStatus.success for branch in response.branches)
    assert response.fusion.status == BranchStatus.success
    assert response.request_id
    assert response.total_processing_time_ms >= 0.0


def test_opt_in_capture_runs_in_the_classifier_worker_and_leaves_no_hooks(tmp_path) -> None:
    real_cnn = HookableRealCnnModel()
    dummy_models = make_registry().models[1:]
    service = make_service(model_registry=ModelRegistry(models=[real_cnn, *dummy_models]))

    execution = service.predict_from_validated_upload_with_processed_audio(
        make_upload_metadata(tmp_path),
        capture_extraction=True,
    )

    assert execution.extraction is not None
    captures = execution.extraction.branches["lfcc_cnn_tcn"]
    assert len(captures) == 3
    assert all(capture.status == "captured" for capture in captures.values())
    assert execution.extraction.metadata["capture_requested"] is True
    assert execution.extraction.metadata["capture_enabled"] is True
    assert all(
        not submodule._forward_hooks
        for _name, submodule in real_cnn._module.named_modules()
    )


def test_real_ssl_temporal_evidence_is_created_during_prediction(tmp_path) -> None:
    ssl_model = OriginalPassTemporalSslModel()
    service = make_service(
        model_registry=ModelRegistry(models=[ssl_model]),
        app_settings=Settings(_env_file=None,xai_enabled=True, xai_mode="mock"),
    )

    execution = service.predict_from_validated_upload_with_processed_audio(
        make_upload_metadata(tmp_path)
    )

    assert ssl_model.temporal_prediction_calls == 1
    assert ssl_model.predict_calls == 0
    assert execution.temporal_evidence is not None
    assert len(execution.temporal_evidence) == 1
    assert execution.temporal_evidence[0].end_seconds == pytest.approx(1.0)


def test_service_one_branch_failure_does_not_crash_pipeline(tmp_path) -> None:
    registry = make_registry(failing_models={"aasist"})
    service = make_service(model_registry=registry)
    upload_metadata = make_upload_metadata(tmp_path)

    response = service.predict_from_validated_upload(upload_metadata)

    failed_branches = [
        branch for branch in response.branches if branch.status == BranchStatus.failed
    ]
    assert [branch.model_name for branch in failed_branches] == ["aasist"]
    assert response.fusion.status == BranchStatus.success
    assert "aasist" not in response.fusion.branch_weights


def test_service_multiple_branch_failures_return_failed_fusion(tmp_path) -> None:
    registry = make_registry(
        failing_models={"aasist", "ssl_wavlm_xlsr", "glottal_features"}
    )
    service = make_service(model_registry=registry)
    upload_metadata = make_upload_metadata(tmp_path)

    response = service.predict_from_validated_upload(upload_metadata)

    assert sum(
        branch.status == BranchStatus.success for branch in response.branches
    ) == 1
    assert response.fusion.status == BranchStatus.failed
    assert response.fusion.prediction is None


def test_service_fusion_is_not_research_eligible_with_dummy_models(tmp_path) -> None:
    service = make_service(model_registry=make_registry())
    upload_metadata = make_upload_metadata(tmp_path)

    response = service.predict_from_validated_upload(upload_metadata)

    assert response.fusion.contains_dummy_branches is True
    assert response.fusion.eligible_for_research_evaluation is False
    assert response.fusion.warning is not None


def test_service_removes_temporary_file(tmp_path) -> None:
    service = make_service(model_registry=make_registry())
    upload_metadata = make_upload_metadata(tmp_path)
    saved_path = upload_metadata.saved_path

    service.predict_from_validated_upload(upload_metadata)

    assert not saved_path.exists()


def test_service_response_schema_validity(tmp_path) -> None:
    service = make_service(model_registry=make_registry())
    upload_metadata = make_upload_metadata(tmp_path)

    response = service.predict_from_validated_upload(upload_metadata)

    assert response.model_validate(response.model_dump()) == response


def test_service_adds_reproducible_prediction_provenance_for_stereo_48khz(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.services.voice_service.audio_tool_versions",
        lambda: {
            "ffmpeg_version": "ffmpeg version 6.1-test",
            "ffprobe_version": "ffprobe version 6.1-test",
        },
    )
    service = VoiceService(
        model_registry=make_registry(),
        preprocess_fn=lambda _metadata: make_stereo_48khz_processed_audio(),
    )

    response = service.predict_from_validated_upload(make_upload_metadata(tmp_path))

    assert response.provenance is not None
    preprocessing = response.provenance.preprocessing
    fixture_path = (
        Path(__file__).parent / "fixtures" / "preprocessing_metadata_48khz_stereo.json"
    )
    assert preprocessing.model_dump(mode="json") == json.loads(
        fixture_path.read_text(encoding="utf-8")
    )
    assert preprocessing.input_sample_rate == 48000
    assert preprocessing.input_channels == 2
    assert preprocessing.target_sample_rate == 16000
    assert preprocessing.target_channels == 1
    assert preprocessing.resampled is True
    assert preprocessing.mono_conversion_applied is True
    assert preprocessing.normalisation_applied is True
    assert preprocessing.ffmpeg_version == "ffmpeg version 6.1-test"
    # The frozen research-detector contract (model_artifacts/fusion/final_detector_v1.json)
    # governs fusion_method regardless of dummy/real branch mode -- see
    # FusionEngine.from_settings.
    assert response.provenance.fusion.fusion_method == "simple_average"
    assert response.provenance.fusion.effective_weights == response.fusion.branch_weights
    assert all(model.mode == ModelMode.dummy for model in response.provenance.models)
    assert all(model.research_result is False for model in response.provenance.models)
    assert all(model.checkpoint_id is None for model in response.provenance.models)
    assert all(model.checkpoint_sha256 is None for model in response.provenance.models)


def test_service_includes_storage_failure_metadata(tmp_path) -> None:
    service = make_service(model_registry=make_registry())
    upload_metadata = make_upload_metadata(tmp_path)
    storage_metadata = AudioStorageMetadata(
        status=BranchStatus.failed,
        error="Audio storage failed.",
    )

    response = service.predict_from_validated_upload(
        upload_metadata,
        storage_metadata=storage_metadata,
    )

    assert response.audio.storage == storage_metadata
    assert response.audio.storage.error == "Audio storage failed."


def test_service_uses_same_processed_audio_for_all_branches(tmp_path) -> None:
    registry = make_registry()
    service = make_service(model_registry=registry)

    service.predict_from_validated_upload(make_upload_metadata(tmp_path))

    seen_ids = {model.seen_audio_ids[0] for model in registry.models}
    assert len(seen_ids) == 1


def test_service_preprocessing_failure_stops_request_and_cleans_up(tmp_path) -> None:
    upload_metadata = make_upload_metadata(tmp_path)
    saved_path = upload_metadata.saved_path

    def fail_preprocessing(_upload_metadata):
        raise UnusableAudioError("audio cannot be used")

    service = VoiceService(
        model_registry=make_registry(),
        preprocess_fn=fail_preprocessing,
    )

    with pytest.raises(UnusableAudioError, match="audio cannot be used"):
        service.predict_from_validated_upload(upload_metadata)

    assert not saved_path.exists()


@pytest.mark.parametrize(
    "error",
    [
        CorruptedAudioError("decode failed"),
        AudioProcessingTimeoutError("decode timed out"),
        DecodedAudioTooLargeError("decode exceeded budget"),
    ],
)
def test_service_cleans_up_after_decode_failure(tmp_path, error) -> None:
    upload_metadata = make_upload_metadata(tmp_path)
    saved_path = upload_metadata.saved_path

    def fail_preprocessing(_upload_metadata):
        raise error

    service = VoiceService(
        model_registry=make_registry(),
        preprocess_fn=fail_preprocessing,
    )

    with pytest.raises(type(error)):
        service.predict_from_validated_upload(upload_metadata)

    assert not saved_path.exists()
