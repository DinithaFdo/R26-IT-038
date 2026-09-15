import numpy as np
import pytest

from app.core.exceptions import ModelLoadError
from app.models.aasist_model import AASISTVoiceModel
from app.models.cnn_model import CNNVoiceModel
from app.models.glottal_model import GlottalVoiceModel
from app.models.ssl_model import SSLVoiceModel
from app.ingestion.audio import ProcessedAudio
from app.schemas.common import BranchStatus, ModelMode


MODEL_CLASSES = [
    CNNVoiceModel,
    AASISTVoiceModel,
    SSLVoiceModel,
    GlottalVoiceModel,
]


def make_processed_audio(frequency: float = 440.0) -> ProcessedAudio:
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    waveform = (0.5 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    peak_amplitude = float(np.max(np.abs(waveform)))
    rms_energy = float(np.sqrt(np.mean(np.square(waveform, dtype=np.float32))))
    return ProcessedAudio(
        waveform=waveform,
        sample_rate=sample_rate,
        original_sample_rate=sample_rate,
        original_channels=1,
        duration_seconds=1.0,
        was_resampled=False,
        was_converted_to_mono=False,
        normalisation_applied=False,
        peak_amplitude=peak_amplitude,
        rms_energy=rms_energy,
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_dummy_prediction_is_deterministic_for_repeated_input(model_class) -> None:
    model = model_class(mode="dummy")
    processed_audio = make_processed_audio()

    first = model.predict(processed_audio)
    second = model.predict(processed_audio)

    assert first == second


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_different_input_can_change_dummy_probability(model_class) -> None:
    model = model_class(mode="dummy")

    first = model.predict(make_processed_audio(frequency=220.0))
    second = model.predict(make_processed_audio(frequency=880.0))

    assert first.probabilities != second.probabilities


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_dummy_probabilities_are_valid(model_class) -> None:
    model = model_class(mode="dummy")

    prediction = model.predict(make_processed_audio())

    assert prediction.status == BranchStatus.success
    assert prediction.probabilities is not None
    assert 0.0 <= prediction.probabilities.bonafide <= 1.0
    assert 0.0 <= prediction.probabilities.spoof <= 1.0
    assert prediction.probabilities.bonafide + prediction.probabilities.spoof == (
        pytest.approx(1.0)
    )
    assert prediction.confidence == pytest.approx(
        max(prediction.probabilities.bonafide, prediction.probabilities.spoof)
    )


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_dummy_mode_is_clearly_marked(model_class) -> None:
    model = model_class(mode="dummy")

    prediction = model.predict(make_processed_audio())

    assert prediction.mode == ModelMode.dummy
    assert prediction.metadata["development_placeholder"] is True
    assert prediction.metadata["research_result"] is False
    assert prediction.metadata["model_provenance"]["mode"] == "dummy"
    assert prediction.metadata["model_provenance"]["research_result"] is False


def test_dummy_branches_use_different_deterministic_methods() -> None:
    methods = {
        model_class(mode="dummy")
        .predict(make_processed_audio())
        .metadata["dummy_method"]
        for model_class in MODEL_CLASSES
    }

    assert len(methods) == len(MODEL_CLASSES)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_real_mode_fails_clearly(model_class) -> None:
    model = model_class(mode="real")

    with pytest.raises(
        ModelLoadError,
        match="Real model integration is not implemented",
    ):
        model.load()

    with pytest.raises(
        ModelLoadError,
        match="Real model integration is not implemented",
    ):
        model.predict(make_processed_audio())
