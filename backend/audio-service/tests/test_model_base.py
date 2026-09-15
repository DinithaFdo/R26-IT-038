import logging

from app.core.exceptions import ModelInferenceError, ModelLoadError
from app.models.base import BaseVoiceModel
from app.schemas.common import BranchStatus, ModelMode, PredictionLabel
from app.schemas.prediction import BranchPrediction, ProbabilityScores


class SimpleVoiceModel(BaseVoiceModel):
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self.loaded = False

    @property
    def model_name(self) -> str:
        return "test_model"

    @property
    def display_name(self) -> str:
        return "Test Model"

    @property
    def mode(self) -> ModelMode:
        return ModelMode.dummy

    @property
    def is_loaded(self) -> bool:
        return self.loaded

    def load(self) -> None:
        self.loaded = True

    def predict(self, processed_audio) -> BranchPrediction:
        if self.failure is not None:
            raise self.failure
        return BranchPrediction(
            model_name=self.model_name,
            display_name=self.display_name,
            status=BranchStatus.success,
            mode=self.mode,
            prediction=PredictionLabel.bonafide,
            confidence=0.8,
            probabilities=ProbabilityScores(bonafide=0.8, spoof=0.2),
            processing_time_ms=0.0,
        )


def test_predict_safe_returns_standard_success_prediction() -> None:
    model = SimpleVoiceModel()

    prediction = model.predict_safe(processed_audio=object())

    assert prediction.status == BranchStatus.success
    assert prediction.model_name == "test_model"
    assert prediction.display_name == "Test Model"
    assert prediction.mode == ModelMode.dummy
    assert prediction.processing_time_ms >= 0.0


def test_predict_safe_returns_failed_prediction_for_expected_inference_error(
    caplog,
) -> None:
    model = SimpleVoiceModel(
        failure=ModelInferenceError(
            "secret internal file path /tmp/private/model.bin",
            public_message="Branch inference failed.",
        )
    )

    with caplog.at_level(logging.ERROR):
        prediction = model.predict_safe(processed_audio=object())

    assert prediction.status == BranchStatus.failed
    assert prediction.prediction is None
    assert prediction.confidence is None
    assert prediction.probabilities is None
    assert prediction.error == "Branch inference failed."
    assert "secret internal file path" not in prediction.error
    assert "Traceback" not in prediction.error
    assert "secret internal file path" in caplog.text


def test_predict_safe_returns_failed_prediction_for_model_load_error() -> None:
    model = SimpleVoiceModel(
        failure=ModelLoadError(
            "internal checkpoint missing",
            public_message="Branch model is not loaded.",
        )
    )

    prediction = model.predict_safe(processed_audio=object())

    assert prediction.status == BranchStatus.failed
    assert prediction.error == "Branch model is not loaded."


def test_predict_safe_sanitizes_unexpected_errors(caplog) -> None:
    model = SimpleVoiceModel(failure=RuntimeError("database password leaked"))

    with caplog.at_level(logging.ERROR):
        prediction = model.predict_safe(processed_audio=object())

    assert prediction.status == BranchStatus.failed
    assert prediction.error == "Model inference failed."
    assert "database password leaked" not in prediction.error
    assert "Traceback" not in prediction.error
    assert "database password leaked" in caplog.text


def test_base_model_health_reports_common_state() -> None:
    model = SimpleVoiceModel()
    model.load()

    assert model.health() == {
        "model_name": "test_model",
        "display_name": "Test Model",
        "mode": "dummy",
        "is_loaded": True,
    }
