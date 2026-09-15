from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.ingestion.audio import ProcessedAudio
from app.schemas.common import PredictionLabel
from app.schemas.xai import SemanticFeatureContribution, ShapDirection
from app.voice_xai.semantic.contracts import SemanticExplanationConfig
from app.voice_xai.semantic.explainer import build_windowed_semantic_explanation
from app.voice_xai.semantic.features import (
    FEATURE_EXTRACTION_VERSION,
    FEATURE_NAMES,
    FeatureExtractionResult,
)
from app.voice_xai.semantic.artifact_loader import (
    LoadedSemanticArtifacts,
    SemanticArtifactManifest,
)
from app.voice_xai.semantic.service import (
    MockSemanticExplanationService,
    ProductionSemanticExplanationService,
    XgboostShapEvidenceProvider,
)
from app.voice_xai.semantic.windows import SemanticWindow, SemanticWindowConfig

MANIFEST_PATH = (
    Path(__file__).parents[1]
    / "app"
    / "voice_xai"
    / "semantic"
    / "manifests"
    / "xgboost-surrogate-v4.json"
)


def test_production_provider_uses_ordered_features_and_raw_margin(monkeypatch) -> None:
    class FakeDMatrix:
        def __init__(self, values, *, feature_names):
            self.values = values
            self.feature_names = feature_names

    class FakeBooster:
        def predict(self, matrix, *, output_margin):
            assert output_margin is True
            assert matrix.feature_names == list(FEATURE_NAMES)
            return np.asarray([sum(range(len(FEATURE_NAMES))) + 0.11])

    class FakeExplainer:
        expected_value = 0.11

        def shap_values(self, row):
            assert row.shape == (1, len(FEATURE_NAMES))
            return np.arange(len(FEATURE_NAMES), dtype=np.float64).reshape(1, -1)

    monkeypatch.setitem(
        __import__("sys").modules,
        "xgboost",
        SimpleNamespace(DMatrix=FakeDMatrix),
    )
    manifest = replace(SemanticArtifactManifest.from_file(MANIFEST_PATH), feature_names=FEATURE_NAMES)
    provider = XgboostShapEvidenceProvider(
        LoadedSemanticArtifacts(
            manifest=manifest,
            booster=FakeBooster(),
            shap_explainer=FakeExplainer(),
            imputer=FakeImputer(),
        )
    )
    features = SimpleNamespace(
        feature_names=FEATURE_NAMES,
        values=np.arange(len(FEATURE_NAMES), dtype=np.float32),
        extractor_version=FEATURE_EXTRACTION_VERSION,
    )
    inference = SimpleNamespace(
        extraction=SimpleNamespace(acoustic_features=features),
    )

    evidence = provider.extract(inference)

    assert evidence.feature_names == FEATURE_NAMES
    assert evidence.predicted_value == sum(range(len(FEATURE_NAMES))) + 0.11
    assert evidence.base_value == 0.11
    assert evidence.shap_values.tolist() == list(range(len(FEATURE_NAMES)))


def test_production_provider_applies_training_median_imputer_before_xgboost(monkeypatch) -> None:
    class FakeDMatrix:
        def __init__(self, values, *, feature_names):
            assert values[0, 0] == 123.0

    class FakeBooster:
        def predict(self, _matrix, *, output_margin):
            assert output_margin is True
            return np.asarray([0.0])

    class FakeExplainer:
        expected_value = 0.0

        def shap_values(self, row):
            return np.zeros_like(row)

    class MedianImputer:
        def transform(self, row):
            assert np.isnan(row[0, 0])
            return np.nan_to_num(row, nan=123.0)

    monkeypatch.setitem(__import__("sys").modules, "xgboost", SimpleNamespace(DMatrix=FakeDMatrix))
    manifest = replace(SemanticArtifactManifest.from_file(MANIFEST_PATH), feature_names=FEATURE_NAMES)
    provider = XgboostShapEvidenceProvider(LoadedSemanticArtifacts(
        manifest=manifest, booster=FakeBooster(), shap_explainer=FakeExplainer(), imputer=MedianImputer(),
    ))
    features = SimpleNamespace(feature_names=FEATURE_NAMES, values=np.asarray([np.nan, *range(1, len(FEATURE_NAMES))], dtype=np.float32), extractor_version=FEATURE_EXTRACTION_VERSION)

    evidence = provider.extract(SimpleNamespace(extraction=SimpleNamespace(acoustic_features=features)))

    assert evidence.feature_values[0] == 123.0
    assert "median_imputed_features=1" in evidence.source


def test_mock_service_returns_deterministic_timestamped_windowed_shap() -> None:
    service = MockSemanticExplanationService(
        feature_extractor=StubFeatureExtractor(),
        window_config=SemanticWindowConfig(duration_seconds=1.0, overlap_seconds=0.5)
    )
    audio = _processed_audio(2.0)

    first = service.analyze_windows(audio, request_id="request-123")
    second = service.analyze_windows(audio, request_id="request-123")

    assert first.development_placeholder is True
    assert [(window.start_seconds, window.end_seconds) for window in first.windows] == [
        (0.0, 1.0),
        (0.5, 1.5),
        (1.0, 2.0),
    ]
    assert all(len(window.contributions) == 10 for window in first.windows)
    assert all(
        contribution.start_seconds == window.start_seconds
        and contribution.end_seconds == window.end_seconds
        for window in first.windows
        for contribution in window.contributions
    )
    assert first.explanation.base_value is None
    assert first.explanation.predicted_value is None
    assert first.explanation.clip_summary is None
    assert all(
        contribution.start_seconds is not None
        for contribution in first.explanation.feature_importance
    )
    assert first.explanation.model_dump(mode="json") == second.explanation.model_dump(
        mode="json"
    )


def test_production_service_runs_xgboost_and_shap_for_each_window(monkeypatch) -> None:
    class FakeDMatrix:
        def __init__(self, values, *, feature_names):
            self.values = values
            self.feature_names = feature_names

    class FakeBooster:
        calls = 0

        def predict(self, matrix, *, output_margin):
            assert output_margin is True
            assert matrix.feature_names == list(FEATURE_NAMES)
            self.calls += 1
            return np.asarray([sum(range(len(FEATURE_NAMES))) + 0.11])

    class FakeExplainer:
        expected_value = 0.11
        calls = 0

        def shap_values(self, row):
            assert row.shape == (1, len(FEATURE_NAMES))
            self.calls += 1
            return np.arange(len(FEATURE_NAMES), dtype=np.float64).reshape(1, -1)

    monkeypatch.setitem(
        __import__("sys").modules,
        "xgboost",
        SimpleNamespace(DMatrix=FakeDMatrix),
    )
    manifest = replace(SemanticArtifactManifest.from_file(MANIFEST_PATH), feature_names=FEATURE_NAMES)
    booster = FakeBooster()
    explainer = FakeExplainer()
    service = ProductionSemanticExplanationService(
        LoadedSemanticArtifacts(
            manifest=manifest,
            booster=booster,
            shap_explainer=explainer,
            imputer=FakeImputer(),
        ),
        feature_extractor=StubFeatureExtractor(),
        window_config=SemanticWindowConfig(duration_seconds=1.0, overlap_seconds=0.5),
    )

    result = service.analyze_windows(_processed_audio(1.5), request_id="request-123")

    assert len(result.windows) == 2
    assert booster.calls == 2
    assert explainer.calls == 2
    assert result.development_placeholder is False
    assert result.research_eligible is False
    assert {item.start_seconds for item in result.explanation.feature_importance} == {
        0.0,
        0.5,
    }
    assert result.explanation.clip_summary is not None
    assert result.explanation.clip_summary.spoof_probability == 1.0
    assert result.explanation.clip_summary.decision_threshold == pytest.approx(
        manifest.decision_threshold
    )
    assert result.explanation.clip_summary.predicted_label == PredictionLabel.spoof


def test_clip_summary_uses_coverage_weights_and_the_trained_threshold() -> None:
    explanation = build_windowed_semantic_explanation(
        (
            _semantic_window(0.0, 1.0, 0.1),
            _semantic_window(0.5, 1.5, 0.9),
            _semantic_window(1.0, 2.0, 0.3),
        ),
        SemanticExplanationConfig(
            output_space="raw_margin",
            decision_threshold=0.3,
        ),
        extractor_version=FEATURE_EXTRACTION_VERSION,
    )

    assert explanation.clip_summary is not None
    # Midpoint coverage is 0.75 s, 0.50 s, 0.75 s: overlapped audio is not
    # counted twice and the short middle support is not over-weighted.
    assert explanation.clip_summary.spoof_probability == pytest.approx(0.375)
    assert explanation.clip_summary.bonafide_probability == pytest.approx(0.625)
    assert explanation.clip_summary.predicted_label == PredictionLabel.spoof
    assert explanation.clip_summary.window_count == 3


def _semantic_window(
    start_seconds: float,
    end_seconds: float,
    spoof_probability: float,
) -> SemanticWindow:
    contribution = SemanticFeatureContribution(
        rank=1,
        feature_name=FEATURE_NAMES[0],
        display_name="Test feature",
        value=1.0,
        shap_value=0.1,
        direction=ShapDirection.toward_spoof,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
    )
    return SemanticWindow(
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        contributions=(contribution,),
        spoof_probability=spoof_probability,
    )


def _processed_audio(seconds: float) -> ProcessedAudio:
    sample_rate = 16_000
    times = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    waveform = (0.5 * np.sin(2 * np.pi * 180 * times)).astype(np.float32)
    return ProcessedAudio(
        waveform=waveform,
        sample_rate=sample_rate,
        original_sample_rate=sample_rate,
        original_channels=1,
        duration_seconds=seconds,
        was_resampled=False,
        was_converted_to_mono=False,
        normalisation_applied=True,
        peak_amplitude=0.5,
        rms_energy=float(np.sqrt(np.mean(waveform**2))),
    )


class StubFeatureExtractor:
    """Keeps service tests hermetic while windowing is tested separately."""

    config = SimpleNamespace(sample_rate=16_000, frame_length_samples=400)

    def extract_waveform(self, waveform, *, sample_rate, metadata):
        return FeatureExtractionResult(
            feature_names=FEATURE_NAMES,
            values=np.arange(len(FEATURE_NAMES), dtype=np.float32),
            sample_rate=sample_rate,
            duration_seconds=len(waveform) / sample_rate,
            extractor_version=FEATURE_EXTRACTION_VERSION,
            metadata=metadata,
        )


class FakeImputer:
    def transform(self, values):
        return np.nan_to_num(values, nan=0.0)
