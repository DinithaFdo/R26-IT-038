"""Mock Phase 3 service: fixture features and SHAP values -> ranked evidence."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Protocol

import numpy as np

from app.ingestion.audio import ProcessedAudio
from app.schemas.xai import SemanticTargetType
from app.voice_xai.contracts import ClassifierInferenceBundle
from app.voice_xai.semantic.features import (
    AcousticFeatureExtractor,
    FEATURE_EXTRACTION_VERSION,
    FEATURE_NAMES,
    FeatureExtractionResult,
)
from app.voice_xai.semantic.contracts import (
    SemanticAnalysisResult,
    SemanticEvidence,
    SemanticExplanationConfig,
    SemanticWindowAnalysisResult,
)
from app.voice_xai.semantic.artifact_loader import (
    LoadedSemanticArtifacts,
    SemanticArtifactCompatibilityError,
    load_production_semantic_artifacts,
    reference_summary_for,
)
from app.voice_xai.semantic.explainer import (
    build_semantic_contributions,
    build_semantic_explanation,
    build_windowed_semantic_explanation,
)
from app.voice_xai.semantic.windows import (
    SemanticWindow,
    SemanticWindowConfig,
    extract_semantic_feature_windows,
    unavailable_window_reason,
)


class SemanticEvidenceProvider(Protocol):
    """Private seam shared by the fixture and XGBoost TreeExplainer providers."""

    def extract(self, inference: ClassifierInferenceBundle) -> SemanticEvidence:
        """Return ordered per-clip feature and attribution values."""

    def extract_features(
        self,
        features: FeatureExtractionResult,
        *,
        evidence_key: str,
    ) -> SemanticEvidence:
        """Return ordered feature and attribution values for one audio window."""


class FixtureSemanticEvidenceProvider:
    """Create deterministic mock SHAP evidence without loading XGBoost or SHAP."""

    def extract(self, inference: ClassifierInferenceBundle) -> SemanticEvidence:
        extraction = inference.extraction
        features = extraction.acoustic_features if extraction is not None else None
        if features is not None:
            return self.extract_features(features, evidence_key=inference.request_id)
        return _fixture_semantic_evidence(
            feature_names=FEATURE_NAMES,
            values=_fixture_feature_values(inference.request_id),
            extractor_version=FEATURE_EXTRACTION_VERSION,
            evidence_key=inference.request_id,
            source="fixture_acoustic_features_and_fixture_shap",
        )

    def extract_features(
        self,
        features: FeatureExtractionResult,
        *,
        evidence_key: str,
    ) -> SemanticEvidence:
        return _fixture_semantic_evidence(
            feature_names=features.feature_names,
            values=features.values,
            extractor_version=features.extractor_version,
            evidence_key=evidence_key,
            source="phase1_acoustic_features_with_fixture_shap",
        )


class MockSemanticExplanationService:
    """Development-only Phase 3 service behind the classifier-to-XAI contract."""

    def __init__(
        self,
        provider: SemanticEvidenceProvider | None = None,
        config: SemanticExplanationConfig | None = None,
        feature_extractor: AcousticFeatureExtractor | None = None,
        window_config: SemanticWindowConfig | None = None,
    ) -> None:
        self._provider = provider or FixtureSemanticEvidenceProvider()
        self._config = config or SemanticExplanationConfig()
        self._feature_extractor = feature_extractor or AcousticFeatureExtractor()
        self._window_config = window_config or SemanticWindowConfig()

    def analyze(self, inference: ClassifierInferenceBundle) -> SemanticAnalysisResult:
        evidence = self._provider.extract(inference)
        explanation = build_semantic_explanation(evidence, self._config)
        return SemanticAnalysisResult(
            explanation=explanation.model_copy(
                update={"development_placeholder": True, "research_eligible": False}
            ),
            warnings=(
                "Development placeholder generated from deterministic fixture SHAP values.",
                unavailable_window_reason(),
            ),
        )

    def analyze_windows(
        self,
        audio: ProcessedAudio,
        *,
        request_id: str,
    ) -> SemanticWindowAnalysisResult:
        return _analyze_windows(
            audio=audio,
            request_id=request_id,
            provider=self._provider,
            config=self._config,
            feature_extractor=self._feature_extractor,
            window_config=self._window_config,
            development_placeholder=True,
            research_eligible=False,
        )


class XgboostShapEvidenceProvider:
    """Read production feature evidence from the approved loaded-once artifacts."""

    def __init__(self, artifacts: LoadedSemanticArtifacts) -> None:
        self._artifacts = artifacts

    def extract(self, inference: ClassifierInferenceBundle) -> SemanticEvidence:
        extraction = inference.extraction
        features = extraction.acoustic_features if extraction is not None else None
        if features is None:
            raise SemanticArtifactCompatibilityError(
                "Production semantic inference requires captured acoustic features."
            )
        return self.extract_features(features, evidence_key="whole-clip")

    def extract_features(
        self,
        features: FeatureExtractionResult,
        *,
        evidence_key: str,
    ) -> SemanticEvidence:
        manifest = self._artifacts.manifest
        if features.feature_names != manifest.feature_names:
            raise SemanticArtifactCompatibilityError(
                "Captured feature names do not match the production manifest."
            )
        if features.extractor_version != manifest.extractor_version:
            raise SemanticArtifactCompatibilityError(
                "Captured extractor version does not match the production manifest."
            )
        raw_values = np.asarray(features.values, dtype=np.float64)
        if raw_values.shape != (len(manifest.feature_names),) or np.isinf(raw_values).any():
            raise SemanticArtifactCompatibilityError(
                "Captured acoustic features must be an ordered v4 feature vector without infinite values."
            )
        missing_feature_count = int(np.isnan(raw_values).sum())
        try:
            values = np.asarray(self._artifacts.imputer.transform(raw_values.reshape(1, -1)), dtype=np.float64)
        except Exception as error:
            raise SemanticArtifactCompatibilityError(
                "The training imputer could not transform the captured acoustic features."
            ) from error
        if values.shape != (1, len(manifest.feature_names)) or not np.isfinite(values).all():
            raise SemanticArtifactCompatibilityError(
                "Training imputation did not produce one finite v4 feature vector."
            )
        row = values
        try:
            import xgboost

            matrix = xgboost.DMatrix(
                row,
                feature_names=list(manifest.feature_names),
            )
            predicted_value = float(
                np.asarray(self._artifacts.booster.predict(matrix, output_margin=True))[0]
            )
            shap_values = np.asarray(self._artifacts.shap_explainer.shap_values(row))
        except Exception as error:
            raise SemanticArtifactCompatibilityError(
                "Production XGBoost or SHAP inference failed."
            ) from error
        shap_values = np.asarray(shap_values, dtype=np.float64).reshape(-1)
        if shap_values.shape != values.shape[1:] or not np.isfinite(shap_values).all():
            raise SemanticArtifactCompatibilityError(
                "SHAP output must contain one finite contribution per feature."
            )
        expected_value = np.asarray(self._artifacts.shap_explainer.expected_value)
        if expected_value.size != 1 or not np.isfinite(expected_value).all():
            raise SemanticArtifactCompatibilityError("SHAP expected value is invalid.")
        base_value = float(expected_value.reshape(-1)[0])
        if not np.isclose(
            base_value + float(shap_values.sum()),
            predicted_value,
            rtol=1e-4,
            atol=1e-4,
        ):
            raise SemanticArtifactCompatibilityError(
                "SHAP contributions do not reconstruct the XGBoost raw margin."
            )
        return SemanticEvidence(
            feature_names=manifest.feature_names,
            feature_values=values.reshape(-1).astype(np.float32),
            shap_values=shap_values.astype(np.float32),
            base_value=base_value,
            predicted_value=predicted_value,
            extractor_version=features.extractor_version,
            source=(
                f"xgboost_shap:{manifest.model_version};"
                f"median_imputed_features={missing_feature_count}"
            ),
            imputed_feature_count=missing_feature_count,
        )


class ProductionSemanticExplanationService:
    """Production semantic inference backed by a startup-loaded XGBoost model."""

    def __init__(
        self,
        artifacts: LoadedSemanticArtifacts,
        *,
        feature_extractor: AcousticFeatureExtractor | None = None,
        window_config: SemanticWindowConfig | None = None,
    ) -> None:
        self._artifacts = artifacts
        manifest = artifacts.manifest
        self._provider = XgboostShapEvidenceProvider(artifacts)
        self._feature_extractor = feature_extractor or AcousticFeatureExtractor()
        self._window_config = window_config or SemanticWindowConfig()
        self._config = SemanticExplanationConfig(
            method_version="xgboost-tree-shap-v4",
            model_version=manifest.model_version,
            feature_schema_version=manifest.feature_schema_version,
            output_space=manifest.output_space,
            decision_threshold=manifest.decision_threshold,
            target_type=SemanticTargetType(manifest.target_type),
            contribution_warning=(
                "Whole-clip feature contributions do not localize evidence in time."
            ),
            reference_summaries={
                name: reference_summary_for(manifest, name)
                for name in manifest.feature_names
            },
        )

    @classmethod
    def load_once(
        cls,
        *,
        manifest_path: Path,
        model_path: Path,
        feature_columns_path: Path,
        threshold_path: Path,
        training_metadata_path: Path,
        imputer_path: Path,
        shap_background_path: Path | None,
        window_config: SemanticWindowConfig | None = None,
    ) -> "ProductionSemanticExplanationService":
        return cls(
            load_production_semantic_artifacts(
                manifest_path=manifest_path,
                model_path=model_path,
                feature_columns_path=feature_columns_path,
                threshold_path=threshold_path,
                training_metadata_path=training_metadata_path,
                imputer_path=imputer_path,
                shap_background_path=shap_background_path,
            ),
            window_config=window_config,
        )

    def analyze(self, inference: ClassifierInferenceBundle) -> SemanticAnalysisResult:
        evidence = self._provider.extract(inference)
        return SemanticAnalysisResult(
            explanation=build_semantic_explanation(evidence, self._config).model_copy(
                update={"development_placeholder": False, "research_eligible": False}
            ),
            development_placeholder=False,
            research_eligible=False,
            warnings=(
                "Whole-clip semantic output does not provide timestamp localization.",
                "The supplied model was trained on ASVspoof2019 LA, ASVspoof5, and WaveFake/LJSpeech; its metadata must be consulted before treating results as benchmark claims.",
            ),
        )

    def analyze_windows(
        self,
        audio: ProcessedAudio,
        *,
        request_id: str,
    ) -> SemanticWindowAnalysisResult:
        return _analyze_windows(
            audio=audio,
            request_id=request_id,
            provider=self._provider,
            config=self._config,
            feature_extractor=self._feature_extractor,
            window_config=self._window_config,
            development_placeholder=False,
            research_eligible=False,
        )


def _fixture_feature_values(request_id: str) -> np.ndarray:
    """Stable non-production values covering the v4 notebook contract."""

    seed = int.from_bytes(
        sha256(f"features:{request_id}".encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    generator = np.random.default_rng(seed)
    values = generator.normal(0.0, 1.0, len(FEATURE_NAMES)).astype(np.float32)
    values[0] = abs(values[0]) / 100.0
    values[1] = abs(values[1]) / 100.0
    values[2] = abs(values[2]) * 10.0
    values[-5] = abs(values[-5]) * 100.0
    return values


def _fixture_semantic_evidence(
    *,
    feature_names: tuple[str, ...],
    values: np.ndarray,
    extractor_version: str,
    evidence_key: str,
    source: str,
) -> SemanticEvidence:
    seed = int.from_bytes(
        sha256(evidence_key.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    )
    generator = np.random.default_rng(seed)
    shap_values = generator.normal(0.0, 0.15, len(FEATURE_NAMES)).astype(np.float32)
    # Ensure the mock includes evidence in both directions for UI coverage.
    shap_values[0] = abs(shap_values[0]) + 0.1
    shap_values[1] = -abs(shap_values[1]) - 0.1
    base_value = float(generator.normal(0.0, 0.1))
    predicted_value = float(base_value + shap_values.sum())
    return SemanticEvidence(
        feature_names=feature_names,
        feature_values=np.asarray(values, dtype=np.float32),
        shap_values=shap_values,
        base_value=base_value,
        predicted_value=predicted_value,
        extractor_version=extractor_version,
        source=source,
    )


def _analyze_windows(
    *,
    audio: ProcessedAudio,
    request_id: str,
    provider: SemanticEvidenceProvider,
    config: SemanticExplanationConfig,
    feature_extractor: AcousticFeatureExtractor,
    window_config: SemanticWindowConfig,
    development_placeholder: bool,
    research_eligible: bool,
) -> SemanticWindowAnalysisResult:
    feature_windows = extract_semantic_feature_windows(
        audio,
        extractor=feature_extractor,
        config=window_config,
    )
    semantic_windows: list[SemanticWindow] = []
    imputed_feature_count = 0
    for window in feature_windows:
        evidence = provider.extract_features(
            window.features,
            evidence_key=(
                f"{request_id}:{window.start_sample}:{window.end_sample}:"
                f"{window_config.version}"
            ),
        )
        imputed_feature_count += evidence.imputed_feature_count
        contributions = build_semantic_contributions(
            evidence,
            config,
            start_seconds=window.start_seconds,
            end_seconds=window.end_seconds,
        )
        semantic_windows.append(
            SemanticWindow(
                start_seconds=window.start_seconds,
                end_seconds=window.end_seconds,
                contributions=tuple(contributions),
                spoof_probability=_spoof_probability(
                    evidence.predicted_value, config.output_space
                ),
            )
        )
    windows = tuple(semantic_windows)
    explanation = build_windowed_semantic_explanation(
        windows,
        config,
        extractor_version=feature_windows[0].features.extractor_version,
    )
    if imputed_feature_count:
        explanation = explanation.model_copy(
            update={
                "warning": (
                    f"{explanation.warning} {imputed_feature_count} unavailable "
                    "feature value(s) were replaced with training median values "
                    "before XGBoost and SHAP."
                )
            }
        )
    explanation = explanation.model_copy(
        update={
            "development_placeholder": development_placeholder,
            "research_eligible": research_eligible,
        }
    )
    return SemanticWindowAnalysisResult(
        explanation=explanation,
        windows=windows,
        development_placeholder=development_placeholder,
        research_eligible=research_eligible,
    )


def _spoof_probability(predicted_value: float, output_space: str) -> float:
    """Map the validated XGBoost binary-logistic output to P(spoof)."""

    if output_space == "probability":
        probability = predicted_value
    elif output_space in {"raw_margin", "log_odds"}:
        if predicted_value >= 0.0:
            probability = 1.0 / (1.0 + np.exp(-predicted_value))
        else:
            exponent = np.exp(predicted_value)
            probability = exponent / (1.0 + exponent)
    else:
        raise SemanticArtifactCompatibilityError("Unsupported semantic output space.")
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise SemanticArtifactCompatibilityError(
            "Semantic model probability is invalid."
        )
    return float(probability)
