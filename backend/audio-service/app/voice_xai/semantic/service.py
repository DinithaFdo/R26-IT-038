"""Run clip-level surrogate inference and produce additive Tree SHAP evidence.

Extract or accept acoustic features, apply the active model's median imputer,
predict the raw regression score, and verify that SHAP reconstructs that score.
"""
from pathlib import Path
from dataclasses import replace
from time import monotonic, perf_counter
import logging
import numpy as np
from app.schemas.xai import SemanticWindowExplanation
from app.voice_xai.semantic.artifact_loader import SemanticArtifactCompatibilityError, load_production_semantic_artifacts, model_fingerprint
from app.voice_xai.semantic.contracts import (
    SemanticAnalysisResult, SemanticEvidence, SemanticWindowOptions,
    WINDOW_EXTRACTION_VERSION, WINDOW_PREPROCESSING_CONTRACT,
)
from app.voice_xai.semantic.feature_extraction import AcousticFeatureExtractor, build_semantic_windows
from app.voice_xai.semantic.schema import FEATURE_NAMES, FEATURE_EXTRACTION_VERSION
from app.voice_xai.semantic.explainer import build_semantic_explanation, build_window_explanation, build_semantic_localization
from app.voice_xai.evaluation.intervals import union_spans

logger = logging.getLogger(__name__)

class XgboostShapEvidenceProvider:
    def __init__(self, artifacts):
        self.artifacts = artifacts

    # Preserve the strict full-clip extraction contract for existing predictions.
    def extract_features(self, features):
        if features.feature_names != FEATURE_NAMES or features.extractor_version != FEATURE_EXTRACTION_VERSION:
            raise SemanticArtifactCompatibilityError("Feature extraction does not match the semantic model contract.")
        return self._evaluate(features)

    # Explicitly accept window-scoped inputs without weakening the full-clip guard.
    def extract_window_features(self, features):
        if (features.feature_names != FEATURE_NAMES or features.extractor_version != WINDOW_EXTRACTION_VERSION
                or features.metadata.get('analysis_scope') != 'window'
                or any(features.metadata.get(key) != value for key, value in WINDOW_PREPROCESSING_CONTRACT.items())):
            raise SemanticArtifactCompatibilityError('Feature extraction does not match the semantic window contract.')
        return self._evaluate(features)

    # Apply the frozen imputer/model and verify additive SHAP for one measured input.
    def _evaluate(self, features):
        import pandas as pd
        import xgboost
        raw = np.asarray(features.values, dtype=np.float64)
        if raw.shape != (len(FEATURE_NAMES),):
            raise SemanticArtifactCompatibilityError("Expected 165 feature values.")
        raw = np.where(np.isfinite(raw), raw, np.nan)
        values = np.asarray(self.artifacts.imputer.transform(pd.DataFrame([raw], columns=FEATURE_NAMES)), dtype=np.float64)
        if values.shape != (1, len(FEATURE_NAMES)) or not np.isfinite(values).all():
            raise SemanticArtifactCompatibilityError("Imputation did not produce 165 finite values.")
        matrix = xgboost.DMatrix(values, feature_names=list(FEATURE_NAMES))
        score = float(self.artifacts.booster.predict(matrix, output_margin=True)[0])
        shap = np.asarray(self.artifacts.shap_explainer.shap_values(values, check_additivity=True), dtype=np.float64).reshape(-1)
        expected = np.asarray(self.artifacts.shap_explainer.expected_value).reshape(-1)
        if expected.size != 1 or shap.shape != (len(FEATURE_NAMES),):
            raise SemanticArtifactCompatibilityError("Unexpected Tree SHAP output shape.")
        base = float(expected[0])
        evidence = SemanticEvidence(values[0], shap, base, score, int(np.isnan(raw).sum()))
        if abs(base + float(shap.sum()) - score) > 1e-4:
            raise SemanticArtifactCompatibilityError("Tree SHAP does not reconstruct the raw regressor score.")
        return evidence

class ProductionSemanticExplanationService:
    """Keep one validated model in memory until the backend restarts."""
    def __init__(self, artifacts, *, feature_extractor=None, window_options=None):
        self.artifacts = artifacts
        self._provider = XgboostShapEvidenceProvider(artifacts)
        self._feature_extractor = feature_extractor or AcousticFeatureExtractor()
        self.window_options = window_options or SemanticWindowOptions()

    @classmethod
    def load_once(cls, *, model_dir: Path, window_options=None):
        return cls(load_production_semantic_artifacts(model_dir), window_options=window_options)

    # Keep full-clip output and optionally add localization over independently supplied coverage.
    def analyze(self, inference, *, analyzed_spans=None, deadline=None):
        try:
            current = model_fingerprint(self.artifacts.directory)
        except SemanticArtifactCompatibilityError as error:
            raise SemanticArtifactCompatibilityError("Model files changed or are incomplete. Restart after replacing the complete model export.") from error
        if current != self.artifacts.fingerprint:
            raise SemanticArtifactCompatibilityError("Model files changed. Restart the backend before creating explanations.")
        if analyzed_spans is None:
            analyzed_spans = [(window.start_seconds, window.end_seconds)
                              for window in (getattr(inference, 'temporal_evidence', None) or ())]
        analyzed_spans = union_spans(analyzed_spans)
        wants_windows = self.window_options.enabled and bool(analyzed_spans)
        audio = getattr(inference, 'processed_audio', None)
        features = inference.acoustic_features
        measured_full_clip = features is None
        prepared = None
        preparation_reason = None
        if wants_windows and audio is not None and audio.source_audio_bytes:
            try:
                prepared = self._feature_extractor.prepare_processed_audio(audio)
            except Exception:
                if features is None:
                    raise
                logger.exception('semantic_window_preparation_failed')
                preparation_reason = 'Original audio could not be prepared for window analysis.'
        if features is None:
            if audio is None:
                raise SemanticArtifactCompatibilityError("Semantic inference requires original audio or extracted features.")
            features = (self._feature_extractor.extract_prepared_audio(prepared) if prepared is not None
                        else self._feature_extractor.extract_processed_audio(audio))
        evidence = self._provider.extract_features(features)
        explanation = build_semantic_explanation(evidence, self.artifacts)
        if wants_windows:
            explanation.localization = self._analyze_windows(
                prepared, analyzed_spans, deadline=deadline, preparation_reason=preparation_reason,
                full_features=features if measured_full_clip and prepared is not None else None)
            if explanation.localization.model_validation_status == 'experimental':
                explanation.warning += ' Window localization uses an experimental application of the full-clip surrogate.'
        return SemanticAnalysisResult(explanation)

    # Analyze each unpadded window, containing failures and checking cooperative deadlines.
    def _analyze_windows(self, prepared, analyzed_spans, *, deadline, preparation_reason=None, full_features=None):
        started = perf_counter()
        options = self.window_options
        if prepared is None:
            return build_semantic_localization([], analyzed_spans, options=options, artifacts=self.artifacts,
                                               reason=preparation_reason or 'Original source audio is required for semantic windows.')
        try:
            plan = build_semantic_windows(prepared, analyzed_spans, window_seconds=options.window_seconds,
                                          hop_seconds=options.hop_seconds, max_windows=options.max_windows)
        except ValueError as error:
            return build_semantic_localization([], analyzed_spans, options=options, artifacts=self.artifacts,
                                               trim_start_sample=prepared.trim_start_sample, reason=str(error))
        stop_at = min(deadline if deadline is not None else float('inf'), monotonic() + options.budget_seconds)
        windows = []
        budget_exhausted = False
        for index, window in enumerate(plan, 1):
            window_started = perf_counter()
            reason = None
            result = None
            if budget_exhausted or monotonic() >= stop_at:
                budget_exhausted = True
                reason = 'window_budget_exhausted'
            else:
                try:
                    # The exact complete cleaned slice has already been measured.
                    # Reuse only a vector extracted from this prepared source in this call.
                    if (full_features is not None and window.start_sample == prepared.trim_start_sample
                            and window.end_sample == prepared.trim_end_sample):
                        features = replace(full_features, extractor_version=WINDOW_EXTRACTION_VERSION,
                                           metadata={**WINDOW_PREPROCESSING_CONTRACT, 'analysis_scope': 'window',
                                                     'start_sample': window.start_sample,
                                                     'end_sample_exclusive': window.end_sample,
                                                     'feature_reuse': 'exact_full_cleaned_waveform'})
                    else:
                        features = self._feature_extractor.extract_prepared_window(prepared, window)
                    if monotonic() >= stop_at:
                        budget_exhausted = True
                        reason = 'window_budget_exhausted'
                    else:
                        evidence = self._provider.extract_window_features(features)
                        if monotonic() >= stop_at:
                            budget_exhausted = True
                            reason = 'window_budget_exhausted'
                        else:
                            result = build_window_explanation(window, evidence, window_id=index,
                                                              sample_rate=prepared.sample_rate,
                                                              processing_seconds=perf_counter() - window_started)
                except Exception:
                    logger.exception('semantic_window_extraction_failed', extra={'window_id': index})
                    reason = 'window_extraction_or_inference_failed'
            windows.append(result or SemanticWindowExplanation(
                window_id=index, start_sample=window.start_sample, end_sample_exclusive=window.end_sample,
                start_seconds=window.start_sample / prepared.sample_rate, end_seconds=window.end_sample / prepared.sample_rate,
                status='not_available', reason=reason, processing_seconds=perf_counter() - window_started,
            ))
        return build_semantic_localization(
            windows, analyzed_spans, options=options, artifacts=self.artifacts,
            sample_rate=prepared.sample_rate, trim_start_sample=prepared.trim_start_sample,
            reason='Semantic window budget was exhausted.' if budget_exhausted else None,
            processing_seconds=perf_counter() - started,
        )
