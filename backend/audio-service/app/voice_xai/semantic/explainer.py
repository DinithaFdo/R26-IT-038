"""Build the semantic API response from raw regression and Tree SHAP evidence.

Rank all 165 signed feature contributions, clip the score for the probability
summary, and attach the active model's decision threshold and fidelity metadata.
"""
import numpy as np
from app.schemas.common import PredictionLabel
from app.schemas.xai import ComponentStatus, MetricEvidence, MetricStatus, SemanticClipSummary, SemanticExplanation, SemanticFeatureContribution, SemanticTargetType, ShapDirection
from app.voice_xai.semantic.schema import FEATURE_NAMES, FEATURE_EXTRACTION_VERSION, FEATURE_SCHEMA_VERSION
from app.voice_xai.semantic.feature_labels import label_for_feature

def build_semantic_explanation(evidence, artifacts):
    probability = float(np.clip(evidence.predicted_value, 0, 1))
    contributions = []
    for rank, index in enumerate(np.argsort(-np.abs(evidence.shap_values), kind="stable"), 1):
        name = FEATURE_NAMES[index]
        value = float(evidence.shap_values[index])
        contributions.append(SemanticFeatureContribution(
            rank=rank, feature_name=name, display_name=label_for_feature(name).display_name,
            value=float(evidence.feature_values[index]), shap_value=value,
            direction=ShapDirection.toward_spoof if value > 0 else ShapDirection.toward_bonafide if value < 0 else ShapDirection.neutral,
        ))
    fidelity = artifacts.fidelity()
    metric = MetricEvidence(status=MetricStatus.available, value=fidelity[0], scope="offline_validation", dataset_version=f"Current model test export (n={fidelity[1]})") if fidelity else MetricEvidence(status=MetricStatus.not_computed, scope="offline_validation", reason="This model bundle has no valid test fidelity metadata.")
    warning = "Tree SHAP explains the surrogate raw score before clipping. It approximates the teacher and provides no time localisation."
    if evidence.imputed_feature_count:
        warning += f" {evidence.imputed_feature_count} missing features used training medians."
    return SemanticExplanation(
        status=ComponentStatus.completed, target_type=SemanticTargetType.classifier_surrogate,
        model_version=artifacts.model_version, feature_schema_version=FEATURE_SCHEMA_VERSION,
        extractor_version=FEATURE_EXTRACTION_VERSION, output_space="raw_regression",
        development_placeholder=False, research_eligible=False,
        base_value=evidence.base_value, predicted_value=evidence.predicted_value,
        clip_summary=SemanticClipSummary(spoof_probability=probability, bonafide_probability=1-probability,
            decision_threshold=artifacts.threshold, predicted_label=PredictionLabel.spoof if probability >= artifacts.threshold else PredictionLabel.bonafide),
        feature_importance=contributions, fidelity_r2=metric, warning=warning,
    )
