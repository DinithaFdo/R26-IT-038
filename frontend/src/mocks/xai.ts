import type {
  SemanticExplanation,
  TemporalExplanation,
  XaiExplanationResponse,
} from "@/types/api";

/**
 * Test fixtures only.
 *
 * These are imported exclusively by tests and are never used as a runtime
 * fallback: shipping demo XAI evidence as a default production response would
 * present fabricated attributions as real model output.
 */

export const temporalFixture: TemporalExplanation = {
  status: "completed",
  method_version: "saved-attention-rollout-v1",
  model_version: "fixture-v1",
  development_placeholder: true,
  research_eligible: false,
  score_kind: "high_attention",
  attention_score_peak: 0.874,
  threshold_percentile: 80,
  attention_threshold: 1.12,
  threshold_crossing_count: 4,
  raw_region_count: 2,
  post_filter_region_count: 2,
  high_attention_region_count: 2,
  high_attention_regions: [
    {
      region_id: 1,
      start_seconds: 0.4,
      end_seconds: 0.9,
      duration_seconds: 0.5,
      attention_score: 0.81,
    },
    {
      region_id: 2,
      start_seconds: 1.6,
      end_seconds: 2.05,
      duration_seconds: 0.45,
      attention_score: 0.74,
    },
  ],
  high_attention_combined_duration_seconds: 0.95,
  artifacts: [],
  warning:
    "Development placeholder generated from deterministic fixture attention. High-attention regions indicate model focus, not confirmed spoof artifacts.",
};

export const semanticFixture: SemanticExplanation = {
  status: "completed",
  target_type: "independent_acoustic_evidence_model",
  model_version: "esvas-xgb-v1",
  feature_schema_version: "esvas-acoustic-28-v1",
  extractor_version: "acoustic-features-v1",
  output_space: "probability",
  development_placeholder: true,
  research_eligible: false,
  base_value: 0.12,
  predicted_value: 0.78,
  feature_importance: [
    {
      rank: 1,
      feature_name: "jitter_local",
      display_name: "Pitch-period jitter",
      value: 0.0031,
      unit: "ratio",
      reference_summary: "ASVspoof2019-LA reference mean 0.0102, standard deviation 0.004.",
      shap_value: 0.184,
      direction: "toward_spoof",
      start_seconds: 2.4,
      end_seconds: 3.1,
    },
    {
      rank: 2,
      feature_name: "hnr_mean",
      display_name: "Harmonic-to-noise ratio",
      value: 21.4,
      unit: "dB",
      reference_summary: null,
      shap_value: -0.092,
      direction: "toward_bonafide",
      start_seconds: null,
      end_seconds: null,
    },
  ],
  windows: [],
  warning: "Whole-clip semantic output does not provide timestamp localization.",
};

export function explanationFixture(
  overrides: Partial<XaiExplanationResponse> = {},
): XaiExplanationResponse {
  return {
    schema_version: "voice-xai-api-v1",
    explanation_id: "explanation-1",
    prediction_id: "prediction-1",
    request_id: "request-1",
    development_placeholder: true,
    research_eligible: false,
    status: "completed",
    component_statuses: {
      temporal: "completed",
      semantic: "completed",
      report: "completed",
      narrative: "not_available",
    },
    component_errors: {
      temporal: null,
      semantic: null,
      report: null,
      narrative: {
        component: "narrative",
        code: "narrative_disabled",
        message: "Narrative generation is disabled for this deployment.",
      },
    },
    classifier_snapshot: {
      verdict: "spoof",
      spoof_probability: 0.8,
      bonafide_probability: 0.2,
      confidence: 0.8,
      decision_threshold: 0.5,
      contains_dummy_branches: true,
      research_eligible: false,
      branches: [],
      auxiliary_evidence: {
        glottal_spoof_probability: 0.61,
        used_for_primary_decision: false,
      },
    },
    temporal: temporalFixture,
    semantic: semanticFixture,
    combined_report: {
      status: "completed",
      authoritative: true,
      disposition: "spoof_suspected",
      requires_human_review: true,
      finding: "Fused classifier output favours spoof.",
      primary_evidence: "Two high-attention regions align with elevated jitter evidence.",
      quality_checks: "Temporal/semantic agreement and surrogate fidelity are available.",
      limitation: "Temporal attention is fixture-backed and not model-derived.",
      recommendation: "Route to a qualified analyst before any downstream action.",
      disclaimer: "This output is decision-support evidence, not proof of identity or fraud.",
    },
    narrative: null,
    quality: {
      temporal_semantic_iou: {
        status: "available",
        value: 0.42,
        scope: "per_analysis",
        reason: null,
        dataset_version: null,
      },
      surrogate_fidelity_r2: {
        status: "available",
        value: 0.6342774342328266,
        scope: "offline_validation",
        reason: null,
        dataset_version: "ASVspoof2019_LA",
      },
    },
    provenance: {
      schema_version: "voice-xai-api-v1",
      pipeline_version: "voice-xai-pipeline-v1",
      classifier_contract_version: "classifier-xai-v1",
      feature_extractor_version: null,
      semantic_model_version: "esvas-xgb-v1",
      semantic_model_hash_short: null,
      temporal_model_version: "fixture-v1",
      temporal_model_hash_short: null,
      configuration_hash: "abc123",
      configuration_hash_inputs: ["xai_mode", "semantic_manifest", "temporal_checkpoint"],
      generated_at: "2026-08-09T10:00:00Z",
    },
    artifacts: [],
    warnings: ["Development placeholder generated from deterministic fixture SHAP values."],
    error: null,
    created_at: "2026-08-09T10:00:00Z",
    updated_at: "2026-08-09T10:00:05Z",
    completed_at: "2026-08-09T10:00:05Z",
    ...overrides,
  };
}
