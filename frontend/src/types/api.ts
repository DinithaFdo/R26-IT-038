/* ---------------------------------------------------------------------------
 * Audio & Deepfake Detection
 * ------------------------------------------------------------------------ */

export type PredictionLabel = "spoof" | "bonafide";
export type BranchStatus = "success" | "failed" | "skipped" | string;

/**
 * `disabled` means the branch has no trained model in this deployment and was
 * never executed — distinct from `failed`, which is a branch that should have
 * worked. The trailing `string` keeps rendering tolerant of modes the backend
 * adds later.
 */
export type ModelMode = "dummy" | "real" | "disabled" | string;

export type SourceType =
  | "dashboard_upload"
  | "live_recording"
  | "public_api"
  | "mcp"
  | string;

export type PredictionStatus =
  | "queued"
  | "validating"
  | "storing"
  | "processing"
  | "completed"
  | "failed"
  | "deleting"
  | "deleted"
  | string;

export type ProbabilityScores = {
  bonafide: number;
  spoof: number;
};

export type BranchPrediction = {
  model_name: string;
  display_name: string;
  status: BranchStatus;
  mode: ModelMode;
  prediction?: PredictionLabel | null;
  confidence?: number | null;
  probabilities?: ProbabilityScores | null;
  processing_time_ms: number;
  error?: string | null;
  metadata: Record<string, unknown>;
};

export type FusionSystemStage = "partial" | "full" | string;

export type FusionResearchBlocker =
  | "dummy_branch_contributed"
  | "incomplete_branch_set"
  | "unverified_branch_contributed"
  | string;

export type FusionResult = {
  status: BranchStatus;
  prediction?: PredictionLabel | null;
  confidence?: number | null;
  probabilities?: ProbabilityScores | null;
  method: string;
  /**
   * The threshold actually applied to `probabilities.spoof` to produce
   * `prediction` (`spoof` iff `probabilities.spoof >= decision_threshold`).
   * Always populated by the backend, including on a failed result -- read
   * this instead of assuming any fixed value in the UI.
   */
  decision_threshold?: number;
  branch_weights: Record<string, number>;
  contains_dummy_branches: boolean;
  eligible_for_research_evaluation: boolean;
  warning?: string | null;
  config_version?: string | null;
  minimum_successful_branches?: number | null;
  contributing_branches: string[];
  excluded_branches: Record<string, string>;
  /**
   * `partial` means fewer than the full designed branch set contributed, so
   * the score must not be read as full-system performance. Optional because
   * older persisted predictions predate the field.
   */
  system_stage?: FusionSystemStage | null;
  /** Machine-readable reasons the result is not research eligible. */
  research_blockers?: FusionResearchBlocker[] | null;
  /**
   * Which fusion contract actually produced this result, e.g.
   * `"fusion-convex-4branch-v3"` or `"legacy-3branch-frozen-v1"`.
   */
  fusion_version?: string;
  /** Coarse fusion strategy label, e.g. `"learned_constrained"`, `"legacy_average"`. */
  fusion_mode?: string;
  /**
   * True only when the primary four-branch detector was configured/available
   * but fell back to the legacy detector for this request.
   */
  fallback_used?: boolean;
};

export type PredictionHistoryAudioMetadata = {
  original_filename?: string | null;
  original_extension?: string | null;
  detected_container?: string | null;
  detected_codec?: string | null;
  duration_seconds?: number | null;
  sample_rate?: number | null;
  channels?: number | null;
  size_bytes?: number | null;
  storage_status?: string | null;
  playback_available: boolean;
};

export type PredictionModeSummary = {
  dummy: number;
  real: number;
  contains_dummy: boolean;
};

export type PredictionHistoryItem = {
  prediction_id: string;
  filename?: string | null;
  source_type: SourceType;
  status: PredictionStatus;
  duration_seconds?: number | null;
  final_prediction?: PredictionLabel | null;
  confidence?: number | null;
  mode_summary: PredictionModeSummary;
  created_at: string;
};

export type PredictionHistoryResponse = {
  items: PredictionHistoryItem[];
  page: number;
  limit: number;
  has_next: boolean;
};

export type PredictionDetailResponse = {
  prediction_id: string;
  request_id: string;
  source_type: SourceType;
  status: PredictionStatus;
  audio: PredictionHistoryAudioMetadata;
  branches: BranchPrediction[];
  fusion?: FusionResult | null;
  preprocessing: Record<string, unknown>;
  total_processing_time_ms?: number | null;
  warnings: string[];
  research_eligible: boolean;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
};

export type PredictionSubmissionResponse = {
  prediction_id: string;
  request_id: string;
  status: PredictionStatus;
  source_type: SourceType;
  audio: PredictionHistoryAudioMetadata;
  branches: BranchPrediction[];
  fusion?: FusionResult | null;
  research_eligible: boolean;
  created_at: string;
};

export type PredictionJobStatusResponse = {
  prediction_id: string;
  request_id: string;
  status: PredictionStatus;
  source_type: SourceType;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
  error_summary: Record<string, unknown>[];
};

export type PredictionAudioPlaybackResponse = {
  playback_url: string;
  expires_in_seconds: number;
};

export type PredictionDeleteResponse = {
  prediction_id: string;
  status: PredictionStatus;
};

export type ReadinessResponse = {
  status: string;
  prediction_ready: boolean;
  research_ready: boolean;
  ffmpeg_available: boolean;
  ffprobe_available: boolean;
  mongodb_configured: boolean;
  mongodb_available: boolean;
  storage_enabled: boolean;
  storage_available: boolean;
  components: Record<string, unknown>;
};

export type ModelHealth = {
  branch_name?: string | null;
  model_name: string;
  display_name: string;
  adapter_type?: string | null;
  mode: ModelMode;
  is_loaded: boolean;
  lifecycle_state?: string | null;
  ready?: boolean | null;
  research_ready?: boolean | null;
  /** Operator attestation that this branch's feature pipeline matches training. */
  preprocessing_verified?: boolean | null;
  /** Operator attestation that the logit-to-label mapping matches training. */
  class_mapping_verified?: boolean | null;
  requested_device?: string | null;
  resolved_device?: string | null;
  fallback_used?: boolean | null;
  precision?: string | null;
  checkpoint_configured?: boolean | null;
  checkpoint_valid?: boolean | null;
  checkpoint_hash_short?: string | null;
  model_version?: string | null;
  architecture?: string | null;
  preprocessing_version?: string | null;
  last_error_code?: string | null;
  last_load_error_code?: string | null;
  last_inference_error_code?: string | null;
  last_transition_at?: string | null;
  branch_timeout_seconds?: number | null;
  uses_dummy_mode: boolean;
  warning?: string | null;
};

export type BackendErrorResponse = {
  request_id?: string;
  error?: {
    code?: string;
    message?: string;
    details?: unknown;
  };
  detail?: unknown;
};

export type PredictionFilters = {
  page: number;
  limit: number;
  status?: string;
  sourceType?: string;
  predictionLabel?: string;
  createdFrom?: string;
  createdTo?: string;
  search?: string;
};

/* ---------------------------------------------------------------------------
 * Voice XAI
 * ------------------------------------------------------------------------ */

export type ExplanationStatus =
  | "queued"
  | "running"
  | "partial"
  | "completed"
  | "failed"
  | "blocked";

export type NarrativeStatus =
  | "not_available"
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "blocked";

export type ComponentStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "blocked"
  | "not_available";

export type MetricStatus = "available" | "not_computed" | "not_applicable";

export type ShapDirection = "toward_spoof" | "toward_bonafide";

export type ReportDisposition = "spoof_suspected" | "bonafide_suspected" | "inconclusive";

export type SemanticTargetType =
  | "classifier_surrogate"
  | "independent_acoustic_evidence_model";

export type CanonicalXaiBranch = "lfcc_cnn_tcn" | "aasist" | "ssl_sequence" | "glottal";

export type ExplanationComponentName = "temporal" | "semantic" | "report" | "narrative";

export type ArtifactKind =
  | "temporal_evidence"
  | "attention_visualization"
  | "attention_spectrogram"
  | "attention_tensor"
  | "intermediate_representations"
  | "semantic_values"
  | "report"
  | "other";

export type ExplanationArtifactReference = {
  artifact_id: string;
  kind: ArtifactKind;
  content_type: string;
  size_bytes: number;
  sha256: string;
  expires_at?: string | null;
  /**
   * Always null in the current backend. Treated as opaque and never used to
   * build a URL: artifacts are fetched through the authenticated
   * `/explanation/artifacts/{artifact_id}` route by ID only.
   */
  download_path?: string | null;
};

export type ExplanationError = {
  component?: ExplanationComponentName | null;
  code: string;
  message: string;
};

export type ExplanationComponentStatuses = {
  temporal: ComponentStatus;
  semantic: ComponentStatus;
  report: ComponentStatus;
  narrative?: NarrativeStatus;
};

export type ExplanationComponentErrors = {
  temporal?: ExplanationError | null;
  semantic?: ExplanationError | null;
  report?: ExplanationError | null;
  narrative?: ExplanationError | null;
};

export type ClassifierBranchSnapshot = {
  branch_name: CanonicalXaiBranch;
  model_name: string;
  status: BranchStatus;
  mode: ModelMode;
  spoof_probability?: number | null;
};

export type ClassifierAuxiliaryEvidence = {
  glottal_spoof_probability?: number | null;
  used_for_primary_decision: boolean;
};

export type ClassifierSnapshot = {
  verdict?: PredictionLabel | null;
  spoof_probability?: number | null;
  bonafide_probability?: number | null;
  confidence?: number | null;
  decision_threshold: number;
  contains_dummy_branches: boolean;
  research_eligible: boolean;
  branches: ClassifierBranchSnapshot[];
  auxiliary_evidence?: ClassifierAuxiliaryEvidence;
};

export type TemporalRegion = {
  region_id: number;
  start_seconds: number;
  start_sec?: number;
  end_seconds: number;
  end_sec?: number;
  duration_seconds?: number | null;
  duration_sec?: number | null;
  attention_score: number;
  score?: number;
  peak_score?: number;
};

export type TemporalExplanation = {
  status: ComponentStatus;
  method_version: string;
  model_version?: string | null;
  development_placeholder: boolean;
  research_eligible: boolean;
  score_kind: "high_attention" | string;
  attention_score_peak?: number | null;
  peak_attention?: number | null;
  threshold_percentile?: number | null;
  attention_threshold?: number | null;
  threshold_crossing_count?: number | null;
  raw_region_count?: number | null;
  post_filter_region_count?: number | null;
  high_attention_region_count?: number | null;
  high_attention_regions?: TemporalRegion[];
  high_attention_combined_duration_seconds?: number | null;
  combined_region_duration_seconds?: number | null;
  regions?: TemporalRegion[];
  /** Per-clip regions used by temporal visualizations. */
  visualization_threshold_percentile?: number | null;
  visualization_attention_threshold?: number | null;
  visualization_high_attention_region_count?: number | null;
  visualization_high_attention_regions?: TemporalRegion[] | null;
  visualization_high_attention_combined_duration_seconds?: number | null;
  artifacts: ExplanationArtifactReference[];
  warning?: string | null;
};

export type SemanticFeatureContribution = {
  rank: number;
  feature_name: string;
  display_name: string;
  value: number;
  unit?: string | null;
  reference_summary?: string | null;
  shap_value: number;
  direction: ShapDirection;
  start_seconds?: number | null;
  end_seconds?: number | null;
};

export type SemanticEvidenceWindow = {
  start_seconds: number;
  end_seconds: number;
  feature_importance: SemanticFeatureContribution[];
};

export type SemanticExplanation = {
  status: ComponentStatus;
  target_type: SemanticTargetType;
  model_version: string;
  feature_schema_version: string;
  extractor_version: string;
  output_space: "raw_margin" | "log_odds" | "probability";
  development_placeholder: boolean;
  research_eligible: boolean;
  base_value?: number | null;
  predicted_value?: number | null;
  feature_importance: SemanticFeatureContribution[];
  windows: SemanticEvidenceWindow[];
  warning?: string | null;
};

export type CombinedExplanationReport = {
  status: ComponentStatus;
  authoritative?: boolean;
  disposition: ReportDisposition;
  requires_human_review: boolean;
  finding: string;
  primary_evidence: string;
  quality_checks: string;
  limitation?: string | null;
  recommendation: string;
  disclaimer: string;
};

export type NarrativeExplanation = {
  status: "completed" | string;
  provider: "alibaba_model_studio" | string;
  model_id: string;
  generated_at: string;
  prompt_version: string;
  input_sha256: string;
  summary: string;
  detailed_explanation: string;
  evidence_references: string[];
};

export type MetricEvidence = {
  status: MetricStatus;
  value?: number | null;
  scope: "per_analysis" | "offline_validation";
  reason?: string | null;
  dataset_version?: string | null;
};

export type ExplanationQuality = {
  surrogate_fidelity_r2?: MetricEvidence | null;
  temporal_localization_iou?: MetricEvidence | null;
  temporal_semantic_iou?: MetricEvidence | null;
  aopc?: MetricEvidence | null;
  robustness_stability?: MetricEvidence | null;
};

export type ExplanationProvenance = {
  schema_version: string;
  pipeline_version: string;
  classifier_contract_version: string;
  feature_extractor_version?: string | null;
  semantic_model_version?: string | null;
  semantic_model_hash_short?: string | null;
  temporal_model_version?: string | null;
  temporal_model_hash_short?: string | null;
  configuration_hash?: string | null;
  configuration_hash_inputs?: string[];
  generated_at: string;
};

export type XaiExplanationResponse = {
  schema_version: string;
  explanation_id: string;
  prediction_id: string;
  request_id: string;
  development_placeholder: boolean;
  research_eligible: boolean;
  status: ExplanationStatus;
  component_statuses: ExplanationComponentStatuses;
  component_errors: ExplanationComponentErrors;
  classifier_snapshot: ClassifierSnapshot;
  temporal?: TemporalExplanation | null;
  semantic?: SemanticExplanation | null;
  combined_report?: CombinedExplanationReport | null;
  narrative?: NarrativeExplanation | null;
  quality: ExplanationQuality;
  provenance: ExplanationProvenance;
  artifacts: ExplanationArtifactReference[];
  warnings: string[];
  error?: ExplanationError | null;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
};

/* ---------------------------------------------------------------------------
 * Text Classification & Text XAI
 * ------------------------------------------------------------------------ */

export type TextClassificationLabel =
  | "AI-Generated"
  | "Human-Written"
  | "Mixed";

export type ConflictLevel = "LOW" | "MODERATE" | "HIGH" | "N/A";
export type SignalDirection = "AI" | "Human";
export type TokenHighlightLevel = "high" | "mid" | "low";

export type TokenHighlight = {
  word: string;
  score: number;
  level: TokenHighlightLevel;
};

export type TextSanitizationReport = {
  was_attacked: boolean;
  attack_report: unknown[];
  clean_text: string;
};

export type TextSignalAnalysis = {
  css_conflict_score: number | null;
  conflict_level: ConflictLevel;
  conflict_detected: boolean;
  deberta_direction: SignalDirection;
  xgboost_direction: SignalDirection | null;
  shap_ai_signals: string[];
  shap_human_signals: string[];
  word_count: number;
  note: string | null;
};

export type TextClassificationRequest = { text: string };

export type TextClassificationResponse = {
  label: TextClassificationLabel;
  prob_ai: number;
  prob_human: number;
  sanitization: TextSanitizationReport;
  token_highlights: TokenHighlight[];
  model_version: string;
  processing_time_ms: number;
  signal_analysis: TextSignalAnalysis | null;
  final_label: TextClassificationLabel;
  leans_toward: "AI-Generated" | "Human-Written" | null;
};

export type TextXaiHeatmapPoint = {
  display_token: string;
  normalized_score: number;
  is_noise: boolean;
};

export type TextXaiAuditRequest = {
  text: string;
  metadata: Record<string, unknown>;
};

export type TextXaiAuditResponse = {
  status: string;
  predicted_class: TextClassificationLabel;
  confidence: number;
  heatmap_data: TextXaiHeatmapPoint[];
  interpretability_report: string;
  model_version: string;
  processing_time_ms: number;
  sanitization: TextSanitizationReport;
  signal_analysis: TextSignalAnalysis | null;
  final_label: TextClassificationLabel;
  leans_toward: "AI-Generated" | "Human-Written" | null;
};

/**
 * View-model for rendering XAI audit results through the classification UI.
 * Deliberately omits `token_highlights` — /xai/audit does not return it.
 */
export type ClassificationView = {
  label: TextClassificationLabel;
  prob_ai: number;
  prob_human: number;
  sanitization: TextSanitizationReport;
  model_version: string;
  processing_time_ms: number;
  signal_analysis: TextSignalAnalysis | null;
  final_label: TextClassificationLabel;
  leans_toward: "AI-Generated" | "Human-Written" | null;
};