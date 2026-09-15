import type {
  BranchPrediction,
  FusionResult,
  ModelHealth,
  PredictionDetailResponse,
  PredictionHistoryResponse,
  ReadinessResponse,
} from "@/types/api";

const branch = (
  model_name: string,
  display_name: string,
  mode: "dummy" | "real",
  spoof: number,
  status: "success" | "failed" = "success",
  /**
   * Mirrors the backend flag: a real branch is only a research result once its
   * feature pipeline and class mapping are attested. Defaults to true so the
   * pre-existing all-real fixtures keep describing a fully attested system.
   */
  researchResult = mode === "real",
): BranchPrediction => ({
  model_name,
  display_name,
  status,
  mode,
  prediction: status === "success" ? (spoof >= 0.5 ? "spoof" : "bonafide") : null,
  confidence: status === "success" ? Math.max(spoof, 1 - spoof) : null,
  probabilities: status === "success" ? { spoof, bonafide: 1 - spoof } : null,
  processing_time_ms: status === "success" ? 128 : 0,
  error: status === "failed" ? "MODEL_BRANCH_FAILED" : null,
  metadata: {
    architecture: display_name,
    device: mode === "real" ? "cpu" : "cpu",
    research_result: status === "success" ? researchResult : false,
  },
});

/** A branch with no trained model in the deployment: runs nothing, scores nothing. */
const disabledBranch = (model_name: string, display_name: string): BranchPrediction => ({
  model_name,
  display_name,
  status: "skipped",
  mode: "disabled",
  prediction: null,
  confidence: null,
  probabilities: null,
  processing_time_ms: 0,
  error: null,
  metadata: {
    adapter_type: "disabled",
    error_code: "branch_disabled",
    reason:
      "This detection branch has no trained model in the current deployment and was not executed.",
    research_result: false,
    architecture: "not-integrated",
    device: "none",
  },
});

export const allDummyBranches = [
  branch("lfcc_cnn_tcn", "LFCC CNN/TCN", "dummy", 0.61),
  branch("aasist", "AASIST", "dummy", 0.54),
  branch("ssl_sequence", "SSL Sequence", "dummy", 0.58),
  branch("glottal", "Glottal", "dummy", 0.49),
];

export const mixedBranches = [
  branch("lfcc_cnn_tcn", "LFCC CNN/TCN", "real", 0.71),
  branch("aasist", "AASIST", "dummy", 0.56),
  branch("ssl_sequence", "SSL Sequence", "real", 0.68),
  branch("glottal", "Glottal", "dummy", 0.44),
];

export const allRealBranches = [
  branch("lfcc_cnn_tcn", "LFCC CNN/TCN", "real", 0.22),
  branch("aasist", "AASIST", "real", 0.31),
  branch("ssl_sequence", "SSL Sequence", "real", 0.26),
  branch("glottal", "Glottal", "real", 0.35),
];

export const partialBranchFailureBranches = [
  branch("lfcc_cnn_tcn", "LFCC CNN/TCN", "real", 0.64),
  branch("aasist", "AASIST", "real", 0, "failed"),
  branch("ssl_sequence", "SSL Sequence", "real", 0.59),
  branch("glottal", "Glottal", "real", 0.47),
];

/**
 * SSL failed -> Fusion V3 was unavailable for this request, so the legacy
 * three-branch detector produced the result (see `fusionLegacyFallbackResult`).
 */
export const fallbackBranches = [
  branch("lfcc_cnn_tcn", "LFCC CNN/TCN", "real", 0.58),
  branch("aasist", "AASIST", "real", 0.63),
  branch("ssl_sequence", "SSL Sequence", "real", 0, "failed"),
  branch("glottal", "Glottal", "real", 0.71),
];

/**
 * The current MULTI-SCOPE stage: CNN and AASIST-Light are real trained
 * checkpoints whose preprocessing/class mapping are not yet attested, and the
 * SSL and glottal branches have no trained model at all.
 */
export const currentStageBranches = [
  branch("cnn_acoustic", "LFCC CNN/TCN", "real", 0.62, "success", false),
  branch("aasist", "AASIST", "real", 0.44, "success", false),
  disabledBranch("ssl_wavlm_xlsr", "SSL Sequence"),
  disabledBranch("glottal_features", "Glottal Features"),
];

export const currentStageFusion: FusionResult = {
  status: "success",
  prediction: "spoof",
  confidence: 0.53,
  probabilities: { spoof: 0.53, bonafide: 0.47 },
  method: "weighted_average",
  decision_threshold: 0.5,
  branch_weights: { cnn_acoustic: 0.5, aasist: 0.5 },
  contains_dummy_branches: false,
  eligible_for_research_evaluation: false,
  warning:
    "At least one contributing branch ran a trained checkpoint whose feature pipeline or class mapping is unverified. Scores are unvalidated.",
  config_version: "fusion-config-v1",
  minimum_successful_branches: 2,
  contributing_branches: ["cnn_acoustic", "aasist"],
  excluded_branches: {
    ssl_wavlm_xlsr: "branch_disabled",
    glottal_features: "branch_disabled",
  },
  system_stage: "partial",
  research_blockers: ["incomplete_branch_set", "unverified_branch_contributed"],
  fusion_version: "score-level-fusion-v1",
  fusion_mode: "development_average",
  fallback_used: false,
};

export const fusionResult = (containsDummy: boolean, spoof = 0.58, status: "success" | "failed" = "success"): FusionResult => ({
  status,
  prediction: status === "success" ? (spoof >= 0.5 ? "spoof" : "bonafide") : null,
  confidence: status === "success" ? Math.max(spoof, 1 - spoof) : null,
  probabilities: status === "success" ? { spoof, bonafide: 1 - spoof } : null,
  method: "weighted_average",
  decision_threshold: 0.5,
  branch_weights: {
    lfcc_cnn_tcn: 0.25,
    aasist: 0.25,
    ssl_sequence: 0.25,
    glottal: 0.25,
  },
  contains_dummy_branches: containsDummy,
  eligible_for_research_evaluation: !containsDummy && status === "success",
  warning: containsDummy ? "Dummy mode is for system development only; not a research result." : null,
  config_version: "fusion-v1",
  minimum_successful_branches: 2,
  contributing_branches: ["lfcc_cnn_tcn", "aasist", "ssl_sequence", "glottal"],
  excluded_branches: {},
  system_stage: "full",
  research_blockers: containsDummy ? ["dummy_branch_contributed"] : [],
  fusion_version: "score-level-fusion-v1",
  fusion_mode: "development_average",
  fallback_used: false,
});

/**
 * A genuine Fusion V3 success: all four branches real and successful, the
 * frozen learned weights, `fallback_used: false`. Use this (not
 * `fusionResult()`) wherever a test needs to assert the primary
 * four-branch-detector UI specifically.
 */
export const fusionV3Result = (spoof = 0.42): FusionResult => ({
  status: "success",
  prediction: spoof >= 0.5 ? "spoof" : "bonafide",
  confidence: Math.max(spoof, 1 - spoof),
  probabilities: { spoof, bonafide: 1 - spoof },
  method: "convex_weighted",
  decision_threshold: 0.5,
  branch_weights: {
    cnn_acoustic: 0.3488902015351922,
    aasist: 0.289776249915524,
    ssl_wavlm_xlsr: 0.22524436863750652,
    glottal_features: 0.13608917991177738,
  },
  contains_dummy_branches: false,
  eligible_for_research_evaluation: true,
  warning: null,
  config_version: "fusion-convex-4branch-v3",
  minimum_successful_branches: 4,
  contributing_branches: ["cnn_acoustic", "aasist", "ssl_wavlm_xlsr", "glottal_features"],
  excluded_branches: {},
  system_stage: "full",
  research_blockers: [],
  fusion_version: "fusion-convex-4branch-v3",
  fusion_mode: "learned_constrained",
  fallback_used: false,
});

/**
 * Fusion V3 was configured/available but unavailable for this request (e.g.
 * SSL failed), so the legacy three-branch detector produced the result
 * instead -- `fallback_used: true`, Glottal excluded, the frozen legacy
 * threshold/method.
 */
export const fusionLegacyFallbackResult = (spoof = 0.61): FusionResult => ({
  status: "success",
  prediction: spoof >= 0.5519237850482265 ? "spoof" : "bonafide",
  confidence: spoof >= 0.5519237850482265 ? spoof : 1 - spoof,
  probabilities: { spoof, bonafide: 1 - spoof },
  method: "simple_average",
  decision_threshold: 0.5519237850482265,
  branch_weights: {
    cnn_acoustic: 0.3333333333333333,
    aasist: 0.3333333333333333,
    ssl_wavlm_xlsr: 0.3333333333333333,
  },
  contains_dummy_branches: false,
  eligible_for_research_evaluation: true,
  warning: null,
  config_version: "final-detector-v1",
  minimum_successful_branches: 3,
  contributing_branches: ["cnn_acoustic", "aasist", "ssl_wavlm_xlsr"],
  excluded_branches: { glottal_features: "auxiliary_not_used_in_primary_fusion" },
  system_stage: "full",
  research_blockers: [],
  fusion_version: "legacy-3branch-frozen-v1",
  fusion_mode: "legacy_average",
  fallback_used: true,
});

export const predictionFixture = (
  prediction_id: string,
  branches: BranchPrediction[],
  fusion: FusionResult | null,
  status: "completed" | "processing" | "failed" = "completed",
): PredictionDetailResponse => ({
  prediction_id,
  request_id: `req_${prediction_id}`,
  source_type: "dashboard_upload",
  status,
  audio: {
    original_filename: "sample-voice.wav",
    original_extension: "wav",
    detected_container: "wav",
    detected_codec: "pcm_s16le",
    duration_seconds: 8.4,
    sample_rate: 16000,
    channels: 1,
    size_bytes: 420000,
    storage_status: "success",
    playback_available: status === "completed",
  },
  branches,
  fusion,
  preprocessing: {},
  total_processing_time_ms: 610,
  warnings: [],
  research_eligible: Boolean(fusion?.eligible_for_research_evaluation),
  created_at: "2026-08-04T08:00:00Z",
  updated_at: "2026-08-04T08:00:04Z",
  completed_at: status === "completed" ? "2026-08-04T08:00:04Z" : null,
});

export const mockPredictionScenarios = {
  currentStageCompleted: predictionFixture(
    "pred_current_stage",
    currentStageBranches,
    currentStageFusion,
  ),
  allDummyCompleted: predictionFixture("pred_all_dummy", allDummyBranches, fusionResult(true, 0.56)),
  mixedCompleted: predictionFixture("pred_mixed", mixedBranches, fusionResult(true, 0.64)),
  allRealCompleted: predictionFixture("pred_all_real", allRealBranches, fusionResult(false, 0.28)),
  fusionV3Completed: predictionFixture("pred_fusion_v3", allRealBranches, fusionV3Result(0.22)),
  fusionFallbackCompleted: predictionFixture(
    "pred_fusion_fallback",
    fallbackBranches,
    fusionLegacyFallbackResult(0.61),
  ),
  partialBranchFailure: predictionFixture("pred_partial_failure", partialBranchFailureBranches, fusionResult(false, 0.57)),
  fusionFailure: predictionFixture("pred_fusion_failed", allRealBranches, fusionResult(false, 0.5, "failed"), "failed"),
  processing: predictionFixture("pred_processing", allDummyBranches, null, "processing"),
  failed: predictionFixture("pred_failed", partialBranchFailureBranches, null, "failed"),
  audioUnavailable: {
    ...predictionFixture("pred_audio_unavailable", allRealBranches, fusionResult(false, 0.34)),
    audio: {
      ...predictionFixture("pred_audio_unavailable", allRealBranches, fusionResult(false, 0.34)).audio,
      playback_available: false,
      storage_status: "failed",
    },
  },
};

export const historyFixture: PredictionHistoryResponse = {
  items: Object.values(mockPredictionScenarios).map((prediction) => ({
    prediction_id: prediction.prediction_id,
    filename: prediction.audio.original_filename,
    source_type: prediction.source_type,
    status: prediction.status,
    duration_seconds: prediction.audio.duration_seconds,
    final_prediction: prediction.fusion?.prediction,
    confidence: prediction.fusion?.confidence,
    mode_summary: {
      dummy: prediction.branches.filter((item) => item.mode === "dummy").length,
      real: prediction.branches.filter((item) => item.mode === "real").length,
      contains_dummy: prediction.branches.some((item) => item.mode === "dummy"),
    },
    created_at: prediction.created_at,
  })),
  page: 1,
  limit: 20,
  has_next: false,
};

export const backendNotReadyFixture: ReadinessResponse = {
  status: "not_ready",
  prediction_ready: false,
  research_ready: false,
  ffmpeg_available: true,
  ffprobe_available: true,
  mongodb_configured: true,
  mongodb_available: false,
  storage_enabled: false,
  storage_available: false,
  components: {
    mongodb: { ready: false, mongodb_configured: true, mongodb_available: false },
    storage: { ready: true, policy: "optional" },
  },
};

export const modelHealthFixture: ModelHealth[] = allDummyBranches.map((item) => ({
  branch_name: item.model_name,
  model_name: item.model_name,
  display_name: item.display_name,
  mode: item.mode,
  is_loaded: true,
  lifecycle_state: "loaded",
  ready: true,
  research_ready: false,
  requested_device: "cpu",
  resolved_device: "cpu",
  checkpoint_configured: false,
  uses_dummy_mode: true,
  warning: "Dummy mode is for system development only; not a research result.",
}));

export const rateLimitedErrorFixture = {
  request_id: "req_rate_limited",
  error: {
    code: "RATE_LIMITED",
    message: "Too many prediction requests. Try again later.",
    details: null,
  },
};
