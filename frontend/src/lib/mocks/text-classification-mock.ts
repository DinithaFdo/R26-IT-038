import type { TextClassificationResponse, TextXaiAuditResponse } from "@/types/api";

export const MOCK_CLASSIFICATION_AI: TextClassificationResponse = {
  label: "AI-Generated",
  prob_ai: 0.9999,
  prob_human: 0.0001,
  sanitization: {
    was_attacked: false,
    attack_report: [],
    clean_text: "The implementation demonstrates significant results.",
  },
  token_highlights: [
    { word: "implementation", score: 0.95, level: "high" },
    { word: "demonstrates", score: 0.82, level: "high" },
    { word: "significant", score: 0.71, level: "high" },
    { word: "results", score: 0.45, level: "mid" },
    { word: "The", score: 0.12, level: "low" },
  ],
  model_version: "deberta-v3-large-model2-domainfix",
  processing_time_ms: 1088.3,
  signal_analysis: {
    css_conflict_score: 0.021,
    conflict_level: "LOW",
    conflict_detected: false,
    deberta_direction: "AI",
    xgboost_direction: "AI",
    shap_ai_signals: [
      "Phrase repetition density",
      "Formal vocabulary density",
      "Sentence length variation",
    ],
    shap_human_signals: ["Contraction usage", "Personal pronoun usage"],
    word_count: 187,
    note: null,
  },
  final_label: "AI-Generated",
  leans_toward: null,
};

export const MOCK_CLASSIFICATION_CONFLICT: TextClassificationResponse = {
  label: "AI-Generated",
  prob_ai: 0.91,
  prob_human: 0.09,
  sanitization: {
    was_attacked: true,
    attack_report: ["Zero-width characters removed: 12 instances"],
    clean_text: "You know, I have been thinking about this a lot lately.",
  },
  token_highlights: [
    { word: "thinking", score: 0.88, level: "high" },
    { word: "know", score: 0.61, level: "high" },
    { word: "lately", score: 0.44, level: "mid" },
    { word: "You", score: 0.21, level: "low" },
  ],
  model_version: "deberta-v3-large-model2-domainfix",
  processing_time_ms: 2876.1,
  signal_analysis: {
    css_conflict_score: 0.641,
    conflict_level: "HIGH",
    conflict_detected: true,
    deberta_direction: "AI",
    xgboost_direction: "Human",
    shap_ai_signals: [
      "Sentence length variation",
      "Vocabulary breadth",
      "Comma usage per sentence",
    ],
    shap_human_signals: [
      "Contraction usage",
      "Personal pronoun usage",
      "Interrogative sentence proportion",
    ],
    word_count: 203,
    note: null,
  },
  final_label: "Mixed",
  leans_toward: "AI-Generated",
};

export const MOCK_XAI_RESPONSE: TextXaiAuditResponse = {
  status: "completed",
  predicted_class: "AI-Generated",
  confidence: 91.0,
  heatmap_data: [
    { display_token: "The", normalized_score: 0.12, is_noise: false },
    { display_token: "implementation", normalized_score: 0.95, is_noise: false },
    { display_token: "demonstrates", normalized_score: 0.82, is_noise: false },
    { display_token: "significant", normalized_score: 0.71, is_noise: false },
    { display_token: "results", normalized_score: 0.45, is_noise: false },
    { display_token: "across", normalized_score: 0.33, is_noise: false },
    { display_token: "multiple", normalized_score: 0.28, is_noise: false },
    { display_token: "domains", normalized_score: 0.51, is_noise: false },
  ],
  interpretability_report:
    "This text exhibits strong markers of AI-generated content. " +
    "The semantic model identified highly formal vocabulary patterns and " +
    "uniform sentence structure that are characteristic of large language model output. " +
    "Surface-level stylometric analysis supports this finding, with low phrase " +
    "repetition variation and consistent lexical density across the passage.",
  model_version: "deberta-v3-large-model2-domainfix",
  processing_time_ms: 4231.5,
  sanitization: {
    was_attacked: false,
    attack_report: [],
    clean_text: "The implementation demonstrates significant results.",
  },
  signal_analysis: {
    css_conflict_score: 0.021,
    conflict_level: "LOW",
    conflict_detected: false,
    deberta_direction: "AI",
    xgboost_direction: "AI",
    shap_ai_signals: ["Phrase repetition density", "Formal vocabulary density"],
    shap_human_signals: ["Contraction usage"],
    word_count: 187,
    note: null,
  },
  final_label: "AI-Generated",
  leans_toward: null,
};
