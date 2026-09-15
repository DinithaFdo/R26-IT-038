import type {
  ComponentStatus,
  ExplanationStatus,
  MetricEvidence,
  ReportDisposition,
  ShapDirection,
} from "@/types/api";

/**
 * Central mapping between backend Voice XAI vocabulary and user-facing text.
 *
 * Kept in one place so no component re-derives semantics locally. In
 * particular, SHAP direction comes from the backend's own `direction` field —
 * never inferred from the sign or magnitude of `shap_value` in the UI.
 */

/** Backend run states that are still progressing; only these should be polled. */
const NON_TERMINAL_EXPLANATION_STATUSES: ReadonlySet<ExplanationStatus> = new Set([
  "queued",
  "running",
  "partial",
]);

/**
 * `blocked` is deliberately terminal for polling purposes: the backend sets it
 * when the bounded queue was at capacity, and it only leaves that state when a
 * user explicitly retries — it will never progress on its own, so polling it
 * would spin forever.
 */
export function isExplanationInProgress(status: ExplanationStatus | undefined): boolean {
  return status !== undefined && NON_TERMINAL_EXPLANATION_STATUSES.has(status);
}

export function isExplanationTerminal(status: ExplanationStatus | undefined): boolean {
  return status !== undefined && !NON_TERMINAL_EXPLANATION_STATUSES.has(status);
}

export const EXPLANATION_STATUS_LABELS: Record<ExplanationStatus, string> = {
  queued: "Queued",
  running: "Running",
  partial: "Partial",
  completed: "Completed",
  failed: "Failed",
  blocked: "Blocked",
};

export const COMPONENT_STATUS_LABELS: Record<ComponentStatus, string> = {
  queued: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  blocked: "Blocked",
  not_available: "Not available",
};

/**
 * Explanatory copy for each run state. `queued`/`running` are explicitly NOT
 * errors and must not be presented as such.
 */
export const EXPLANATION_STATUS_DESCRIPTIONS: Record<ExplanationStatus, string> = {
  queued: "The explanation run is waiting for a background worker.",
  running: "The explanation run is in progress.",
  partial: "Some explanation components finished; others did not.",
  completed: "All explanation components finished.",
  failed: "The explanation run did not finish. The classifier prediction is unaffected.",
  blocked:
    "The explanation queue was at capacity when this run was created. Retry to create a new run.",
};

export type StatusTone = "neutral" | "progress" | "positive" | "warning" | "critical";

export const EXPLANATION_STATUS_TONES: Record<ExplanationStatus, StatusTone> = {
  queued: "neutral",
  running: "progress",
  partial: "warning",
  completed: "positive",
  failed: "critical",
  blocked: "warning",
};

export const COMPONENT_STATUS_TONES: Record<ComponentStatus, StatusTone> = {
  queued: "neutral",
  running: "progress",
  completed: "positive",
  failed: "critical",
  blocked: "warning",
  not_available: "neutral",
};

export const REPORT_DISPOSITION_LABELS: Record<ReportDisposition, string> = {
  spoof_suspected: "Spoof suspected",
  bonafide_suspected: "Bonafide suspected",
  inconclusive: "Inconclusive — manual review required",
};

export const REPORT_DISPOSITION_TONES: Record<ReportDisposition, StatusTone> = {
  spoof_suspected: "critical",
  bonafide_suspected: "positive",
  inconclusive: "warning",
};

/** Direction is authoritative from the backend, never inferred from magnitude. */
export const SHAP_DIRECTION_LABELS: Record<ShapDirection, string> = {
  toward_spoof: "Toward spoof",
  toward_bonafide: "Toward bonafide",
};

export const SHAP_DIRECTION_TONES: Record<ShapDirection, StatusTone> = {
  toward_spoof: "critical",
  toward_bonafide: "positive",
};

/**
 * Human-readable status for a quality metric.
 *
 * A metric that is absent (`undefined`/`null`) or not computed must never be
 * rendered as `0` — zero is a legitimate measured value and conflating the two
 * would misrepresent the evaluation state.
 */
export function describeMetric(metric: MetricEvidence | null | undefined): {
  available: boolean;
  value: number | null;
  label: string;
  reason: string | null;
} {
  if (!metric) {
    return {
      available: false,
      value: null,
      label: "Not available",
      reason: "This metric is not provided by the current backend.",
    };
  }
  if (metric.status === "available" && typeof metric.value === "number") {
    return { available: true, value: metric.value, label: "Available", reason: metric.reason ?? null };
  }
  return {
    available: false,
    value: null,
    label: metric.status === "not_applicable" ? "Not applicable" : "Not computed",
    reason: metric.reason ?? null,
  };
}

/** Human label for the classifier verdict. Backend stored values are unchanged. */
export function predictionLabelText(label: string | null | undefined): string {
  if (label === "spoof") return "Spoof suspected";
  if (label === "bonafide") return "Bonafide";
  return "Not available";
}

export const DECISION_SUPPORT_NOTICE =
  "Decision-support evidence. Requires qualified human review.";
