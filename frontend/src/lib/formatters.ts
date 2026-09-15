import type { PredictionLabel, PredictionStatus, SourceType } from "@/types/api";

export function formatPercent(value?: number | null) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Unavailable";
  return `${Math.round(value * 100)}%`;
}

export function formatProbability(value?: number | null) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Not reported";
  return `${(value * 100).toFixed(2)}%`;
}

/** Two decimal places -- fusion weights are frozen constants precise enough
 * to be worth showing more precisely than a general probability estimate. */
export function formatWeightPercent(value?: number | null) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "Not reported";
  return `${(value * 100).toFixed(2)}%`;
}

export function formatBytes(bytes?: number | null) {
  if (typeof bytes !== "number" || !Number.isFinite(bytes)) return "Unknown size";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function formatDuration(seconds?: number | null) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "Unknown duration";
  const minutes = Math.floor(seconds / 60);
  const remaining = Math.round(seconds % 60).toString().padStart(2, "0");
  return `${minutes}:${remaining}`;
}

export function formatDateTime(value?: string | null) {
  if (!value) return "Not available";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Not available";
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function labelText(label?: PredictionLabel | null) {
  if (label === "spoof") return "Spoof";
  if (label === "bonafide") return "Bonafide";
  return "No decision";
}

export function statusText(status?: PredictionStatus | null) {
  if (!status) return "Unknown";
  return status.replaceAll("_", " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export function sourceText(source?: SourceType | null) {
  if (source === "dashboard_upload") return "File upload";
  if (source === "live_recording") return "Browser recording";
  if (source === "public_api") return "Public API";
  if (source === "mcp") return "MCP";
  return source ? statusText(source) : "Unknown source";
}

export function isTerminalStatus(status?: PredictionStatus | null) {
  return status === "completed" || status === "failed" || status === "deleted";
}
