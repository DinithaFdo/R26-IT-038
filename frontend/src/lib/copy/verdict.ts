import type { PredictionLabel } from "@/types/api";

/**
 * Plain-language primary verdict text. Backend values (bonafide/spoof) are
 * unchanged -- this only controls what a normal reader sees as the headline.
 */
export function getVerdictHeadline(label: PredictionLabel | null | undefined): string {
  if (label === "spoof") return "Likely AI-Generated or Manipulated Voice";
  if (label === "bonafide") return "Likely Authentic Voice";
  return "No Result Available";
}

/** One-sentence plain explanation of what the verdict means, for below the headline. */
export function getVerdictSentence(label: PredictionLabel | null | undefined): string | null {
  if (label === "spoof") {
    return "The system found patterns commonly associated with AI-generated or manipulated speech.";
  }
  if (label === "bonafide") {
    return "The system found patterns commonly associated with genuine, unmodified speech.";
  }
  return null;
}
