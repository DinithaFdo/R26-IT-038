import type { PredictionFilters } from "@/types/api";

export const queryKeys = {
  readiness: ["system", "readiness"] as const,
  modelHealth: ["system", "models"] as const,
  predictions: (filters: PredictionFilters) => ["predictions", filters] as const,
  prediction: (id: string) => ["prediction", id] as const,
  predictionStatus: (id: string) => ["prediction", id, "status"] as const,
  predictionAudio: (id: string) => ["prediction", id, "audio"] as const,
  explanation: (id: string) => ["prediction", id, "explanation"] as const,
  explanationArtifact: (id: string, artifactId: string) =>
    ["prediction", id, "explanation", "artifact", artifactId] as const,
};
