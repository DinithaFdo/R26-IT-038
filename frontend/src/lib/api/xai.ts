import { apiClient } from "@/lib/api/client";
import type {
  CombinedExplanationReport,
  SemanticExplanation,
  TemporalExplanation,
  XaiExplanationResponse,
} from "@/types/api";

/**
 * Owner-scoped Voice XAI endpoints.
 *
 * Every route below is authenticated and owner-scoped server-side; the backend
 * resolves the owner from the auth principal and returns 404 (not 403) for
 * another user's prediction, so a manually edited `predictionId` in the URL
 * cannot disclose whether that prediction exists. The frontend never sends an
 * owner/user identifier of its own.
 */

const basePath = (predictionId: string) =>
  `/api/v1/me/predictions/${encodeURIComponent(predictionId)}/explanation`;

/** Trigger (or return the existing) explanation run for a completed prediction. */
export async function triggerExplanation(predictionId: string) {
  const response = await apiClient.post<XaiExplanationResponse>(basePath(predictionId));
  return response.data;
}

/** Create a NEW explanation run. The backend never overwrites terminal evidence. */
export async function retryExplanation(predictionId: string) {
  const response = await apiClient.post<XaiExplanationResponse>(
    `${basePath(predictionId)}/retry`,
  );
  return response.data;
}

/** Retry only the optional AI narrative using persisted XAI evidence. */
export async function retryNarrativeExplanation(predictionId: string) {
  const response = await apiClient.post<XaiExplanationResponse>(
    `${basePath(predictionId)}/narrative/retry`,
  );
  return response.data;
}

/** Read the latest explanation. 404 means "no explanation run exists yet". */
export async function getExplanation(predictionId: string) {
  const response = await apiClient.get<XaiExplanationResponse>(basePath(predictionId));
  return response.data;
}

export async function getTemporalExplanation(predictionId: string) {
  const response = await apiClient.get<TemporalExplanation>(
    `${basePath(predictionId)}/temporal`,
  );
  return response.data;
}

export async function getSemanticExplanation(predictionId: string) {
  const response = await apiClient.get<SemanticExplanation>(
    `${basePath(predictionId)}/semantic`,
  );
  return response.data;
}

export async function getExplanationReport(predictionId: string) {
  const response = await apiClient.get<CombinedExplanationReport>(
    `${basePath(predictionId)}/report`,
  );
  return response.data;
}

/**
 * Fetch a private artifact's bytes through the authenticated route.
 *
 * `artifactId` is treated as an opaque identifier and is only ever used as a
 * path segment here — the backend stores artifacts under hashed paths and does
 * not expose a filesystem location, so there is nothing to construct locally.
 * Returns a Blob; callers are responsible for revoking any object URL created
 * from it.
 */
export async function getExplanationArtifact(predictionId: string, artifactId: string) {
  const response = await apiClient.get<Blob>(
    `${basePath(predictionId)}/artifacts/${encodeURIComponent(artifactId)}`,
    { responseType: "blob" },
  );
  return response.data;
}
