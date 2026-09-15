import { apiClient } from "@/lib/api/client";
import type {
  PredictionAudioPlaybackResponse,
  PredictionDeleteResponse,
  PredictionDetailResponse,
  PredictionFilters,
  PredictionHistoryResponse,
  PredictionJobStatusResponse,
  PredictionSubmissionResponse,
} from "@/types/api";

export type CreatePredictionInput = {
  file: File;
  sourceType: "dashboard_upload" | "live_recording";
  clientFilename?: string;
  idempotencyKey?: string;
  onUploadProgress?: (progress: number) => void;
};

export async function createPrediction(input: CreatePredictionInput) {
  const formData = new FormData();
  formData.append("file", input.file);
  formData.append("source_type", input.sourceType);
  if (input.clientFilename) formData.append("client_filename", input.clientFilename);
  if (input.idempotencyKey) formData.append("idempotency_key", input.idempotencyKey);

  const response = await apiClient.post<PredictionSubmissionResponse>(
    "/api/v1/predictions",
    formData,
    {
      headers: { "Content-Type": "multipart/form-data" },
      // This call blocks synchronously for the full real 4-branch inference
      // pipeline (CNN + AASIST + SSL + Glottal + fusion), not just the
      // upload. Each branch has its own ~30s server-side timeout, and Glottal
      // alone routinely takes ~7s on CPU even when nothing fails -- 120s left
      // too little headroom if several branches individually hit their
      // timeout in the same request. 180s keeps clear margin beyond that
      // worst case while staying a bounded, demo-safe timeout.
      timeout: 180_000,
      onUploadProgress: (event) => {
        if (!input.onUploadProgress || !event.total) return;
        input.onUploadProgress(Math.round((event.loaded / event.total) * 100));
      },
    },
  );
  return response.data;
}

export async function getPredictionStatus(predictionId: string) {
  const response = await apiClient.get<PredictionJobStatusResponse>(
    `/api/v1/predictions/${encodeURIComponent(predictionId)}/status`,
  );
  return response.data;
}

export async function getPredictionHistory(filters: PredictionFilters) {
  const response = await apiClient.get<PredictionHistoryResponse>("/api/v1/me/predictions", {
    params: {
      page: filters.page,
      limit: filters.limit,
      status: filters.status || undefined,
      source_type: filters.sourceType || undefined,
      prediction_label: filters.predictionLabel || undefined,
      created_from: filters.createdFrom || undefined,
      created_to: filters.createdTo || undefined,
    },
  });
  return response.data;
}

export async function getPrediction(predictionId: string) {
  const response = await apiClient.get<PredictionDetailResponse>(
    `/api/v1/me/predictions/${encodeURIComponent(predictionId)}`,
  );
  return response.data;
}

export async function getPredictionAudio(predictionId: string) {
  const response = await apiClient.get<PredictionAudioPlaybackResponse>(
    `/api/v1/me/predictions/${encodeURIComponent(predictionId)}/audio`,
  );
  return response.data;
}

export async function rerunPrediction(predictionId: string, reason?: string) {
  const formData = new FormData();
  if (reason) formData.append("rerun_reason", reason);
  const response = await apiClient.post<PredictionSubmissionResponse>(
    `/api/v1/me/predictions/${encodeURIComponent(predictionId)}/rerun`,
    formData,
  );
  return response.data;
}

export async function deletePrediction(predictionId: string) {
  const response = await apiClient.delete<PredictionDeleteResponse>(
    `/api/v1/me/predictions/${encodeURIComponent(predictionId)}`,
  );
  return response.data;
}
