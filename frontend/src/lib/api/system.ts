import { apiClient } from "@/lib/api/client";
import type { ModelHealth, ReadinessResponse } from "@/types/api";

export async function getHealth() {
  const response = await apiClient.get<{ status: string }>("/health");
  return response.data;
}

export async function getReadiness() {
  const response = await apiClient.get<ReadinessResponse>("/ready", {
    validateStatus: (status) => (status >= 200 && status < 300) || status === 503,
  });
  return response.data;
}

export async function getModelHealth() {
  const response = await apiClient.get<ModelHealth[]>("/api/v1/voice/models/health");
  return response.data;
}
