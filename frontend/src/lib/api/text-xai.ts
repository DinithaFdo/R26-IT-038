import { textApiClient } from "@/lib/api/client";
import type { TextXaiAuditRequest, TextXaiAuditResponse } from "@/types/api";

export async function auditTextExplanation(input: TextXaiAuditRequest): Promise<TextXaiAuditResponse> {
  const response = await textApiClient.post<TextXaiAuditResponse>("/xai/audit", input);
  return response.data;
}
