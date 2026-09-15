import { textApiClient } from "@/lib/api/client";
import type { TextClassificationRequest, TextClassificationResponse } from "@/types/api";

export async function classifyText(input: TextClassificationRequest): Promise<TextClassificationResponse> {
  const response = await textApiClient.post<TextClassificationResponse>("/classify", input);
  return response.data;
}
