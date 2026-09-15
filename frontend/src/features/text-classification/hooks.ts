"use client";

import { useMutation } from "@tanstack/react-query";
import { classifyText } from "@/lib/api/text-classification";
import { auditTextExplanation } from "@/lib/api/text-xai";
import type { ApiError } from "@/lib/api/errors";
import type {
  TextClassificationRequest,
  TextClassificationResponse,
  TextXaiAuditRequest,
  TextXaiAuditResponse,
} from "@/types/api";

export function useClassifyText() {
  return useMutation<TextClassificationResponse, Error, TextClassificationRequest>({
    mutationFn: classifyText,
  });
}

export function useAuditTextExplanation() {
  return useMutation<TextXaiAuditResponse, ApiError, TextXaiAuditRequest>({
    mutationFn: auditTextExplanation,
  });
}
