"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ApiError, normalizeApiError, shouldRetryApiError } from "@/lib/api/errors";
import {
  getExplanation,
  retryNarrativeExplanation,
  retryExplanation,
  triggerExplanation,
} from "@/lib/api/xai";
import { queryKeys } from "@/lib/query/keys";
import { isExplanationInProgress } from "@/lib/xai/status";
import type { XaiExplanationResponse } from "@/types/api";

/** Poll interval while a run is queued/running. Deliberately unaggressive. */
const EXPLANATION_POLL_MS = 3_000;

/**
 * Read the latest explanation for a prediction.
 *
 * A 404 from this endpoint means "no explanation run exists yet" — a normal
 * state, not an error — so it resolves to `null` instead of an error state and
 * is not retried. Any other error propagates normally.
 */
export function useExplanation(predictionId: string, enabled = true) {
  return useQuery<XaiExplanationResponse | null>({
    queryKey: queryKeys.explanation(predictionId),
    enabled: enabled && Boolean(predictionId),
    queryFn: async () => {
      try {
        return await getExplanation(predictionId);
      } catch (error) {
        const normalized = normalizeApiError(error);
        if (normalized.status === 404) return null;
        throw normalized;
      }
    },
    // Poll only while the backend reports a genuinely progressing run.
    // TanStack Query stops the interval automatically when the component
    // unmounts, so navigating away aborts polling.
    refetchInterval: (query) => {
      const current = query.state.data;
      const narrativeInProgress = ["queued", "running"].includes(
        current?.component_statuses.narrative ?? "",
      );
      return isExplanationInProgress(current?.status) || narrativeInProgress
        ? EXPLANATION_POLL_MS
        : false;
    },
    refetchIntervalInBackground: false,
    retry: (failureCount, error) => {
      if (error instanceof ApiError && error.status === 404) return false;
      return shouldRetryApiError(failureCount, error);
    },
    // Explanation payloads are owner-scoped evidence; do not retain them in
    // cache indefinitely after the UI stops using them.
    gcTime: 5 * 60_000,
  });
}

/**
 * Trigger an explanation. The backend returns the existing run if one is
 * already present, so this is safe to call more than once.
 */
export function useTriggerExplanation(predictionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => triggerExplanation(predictionId),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.explanation(predictionId), data);
    },
  });
}

/**
 * Retry creates a NEW run server-side. The previous run's terminal evidence is
 * never overwritten, so the cache is replaced with the new run rather than
 * merged into the old one.
 */
export function useRetryExplanation(predictionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => retryExplanation(predictionId),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.explanation(predictionId), data);
      void queryClient.invalidateQueries({ queryKey: queryKeys.explanation(predictionId) });
    },
  });
}

/** Retry the optional provider call without recreating deterministic XAI evidence. */
export function useNarrativeRetry(predictionId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => retryNarrativeExplanation(predictionId),
    onSuccess: (data) => {
      queryClient.setQueryData(queryKeys.explanation(predictionId), data);
      void queryClient.invalidateQueries({ queryKey: queryKeys.explanation(predictionId) });
    },
  });
}
