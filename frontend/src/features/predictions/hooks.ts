"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { env } from "@/lib/config/env";
import { shouldRetryApiError } from "@/lib/api/errors";
import {
  createPrediction,
  deletePrediction,
  getPrediction,
  getPredictionAudio,
  getPredictionHistory,
  getPredictionStatus,
  rerunPrediction,
  type CreatePredictionInput,
} from "@/lib/api/predictions";
import { isTerminalStatus } from "@/lib/formatters";
import { queryKeys } from "@/lib/query/keys";
import type { PredictionFilters } from "@/types/api";

export function usePredictionHistory(filters: PredictionFilters) {
  return useQuery({
    queryKey: queryKeys.predictions(filters),
    queryFn: () => getPredictionHistory(filters),
    retry: shouldRetryApiError,
  });
}

export function usePrediction(predictionId: string) {
  return useQuery({
    queryKey: queryKeys.prediction(predictionId),
    queryFn: () => getPrediction(predictionId),
    enabled: Boolean(predictionId),
    retry: shouldRetryApiError,
  });
}

export function usePredictionStatus(predictionId: string, enabled = true) {
  return useQuery({
    queryKey: queryKeys.predictionStatus(predictionId),
    queryFn: () => getPredictionStatus(predictionId),
    enabled: enabled && Boolean(predictionId),
    refetchInterval: (query) =>
      isTerminalStatus(query.state.data?.status) ? false : env.predictionStatusPollMs,
    retry: shouldRetryApiError,
  });
}

export function usePredictionAudio(predictionId: string, enabled: boolean) {
  return useQuery({
    queryKey: queryKeys.predictionAudio(predictionId),
    queryFn: () => getPredictionAudio(predictionId),
    enabled: enabled && Boolean(predictionId),
    staleTime: 30_000,
    gcTime: 60_000,
    retry: shouldRetryApiError,
  });
}

export function useCreatePrediction() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (input: CreatePredictionInput) => createPrediction(input),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["predictions"] });
    },
  });
}

export function useRerunPrediction() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ predictionId, reason }: { predictionId: string; reason?: string }) =>
      rerunPrediction(predictionId, reason),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ["predictions"] });
    },
  });
}

export function useDeletePrediction() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (predictionId: string) => deletePrediction(predictionId),
    onSuccess: (_data, predictionId) => {
      void queryClient.invalidateQueries({ queryKey: ["predictions"] });
      queryClient.removeQueries({ queryKey: queryKeys.prediction(predictionId) });
      queryClient.removeQueries({ queryKey: queryKeys.predictionAudio(predictionId) });
      queryClient.removeQueries({ queryKey: queryKeys.predictionStatus(predictionId) });
    },
  });
}
