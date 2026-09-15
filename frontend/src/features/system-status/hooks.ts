"use client";

import { useQuery } from "@tanstack/react-query";
import { env } from "@/lib/config/env";
import { shouldRetryApiError } from "@/lib/api/errors";
import { getModelHealth, getReadiness } from "@/lib/api/system";
import { queryKeys } from "@/lib/query/keys";

export function useSystemReadiness() {
  return useQuery({
    queryKey: queryKeys.readiness,
    queryFn: getReadiness,
    refetchInterval: env.systemStatusRefreshMs,
    refetchIntervalInBackground: false,
    retry: shouldRetryApiError,
  });
}

export function useModelHealth() {
  return useQuery({
    queryKey: queryKeys.modelHealth,
    queryFn: getModelHealth,
    refetchInterval: env.systemStatusRefreshMs,
    refetchIntervalInBackground: false,
    retry: shouldRetryApiError,
  });
}
