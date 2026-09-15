import axios, { AxiosInstance, CreateAxiosDefaults } from "axios";
import { env } from "@/lib/config/env";
import { normalizeApiError } from "@/lib/api/errors";

export type AuthTokenGetter = () => Promise<string | null> | string | null;

let authTokenGetter: AuthTokenGetter | null = null;

export function setApiAuthTokenGetter(getter: AuthTokenGetter | null) {
  authTokenGetter = getter;
}

export interface ApiClientOptions extends CreateAxiosDefaults {
  serviceName: string;
  getBaseUrl: () => string;
  envVarName: string;
}

/**
 * Creates an enterprise-grade Axios instance configured with standardized
 * request verification, timeout handling, and unified error normalization.
 */
export function createApiClient({
  serviceName,
  getBaseUrl,
  envVarName,
  ...axiosConfig
}: ApiClientOptions): AxiosInstance {
  const client = axios.create({
    baseURL: getBaseUrl() || undefined,
    timeout: env.apiTimeoutMs,
    headers: {
      Accept: "application/json",
    },
    ...axiosConfig,
  });

  client.interceptors.request.use(async (config) => {
    const baseUrl = getBaseUrl();
    if (!baseUrl) {
      throw new Error(
        `[${serviceName}] API Base URL is not configured. Please set ${envVarName} in your environment variables.`
      );
    }
    if (!config.baseURL) {
      config.baseURL = baseUrl;
    }

    // Auth token hook (ready for Clerk or custom auth token injection)
    // const token = authTokenGetter ? await authTokenGetter() : null;
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }

    return config;
  });

  client.interceptors.response.use(
    (response) => response,
    (error) => {
      // 401 redirect hook (ready for session expiration handling)
      // if (axios.isAxiosError(error) && error.response?.status === 401) {
      //   if (typeof window !== "undefined") {
      //     const currentPath = window.location.pathname;
      //     const search = window.location.search;
      //     const redirectUrl = encodeURIComponent(`${currentPath}${search}`);
      //     window.location.href = `/sign-in?redirect_url=${redirectUrl}`;
      //   }
      // }
      return Promise.reject(normalizeApiError(error));
    },
  );

  return client;
}

/**
 * Dedicated API client for Voice Classification & Voice XAI services.
 */
export const voiceApiClient = createApiClient({
  serviceName: "Voice Service",
  getBaseUrl: () => env.voiceApiBaseUrl,
  envVarName: "NEXT_PUBLIC_VOICE_API_BASE_URL (or NEXT_PUBLIC_API_BASE_URL)",
});

/**
 * Dedicated API client for Text Classification & Text XAI services.
 */
export const textApiClient = createApiClient({
  serviceName: "Text Service",
  getBaseUrl: () => env.textApiBaseUrl,
  envVarName: "NEXT_PUBLIC_TEXT_API_BASE_URL (or NEXT_PUBLIC_API_BASE_URL)",
});

/**
 * Default / legacy API client alias for backwards compatibility across existing modules.
 */
export const apiClient = voiceApiClient;
