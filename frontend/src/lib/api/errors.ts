import axios, { AxiosError } from "axios";
import type { BackendErrorResponse } from "@/types/api";

export class ApiError extends Error {
  status?: number;
  code?: string;
  requestId?: string;
  details?: unknown;
  retryAfterSeconds?: number;

  constructor(
    message: string,
    options: {
      status?: number;
      code?: string;
      requestId?: string;
      details?: unknown;
      retryAfterSeconds?: number;
    } = {},
  ) {
    super(message);
    this.name = "ApiError";
    this.status = options.status;
    this.code = options.code;
    this.requestId = options.requestId;
    this.details = options.details;
    this.retryAfterSeconds = options.retryAfterSeconds;
  }
}

function parseRetryAfterSeconds(value: unknown): number | undefined {
  if (typeof value !== "string" || value.trim() === "") return undefined;
  const seconds = Number(value);
  return Number.isFinite(seconds) && seconds >= 0 ? seconds : undefined;
}

export function normalizeApiError(error: unknown): ApiError {
  if (error instanceof ApiError) return error;
  if (axios.isAxiosError(error)) {
    const axiosError = error as AxiosError<BackendErrorResponse>;
    const data = axiosError.response?.data;
    const message =
      data?.error?.message ||
      (typeof data?.detail === "string" ? data.detail : undefined) ||
      axiosError.message ||
      "Request failed.";
    return new ApiError(message, {
      status: axiosError.response?.status,
      code: data?.error?.code,
      requestId:
        data?.request_id ||
        axiosError.response?.headers?.["x-request-id"] ||
        axiosError.response?.headers?.["x-correlation-id"],
      details: data?.error?.details || data?.detail,
      retryAfterSeconds: parseRetryAfterSeconds(axiosError.response?.headers?.["retry-after"]),
    });
  }
  if (error instanceof Error) return new ApiError(error.message);
  return new ApiError("Unexpected error.");
}

export function shouldRetryApiError(failureCount: number, error: unknown) {
  const normalized = normalizeApiError(error);
  if (normalized.status && normalized.status >= 400 && normalized.status < 500) {
    return (normalized.status === 408 || normalized.status === 429) && failureCount < 2;
  }
  return failureCount < 2;
}

/**
 * Plain-language text for a known HTTP status/code, for surfaces a normal
 * user sees (toasts, inline error states). Falls back to the backend's own
 * message for anything not in this list, so an unrecognized failure still
 * shows real information instead of a generic dead end.
 *
 * Technical detail (status code, error code, request ID) is preserved on the
 * underlying ApiError for logs/support -- this function only controls what
 * is rendered as the primary sentence.
 */
export function getUserFriendlyErrorMessage(error: unknown): string {
  const normalized = normalizeApiError(error);

  const byCode: Record<string, string> = {
    audio_file_too_large: "This audio file is too large. Please upload a smaller file.",
    decoded_audio_too_large: "This audio file is too large to analyze. Please upload a smaller file.",
    unsupported_audio_format:
      "We can't analyze this audio format. Please upload a supported audio file.",
    video_stream_not_allowed:
      "This file contains video. Please upload an audio-only file.",
    corrupted_audio: "This audio file appears to be damaged and couldn't be read.",
    empty_audio_file: "This audio file is empty.",
    audio_too_short: "This audio clip is too short to analyze.",
    audio_duration_exceeded: "This audio clip is too long. Please upload a shorter clip.",
    silent_audio: "This audio appears to be silent, so it can't be analyzed.",
    audio_processing_unavailable:
      "Audio analysis is temporarily unavailable. Please try again shortly.",
    audio_processing_timeout: "Analyzing this audio took too long. Please try again.",
    model_unavailable: "The analysis service is temporarily unavailable. Please try again shortly.",
    prediction_queue_full: "The analysis service is busy right now. Please try again shortly.",
    prediction_runner_shutting_down: "The analysis service is restarting. Please try again shortly.",
    principal_prediction_limit_exceeded:
      "You have reached the limit of analyses running at once. Please wait for one to finish.",
    authentication_failed: "Your session has expired. Please sign in again.",
    request_validation_error: "That request wasn't valid. Please check the form and try again.",
    not_found: "We couldn't find what you were looking for.",
    internal_server_error: "Something went wrong on our end. Please try again shortly.",
  };

  if (normalized.code && byCode[normalized.code]) {
    return byCode[normalized.code];
  }

  // Voice XAI's own "disabled"/mode-unavailable text is accurate but
  // technical; a normal user just needs to know explanation isn't available.
  if (/voice xai is disabled|voice xai mode is not available/i.test(normalized.message)) {
    return "An explanation isn't available for this result right now.";
  }

  const byStatus: Record<number, string> = {
    401: "Your session has expired. Please sign in again.",
    403: "You don't have access to do that.",
    404: "We couldn't find what you were looking for.",
    408: "That took too long. Please try again.",
    413: "This file is too large.",
    415: "We can't analyze this audio format. Please upload a supported audio file.",
    422: "That request wasn't valid. Please check the form and try again.",
    429: "You're doing that a bit too fast. Please wait a moment and try again.",
    503: "The service is busy or temporarily unavailable. Please try again shortly.",
  };

  if (normalized.status && byStatus[normalized.status]) {
    return byStatus[normalized.status];
  }

  if (normalized.status && normalized.status >= 500) {
    return "Something went wrong on our end. Please try again shortly.";
  }

  return normalized.message;
}
