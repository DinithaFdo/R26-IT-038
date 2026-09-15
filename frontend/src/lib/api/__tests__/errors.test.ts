import { describe, expect, it } from "vitest";
import { AxiosError, AxiosHeaders } from "axios";
import {
  ApiError,
  getUserFriendlyErrorMessage,
  normalizeApiError,
  shouldRetryApiError,
} from "@/lib/api/errors";

function axiosErrorWithResponse(status: number, data: unknown, headers: Record<string, string> = {}) {
  const error = new AxiosError("Request failed");
  error.response = {
    status,
    statusText: "",
    headers: new AxiosHeaders(headers),
    config: {} as never,
    data,
  };
  return error;
}

describe("normalizeApiError", () => {
  it("passes an ApiError through unchanged", () => {
    const original = new ApiError("already normalized");
    expect(normalizeApiError(original)).toBe(original);
  });

  it("extracts the backend error envelope shape (request_id, error.code, error.message)", () => {
    const error = axiosErrorWithResponse(429, {
      request_id: "req_abc",
      error: { code: "rate_limit_exceeded", message: "Rate limit exceeded.", details: null },
    });
    const normalized = normalizeApiError(error);

    expect(normalized.status).toBe(429);
    expect(normalized.code).toBe("rate_limit_exceeded");
    expect(normalized.message).toBe("Rate limit exceeded.");
    expect(normalized.requestId).toBe("req_abc");
  });

  it("parses the Retry-After header into retryAfterSeconds", () => {
    const error = axiosErrorWithResponse(
      429,
      { request_id: "rate-limit", error: { code: "rate_limit_exceeded", message: "Rate limit exceeded." } },
      { "retry-after": "5" },
    );
    expect(normalizeApiError(error).retryAfterSeconds).toBe(5);
  });

  it("leaves retryAfterSeconds undefined when no Retry-After header is present", () => {
    const error = axiosErrorWithResponse(500, { error: { code: "internal_server_error", message: "Oops." } });
    expect(normalizeApiError(error).retryAfterSeconds).toBeUndefined();
  });

  it("falls back to the request_id from response headers when the body omits it", () => {
    const error = axiosErrorWithResponse(
      503,
      { error: { code: "prediction_queue_full", message: "Queue is full." } },
      { "x-request-id": "req_from_header" },
    );
    expect(normalizeApiError(error).requestId).toBe("req_from_header");
  });

  it("never throws for a non-axios, non-Error value", () => {
    expect(normalizeApiError("some string").message).toBe("Unexpected error.");
  });

  it("has no status/code for a client-side timeout (the request never got a response)", () => {
    const error = new AxiosError("timeout of 180000ms exceeded");
    const normalized = normalizeApiError(error);
    expect(normalized.status).toBeUndefined();
    expect(normalized.code).toBeUndefined();
    expect(normalized.message).toBe("timeout of 180000ms exceeded");
  });
});

describe("shouldRetryApiError", () => {
  it("does not retry 401/403/404/409/413/415/422", () => {
    for (const status of [401, 403, 404, 409, 413, 415, 422]) {
      const error = axiosErrorWithResponse(status, {});
      expect(shouldRetryApiError(0, error)).toBe(false);
    }
  });

  it("retries 408 and 429 only within the bounded retry budget", () => {
    expect(shouldRetryApiError(0, axiosErrorWithResponse(408, {}))).toBe(true);
    expect(shouldRetryApiError(0, axiosErrorWithResponse(429, {}))).toBe(true);
    expect(shouldRetryApiError(2, axiosErrorWithResponse(408, {}))).toBe(false);
    expect(shouldRetryApiError(2, axiosErrorWithResponse(429, {}))).toBe(false);
  });

  it("allows limited retries for a temporary 503", () => {
    const error = axiosErrorWithResponse(503, {});
    expect(shouldRetryApiError(0, error)).toBe(true);
    expect(shouldRetryApiError(1, error)).toBe(true);
    expect(shouldRetryApiError(2, error)).toBe(false);
  });
});

describe("getUserFriendlyErrorMessage", () => {
  it("maps a known error code to plain language, never the raw backend message", () => {
    const error = axiosErrorWithResponse(413, {
      error: { code: "audio_file_too_large", message: "Uploaded file exceeds MAX_UPLOAD_BYTES." },
    });
    expect(getUserFriendlyErrorMessage(error)).toBe(
      "This audio file is too large. Please upload a smaller file.",
    );
  });

  it("maps session expiry (401 / authentication_failed) to a plain sign-in prompt", () => {
    const error = axiosErrorWithResponse(401, {
      error: { code: "authentication_failed", message: "Authentication failed." },
    });
    expect(getUserFriendlyErrorMessage(error)).toBe("Your session has expired. Please sign in again.");
  });

  it("maps Voice XAI's own technical disabled/mode text to a plain explanation-unavailable sentence", () => {
    const error = axiosErrorWithResponse(503, {
      error: { code: undefined, message: "Voice XAI is disabled." },
    });
    expect(getUserFriendlyErrorMessage(error)).toBe(
      "An explanation isn't available for this result right now.",
    );
  });

  it("falls back to a generic status-based message for an unmapped 5xx", () => {
    const error = axiosErrorWithResponse(502, { error: { code: "bad_gateway", message: "Bad gateway." } });
    expect(getUserFriendlyErrorMessage(error)).toBe(
      "Something went wrong on our end. Please try again shortly.",
    );
  });

  it("falls back to the backend's own message for a status/code this mapping does not recognise", () => {
    const error = axiosErrorWithResponse(409, {
      error: { code: "idempotency_conflict", message: "A prediction with this key already exists." },
    });
    expect(getUserFriendlyErrorMessage(error)).toBe("A prediction with this key already exists.");
  });
});
