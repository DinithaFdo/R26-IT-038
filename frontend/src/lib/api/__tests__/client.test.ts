import { describe, expect, it } from "vitest";
import { createApiClient, voiceApiClient, textApiClient, apiClient } from "@/lib/api/client";
import { env } from "@/lib/config/env";

describe("API Clients Configuration", () => {
  it("exports configured voiceApiClient, textApiClient, and apiClient alias", () => {
    expect(voiceApiClient).toBeDefined();
    expect(textApiClient).toBeDefined();
    expect(apiClient).toBe(voiceApiClient);
  });

  it("fails fast in request interceptor when service base URL is missing", async () => {
    const unconfiguredClient = createApiClient({
      serviceName: "Test Service",
      getBaseUrl: () => "",
      envVarName: "NEXT_PUBLIC_TEST_API_BASE_URL",
    });

    await expect(unconfiguredClient.get("/test-endpoint")).rejects.toThrow(
      "[Test Service] API Base URL is not configured. Please set NEXT_PUBLIC_TEST_API_BASE_URL in your environment variables.",
    );
  });

  it("sets baseURL properly when configured", async () => {
    const configuredClient = createApiClient({
      serviceName: "Custom Service",
      getBaseUrl: () => "https://api.example.com",
      envVarName: "NEXT_PUBLIC_CUSTOM_API_BASE_URL",
    });

    expect(configuredClient.defaults.baseURL).toBe("https://api.example.com");
  });
});
