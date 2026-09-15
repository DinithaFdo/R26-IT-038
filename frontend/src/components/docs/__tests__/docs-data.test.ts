import { describe, expect, it } from "vitest";
import { endpointDocs, mcpTools } from "@/components/docs/docs-data";

describe("developer documentation data", () => {
  it("documents verified prediction and model endpoints", () => {
    expect(endpointDocs.map((endpoint) => endpoint.path)).toEqual(
      expect.arrayContaining([
        "/api/v1/predictions",
        "/api/v1/me/predictions",
        "/api/v1/voice/models/health",
        "/api/v1/external/predictions",
      ]),
    );
  });

  it("documents actual MCP tool names and scopes", () => {
    expect(mcpTools.map((tool) => tool.name)).toEqual([
      "multiscope_create_prediction",
      "multiscope_get_prediction",
      "multiscope_list_predictions",
      "multiscope_get_model_status",
      "multiscope_delete_prediction",
    ]);
    expect(mcpTools.every((tool) => tool.scope.startsWith("prediction:"))).toBe(true);
  });
});
