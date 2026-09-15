import { describe, expect, it } from "vitest";
import { getDetectorDescription, getDetectorDisplayName } from "@/lib/copy/detector-names";

describe("getDetectorDisplayName", () => {
  it("maps the real backend model_name values to plain-language names", () => {
    expect(getDetectorDisplayName("cnn_acoustic")).toBe("Acoustic Pattern Analysis");
    expect(getDetectorDisplayName("aasist")).toBe("Voice Structure Analysis");
    expect(getDetectorDisplayName("ssl_wavlm_xlsr")).toBe("Temporal Voice Analysis");
    expect(getDetectorDisplayName("glottal_features")).toBe("Vocal Source Analysis");
  });

  it("also maps the canonical/fixture-style model_name aliases", () => {
    expect(getDetectorDisplayName("lfcc_cnn_tcn")).toBe("Acoustic Pattern Analysis");
    expect(getDetectorDisplayName("ssl_sequence")).toBe("Temporal Voice Analysis");
    expect(getDetectorDisplayName("glottal")).toBe("Vocal Source Analysis");
  });

  it("falls back to the raw model name for an unknown detector instead of rendering blank", () => {
    expect(getDetectorDisplayName("future_branch_x")).toBe("future_branch_x");
  });
});

describe("getDetectorDescription", () => {
  it("returns a one-sentence description for a known detector", () => {
    expect(getDetectorDescription("cnn_acoustic")).toMatch(/acoustic texture/i);
  });

  it("returns null for an unknown detector rather than inventing a description", () => {
    expect(getDetectorDescription("future_branch_x")).toBeNull();
  });
});
