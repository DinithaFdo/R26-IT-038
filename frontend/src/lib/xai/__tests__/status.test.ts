import { describe, expect, it } from "vitest";
import {
  describeMetric,
  isExplanationInProgress,
  isExplanationTerminal,
  predictionLabelText,
  SHAP_DIRECTION_LABELS,
} from "@/lib/xai/status";

describe("explanation lifecycle", () => {
  it("treats queued/running/partial as in progress so they are polled, not errored", () => {
    expect(isExplanationInProgress("queued")).toBe(true);
    expect(isExplanationInProgress("running")).toBe(true);
    expect(isExplanationInProgress("partial")).toBe(true);
  });

  it("stops polling on completed and failed", () => {
    expect(isExplanationInProgress("completed")).toBe(false);
    expect(isExplanationInProgress("failed")).toBe(false);
    expect(isExplanationTerminal("completed")).toBe(true);
    expect(isExplanationTerminal("failed")).toBe(true);
  });

  it("stops polling on blocked, which only leaves that state via an explicit retry", () => {
    expect(isExplanationInProgress("blocked")).toBe(false);
    expect(isExplanationTerminal("blocked")).toBe(true);
  });

  it("does not poll when the status is unknown", () => {
    expect(isExplanationInProgress(undefined)).toBe(false);
  });
});

describe("describeMetric", () => {
  it("reports an absent metric as unavailable rather than zero", () => {
    const described = describeMetric(null);
    expect(described.available).toBe(false);
    expect(described.value).toBeNull();
    expect(described.label).toBe("Not available");
  });

  it("reports a not_computed metric as unavailable and keeps the backend reason", () => {
    const described = describeMetric({
      status: "not_computed",
      value: null,
      scope: "offline_validation",
      reason: "Requires timestamp-labelled evaluation audio.",
      dataset_version: null,
    });
    expect(described.available).toBe(false);
    expect(described.value).toBeNull();
    expect(described.label).toBe("Not computed");
    expect(described.reason).toContain("timestamp-labelled");
  });

  it("distinguishes a genuine measured zero from a missing metric", () => {
    const described = describeMetric({
      status: "available",
      value: 0,
      scope: "per_analysis",
      reason: null,
      dataset_version: null,
    });
    expect(described.available).toBe(true);
    expect(described.value).toBe(0);
  });

  it("reports not_applicable distinctly from not_computed", () => {
    const described = describeMetric({
      status: "not_applicable",
      value: null,
      scope: "per_analysis",
      reason: null,
      dataset_version: null,
    });
    expect(described.label).toBe("Not applicable");
  });
});

describe("label mapping", () => {
  it("maps backend labels to user-facing text without altering stored values", () => {
    expect(predictionLabelText("spoof")).toBe("Spoof suspected");
    expect(predictionLabelText("bonafide")).toBe("Bonafide");
    expect(predictionLabelText(null)).toBe("Not available");
  });

  it("maps SHAP direction from the backend field, in both directions", () => {
    expect(SHAP_DIRECTION_LABELS.toward_spoof).toBe("Toward spoof");
    expect(SHAP_DIRECTION_LABELS.toward_bonafide).toBe("Toward bonafide");
  });
});
