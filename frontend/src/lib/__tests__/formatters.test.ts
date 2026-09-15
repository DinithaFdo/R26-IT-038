import { describe, expect, it } from "vitest";
import { formatBytes, formatDuration, formatProbability, isTerminalStatus } from "@/lib/formatters";

describe("formatters", () => {
  it("formats probabilities without implying certainty", () => {
    expect(formatProbability(0.625)).toBe("62.50%");
    expect(formatProbability(null)).toBe("Not reported");
  });

  it("formats bytes and durations", () => {
    expect(formatBytes(1024 * 1024)).toBe("1.0 MB");
    expect(formatDuration(67)).toBe("1:07");
  });

  it("recognizes terminal statuses", () => {
    expect(isTerminalStatus("completed")).toBe(true);
    expect(isTerminalStatus("failed")).toBe(true);
    expect(isTerminalStatus("processing")).toBe(false);
  });
});
