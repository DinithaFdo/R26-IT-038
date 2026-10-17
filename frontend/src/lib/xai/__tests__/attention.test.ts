import { describe, expect, it } from "vitest";
import { attentionAtTime, attentionSegments, attentionVector, interpolateAttention } from "../attention";

describe("source-time attention gaps", () => {
  it("preserves null frames instead of zero scores or dropping the series", () => {
    expect(attentionVector([null, 0, 1.2, null])).toEqual([null, 0, 1.2, null]);
    expect(attentionVector([0, Number.NaN])).toBeNull();
  });
  it("does not interpolate missing frames", () => {
    expect(interpolateAttention([null, 1, 2, null], 0, 7)).toBeNull();
    expect(interpolateAttention([null, 1, 2, null], 1, 7)).toBeNull();
    expect(interpolateAttention([null, 1, 2, null], 3, 7)).toBe(1.5);
    expect(interpolateAttention([null, 1, 2, null], 5, 7)).toBeNull();
  });
  it("maps source timestamps without extending endpoints", () => {
    const values = [1, 2];
    const times = [7, 13];
    expect(attentionAtTime(values, times, 2)).toBeNull();
    expect(attentionAtTime(values, times, 7)).toBe(1);
    expect(attentionAtTime(values, times, 10)).toBe(1.5);
    expect(attentionAtTime(values, times, 18)).toBeNull();
  });
  it("does not draw or smooth across unknown gaps", () => {
    expect(attentionSegments([null, 1, 2, null, 3, null], [0, 7, 8, 9, 12, 20], 20)).toEqual([
      [{ time: 7, value: 1 }, { time: 8, value: 2 }],
      [{ time: 12, value: 3 }],
    ]);
    expect(attentionAtTime([1, null, 2], [7, 10, 13], 11)).toBeNull();
  });
  it("keeps legacy complete series usable", () => {
    expect(attentionSegments([1, 2, 3], null, 2)).toEqual([
      [{ time: 0, value: 1 }, { time: 1, value: 2 }, { time: 2, value: 3 }],
    ]);
  });
});
