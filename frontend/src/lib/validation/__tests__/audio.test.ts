import { describe, expect, it } from "vitest";
import { validateAudioFile, validateKnownDuration } from "@/lib/validation/audio";

describe("audio validation", () => {
  it("accepts backend-supported extensions", () => {
    const file = new File([new Uint8Array([1, 2, 3])], "sample.webm", { type: "audio/webm" });
    expect(validateAudioFile(file).valid).toBe(true);
  });

  it("rejects unsupported extensions", () => {
    const file = new File([new Uint8Array([1])], "sample.txt", { type: "text/plain" });
    expect(validateAudioFile(file).valid).toBe(false);
  });

  it("validates known browser durations as frontend hints", () => {
    expect(validateKnownDuration(0.2).valid).toBe(false);
    expect(validateKnownDuration(5).valid).toBe(true);
  });
});
