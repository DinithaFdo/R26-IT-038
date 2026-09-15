import { describe, expect, it, vi } from "vitest";
import { selectRecordingMimeType } from "@/features/audio-recorder/mime";

describe("selectRecordingMimeType", () => {
  it("returns the first supported recording MIME type", () => {
    vi.stubGlobal("MediaRecorder", {
      isTypeSupported: (mime: string) => mime === "audio/ogg;codecs=opus",
    });
    expect(selectRecordingMimeType()).toBe("audio/ogg;codecs=opus");
    vi.unstubAllGlobals();
  });
});
