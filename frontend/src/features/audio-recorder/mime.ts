import { RECORDING_MIME_CANDIDATES } from "@/lib/constants/audio";

export function selectRecordingMimeType() {
  if (typeof window === "undefined" || typeof MediaRecorder === "undefined") {
    return "";
  }
  return RECORDING_MIME_CANDIDATES.find((mimeType) =>
    MediaRecorder.isTypeSupported(mimeType),
  ) || "";
}
