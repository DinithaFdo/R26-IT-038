import { z } from "zod";
import { SUPPORTED_AUDIO_EXTENSIONS } from "@/lib/constants/audio";
import { env } from "@/lib/config/env";

export const sourceTypeSchema = z.enum(["dashboard_upload", "live_recording"]);

export const predictionSubmitSchema = z.object({
  sourceType: sourceTypeSchema,
  clientFilename: z.string().trim().max(180).optional(),
});

export function getFileExtension(filename: string) {
  const extension = filename.split(".").pop()?.toLowerCase() || "";
  return extension;
}

export type FileValidationResult = {
  valid: boolean;
  message?: string;
};

export function validateAudioFile(file: File | null): FileValidationResult {
  if (!file) return { valid: false, message: "Choose an audio file first." };
  if (file.size <= 0) return { valid: false, message: "The selected file is empty." };
  const maxBytes = env.maxUploadSizeMb * 1024 * 1024;
  if (file.size > maxBytes) {
    return {
      valid: false,
      message: `The selected file is larger than the ${env.maxUploadSizeMb} MB frontend limit.`,
    };
  }
  const extension = getFileExtension(file.name);
  if (!SUPPORTED_AUDIO_EXTENSIONS.includes(extension as (typeof SUPPORTED_AUDIO_EXTENSIONS)[number])) {
    return {
      valid: false,
      message: `Unsupported extension .${extension || "unknown"}. Accepted: ${SUPPORTED_AUDIO_EXTENSIONS.join(", ")}.`,
    };
  }
  if (file.type && !file.type.startsWith("audio/") && file.type !== "video/webm") {
    return {
      valid: false,
      message: "The browser did not identify this as an audio file.",
    };
  }
  return { valid: true };
}

export function validateKnownDuration(durationSeconds?: number | null): FileValidationResult {
  if (typeof durationSeconds !== "number" || !Number.isFinite(durationSeconds)) {
    return { valid: true };
  }
  if (durationSeconds < env.minAudioDurationSeconds) {
    return {
      valid: false,
      message: `Audio is shorter than ${env.minAudioDurationSeconds} second(s).`,
    };
  }
  if (durationSeconds > env.maxAudioDurationSeconds) {
    return {
      valid: false,
      message: `Audio is longer than ${env.maxAudioDurationSeconds} second(s).`,
    };
  }
  return { valid: true };
}
