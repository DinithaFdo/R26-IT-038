const numberFromEnv = (value: string | undefined, fallback: number) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
};

export const env = {
  appName: process.env.NEXT_PUBLIC_APP_NAME || "MULTI-SCOPE",
  apiBaseUrl: process.env.NEXT_PUBLIC_API_BASE_URL || "",
  voiceApiBaseUrl:
    process.env.NEXT_PUBLIC_VOICE_API_BASE_URL ||
    process.env.NEXT_PUBLIC_API_BASE_URL ||
    "",
  textApiBaseUrl:
    process.env.NEXT_PUBLIC_TEXT_API_BASE_URL ||
    process.env.NEXT_PUBLIC_API_BASE_URL ||
    "",
  enableAudioRecording:
    process.env.NEXT_PUBLIC_ENABLE_AUDIO_RECORDING !== "false",
  systemStatusRefreshMs: numberFromEnv(
    process.env.NEXT_PUBLIC_SYSTEM_STATUS_REFRESH_MS,
    30_000,
  ),
  predictionStatusPollMs: numberFromEnv(
    process.env.NEXT_PUBLIC_PREDICTION_STATUS_POLL_MS,
    2_000,
  ),
  apiTimeoutMs: numberFromEnv(
    process.env.NEXT_PUBLIC_API_TIMEOUT_MS,
    120_000,
  ),
  maxUploadSizeMb: numberFromEnv(
    process.env.NEXT_PUBLIC_MAX_UPLOAD_SIZE_MB,
    25,
  ),
  minAudioDurationSeconds: numberFromEnv(
    process.env.NEXT_PUBLIC_MIN_AUDIO_DURATION_SECONDS,
    1,
  ),
  maxAudioDurationSeconds: numberFromEnv(
    process.env.NEXT_PUBLIC_MAX_AUDIO_DURATION_SECONDS,
    180,
  ),
  clerkPublishableKey: process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY || "",
};

export const hasApiBaseUrl = Boolean(env.apiBaseUrl || env.voiceApiBaseUrl);
export const hasVoiceApiBaseUrl = Boolean(env.voiceApiBaseUrl);
export const hasTextApiBaseUrl = Boolean(env.textApiBaseUrl);
export const isClerkConfigured = Boolean(env.clerkPublishableKey);
