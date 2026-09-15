export const SUPPORTED_AUDIO_EXTENSIONS = [
  "wav",
  "flac",
  "mp3",
  "m4a",
  "aac",
  "opus",
  "ogg",
  "webm",
] as const;

export const SUPPORTED_AUDIO_MIME_TYPES = [
  "audio/wav",
  "audio/x-wav",
  "audio/flac",
  "audio/mpeg",
  "audio/mp4",
  "audio/aac",
  "audio/ogg",
  "audio/opus",
  "audio/webm",
] as const;

export const RECORDING_MIME_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/ogg;codecs=opus",
  "audio/mp4",
  "audio/webm",
] as const;

export const FILE_ACCEPT = SUPPORTED_AUDIO_EXTENSIONS.map((extension) => `.${extension}`).join(",");
