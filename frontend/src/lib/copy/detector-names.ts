/**
 * Plain-language names for classifier branches, keyed by the backend's
 * public `model_name`.
 *
 * The backend's own `display_name` (e.g. "LFCC CNN/TCN", "AASIST") stays
 * available as the technical name in advanced/secondary UI -- this file only
 * supplies what a non-technical reader sees first. Multiple keys map to the
 * same detector because the backend has used slightly different `model_name`
 * values across versions/fixtures (e.g. "ssl_sequence" vs "ssl_wavlm_xlsr");
 * listing both keeps this safe rather than silently falling back to the raw
 * identifier for a name that has simply changed spelling.
 */

type DetectorCopy = {
  friendlyName: string;
  /** One sentence: what this detector actually looks at. */
  description: string;
};

const DETECTOR_COPY: Record<string, DetectorCopy> = {
  cnn_acoustic: {
    friendlyName: "Acoustic Pattern Analysis",
    description: "Looks at short-term acoustic texture for patterns common in synthetic speech.",
  },
  lfcc_cnn_tcn: {
    friendlyName: "Acoustic Pattern Analysis",
    description: "Looks at short-term acoustic texture for patterns common in synthetic speech.",
  },
  aasist: {
    friendlyName: "Voice Structure Analysis",
    description: "Looks at how consistent the voice's structure is across the whole clip.",
  },
  ssl_wavlm_xlsr: {
    friendlyName: "Temporal Voice Analysis",
    description: "Looks at how the voice changes moment to moment across the recording.",
  },
  ssl_sequence: {
    friendlyName: "Temporal Voice Analysis",
    description: "Looks at how the voice changes moment to moment across the recording.",
  },
  glottal_features: {
    friendlyName: "Vocal Source Analysis",
    description: "Looks at signals from the vocal source that are difficult to fake convincingly.",
  },
  glottal: {
    friendlyName: "Vocal Source Analysis",
    description: "Looks at signals from the vocal source that are difficult to fake convincingly.",
  },
};

/** Falls back to the raw model name so an unknown detector never renders blank. */
export function getDetectorDisplayName(modelName: string): string {
  return DETECTOR_COPY[modelName]?.friendlyName ?? modelName;
}

export function getDetectorDescription(modelName: string): string | null {
  return DETECTOR_COPY[modelName]?.description ?? null;
}
