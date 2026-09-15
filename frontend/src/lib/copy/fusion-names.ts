/**
 * Plain-language names for fusion contracts, keyed by the backend's public
 * `fusion_version` string.
 *
 * Same convention as `detector-names.ts`: never invents data, only relabels
 * a real backend-provided identifier. An unrecognised version falls back to
 * the raw string so the UI stays correct if the backend adds a new contract.
 */

type FusionVersionCopy = {
  friendlyName: string;
  /** Short technical descriptor shown alongside the friendly name. */
  descriptor: string;
};

const FUSION_VERSION_COPY: Record<string, FusionVersionCopy> = {
  "fusion-convex-4branch-v3": {
    friendlyName: "Fusion V3",
    descriptor: "Four-branch learned fusion",
  },
  "legacy-3branch-frozen-v1": {
    friendlyName: "Legacy detector",
    descriptor: "Three-branch fallback fusion",
  },
};

export function getFusionVersionDisplayName(version: string): string {
  return FUSION_VERSION_COPY[version]?.friendlyName ?? version;
}

export function getFusionVersionDescriptor(version: string): string | null {
  return FUSION_VERSION_COPY[version]?.descriptor ?? null;
}
