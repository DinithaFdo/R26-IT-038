import { Info } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { getFusionVersionDisplayName } from "@/lib/copy/fusion-names";
import type { FusionResult } from "@/types/api";

/**
 * Shown only when the backend reports `fallback_used: true` -- the complete
 * four-branch detector (Fusion V3) was unavailable for this specific
 * request, so the legacy three-branch detector produced the final decision
 * instead. Deliberately the neutral `Alert` variant, not `warning` or
 * `destructive`: a working fallback is not an error, and must not be
 * presented like one. Renders nothing when Fusion V3 succeeded.
 */
export function FusionFallbackNotice({ fusion }: { fusion?: FusionResult | null }) {
  if (!fusion || !fusion.fallback_used) return null;

  return (
    <Alert variant="default">
      <Info className="mb-2 h-4 w-4" aria-hidden="true" />
      <AlertTitle>Fallback detector used</AlertTitle>
      <AlertDescription>
        <p>
          The complete four-branch detector was unavailable for this request, so the legacy
          three-branch detector produced the final decision.
        </p>
        <p className="mt-2 font-mono text-xs text-muted-foreground">
          fusion_version: {fusion.fusion_version} ({getFusionVersionDisplayName(fusion.fusion_version)})
        </p>
      </AlertDescription>
    </Alert>
  );
}
