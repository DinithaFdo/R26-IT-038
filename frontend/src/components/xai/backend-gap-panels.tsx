import { ClipboardCheck, Database } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

/**
 * Panels for report sections whose backing API does not exist yet.
 *
 * These are rendered as explicit, honest gaps rather than as interactive
 * features that appear to work. Building a human-review form that silently
 * discards input, or showing model-wide validation numbers next to a
 * single-clip result, would both misrepresent the system.
 */

export function HumanReviewPanel() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Human review</CardTitle>
        <CardDescription>Analyst sign-off for this analysis.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="rounded-lg border border-dashed p-6 text-center">
          <ClipboardCheck className="mx-auto h-8 w-8 text-muted-foreground" aria-hidden="true" />
          <h3 className="mt-4 text-base font-semibold">
            Human-review persistence is not yet available
          </h3>
          <p className="mx-auto mt-2 max-w-lg text-sm text-muted-foreground">
            The backend does not currently expose an API for storing analyst decisions,
            annotations, or sign-off. No review form is shown here because any input would be
            discarded rather than recorded against this prediction.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

export function ValidationContextPanel() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Validation context</CardTitle>
        <CardDescription>
          Model-level evaluation evidence: accuracy, false-positive and false-negative rates,
          calibration, per-attack-type and subgroup performance.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="rounded-lg border border-dashed p-6 text-center">
          <Database className="mx-auto h-8 w-8 text-muted-foreground" aria-hidden="true" />
          <h3 className="mt-4 text-base font-semibold">Unavailable for this deployment</h3>
          <p className="mx-auto mt-2 max-w-lg text-sm text-muted-foreground">
            The backend does not expose evaluation metadata, so no validation metrics are shown.
            Model-wide validation figures would not describe this individual analysis in any case,
            and are deliberately not displayed alongside a single-clip result.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
