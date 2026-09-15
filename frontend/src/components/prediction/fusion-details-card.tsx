import { GitMerge } from "lucide-react";
import { Card, CardContent, CardHeader, CardDescription, CardTitle } from "@/components/ui/card";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { StatusBadge } from "@/components/prediction/status-badge";
import { getDetectorDisplayName } from "@/lib/copy/detector-names";
import { formatProbability } from "@/lib/formatters";
import type { FusionResult } from "@/types/api";

export function FusionDetailsCard({ fusion }: { fusion?: FusionResult | null }) {
  if (!fusion) {
    return (
      <Card>
        <CardHeader><CardTitle>Combined analysis details</CardTitle></CardHeader>
        <CardContent><p className="text-sm text-muted-foreground">Fusion details are unavailable for this prediction.</p></CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <GitMerge className="h-5 w-5 text-primary" aria-hidden="true" />
          Combined analysis details
        </CardTitle>
        <CardDescription>How the individual detection methods above were combined into one result.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="flex flex-wrap gap-2">
          <StatusBadge status={fusion.status} />
        </div>
        <div className="rounded-lg border p-4">
          <h4 className="text-sm font-medium">Detection methods included</h4>
          <p className="mt-3 text-sm text-muted-foreground">
            {fusion.contributing_branches.length
              ? fusion.contributing_branches.map((name) => getDetectorDisplayName(name)).join(", ")
              : "None reported"}
          </p>
        </div>

        <Accordion type="single" collapsible>
          <AccordionItem value="technical-details" className="border-0">
            <AccordionTrigger className="py-2 text-xs text-muted-foreground hover:no-underline">
              Technical details
            </AccordionTrigger>
            <AccordionContent className="space-y-4">
              <dl className="grid gap-3 text-sm md:grid-cols-3">
                <div><dt className="text-muted-foreground">Method</dt><dd>{fusion.method}</dd></div>
                <div><dt className="text-muted-foreground">Config version</dt><dd>{fusion.config_version || "Not reported"}</dd></div>
                <div><dt className="text-muted-foreground">Minimum successful branches</dt><dd>{fusion.minimum_successful_branches ?? "Not reported"}</dd></div>
              </dl>
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-lg border p-4">
                  <h4 className="text-sm font-medium">Configured weights</h4>
                  <dl className="mt-3 space-y-2 text-sm">
                    {Object.entries(fusion.branch_weights).map(([branch, weight]) => (
                      <div key={branch} className="flex justify-between gap-3">
                        <dt className="break-all text-muted-foreground">{branch}</dt>
                        <dd>{formatProbability(weight)}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
                <div className="rounded-lg border p-4">
                  <h4 className="text-sm font-medium">Contributors</h4>
                  <p className="mt-3 text-sm text-muted-foreground">Included: {fusion.contributing_branches.join(", ") || "None reported"}</p>
                  <div className="mt-3 space-y-1 text-sm">
                    {Object.entries(fusion.excluded_branches).length ? Object.entries(fusion.excluded_branches).map(([branch, reason]) => (
                      <p key={branch}><span className="text-muted-foreground">{branch}:</span> {reason}</p>
                    )) : <p className="text-muted-foreground">No excluded branches reported.</p>}
                  </div>
                </div>
              </div>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
        {fusion.warning ? <p className="rounded-md bg-amber-500/10 p-3 text-sm text-amber-200">{fusion.warning}</p> : null}
      </CardContent>
    </Card>
  );
}
