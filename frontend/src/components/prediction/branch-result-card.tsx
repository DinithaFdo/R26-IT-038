import { Cpu } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { BranchStateBadge } from "@/components/prediction/branch-state-badge";
import { ProbabilityBar } from "@/components/prediction/probability-bar";
import { LabelBadge } from "@/components/prediction/status-badge";
import { describeBranchState } from "@/lib/models/branch-state";
import { getDetectorDescription, getDetectorDisplayName } from "@/lib/copy/detector-names";
import { formatDuration, formatProbability } from "@/lib/formatters";
import type { BranchPrediction } from "@/types/api";

function metadataString(branch: BranchPrediction, key: string): string | null {
  const value = branch.metadata?.[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

export function BranchResultCard({ branch }: { branch: BranchPrediction }) {
  const branchState = describeBranchState(branch);
  const friendlyName = getDetectorDisplayName(branch.model_name);
  const detectorDescription = getDetectorDescription(branch.model_name);
  const architecture = metadataString(branch, "architecture");
  const device = metadataString(branch, "device");
  const modelVersion =
    metadataString(branch, "model_version") ??
    (typeof branch.metadata?.model_provenance === "object" &&
    branch.metadata.model_provenance !== null
      ? ((branch.metadata.model_provenance as Record<string, unknown>).model_version as
          | string
          | undefined) ?? null
      : null);

  return (
    <Card className="h-full">
      <CardHeader>
        <div className="flex items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <Cpu className="h-4 w-4 text-primary" aria-hidden="true" />
              {friendlyName}
            </CardTitle>
            <p className="mt-1 text-xs text-muted-foreground">
              Technical name: {branch.display_name}
            </p>
          </div>
          <BranchStateBadge state={branchState.state} label={branchState.label} />
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        <p className="text-xs text-muted-foreground">
          {(branchState.state === "real" || branchState.state === "real_unverified") &&
          detectorDescription
            ? detectorDescription
            : branchState.description}
        </p>

        {branchState.showsScores ? (
          <>
            <div className="flex flex-wrap gap-2">
              <LabelBadge label={branch.prediction} />
            </div>
            <ProbabilityBar probabilities={branch.probabilities} />
          </>
        ) : (
          /* No scores are invented for a branch that produced none. */
          <p className="rounded-md border border-dashed border-border p-3 text-xs text-muted-foreground">
            No scores were produced by this branch.
          </p>
        )}

        {branch.error ? (
          <p className="rounded-md bg-destructive/10 p-3 text-xs text-destructive">
            {branch.error}
          </p>
        ) : null}

        <Accordion type="single" collapsible>
          <AccordionItem value="technical-details" className="border-0">
            <AccordionTrigger className="py-2 text-xs text-muted-foreground hover:no-underline">
              Technical details
            </AccordionTrigger>
            <AccordionContent>
              <dl className="grid grid-cols-2 gap-3 text-xs">
                <div className="col-span-2">
                  <dt className="text-muted-foreground">Model name</dt>
                  <dd className="break-all font-medium">{branch.model_name}</dd>
                </div>
                {branchState.showsScores ? (
                  <>
                    <div>
                      <dt className="text-muted-foreground">Confidence</dt>
                      <dd className="font-medium">{formatProbability(branch.confidence)}</dd>
                    </div>
                    <div>
                      <dt className="text-muted-foreground">Inference</dt>
                      <dd className="font-medium">
                        {formatDuration(branch.processing_time_ms / 1000)}
                      </dd>
                    </div>
                  </>
                ) : null}
                <div>
                  <dt className="text-muted-foreground">Architecture</dt>
                  <dd className="break-all font-medium">{architecture ?? "Not reported"}</dd>
                </div>
                <div>
                  <dt className="text-muted-foreground">Device</dt>
                  <dd className="font-medium">{device ?? "Not reported"}</dd>
                </div>
                {modelVersion ? (
                  <div className="col-span-2">
                    <dt className="text-muted-foreground">Model version</dt>
                    <dd className="break-all font-medium">{modelVersion}</dd>
                  </div>
                ) : null}
                {branchState.technicalNote ? (
                  <div className="col-span-2">
                    <dt className="text-muted-foreground">Verification status</dt>
                    <dd className="font-medium">{branchState.technicalNote}</dd>
                  </div>
                ) : null}
              </dl>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  );
}
