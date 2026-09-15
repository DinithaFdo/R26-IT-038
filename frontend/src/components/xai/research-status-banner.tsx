import { FlaskConical, Info, ShieldAlert } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { DECISION_SUPPORT_NOTICE } from "@/lib/xai/status";

/**
 * Surfaces the backend's own `development_placeholder` / `research_eligible`
 * flags. These are never hidden and never styled to look like validated
 * research output — a polished UI must not imply the scientific pipeline is
 * more complete than the backend reports it to be.
 */
export function ResearchStatusBanner({
  developmentPlaceholder,
  researchEligible,
  containsDummyBranches,
}: {
  developmentPlaceholder: boolean;
  researchEligible: boolean;
  containsDummyBranches?: boolean;
}) {
  if (!developmentPlaceholder && researchEligible && !containsDummyBranches) return null;

  return (
    <Alert variant="warning">
      <FlaskConical className="h-4 w-4" aria-hidden="true" />
      <AlertTitle className="flex flex-wrap items-center gap-2">
        Development-stage result — research validation pending
        <span className="flex flex-wrap gap-1.5">
          {developmentPlaceholder ? (
            <Badge variant="warning" className="gap-1">
              <FlaskConical className="h-3 w-3" aria-hidden="true" />
              Development evidence
            </Badge>
          ) : null}
          {!researchEligible ? (
            <Badge variant="warning" className="gap-1">
              <ShieldAlert className="h-3 w-3" aria-hidden="true" />
              Research validation pending
            </Badge>
          ) : null}
          {containsDummyBranches ? (
            <Badge variant="warning" className="gap-1">
              <Info className="h-3 w-3" aria-hidden="true" />
              Contains dummy branch
            </Badge>
          ) : null}
        </span>
      </AlertTitle>
      <AlertDescription className="space-y-1.5">
        <p>
          This explanation is reported by the backend as development output. It must not be used as
          evidence in a research result, experiment, or publication.
        </p>
        {containsDummyBranches ? (
          <p>
            At least one classifier branch ran in dummy mode, so the underlying prediction is not a
            real model result.
          </p>
        ) : null}
        <p className="font-medium">{DECISION_SUPPORT_NOTICE}</p>
      </AlertDescription>
    </Alert>
  );
}

/**
 * Compact inline pair of flags for headers and list rows, where the full
 * banner would be too heavy but the state must still be visible.
 */
export function ResearchStatusBadges({
  developmentPlaceholder,
  researchEligible,
}: {
  developmentPlaceholder: boolean;
  researchEligible: boolean;
}) {
  return (
    <span className="flex flex-wrap gap-1.5">
      <Badge variant={developmentPlaceholder ? "warning" : "success"} className="gap-1">
        <FlaskConical className="h-3 w-3" aria-hidden="true" />
        {developmentPlaceholder ? "Development evidence" : "Model-backed evidence"}
      </Badge>
      <Badge variant={researchEligible ? "success" : "warning"} className="gap-1">
        <ShieldAlert className="h-3 w-3" aria-hidden="true" />
        {researchEligible ? "Research eligible" : "Research validation pending"}
      </Badge>
    </span>
  );
}
