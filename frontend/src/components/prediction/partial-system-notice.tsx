import { Info } from "lucide-react";
import {
  describeBranchState,
  describeResearchBlocker,
  isPartialSystemResult,
} from "@/lib/models/branch-state";
import { getDetectorDisplayName } from "@/lib/copy/detector-names";
import type { BranchPrediction, FusionResult } from "@/types/api";

/**
 * States plainly which detection methods produced the result.
 *
 * Rendered whenever fewer than the full detection-method set contributed —
 * the reader should never have to infer from a positive-looking result that
 * every planned method was actually used. Scientific-verification detail
 * (research eligibility blockers) is precise/technical and only shown in a
 * secondary line, not as the primary message.
 */
export function PartialSystemNotice({
  branches,
  fusion,
}: {
  branches: BranchPrediction[];
  fusion?: FusionResult | null;
}) {
  const contributing = branches.filter((branch) => describeBranchState(branch).contributes);
  const unavailable = branches.filter(
    (branch) => describeBranchState(branch).state === "disabled",
  );

  const partial = isPartialSystemResult(fusion) || unavailable.length > 0;
  if (!partial) return null;

  const blockers = fusion?.research_blockers ?? [];

  return (
    <div
      className="rounded-lg border border-amber-200 bg-amber-50 p-4 dark:border-amber-900 dark:bg-amber-950"
      role="note"
      aria-labelledby="partial-system-heading"
    >
      <div className="flex items-start gap-3">
        <Info
          className="mt-0.5 h-4 w-4 shrink-0 text-amber-700 dark:text-amber-300"
          aria-hidden="true"
        />
        <div className="space-y-2 text-sm">
          <p
            id="partial-system-heading"
            className="font-semibold text-amber-900 dark:text-amber-200"
          >
            Based on the methods currently available
          </p>

          <p className="text-amber-900 dark:text-amber-200">
            This result combines{" "}
            {contributing.map((branch) => getDetectorDisplayName(branch.model_name)).join(" and ")}.
            {unavailable.length > 0 ? (
              <>
                {" "}
                {unavailable.map((branch) => getDetectorDisplayName(branch.model_name)).join(" and ")}{" "}
                {unavailable.length === 1 ? "is" : "are"} still in development and not
                included yet.
              </>
            ) : null}
          </p>

          {blockers.length > 0 ? (
            <details className="text-amber-900 dark:text-amber-200">
              <summary className="cursor-pointer text-xs font-medium">Technical details</summary>
              <ul className="mt-2 list-disc space-y-1 pl-5">
                {blockers.map((blocker) => (
                  <li key={blocker}>{describeResearchBlocker(blocker)}</li>
                ))}
              </ul>
            </details>
          ) : null}

          <p className="text-xs text-amber-800 dark:text-amber-300">
            This result must not be interpreted as full MULTI-SCOPE system performance.
          </p>
        </div>
      </div>
    </div>
  );
}
