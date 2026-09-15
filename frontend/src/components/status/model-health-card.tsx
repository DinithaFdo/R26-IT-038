import { Cpu } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { BranchStateBadge } from "@/components/prediction/branch-state-badge";
import type { BranchState } from "@/lib/models/branch-state";
import type { ModelHealth } from "@/types/api";

/**
 * Health reports a *configured* branch rather than a completed run, so it maps
 * to a display state slightly differently from a branch prediction: there is no
 * "failed run" here, but there is "configured real but not yet loaded".
 */
function healthState(model: ModelHealth): { state: BranchState; label: string } {
  if (model.mode === "disabled") {
    return { state: "disabled", label: "Not available" };
  }
  if (model.mode === "dummy") {
    return { state: "dummy", label: "Development placeholder" };
  }
  if (model.mode === "real") {
    if (!model.is_loaded) {
      return { state: "failed", label: "Not loaded" };
    }
    return model.research_ready
      ? { state: "real", label: "Trained model" }
      : { state: "real_unverified", label: "Trained model" };
  }
  return { state: "failed", label: String(model.mode) };
}

function verificationText(model: ModelHealth): string | null {
  if (model.mode !== "real") return null;
  const preprocessing = model.preprocessing_verified;
  const classMapping = model.class_mapping_verified;
  if (preprocessing == null && classMapping == null) return null;
  if (preprocessing && classMapping) return "Feature pipeline and class mapping verified";

  const outstanding = [
    preprocessing ? null : "feature pipeline",
    classMapping ? null : "class mapping",
  ].filter(Boolean);
  return `Verification pending: ${outstanding.join(" and ")}`;
}

export function ModelHealthCard({ models }: { models: ModelHealth[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Model branch health</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {models.map((model) => {
          const { state, label } = healthState(model);
          const verification = verificationText(model);
          const isDisabled = state === "disabled";
          return (
            <div key={model.branch_name || model.model_name} className="rounded-lg border p-4">
              <div className="flex items-start justify-between gap-3">
                <Cpu className="h-5 w-5 text-primary" aria-hidden="true" />
                <BranchStateBadge state={state} label={label} />
              </div>
              <h3 className="mt-3 font-medium">{model.display_name}</h3>
              <p className="mt-1 break-all text-xs text-muted-foreground">
                {model.branch_name || model.model_name}
              </p>
              <dl className="mt-4 space-y-2 text-xs">
                <div className="flex justify-between gap-3">
                  <dt className="text-muted-foreground">Loaded</dt>
                  <dd>{isDisabled ? "Not applicable" : model.is_loaded ? "Yes" : "No"}</dd>
                </div>
                <div className="flex justify-between gap-3">
                  <dt className="text-muted-foreground">Lifecycle</dt>
                  <dd>{isDisabled ? "Not integrated" : model.lifecycle_state || "Unknown"}</dd>
                </div>
                <div className="flex justify-between gap-3">
                  <dt className="text-muted-foreground">Device</dt>
                  <dd>
                    {isDisabled
                      ? "None"
                      : model.resolved_device || model.requested_device || "Not reported"}
                  </dd>
                </div>
                <div className="flex justify-between gap-3">
                  <dt className="text-muted-foreground">Checkpoint</dt>
                  <dd>
                    {isDisabled
                      ? "None"
                      : model.checkpoint_configured
                        ? "Configured"
                        : "Not configured"}
                  </dd>
                </div>
              </dl>
              {verification ? (
                <p className="mt-3 text-xs text-muted-foreground">{verification}</p>
              ) : null}
              {isDisabled ? (
                <p className="mt-3 rounded-md border border-dashed border-border p-2 text-xs text-muted-foreground">
                  No trained model in this deployment. This branch is excluded from results.
                </p>
              ) : null}
              {model.warning ? (
                <p className="mt-3 rounded-md bg-amber-500/10 p-2 text-xs text-amber-700 dark:text-amber-300">
                  {model.warning}
                </p>
              ) : null}
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}
