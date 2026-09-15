import { CircleCheck, CircleX, Gauge, GitBranch } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { NotAvailable } from "@/components/xai/evidence-availability";
import { cn } from "@/lib/utils";
import { formatProbability } from "@/lib/formatters";
import { predictionLabelText } from "@/lib/xai/status";
import type { ClassifierSnapshot } from "@/types/api";

function GlottalRole({
  auxiliary,
}: {
  auxiliary: ClassifierSnapshot["auxiliary_evidence"];
}) {
  const probability = auxiliary.glottal_spoof_probability;

  return (
    <div className="rounded-md border p-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="text-xs text-muted-foreground">Glottal auxiliary evidence</p>
          <p className="mt-1 text-sm font-medium">
            {auxiliary.used_for_primary_decision
              ? "Used for primary decision"
              : "Not used for primary decision"}
          </p>
        </div>
        <Badge variant={auxiliary.used_for_primary_decision ? "info" : "secondary"}>
          {auxiliary.used_for_primary_decision ? "Decision input" : "Auxiliary only"}
        </Badge>
      </div>
      <p className="mt-2 text-sm tabular-nums">
        {typeof probability === "number" ? (
          <>Glottal spoof probability: {formatProbability(probability)}</>
        ) : (
          <NotAvailable
            label="Glottal score not reported"
            reason="the backend did not include an auxiliary glottal probability"
          />
        )}
      </p>
    </div>
  );
}

function branchPresentation(status: string) {
  if (status === "success") {
    return {
      card: "border-emerald-200 dark:border-emerald-900",
      icon: "text-emerald-600 dark:text-emerald-400",
      badge: "success" as const,
      Icon: CircleCheck,
    };
  }
  if (status === "failed") {
    return {
      card: "border-destructive/35",
      icon: "text-destructive",
      badge: "destructive" as const,
      Icon: CircleX,
    };
  }
  return {
    card: "border-border",
    icon: "text-muted-foreground",
    badge: "secondary" as const,
    Icon: Gauge,
  };
}

export function ClassifierSnapshotPanel({
  snapshot,
}: {
  snapshot: ClassifierSnapshot;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Classifier snapshot</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_minmax(30rem,1.15fr)] xl:items-start">
          <section className="space-y-4">
            <dl className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-md border border-primary/25 bg-primary/5 p-3">
                <dt className="text-xs text-muted-foreground">Verdict</dt>
                <dd className="mt-1 text-lg font-semibold text-primary">
                  {predictionLabelText(snapshot.verdict)}
                </dd>
              </div>
              <div className="rounded-md border bg-muted/40 p-3">
                <dt className="text-xs text-muted-foreground">Confidence</dt>
                <dd className="mt-1 text-lg font-semibold tabular-nums">
                  {formatProbability(snapshot.confidence)}
                </dd>
              </div>
              <div className="rounded-md border bg-muted/40 p-3">
                <dt className="text-xs text-muted-foreground">Spoof probability</dt>
                <dd className="mt-1 text-base font-semibold tabular-nums text-destructive">
                  {formatProbability(snapshot.spoof_probability)}
                </dd>
              </div>
              <div className="rounded-md border bg-muted/40 p-3">
                <dt className="text-xs text-muted-foreground">Bonafide probability</dt>
                <dd className="mt-1 text-base font-semibold tabular-nums text-emerald-700 dark:text-emerald-400">
                  {formatProbability(snapshot.bonafide_probability)}
                </dd>
              </div>
              <div className="rounded-md border bg-muted/40 p-3 sm:col-span-2">
                <dt className="text-xs text-muted-foreground">Decision threshold</dt>
                <dd className="mt-1 text-sm font-semibold tabular-nums">
                  {formatProbability(snapshot.decision_threshold)}
                </dd>
              </div>
            </dl>

            <GlottalRole auxiliary={snapshot.auxiliary_evidence} />
          </section>

          <section className="rounded-lg border border-primary/15 bg-muted/30 p-4">
            <h4 className="flex items-center gap-1.5 text-sm font-semibold">
              <GitBranch className="h-4 w-4 text-primary" aria-hidden="true" />
              Branch snapshot
            </h4>
            {snapshot.branches.length === 0 ? (
              <p className="mt-2 text-sm text-muted-foreground">
                No branch-level snapshot was persisted for this explanation run.
              </p>
            ) : (
              <ul className="mt-3 grid gap-3 sm:grid-cols-2">
                {snapshot.branches.map((branch) => {
                  const presentation = branchPresentation(branch.status);
                  const StatusIcon = presentation.Icon;
                  return (
                    <li key={branch.branch_name} className={cn("rounded-md border p-3", presentation.card)}>
                      <div className="flex items-start justify-between gap-2">
                        <div className="min-w-0">
                          <p className="truncate text-xs text-muted-foreground">{branch.model_name}</p>
                          <p className="mt-1 text-xs text-muted-foreground">Spoof probability</p>
                          <p className="mt-0.5 flex items-center gap-1.5 text-base font-semibold tabular-nums">
                            <Gauge className={cn("h-4 w-4", presentation.icon)} aria-hidden="true" />
                            {formatProbability(branch.spoof_probability)}
                          </p>
                        </div>
                        <StatusIcon className={cn("h-4 w-4 shrink-0", presentation.icon)} aria-hidden="true" />
                      </div>
                      <dl className="mt-3 text-xs text-muted-foreground">
                        <div>
                          <dt>Status</dt>
                          <dd className="mt-1">
                            <Badge variant={presentation.badge}>{branch.status}</Badge>
                          </dd>
                        </div>
                      </dl>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>
        </div>
      </CardContent>
    </Card>
  );
}
