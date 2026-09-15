"use client";

import { Loader2, Play, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { ComponentStatusBadge, ExplanationStatusBadge } from "@/components/xai/xai-status-badge";
import { EXPLANATION_STATUS_DESCRIPTIONS } from "@/lib/xai/status";
import type { ComponentStatus, XaiExplanationResponse } from "@/types/api";

/**
 * Run state, per-component progress, and the trigger/retry controls.
 *
 * Both mutations are disabled while pending so a double click cannot create
 * duplicate background jobs (retry in particular creates a new run server-side
 * each time it is called).
 */
export function ExplanationLifecycleCard({
  explanation,
  onTrigger,
  onRetry,
  triggerPending,
  retryPending,
}: {
  explanation: XaiExplanationResponse;
  onTrigger?: () => void;
  onRetry: () => void;
  triggerPending?: boolean;
  retryPending: boolean;
}) {
  const components: Array<{ name: string; status: ComponentStatus }> = [
    { name: "Temporal", status: explanation.component_statuses.temporal },
    { name: "Semantic", status: explanation.component_statuses.semantic },
    { name: "Report", status: explanation.component_statuses.report },
  ];

  const blocked = explanation.status === "blocked";

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <CardTitle>Explanation run</CardTitle>
            <CardDescription>{EXPLANATION_STATUS_DESCRIPTIONS[explanation.status]}</CardDescription>
          </div>
          <ExplanationStatusBadge status={explanation.status} />
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        <ul className="grid gap-3 sm:grid-cols-3">
          {components.map((component) => (
            <li key={component.name} className="rounded-md border p-3">
              <p className="text-xs text-muted-foreground">{component.name}</p>
              <div className="mt-1.5">
                <ComponentStatusBadge status={component.status} />
              </div>
            </li>
          ))}
        </ul>

        <div className="flex flex-wrap gap-2">
          {blocked && onTrigger ? (
            <Button onClick={onTrigger} disabled={triggerPending}>
              {triggerPending ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              ) : (
                <Play className="h-4 w-4" aria-hidden="true" />
              )}
              Request again
            </Button>
          ) : null}
          <Button variant="outline" onClick={onRetry} disabled={retryPending}>
            {retryPending ? (
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
            ) : (
              <RotateCcw className="h-4 w-4" aria-hidden="true" />
            )}
            {retryPending ? "Starting new run" : "Retry explanation"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
