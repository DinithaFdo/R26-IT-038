"use client";

import Link from "next/link";
import { useRef } from "react";
import { FileText, Loader2, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ComponentStatusBadge, ExplanationStatusBadge } from "@/components/xai/xai-status-badge";
import { ResearchStatusBadges } from "@/components/xai/research-status-banner";
import { useExplanation, useTriggerExplanation } from "@/features/xai/hooks";
import { getUserFriendlyErrorMessage } from "@/lib/api/errors";
import { DECISION_SUPPORT_NOTICE, EXPLANATION_STATUS_DESCRIPTIONS } from "@/lib/xai/status";
import type { ComponentStatus } from "@/types/api";

/**
 * Compact Voice XAI status shown on the prediction detail page, with a link
 * into the full explanation view. Polling is driven by the shared
 * `useExplanation` hook, so it stops on terminal states.
 */
export function ExplanationSummaryCard({
  predictionId,
  predictionCompleted,
}: {
  predictionId: string;
  predictionCompleted: boolean;
}) {
  const explanation = useExplanation(predictionId);
  const trigger = useTriggerExplanation(predictionId);
  const triggerInFlight = useRef(false);

  const onTrigger = async () => {
    if (triggerInFlight.current) return;
    triggerInFlight.current = true;
    try {
      await trigger.mutateAsync();
      toast.success("Explanation requested");
    } catch (error) {
      toast.error(getUserFriendlyErrorMessage(error));
    } finally {
      triggerInFlight.current = false;
    }
  };

  if (explanation.isLoading) return <Skeleton className="h-40" />;

  // An explanation failure must never be presented as a prediction failure.
  if (explanation.isError) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Explanation</CardTitle>
          <CardDescription>
            The explanation service could not be reached. The classification result above is
            unaffected.
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const data = explanation.data;
  const componentRows: Array<[string, ComponentStatus]> = data
    ? [
        ["Temporal", data.component_statuses.temporal],
        ["Semantic", data.component_statuses.semantic],
        ["Report", data.component_statuses.report],
      ]
    : [];

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <CardTitle>Explanation</CardTitle>
            <CardDescription>
              {data ? EXPLANATION_STATUS_DESCRIPTIONS[data.status] : DECISION_SUPPORT_NOTICE}
            </CardDescription>
          </div>
          {data ? <ExplanationStatusBadge status={data.status} /> : null}
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {!data ? (
          <>
            <p className="text-sm text-muted-foreground">
              {predictionCompleted
                ? "No explanation run exists for this prediction yet."
                : "An explanation can only be created for a completed prediction."}
            </p>
            {predictionCompleted ? (
              <Button onClick={onTrigger} disabled={trigger.isPending}>
                {trigger.isPending ? (
                  <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
                ) : (
                  <Sparkles className="h-4 w-4" aria-hidden="true" />
                )}
                {trigger.isPending ? "Requesting" : "Explain this result"}
              </Button>
            ) : null}
          </>
        ) : (
          <>
            <ResearchStatusBadges
              developmentPlaceholder={data.development_placeholder}
              researchEligible={data.research_eligible}
            />
            <ul className="grid gap-2 sm:grid-cols-3">
              {componentRows.map(([name, status]) => (
                <li
                  key={name}
                  className="flex items-center justify-between gap-2 rounded-md border p-2.5"
                >
                  <span className="text-xs text-muted-foreground">{name}</span>
                  <ComponentStatusBadge status={status} />
                </li>
              ))}
            </ul>
            <div className="flex flex-wrap gap-2">
              <Button asChild>
                <Link href={`/dashboard/predictions/${predictionId}/explanation`}>
                  <Sparkles className="h-4 w-4" aria-hidden="true" />
                  View explanation
                </Link>
              </Button>
              <Button asChild variant="outline">
                <Link href={`/dashboard/predictions/${predictionId}/report`}>
                  <FileText className="h-4 w-4" aria-hidden="true" />
                  View report
                </Link>
              </Button>
            </div>
          </>
        )}
      </CardContent>
    </Card>
  );
}
