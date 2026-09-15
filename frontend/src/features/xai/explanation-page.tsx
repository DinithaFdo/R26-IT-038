"use client";

import Link from "next/link";
import { useRef, useState } from "react";
import { FileText, Loader2, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ErrorState } from "@/components/shared/error-state";
import { PageHeader } from "@/components/shared/page-header";
import { CombinedFindingPanel } from "@/components/xai/combined-finding-panel";
import { ClassifierSnapshotPanel } from "@/components/xai/classifier-snapshot-panel";
import { ExplanationLifecycleCard } from "@/components/xai/explanation-lifecycle-card";
import { NarrativePanel } from "@/components/xai/narrative-panel";
import { QualityEvidencePanel } from "@/components/xai/quality-evidence-panel";
import { ProvenancePanel } from "@/components/xai/provenance-panel";
import { SemanticEvidencePanel } from "@/components/xai/semantic-evidence-panel";
import { TemporalEvidencePanel } from "@/components/xai/temporal-evidence-panel";
import { WarningsPanel } from "@/components/xai/warnings-panel";
import { usePrediction } from "@/features/predictions/hooks";
import { useExplanation, useNarrativeRetry, useRetryExplanation, useTriggerExplanation } from "@/features/xai/hooks";
import { getUserFriendlyErrorMessage } from "@/lib/api/errors";
import { normalizeTemporalRegions } from "@/lib/xai/regions";

function NoExplanationYet({
  predictionId,
  onTrigger,
  pending,
  predictionCompleted,
}: {
  predictionId: string;
  onTrigger: () => void;
  pending: boolean;
  predictionCompleted: boolean;
}) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>No explanation yet</CardTitle>
        <CardDescription>
          An explanation shows why the system reached this result. It runs separately from the
          analysis above and never changes that result.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {predictionCompleted ? (
          <>
            <p className="text-sm text-muted-foreground">
              No explanation has been generated for this result yet.
            </p>
            <Button onClick={onTrigger} disabled={pending}>
              {pending ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              ) : (
                <Sparkles className="h-4 w-4" aria-hidden="true" />
              )}
              {pending ? "Requesting" : "Explain this result"}
            </Button>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">
            An explanation can only be created for a completed prediction. This prediction is not in
            a completed state.
          </p>
        )}
        <p className="text-xs text-muted-foreground">
          Prediction:{" "}
          <Link className="underline" href={`/dashboard/predictions/${predictionId}`}>
            {predictionId}
          </Link>
        </p>
      </CardContent>
    </Card>
  );
}

export function ExplanationPage({ predictionId }: { predictionId: string }) {
  const prediction = usePrediction(predictionId);
  const explanation = useExplanation(predictionId);
  const trigger = useTriggerExplanation(predictionId);
  const retry = useRetryExplanation(predictionId);
  const narrativeRetry = useNarrativeRetry(predictionId);
  const triggerInFlight = useRef(false);
  const retryInFlight = useRef(false);
  const narrativeRetryInFlight = useRef(false);
  const [selectedRegionId, setSelectedRegionId] = useState<number | null>(null);

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

  const onRetry = async () => {
    if (retryInFlight.current) return;
    retryInFlight.current = true;
    try {
      const created = await retry.mutateAsync();
      toast.success("New explanation run created", { description: created.explanation_id });
    } catch (error) {
      toast.error(getUserFriendlyErrorMessage(error));
    } finally {
      retryInFlight.current = false;
    }
  };

  const onNarrativeRetry = async () => {
    if (narrativeRetryInFlight.current) return;
    narrativeRetryInFlight.current = true;
    try {
      await narrativeRetry.mutateAsync();
      toast.success("AI narrative retry requested");
    } catch (error) {
      toast.error(getUserFriendlyErrorMessage(error));
    } finally {
      narrativeRetryInFlight.current = false;
    }
  };

  if (explanation.isLoading || prediction.isLoading) {
    return <Skeleton className="h-[34rem]" />;
  }
  if (explanation.isError) {
    return <ErrorState error={explanation.error} title="Explanation unavailable" />;
  }

  const data = explanation.data;
  const header = (
    <PageHeader
      title="Explanation"
      description="Review the model's temporal, semantic, and quality evidence."
      action={
        <div className="flex flex-wrap gap-2">
          <Button asChild variant="outline">
            <Link href={`/dashboard/predictions/${predictionId}`}>Back to prediction</Link>
          </Button>
          {data?.combined_report ? (
            <Button asChild>
              <Link href={`/dashboard/predictions/${predictionId}/report`}>
                <FileText className="h-4 w-4" aria-hidden="true" />
                View report
              </Link>
            </Button>
          ) : null}
        </div>
      }
    />
  );

  if (!data) {
    return (
      <div className="space-y-6">
        {header}
        <NoExplanationYet
          predictionId={predictionId}
          onTrigger={onTrigger}
          pending={trigger.isPending}
          predictionCompleted={prediction.data?.status === "completed"}
        />
      </div>
    );
  }

  const temporalRegions = normalizeTemporalRegions(data.temporal).regions;

  return (
    <div className="space-y-6">
      {header}

      <ExplanationLifecycleCard
        explanation={data}
        onTrigger={onTrigger}
        onRetry={onRetry}
        triggerPending={trigger.isPending}
        retryPending={retry.isPending}
      />

      <ClassifierSnapshotPanel snapshot={data.classifier_snapshot} />

      <Tabs defaultValue="temporal">
        <TabsList>
          <TabsTrigger value="temporal">Temporal</TabsTrigger>
          <TabsTrigger value="semantic">Semantic</TabsTrigger>
          <TabsTrigger value="finding">Finding</TabsTrigger>
        </TabsList>

        <TabsContent value="temporal" className="mt-4">
          <TemporalEvidencePanel
            predictionId={predictionId}
            temporal={data.temporal}
            status={data.component_statuses.temporal}
            error={data.component_errors.temporal}
            clipDurationSeconds={prediction.data?.audio.duration_seconds}
            semantic={data.semantic}
            selectedRegionId={selectedRegionId}
            onSelectedRegionIdChange={setSelectedRegionId}
            audioAvailable={prediction.data?.audio.playback_available}
            warnings={data.warnings}
            componentErrors={data.component_errors}
            runError={data.error}
            provenance={data.provenance}
            explanationId={data.explanation_id}
            requestId={data.request_id}
            createdAt={data.created_at}
            completedAt={data.completed_at}
          />
        </TabsContent>

        <TabsContent value="semantic" className="mt-4">
          <div className="space-y-4">
            <SemanticEvidencePanel
              semantic={data.semantic}
              status={data.component_statuses.semantic}
              error={data.component_errors.semantic}
              temporalRegions={temporalRegions}
              selectedRegionId={selectedRegionId}
            />
            <WarningsPanel
              warnings={data.warnings}
              componentErrors={data.component_errors}
              runError={data.error}
              limitationFor="semantic"
            />
            <ProvenancePanel
              provenance={data.provenance}
              explanationId={data.explanation_id}
              requestId={data.request_id}
              createdAt={data.created_at}
              completedAt={data.completed_at}
            />
          </div>
        </TabsContent>

        <TabsContent value="finding" className="mt-4">
          <div className="space-y-4">
            <CombinedFindingPanel
              report={data.combined_report}
              status={data.component_statuses.report}
              error={data.component_errors.report}
            />
            <QualityEvidencePanel quality={data.quality} compact />
            <NarrativePanel
              narrative={data.narrative}
              status={data.component_statuses.narrative}
              error={data.component_errors.narrative}
              onRetry={onNarrativeRetry}
              retryPending={narrativeRetry.isPending}
            />
            <WarningsPanel
              warnings={data.warnings}
              componentErrors={data.component_errors}
              runError={data.error}
              limitationFor="report"
            />
            <ProvenancePanel
              provenance={data.provenance}
              explanationId={data.explanation_id}
              requestId={data.request_id}
              createdAt={data.created_at}
              completedAt={data.completed_at}
            />
          </div>
        </TabsContent>
      </Tabs>

    </div>
  );
}
