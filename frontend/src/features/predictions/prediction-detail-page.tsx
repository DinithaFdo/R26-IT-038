"use client";

import Link from "next/link";
import { RotateCcw, Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { PageHeader } from "@/components/shared/page-header";
import { ErrorState } from "@/components/shared/error-state";
import { Skeleton } from "@/components/ui/skeleton";
import { ConfirmActionDialog } from "@/components/shared/confirm-action-dialog";
import { PredictionSummaryCard } from "@/components/prediction/prediction-summary-card";
import { BranchResultCard } from "@/components/prediction/branch-result-card";
import { FusionDetailsCard } from "@/components/prediction/fusion-details-card";
import { DummyModeWarning } from "@/components/prediction/dummy-mode-warning";
import { PartialSystemNotice } from "@/components/prediction/partial-system-notice";
import { FusionFallbackNotice } from "@/components/prediction/fusion-fallback-notice";
import { AudioPlaybackCard } from "@/components/prediction/audio-playback-card";
import { EvidenceIdentityPanel } from "@/components/xai/evidence-identity-panel";
import { ExplanationSummaryCard } from "@/components/xai/explanation-summary-card";
import { useDeletePrediction, usePrediction, useRerunPrediction } from "@/features/predictions/hooks";
import { getUserFriendlyErrorMessage } from "@/lib/api/errors";

export function PredictionDetailPage({ predictionId }: { predictionId: string }) {
  const router = useRouter();
  const prediction = usePrediction(predictionId);
  const rerun = useRerunPrediction();
  const remove = useDeletePrediction();

  const onRerun = async () => {
    try {
      const response = await rerun.mutateAsync({ predictionId });
      toast.success("Rerun started", { description: response.prediction_id });
      router.push(`/dashboard/predictions/${response.prediction_id}`);
    } catch (error) {
      toast.error(getUserFriendlyErrorMessage(error));
    }
  };

  const onDelete = async () => {
    try {
      await remove.mutateAsync(predictionId);
      toast.success("Prediction deleted");
      router.push("/dashboard/history");
    } catch (error) {
      toast.error(getUserFriendlyErrorMessage(error));
    }
  };

  if (prediction.isLoading) return <Skeleton className="h-[34rem]" />;
  if (prediction.isError) return <ErrorState error={prediction.error} title="Prediction unavailable" />;
  if (!prediction.data) return null;
  const containsDummy = prediction.data.branches.some((branch) => branch.mode === "dummy") || Boolean(prediction.data.fusion?.contains_dummy_branches);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Prediction detail"
        description="Final result, branch-level classifiers, fusion details, and safe actions."
        action={
          <div className="flex flex-wrap gap-2">
            <Button asChild variant="outline"><Link href="/dashboard/history">Return to history</Link></Button>
            <Button asChild><Link href="/dashboard/analyze">Analyze another</Link></Button>
          </div>
        }
      />
      <DummyModeWarning show={containsDummy} reason={!prediction.data.research_eligible ? prediction.data.fusion?.warning : null} />
      <FusionFallbackNotice fusion={prediction.data.fusion} />
      <PartialSystemNotice branches={prediction.data.branches} fusion={prediction.data.fusion} />
      <PredictionSummaryCard prediction={prediction.data} />
      <section>
        <h2 className="mb-3 text-lg font-semibold">Branch results</h2>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          {prediction.data.branches.map((branch) => (
            <BranchResultCard key={branch.model_name} branch={branch} />
          ))}
        </div>
      </section>
      <FusionDetailsCard fusion={prediction.data.fusion} />
      <ExplanationSummaryCard
        predictionId={predictionId}
        predictionCompleted={prediction.data.status === "completed"}
      />
      <EvidenceIdentityPanel prediction={prediction.data} />
      <AudioPlaybackCard predictionId={predictionId} available={prediction.data.audio.playback_available} />
      <div className="flex flex-wrap gap-2">
        <ConfirmActionDialog
          trigger={<Button variant="outline"><RotateCcw className="h-4 w-4" /> Rerun analysis</Button>}
          title="Rerun analysis?"
          description="This creates a new prediction if the backend can access retained source audio."
          actionLabel="Rerun"
          onConfirm={onRerun}
        />
        <ConfirmActionDialog
          trigger={<Button variant="outline"><Trash2 className="h-4 w-4" /> Delete prediction</Button>}
          title="Delete prediction?"
          description="This deletes the owner-scoped prediction record and retained audio when present."
          actionLabel="Delete"
          destructive
          onConfirm={onDelete}
        />
      </div>
    </div>
  );
}
