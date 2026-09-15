"use client";

import Link from "next/link";
import { FileText, Printer } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { EmptyState } from "@/components/shared/empty-state";
import { ErrorState } from "@/components/shared/error-state";
import { PageHeader } from "@/components/shared/page-header";
import { PredictionSummaryCard } from "@/components/prediction/prediction-summary-card";
import { ClassifierSnapshotPanel } from "@/components/xai/classifier-snapshot-panel";
import { CombinedFindingPanel } from "@/components/xai/combined-finding-panel";
import { EvidenceIdentityPanel } from "@/components/xai/evidence-identity-panel";
import { NarrativePanel } from "@/components/xai/narrative-panel";
import { ProvenancePanel } from "@/components/xai/provenance-panel";
import { QualityEvidencePanel } from "@/components/xai/quality-evidence-panel";
import { SemanticEvidencePanel } from "@/components/xai/semantic-evidence-panel";
import { TemporalEvidencePanel } from "@/components/xai/temporal-evidence-panel";
import { WarningsPanel } from "@/components/xai/warnings-panel";
import { usePrediction } from "@/features/predictions/hooks";
import { useExplanation, useNarrativeRetry } from "@/features/xai/hooks";
import { getUserFriendlyErrorMessage } from "@/lib/api/errors";
import { openBrowserPrintDialog } from "@/lib/print/browser-print";
import { normalizeTemporalRegions } from "@/lib/xai/regions";

/**
 * Full analysis report.
 *
 * There is no backend report-export artifact, so this is a browser-printable
 * layout rather than a generated forensic PDF. That limitation is documented
 * rather than hidden behind a download button implying a signed export exists.
 */
export function ReportPage({ predictionId }: { predictionId: string }) {
  const prediction = usePrediction(predictionId);
  const explanation = useExplanation(predictionId);
  const narrativeRetry = useNarrativeRetry(predictionId);

  const onNarrativeRetry = async () => {
    try {
      await narrativeRetry.mutateAsync();
      toast.success("AI narrative retry requested");
    } catch (error) {
      toast.error(getUserFriendlyErrorMessage(error));
    }
  };

  const onPrint = () => {
    openBrowserPrintDialog(() => {
      toast.success("Print dialog closed. Check your selected folder for the saved PDF report.");
    });
  };

  if (prediction.isLoading || explanation.isLoading) return <Skeleton className="h-[34rem]" />;
  if (prediction.isError)
    return <ErrorState error={prediction.error} title="Prediction unavailable" />;
  if (explanation.isError)
    return <ErrorState error={explanation.error} title="Explanation unavailable" />;
  if (!prediction.data) return null;

  const data = explanation.data;
  const temporalRegions = normalizeTemporalRegions(data?.temporal).regions;

  return (
    <div className="space-y-6" data-print-report>
      <PageHeader
        title="Analysis report"
        description="Combined classification and explanation record for this prediction."
        action={
          <div className="flex flex-wrap gap-2 print:hidden">
            <Button asChild variant="outline">
              <Link href={`/dashboard/predictions/${predictionId}/explanation`}>
                Back to explanation
              </Link>
            </Button>
            <Button onClick={onPrint}>
              <Printer className="h-4 w-4" aria-hidden="true" />
              Print report
            </Button>
          </div>
        }
      />

      <section className="print-report-section" aria-labelledby="report-decision">
        <h2 id="report-decision" className="mb-3 text-lg font-semibold">
          Decision
        </h2>
        <PredictionSummaryCard prediction={prediction.data} />
      </section>

      <section className="print-report-section" aria-labelledby="report-identity">
        <h2 id="report-identity" className="mb-3 text-lg font-semibold">
          Original evidence identity
        </h2>
        <EvidenceIdentityPanel prediction={prediction.data} />
      </section>

      {!data ? (
        <EmptyState
          icon={FileText}
          title="No explanation for this prediction"
          description="Generate a Voice XAI explanation to populate the evidence, quality, and reproducibility sections of this report."
          action={{
            href: `/dashboard/predictions/${predictionId}/explanation`,
            label: "Go to explanation",
          }}
        />
      ) : (
        <>
          <section className="print-report-section" aria-labelledby="report-temporal">
            <h2 id="report-temporal" className="mb-3 text-lg font-semibold">
              Temporal evidence
            </h2>
            <TemporalEvidencePanel
              predictionId={predictionId}
              temporal={data.temporal}
              status={data.component_statuses.temporal}
              error={data.component_errors.temporal}
              clipDurationSeconds={prediction.data.audio.duration_seconds}
              semantic={data.semantic}
            />
          </section>

          <section className="print-report-section" aria-labelledby="report-semantic">
            <h2 id="report-semantic" className="mb-3 text-lg font-semibold">
              Semantic evidence
            </h2>
            <SemanticEvidencePanel
              semantic={data.semantic}
              status={data.component_statuses.semantic}
              error={data.component_errors.semantic}
              temporalRegions={temporalRegions}
            />
          </section>

          <section className="print-report-section" aria-labelledby="report-finding">
            <h2 id="report-finding" className="mb-3 text-lg font-semibold">
              Combined finding
            </h2>
            <CombinedFindingPanel
              report={data.combined_report}
              status={data.component_statuses.report}
              error={data.component_errors.report}
            />
          </section>

          <section className="print-report-section" aria-labelledby="report-classifier-snapshot">
            <h2 id="report-classifier-snapshot" className="mb-3 text-lg font-semibold">
              Classifier snapshot
            </h2>
            <ClassifierSnapshotPanel snapshot={data.classifier_snapshot} />
          </section>

          <section className="print-report-section" aria-labelledby="report-narrative">
            <h2 id="report-narrative" className="mb-3 text-lg font-semibold">
              AI narrative
            </h2>
            <NarrativePanel
              narrative={data.narrative}
              status={data.component_statuses.narrative}
              error={data.component_errors.narrative}
              onRetry={onNarrativeRetry}
              retryPending={narrativeRetry.isPending}
            />
          </section>

          <section className="print-report-section" aria-labelledby="report-quality">
            <h2 id="report-quality" className="mb-3 text-lg font-semibold">
              Explanation quality
            </h2>
            <QualityEvidencePanel quality={data.quality} />
          </section>

          <section className="print-report-section" aria-labelledby="report-warnings">
            <h2 id="report-warnings" className="mb-3 text-lg font-semibold">
              Warnings and limitations
            </h2>
            <WarningsPanel
              warnings={data.warnings}
              componentErrors={data.component_errors}
              runError={data.error}
              limitationFor="both"
            />
          </section>

          <section className="print-report-section" aria-labelledby="report-reproducibility">
            <h2 id="report-reproducibility" className="mb-3 text-lg font-semibold">
              Reproducibility
            </h2>
            <ProvenancePanel
              provenance={data.provenance}
              explanationId={data.explanation_id}
              requestId={data.request_id}
              createdAt={data.created_at}
              completedAt={data.completed_at}
            />
          </section>
        </>
      )}

    </div>
  );
}
