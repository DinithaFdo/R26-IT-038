import { ShieldQuestion, UserCheck } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EvidenceUnavailablePanel } from "@/components/xai/evidence-availability";
import { ComponentStatusBadge } from "@/components/xai/xai-status-badge";
import {
  COMPONENT_STATUS_LABELS,
  REPORT_DISPOSITION_LABELS,
  REPORT_DISPOSITION_TONES,
} from "@/lib/xai/status";
import type { CombinedExplanationReport, ComponentStatus, ExplanationError } from "@/types/api";

const TONE_VARIANT = {
  neutral: "secondary",
  progress: "info",
  positive: "success",
  warning: "warning",
  critical: "destructive",
} as const;

function ReportField({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <h4 className="text-xs font-medium uppercase text-muted-foreground">{label}</h4>
      {/* Rendered as plain text. Report content is never injected as HTML. */}
      <p className="mt-1 whitespace-pre-line text-sm">{value}</p>
    </div>
  );
}

/**
 * The combined finding is composed server-side by the backend's deterministic
 * report composer. The UI renders it verbatim and never synthesises a
 * scientific conclusion by combining values client-side.
 */
export function CombinedFindingPanel({
  report,
  status,
  error,
}: {
  report: CombinedExplanationReport | null | undefined;
  status: ComponentStatus;
  error?: ExplanationError | null;
}) {
  if (!report) {
    return (
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <CardTitle>Combined finding</CardTitle>
            <ComponentStatusBadge status={status} />
          </div>
        </CardHeader>
        <CardContent>
          <EvidenceUnavailablePanel
            title="No combined finding"
            description={
              error?.message ??
              `The report component is ${COMPONENT_STATUS_LABELS[
                status
              ].toLowerCase()}. The backend has not composed a combined finding for this run.`
            }
          />
        </CardContent>
      </Card>
    );
  }

  const tone = REPORT_DISPOSITION_TONES[report.disposition];

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <CardTitle>Combined finding</CardTitle>
          </div>
          <ComponentStatusBadge status={report.status} />
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={TONE_VARIANT[tone]} className="gap-1.5">
            <ShieldQuestion className="h-3.5 w-3.5" aria-hidden="true" />
            {REPORT_DISPOSITION_LABELS[report.disposition]}
          </Badge>
          {report.requires_human_review ? (
            <Badge variant="warning" className="gap-1.5">
              <UserCheck className="h-3.5 w-3.5" aria-hidden="true" />
              Human review required
            </Badge>
          ) : null}
        </div>

        <div className="grid gap-5 sm:grid-cols-2">
          <ReportField label="Finding" value={report.finding} />
          <ReportField label="Primary evidence" value={report.primary_evidence} />
          <ReportField label="Quality checks" value={report.quality_checks} />
          <ReportField label="Recommendation" value={report.recommendation} />
        </div>
      </CardContent>
    </Card>
  );
}
