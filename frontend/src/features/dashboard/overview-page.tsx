"use client";

import Link from "next/link";
import { FileAudio, History, ShieldCheck, TriangleAlert, type LucideIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { PageHeader } from "@/components/shared/page-header";
import { ErrorState } from "@/components/shared/error-state";
import { StatusBadge, LabelBadge } from "@/components/prediction/status-badge";
import { usePredictionHistory } from "@/features/predictions/hooks";
import { useSystemReadiness } from "@/features/system-status/hooks";
import { formatDateTime, formatProbability, sourceText } from "@/lib/formatters";

export function OverviewPage() {
  const history = usePredictionHistory({ page: 1, limit: 5 });
  const readiness = useSystemReadiness();
  const items = history.data?.items || [];
  const completed = items.filter((item) => item.status === "completed");
  const spoof = completed.filter((item) => item.final_prediction === "spoof").length;
  const bonafide = completed.filter((item) => item.final_prediction === "bonafide").length;
  const incomplete = items.filter((item) => item.status !== "completed").length;
  const researchEligible = items.filter((item) => !item.mode_summary.contains_dummy && item.status === "completed").length;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dashboard overview"
        description="Summary values are derived from the currently loaded history page because no aggregate endpoint was verified."
        action={<Button asChild><Link href="/dashboard/analyze">Analyze audio</Link></Button>}
      />
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard icon={ShieldCheck} label="Completed on this page" value={completed.length.toString()} />
        <MetricCard icon={TriangleAlert} label="Spoof results on this page" value={spoof.toString()} />
        <MetricCard icon={FileAudio} label="Bonafide results on this page" value={bonafide.toString()} />
        <MetricCard icon={History} label="Failed or incomplete on this page" value={incomplete.toString()} />
      </div>
      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>Recent predictions</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            {history.isError ? <ErrorState error={history.error} /> : null}
            {items.map((item) => (
              <Link key={item.prediction_id} href={`/dashboard/predictions/${item.prediction_id}`} className="block rounded-lg border p-4 transition-colors hover:bg-accent/50">
                <div className="flex flex-wrap items-center justify-between gap-3">
                  <div>
                    <p className="font-medium">{item.filename || "Untitled audio"}</p>
                    <p className="text-xs text-muted-foreground">{formatDateTime(item.created_at)} · {sourceText(item.source_type)}</p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <StatusBadge status={item.status} />
                    <LabelBadge label={item.final_prediction} />
                  </div>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">Confidence: {formatProbability(item.confidence)}</p>
              </Link>
            ))}
            {!history.isLoading && !items.length && !history.isError ? <p className="text-sm text-muted-foreground">No predictions have been returned for this page.</p> : null}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>Readiness summary</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            {readiness.isError ? <ErrorState error={readiness.error} title="Backend status unavailable" /> : null}
            <div className="rounded-lg border p-4">
              <p className="text-sm text-muted-foreground">Prediction ready</p>
              <p className="mt-1 text-2xl font-semibold">{readiness.data?.prediction_ready ? "Yes" : "No"}</p>
            </div>
            <div className="rounded-lg border p-4">
              <p className="text-sm text-muted-foreground">Research eligible completed items</p>
              <p className="mt-1 text-2xl font-semibold">{researchEligible}</p>
            </div>
            <div className="rounded-lg border p-4">
              <p className="text-sm text-muted-foreground">Model mode summary</p>
              <p className="mt-1 text-sm">{items.some((item) => item.mode_summary.contains_dummy) ? "Dummy or mixed results present" : "No dummy results on loaded page"}</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

function MetricCard({ icon: Icon, label, value }: { icon: LucideIcon; label: string; value: string }) {
  return (
    <Card>
      <CardContent className="p-5">
        <Icon className="h-5 w-5 text-primary" aria-hidden="true" />
        <p className="mt-4 text-sm text-muted-foreground">{label}</p>
        <p className="mt-1 text-3xl font-semibold">{value}</p>
      </CardContent>
    </Card>
  );
}
