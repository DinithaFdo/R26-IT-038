import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { NotAvailable } from "@/components/xai/evidence-availability";
import { formatDateTime } from "@/lib/formatters";
import type { ExplanationProvenance } from "@/types/api";

type ProvenanceRow = {
  label: string;
  value: string | null | undefined;
  mono?: boolean;
};

/**
 * Explanation record.
 *
 * Only fields the backend actually reports are shown as values; anything the
 * backend leaves null renders as an explicit unavailable state so a reader
 * cannot mistake a gap for a recorded provenance value.
 */
export function ProvenancePanel({
  provenance,
  explanationId,
  requestId,
  createdAt,
  completedAt,
}: {
  provenance: ExplanationProvenance;
  explanationId: string;
  requestId: string;
  createdAt: string;
  completedAt?: string | null;
}) {
  const rows: ProvenanceRow[] = [
    { label: "Explanation ID", value: explanationId, mono: true },
    { label: "Request ID", value: requestId, mono: true },
    { label: "Semantic model version", value: provenance.semantic_model_version },
    { label: "Temporal model version", value: provenance.temporal_model_version },
    { label: "Generated at", value: formatDateTime(provenance.generated_at) },
    { label: "Run created at", value: formatDateTime(createdAt) },
    { label: "Run completed at", value: completedAt ? formatDateTime(completedAt) : null },
  ];
  return (
    <Card>
      <CardHeader>
        <CardTitle>Explanation record</CardTitle>
        <CardDescription>
          Key identifiers, model versions, and timestamps for this explanation run.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="grid gap-x-6 gap-y-4 sm:grid-cols-2">
          {rows.map((row) => (
            <div key={row.label} className="min-w-0">
              <dt className="text-xs text-muted-foreground">{row.label}</dt>
              <dd className={row.mono ? "break-all font-mono text-xs" : "break-words text-sm"}>
                {row.value ? row.value : <NotAvailable label="Not recorded" />}
              </dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  );
}
