import type { ReactNode } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { NotAvailable } from "@/components/xai/evidence-availability";
import { formatBytes, formatDateTime, formatDuration } from "@/lib/formatters";
import type { PredictionDetailResponse } from "@/types/api";

/**
 * Original evidence identity.
 *
 * Only fields the backend actually returns are populated. Forensic fields the
 * backend does not implement (content hash of the original upload, formal
 * chain-of-custody reference) are stated as not recorded — claiming hash
 * preservation or custody tracking the backend does not perform would be a
 * false evidential claim.
 */
export function EvidenceIdentityPanel({ prediction }: { prediction: PredictionDetailResponse }) {
  const audio = prediction.audio;

  const rows: Array<{ label: string; value: ReactNode }> = [
    {
      label: "Prediction / case ID",
      value: <span className="break-all font-mono text-xs">{prediction.prediction_id}</span>,
    },
    {
      label: "Request ID",
      value: <span className="break-all font-mono text-xs">{prediction.request_id}</span>,
    },
    {
      label: "Original filename",
      value: audio.original_filename ?? <NotAvailable label="Not recorded" />,
    },
    {
      label: "Acquisition time",
      value: prediction.created_at ? (
        formatDateTime(prediction.created_at)
      ) : (
        <NotAvailable label="Not recorded" />
      ),
    },
    {
      label: "Duration",
      value:
        typeof audio.duration_seconds === "number" ? (
          formatDuration(audio.duration_seconds)
        ) : (
          <NotAvailable label="Not recorded" />
        ),
    },
    {
      label: "Container / codec",
      value:
        audio.detected_container || audio.detected_codec ? (
          `${audio.detected_container ?? "unknown"} / ${audio.detected_codec ?? "unknown"}`
        ) : (
          <NotAvailable label="Not recorded" />
        ),
    },
    {
      label: "Sample rate",
      value:
        typeof audio.sample_rate === "number" ? (
          `${audio.sample_rate} Hz`
        ) : (
          <NotAvailable label="Not recorded" />
        ),
    },
    {
      label: "Channels",
      value:
        typeof audio.channels === "number" ? audio.channels : <NotAvailable label="Not recorded" />,
    },
    {
      label: "File size",
      value:
        typeof audio.size_bytes === "number" ? (
          formatBytes(audio.size_bytes)
        ) : (
          <NotAvailable label="Not recorded" />
        ),
    },
    { label: "Source", value: prediction.source_type },
    {
      label: "Original audio SHA-256",
      value: (
        <NotAvailable
          label="Not recorded"
          reason="the current backend does not persist a content hash of the original upload"
        />
      ),
    },
    {
      label: "Chain-of-custody reference",
      value: (
        <NotAvailable
          label="Not recorded"
          reason="the current backend does not implement chain-of-custody tracking"
        />
      ),
    },
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle>Original evidence identity</CardTitle>
        <CardDescription>
          Metadata recorded for the submitted audio. Analysis operates on a derived working copy;
          the retained original is not modified by this system.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <dl className="grid gap-x-6 gap-y-4 sm:grid-cols-2">
          {rows.map((row) => (
            <div key={row.label} className="min-w-0">
              <dt className="text-xs text-muted-foreground">{row.label}</dt>
              <dd className="break-words text-sm">{row.value}</dd>
            </div>
          ))}
        </dl>
      </CardContent>
    </Card>
  );
}
