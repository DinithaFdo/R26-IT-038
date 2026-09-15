"use client";

import { useState } from "react";
import { Download, Loader2 } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { getUserFriendlyErrorMessage } from "@/lib/api/errors";
import { getExplanationArtifact } from "@/lib/api/xai";
import { formatBytes, formatDateTime } from "@/lib/formatters";
import type { ExplanationArtifactReference } from "@/types/api";

const ARTIFACT_KIND_LABELS: Record<string, string> = {
  attention_visualization: "Attention visualisation",
  attention_spectrogram: "Attention spectrogram",
  attention_tensor: "Attention tensor",
  semantic_values: "Semantic values",
  report: "Report",
  other: "Artifact",
};

function isExpired(artifact: ExplanationArtifactReference) {
  if (!artifact.expires_at) return false;
  return new Date(artifact.expires_at).getTime() <= Date.now();
}

/**
 * Downloads a private artifact through the authenticated backend route.
 *
 * The artifact ID is opaque: no filesystem path is ever constructed, and
 * `download_path` from the API is deliberately ignored. The object URL is
 * revoked immediately after the download is triggered so artifact bytes are not
 * retained in memory.
 */
export function ArtifactDownloadButton({
  predictionId,
  artifact,
}: {
  predictionId: string;
  artifact: ExplanationArtifactReference;
}) {
  const [downloading, setDownloading] = useState(false);
  const expired = isExpired(artifact);
  const label = ARTIFACT_KIND_LABELS[artifact.kind] ?? ARTIFACT_KIND_LABELS.other;

  const onDownload = async () => {
    setDownloading(true);
    try {
      const blob = await getExplanationArtifact(predictionId, artifact.artifact_id);
      const objectUrl = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = objectUrl;
      anchor.download = `${artifact.kind}-${artifact.artifact_id}`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(objectUrl);
    } catch (error) {
      toast.error(getUserFriendlyErrorMessage(error));
    } finally {
      setDownloading(false);
    }
  };

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-md border p-3">
      <div className="min-w-0">
        <p className="text-sm font-medium">{label}</p>
        <p className="text-xs text-muted-foreground">
          {artifact.content_type} · {formatBytes(artifact.size_bytes)} · Expires{" "}
          {artifact.expires_at ? formatDateTime(artifact.expires_at) : "not reported"}
        </p>
        <details className="mt-1">
          <summary className="cursor-pointer text-xs text-muted-foreground">
            Technical details
          </summary>
          <dl className="mt-1 grid gap-1 text-xs text-muted-foreground">
            <div>
              <dt className="inline">Artifact type: </dt>
              <dd className="inline">{artifact.kind}</dd>
            </div>
            <div>
              <dt className="inline">SHA-256: </dt>
              <dd className="inline break-all">{artifact.sha256}</dd>
            </div>
          </dl>
        </details>
        {expired ? (
          <p className="mt-1 text-xs text-muted-foreground">
            Retention window has passed; this artifact is no longer retrievable.
          </p>
        ) : null}
      </div>
      <Button
        variant="outline"
        size="sm"
        onClick={onDownload}
        disabled={downloading || expired}
        aria-label={`Download ${label}`}
      >
        {downloading ? (
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
        ) : (
          <Download className="h-4 w-4" aria-hidden="true" />
        )}
        {downloading ? "Downloading" : "Download"}
      </Button>
    </div>
  );
}
