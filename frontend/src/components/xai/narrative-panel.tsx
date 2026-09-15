import { Bot, FileText, Loader2, RotateCcw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { EvidenceUnavailablePanel } from "@/components/xai/evidence-availability";
import { formatDateTime } from "@/lib/formatters";
import type { ExplanationError, NarrativeExplanation, NarrativeStatus } from "@/types/api";

const NARRATIVE_STATUS_LABELS: Record<NarrativeStatus, string> = {
  not_available: "Not available",
  queued: "Queued",
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  blocked: "Blocked",
};

export function NarrativePanel({
  narrative,
  status,
  error,
  onRetry,
  retryPending = false,
}: {
  narrative?: NarrativeExplanation | null;
  status: NarrativeStatus;
  error?: ExplanationError | null;
  onRetry?: () => void;
  retryPending?: boolean;
}) {
  const retryable =
    (status === "failed" || status === "not_available") &&
    error?.code !== "narrative_disabled";
  const inProgress = status === "queued" || status === "running";
  if (status !== "completed" || !narrative) {
    return (
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-2">
            <div>
              <CardTitle>AI narrative</CardTitle>
              <CardDescription>
                Optional language-model summary. It is never authoritative evidence.
              </CardDescription>
            </div>
            <Badge variant="secondary">{NARRATIVE_STATUS_LABELS[status]}</Badge>
          </div>
        </CardHeader>
        <CardContent>
          <EvidenceUnavailablePanel
            title={inProgress ? "AI narrative in progress" : "Narrative not available"}
            description={
              inProgress
                ? "The AI narrative is being generated. This page will update automatically."
                : error?.message ??
                  "The backend did not generate an AI narrative for this explanation run."
            }
          />
          {retryable && onRetry ? (
            <Button className="mt-4" variant="outline" onClick={onRetry} disabled={retryPending}>
              {retryPending ? (
                <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
              ) : (
                <RotateCcw className="h-4 w-4" aria-hidden="true" />
              )}
              {retryPending ? "Requesting retry" : "Retry AI narrative"}
            </Button>
          ) : null}
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <CardTitle>AI narrative</CardTitle>
            <CardDescription>
              Non-authoritative summary generated from backend evidence references.
            </CardDescription>
          </div>
          <Badge variant="warning">Not authoritative</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="rounded-md border bg-muted/40 p-4">
          <p className="whitespace-pre-line text-sm">{narrative.summary}</p>
          <p className="mt-3 whitespace-pre-line text-sm text-muted-foreground">
            {narrative.detailed_explanation}
          </p>
        </div>
        <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <dt className="text-xs text-muted-foreground">Model</dt>
            <dd className="text-sm font-medium">{narrative.model_id ?? "Not reported"}</dd>
          </div>
          <div>
            <dt className="text-xs text-muted-foreground">Prompt version</dt>
            <dd className="text-sm font-medium">{narrative.prompt_version ?? "Not reported"}</dd>
          </div>
          <div>
            <dt className="text-xs text-muted-foreground">Generated at</dt>
            <dd className="text-sm font-medium">{formatDateTime(narrative.generated_at)}</dd>
          </div>
          <div>
            <dt className="text-xs text-muted-foreground">Evidence references</dt>
            <dd className="flex items-center gap-1.5 text-sm font-medium">
              <FileText className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
              {narrative.evidence_references.length}
            </dd>
          </div>
        </dl>
        <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Bot className="h-3.5 w-3.5" aria-hidden="true" />
          This narrative is a reading aid only; rely on the deterministic panels above for evidence.
        </p>
      </CardContent>
    </Card>
  );
}
