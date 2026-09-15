import { Fingerprint } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { ConfidenceDisplay } from "@/components/prediction/confidence-display";
import { ResearchEligibilityBadge } from "@/components/prediction/research-eligibility-badge";
import { StatusBadge } from "@/components/prediction/status-badge";
import { getFusionVersionDescriptor, getFusionVersionDisplayName } from "@/lib/copy/fusion-names";
import { getVerdictHeadline, getVerdictSentence } from "@/lib/copy/verdict";
import { formatDateTime, formatDuration, formatProbability, sourceText } from "@/lib/formatters";
import type { PredictionDetailResponse } from "@/types/api";

export function PredictionSummaryCard({ prediction }: { prediction: PredictionDetailResponse }) {
  const fusion = prediction.fusion;
  const verdictSentence = getVerdictSentence(fusion?.prediction);
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Fingerprint className="h-5 w-5 text-primary" aria-hidden="true" />
          Result summary
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="space-y-1">
          <p className="text-2xl font-semibold">{getVerdictHeadline(fusion?.prediction)}</p>
          {verdictSentence ? <p className="text-sm text-muted-foreground">{verdictSentence}</p> : null}
        </div>
        <div className="flex flex-wrap gap-2">
          <StatusBadge status={prediction.status} />
          <ResearchEligibilityBadge eligible={prediction.research_eligible} />
          {fusion && !fusion.fallback_used ? (
            <Badge variant="info">
              {getFusionVersionDisplayName(fusion.fusion_version)}
              {getFusionVersionDescriptor(fusion.fusion_version)
                ? ` · ${getFusionVersionDescriptor(fusion.fusion_version)}`
                : ""}
              {typeof fusion.decision_threshold === "number"
                ? ` · threshold ${fusion.decision_threshold}`
                : ""}
            </Badge>
          ) : null}
        </div>
        <div className="max-w-xs">
          <ConfidenceDisplay value={fusion?.confidence} />
        </div>

        <Accordion type="single" collapsible>
          <AccordionItem value="technical-details" className="border-0">
            <AccordionTrigger className="py-2 text-xs text-muted-foreground hover:no-underline">
              Technical details
            </AccordionTrigger>
            <AccordionContent>
              <div className="grid gap-4 pb-2 md:grid-cols-2">
                <div className="rounded-lg border bg-muted/40 p-4">
                  <p className="text-xs text-muted-foreground">Spoof probability</p>
                  <p className="text-xl font-semibold text-red-300">{formatProbability(fusion?.probabilities?.spoof)}</p>
                </div>
                <div className="rounded-lg border bg-muted/40 p-4">
                  <p className="text-xs text-muted-foreground">Bonafide probability</p>
                  <p className="text-xl font-semibold text-emerald-300">{formatProbability(fusion?.probabilities?.bonafide)}</p>
                </div>
              </div>
              <dl className="grid gap-3 text-sm md:grid-cols-3">
                <div><dt className="text-muted-foreground">Prediction ID</dt><dd className="break-all font-mono text-xs">{prediction.prediction_id}</dd></div>
                <div><dt className="text-muted-foreground">Request ID</dt><dd className="break-all font-mono text-xs">{prediction.request_id}</dd></div>
                <div><dt className="text-muted-foreground">Source</dt><dd>{sourceText(prediction.source_type)}</dd></div>
                <div><dt className="text-muted-foreground">Completed</dt><dd>{formatDateTime(prediction.completed_at)}</dd></div>
                <div><dt className="text-muted-foreground">Audio duration</dt><dd>{formatDuration(prediction.audio.duration_seconds)}</dd></div>
                <div><dt className="text-muted-foreground">Decision threshold</dt><dd>{typeof prediction.preprocessing?.decision_threshold === "number" ? prediction.preprocessing.decision_threshold : "Backend did not report"}</dd></div>
              </dl>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </CardContent>
    </Card>
  );
}
