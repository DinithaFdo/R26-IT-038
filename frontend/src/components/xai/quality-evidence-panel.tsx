import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { NotAvailable } from "@/components/xai/evidence-availability";
import { describeMetric } from "@/lib/xai/status";
import type { ExplanationQuality, MetricEvidence } from "@/types/api";

/**
 * Explanation-quality metrics.
 *
 * Every metric renders its availability explicitly. A metric the backend has
 * not computed shows "Not computed" — never `0`, and never an inferred
 * substitute, because zero is itself a meaningful measured value here.
 */

type MetricDefinition = {
  key: keyof ExplanationQuality;
  label: string;
  description: string;
};

const METRICS: MetricDefinition[] = [
  {
    key: "temporal_semantic_iou",
    label: "Temporal / semantic agreement (IoU)",
    description:
      "Overlap between high-attention temporal regions and suspicious semantic windows.",
  },
  {
    key: "surrogate_fidelity_r2",
    label: "Surrogate fidelity (R²)",
    description: "How closely the semantic surrogate reproduces the classifier it explains.",
  },
];

function MetricRow({
  definition,
  metric,
}: {
  definition: MetricDefinition;
  metric?: MetricEvidence | null;
}) {
  const described = describeMetric(metric);

  return (
    <div className="border-b py-4 last:border-0">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h4 className="text-sm font-medium">{definition.label}</h4>
        <span className="text-sm font-semibold tabular-nums">
          {described.available && described.value !== null ? (
            described.value.toFixed(3)
          ) : (
            <NotAvailable label={described.label} />
          )}
        </span>
      </div>
      <p className="mt-1 text-xs text-muted-foreground">{definition.description}</p>
      {described.reason ? (
        <p className="mt-1 text-xs text-muted-foreground">Reason: {described.reason}</p>
      ) : null}
      {metric ? (
        <p className="mt-1 text-xs text-muted-foreground">
          Scope: {metric.scope === "per_analysis" ? "This analysis" : "Offline validation"}
          {metric.dataset_version ? ` · Dataset: ${metric.dataset_version}` : ""}
        </p>
      ) : null}
    </div>
  );
}

export function QualityEvidencePanel({
  quality,
  compact = false,
}: {
  quality: ExplanationQuality;
  compact?: boolean;
}) {
  const availableMetrics = METRICS.filter((definition) =>
    describeMetric(quality[definition.key]).available,
  );
  const unavailableMetrics = METRICS.filter(
    (definition) => !describeMetric(quality[definition.key]).available,
  );
  const availableCount = availableMetrics.length;

  return (
    <Card>
      <CardHeader>
        <CardTitle>{compact ? "Research quality" : "Explanation quality"}</CardTitle>
        <CardDescription>
          {availableCount} of {METRICS.length} quality metrics are computed.
         
        </CardDescription>
      </CardHeader>
      <CardContent className="pt-0">
        {(compact ? availableMetrics : METRICS).map((definition) => (
          <MetricRow
            key={definition.key}
            definition={definition}
            metric={quality[definition.key]}
          />
        ))}
        {compact && unavailableMetrics.length > 0 ? (
          <Accordion type="single" collapsible>
            <AccordionItem value="not-computed" className="border-b-0">
              <AccordionTrigger>
                Not computed ({unavailableMetrics.length})
              </AccordionTrigger>
              <AccordionContent>
                {unavailableMetrics.map((definition) => (
                  <MetricRow
                    key={definition.key}
                    definition={definition}
                    metric={quality[definition.key]}
                  />
                ))}
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        ) : null}
      </CardContent>
    </Card>
  );
}
