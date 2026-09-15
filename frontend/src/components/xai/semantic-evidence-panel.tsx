"use client";

import { useState } from "react";
import { ArrowDownRight, ArrowUpRight, ChevronDown, ChevronLeft, ChevronRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { EvidenceUnavailablePanel, NotAvailable } from "@/components/xai/evidence-availability";
import { ComponentStatusBadge } from "@/components/xai/xai-status-badge";
import { cn } from "@/lib/utils";
import {
  formatDecimal,
  formatSeconds,
  overlapsRegion,
  type RenderedTemporalRegion,
} from "@/lib/xai/regions";
import { COMPONENT_STATUS_LABELS, SHAP_DIRECTION_LABELS } from "@/lib/xai/status";
import type {
  ComponentStatus,
  ExplanationError,
  SemanticExplanation,
  SemanticFeatureContribution,
  SemanticEvidenceWindow,
} from "@/types/api";

/**
 * Direction badge.
 *
 * The label comes from the backend's `direction` field, never from the sign or
 * absolute magnitude of `shap_value` — the backend owns the mapping between a
 * contribution and the spoof/bonafide direction, and duplicating that logic
 * here risks silently inverting the meaning of the evidence.
 */
function DirectionBadge({ direction }: { direction: SemanticFeatureContribution["direction"] }) {
  const towardSpoof = direction === "toward_spoof";
  const Icon = towardSpoof ? ArrowUpRight : ArrowDownRight;
  return (
    <Badge variant={towardSpoof ? "destructive" : "success"} className="gap-1">
      <Icon className="h-3 w-3" aria-hidden="true" />
      {SHAP_DIRECTION_LABELS[direction]}
    </Badge>
  );
}

/** Magnitude bar, scaled against the largest absolute SHAP in the same list. */
function ContributionBar({ value, max }: { value: number; max: number }) {
  const width = max > 0 ? Math.min(100, (Math.abs(value) / max) * 100) : 0;
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted" aria-hidden="true">
      <div
        className={value >= 0 ? "h-full bg-destructive/70" : "h-full bg-emerald-600/70"}
        style={{ width: `${width}%` }}
      />
    </div>
  );
}

function ContributionRow({
  contribution,
  maxAbsShap,
  selected,
  highlighted,
  dimmed,
}: {
  contribution: SemanticFeatureContribution;
  maxAbsShap: number;
  selected?: boolean;
  highlighted?: boolean;
  dimmed?: boolean;
}) {
  const hasInterval =
    typeof contribution.start_seconds === "number" && typeof contribution.end_seconds === "number";

  return (
    <li
      className={cn(
        "rounded-xl border p-4 transition-all duration-200",
        selected
          ? "border-primary/20 bg-card shadow-sm"
          : null,
        highlighted && !selected ? "border-primary/25 bg-primary/[0.045]" : null,
        dimmed ? "opacity-60" : null,
      )}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <p className="font-medium">
            <span className="text-muted-foreground">#{contribution.rank}</span>{" "}
            {contribution.display_name}
          </p>
          <p className="break-all text-xs text-muted-foreground">{contribution.feature_name}</p>
        </div>
        <DirectionBadge direction={contribution.direction} />
      </div>

      <dl className="mt-3 grid gap-3 sm:grid-cols-3">
        <div>
          <dt className="text-xs text-muted-foreground">Measured value</dt>
          <dd className="text-sm font-medium tabular-nums">
            {contribution.value.toPrecision(4)}
            {contribution.unit ? (
              <span className="ml-1 text-xs text-muted-foreground">{contribution.unit}</span>
            ) : null}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-muted-foreground">SHAP contribution</dt>
          <dd className="text-sm font-medium tabular-nums">
            {contribution.shap_value >= 0 ? "+" : ""}
            {contribution.shap_value.toFixed(4)}
          </dd>
        </div>
        <div>
          <dt className="text-xs text-muted-foreground">Time interval</dt>
          <dd className="text-sm font-medium tabular-nums">
            {hasInterval ? (
              `${formatSeconds(contribution.start_seconds)} - ${formatSeconds(
                contribution.end_seconds,
              )}`
            ) : (
              <NotAvailable label="Whole clip" reason="no per-window localisation" />
            )}
          </dd>
        </div>
      </dl>

      <div className="mt-3">
        <ContributionBar value={contribution.shap_value} max={maxAbsShap} />
      </div>

      <p className="mt-2 text-xs text-muted-foreground">
        {contribution.reference_summary ?? "No reference range reported for this feature."}
      </p>
    </li>
  );
}

function formatFeatureTime(contribution: SemanticFeatureContribution) {
  const hasInterval =
    typeof contribution.start_seconds === "number" && typeof contribution.end_seconds === "number";

  return hasInterval
    ? `${formatSeconds(contribution.start_seconds)} - ${formatSeconds(contribution.end_seconds)}`
    : "Whole clip";
}

/** Compact, selectable diverging SHAP plot. Negative contributions extend left; positive extend right. */
function ShapRankingPlot({
  features,
  maxAbsShap,
  selectedRank,
  onSelect,
  selectedRegion,
}: {
  features: SemanticFeatureContribution[];
  maxAbsShap: number;
  selectedRank: number;
  onSelect: (rank: number) => void;
  selectedRegion: RenderedTemporalRegion | null;
}) {
  return (
    <section aria-labelledby="shap-ranking-heading">
      <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h4 id="shap-ranking-heading" className="text-sm font-semibold">
          Top feature ranking ({features.length})
        </h4>
        <p className="text-xs text-muted-foreground">
          <span className="text-emerald-700">← Toward bonafide</span>
          <span className="mx-2 text-border">|</span>
          <span className="text-destructive">Toward spoof →</span>
        </p>
      </div>
      <p className="mt-1 text-xs text-muted-foreground">
        Select a bar to inspect its measurement and time-specific details.
      </p>

      <div className="mt-3 space-y-1.5" role="list" aria-label="Ranked SHAP contributions">
        {features.map((contribution) => {
          const magnitude = maxAbsShap > 0 ? (Math.abs(contribution.shap_value) / maxAbsShap) * 100 : 0;
          const towardSpoof = contribution.direction === "toward_spoof";
          const isSelected = contribution.rank === selectedRank;
          const overlapsSelectedRegion = selectedRegion
            ? overlapsRegion(contribution, selectedRegion)
            : false;

          return (
            <div key={`${contribution.rank}-${contribution.feature_name}`} role="listitem">
              <button
                type="button"
                aria-pressed={isSelected}
                aria-label={`Rank ${contribution.rank}: ${contribution.display_name}, ${
                  SHAP_DIRECTION_LABELS[contribution.direction]
                }, ${formatFeatureTime(contribution)}`}
                onClick={() => onSelect(contribution.rank)}
                className={cn(
                  "relative grid w-full grid-cols-[minmax(0,1fr)_minmax(7rem,1.15fr)] items-center gap-3 overflow-hidden rounded-lg border px-3 py-2.5 text-left transition-all duration-200 hover:border-primary/20 hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  isSelected
                    ? "border-border bg-card shadow-sm"
                    : "border-transparent",
                  selectedRegion && !overlapsSelectedRegion && !isSelected ? "opacity-60" : null,
                )}
              >
                {isSelected ? (
                  <span className="absolute inset-y-0 left-0 w-1 bg-foreground" aria-hidden="true" />
                ) : null}
              <span className="min-w-0">
                <span className="block truncate text-sm font-medium" title={contribution.display_name}>
                  <span className="mr-1 text-muted-foreground">#{contribution.rank}</span>
                  {contribution.display_name}
                </span>
                <span className="block truncate text-xs text-muted-foreground">
                  {formatFeatureTime(contribution)}
                </span>
              </span>

              <span className="flex items-center gap-1.5" aria-hidden="true">
                <span className="flex h-2 flex-1 justify-end rounded-l-sm">
                  {!towardSpoof ? (
                    <span
                      className="h-full rounded-l-sm bg-emerald-600/70"
                      style={{ width: `${magnitude}%` }}
                    />
                  ) : null}
                </span>
                <span className="h-4 w-px bg-border" />
                <span className="flex h-2 flex-1 rounded-r-sm">
                  {towardSpoof ? (
                    <span
                      className="h-full rounded-r-sm bg-destructive/70"
                      style={{ width: `${magnitude}%` }}
                    />
                  ) : null}
                </span>
              </span>
              </button>
            </div>
          );
        })}
      </div>
    </section>
  );
}

/** A complete, non-interactive record for browser-printed analysis reports. */
function SemanticFeaturePrintDetails({
  features,
  maxAbsShap,
}: {
  features: SemanticFeatureContribution[];
  maxAbsShap: number;
}) {
  return (
    <section className="hidden print:block" aria-labelledby="semantic-print-details-heading">
      <h4 id="semantic-print-details-heading" className="text-sm font-semibold">
        All semantic feature details
      </h4>
      <ul className="print-semantic-details mt-3 space-y-3">
        {features.map((feature) => (
          <ContributionRow
            key={`${feature.rank}-${feature.feature_name}`}
            contribution={feature}
            maxAbsShap={maxAbsShap}
          />
        ))}
      </ul>
    </section>
  );
}

function formatWindowInterval(window: SemanticEvidenceWindow) {
  return `${formatSeconds(window.start_seconds)} - ${formatSeconds(window.end_seconds)}`;
}

/** Compact navigator for the technical sliding windows used by the semantic model. */
function SemanticWindowNavigator({
  windows,
  selectedWindowIndex,
  onSelect,
  selectedRegion,
}: {
  windows: SemanticEvidenceWindow[];
  selectedWindowIndex: number;
  onSelect: (index: number) => void;
  selectedRegion: RenderedTemporalRegion | null;
}) {
  const timelineStart = Math.min(...windows.map((window) => window.start_seconds));
  const timelineEnd = Math.max(...windows.map((window) => window.end_seconds));
  const timelineDuration = Math.max(timelineEnd - timelineStart, Number.EPSILON);
  const selectedWindow = windows[selectedWindowIndex] ?? windows[0];
  const selectedStart = ((selectedWindow.start_seconds - timelineStart) / timelineDuration) * 100;
  const selectedWidth = Math.max(
    1,
    ((selectedWindow.end_seconds - selectedWindow.start_seconds) / timelineDuration) * 100,
  );
  const overlapsSelectedRegion = selectedRegion ? overlapsRegion(selectedWindow, selectedRegion) : false;

  return (
    <div className="rounded-xl border bg-muted/20 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          aria-label="Previous analysis window"
          disabled={selectedWindowIndex === 0}
          onClick={() => onSelect(selectedWindowIndex - 1)}
        >
          <ChevronLeft className="h-4 w-4" aria-hidden="true" />
          Previous
        </Button>
        <Select value={String(selectedWindowIndex)} onValueChange={(value) => onSelect(Number(value))}>
          <SelectTrigger aria-label="Jump to analysis window" className="min-w-56 flex-1 sm:flex-none">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {windows.map((window, index) => (
              <SelectItem key={`${window.start_seconds}-${window.end_seconds}`} value={String(index)}>
                Window {String(index + 1).padStart(2, "0")} · {formatWindowInterval(window)}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button
          type="button"
          variant="outline"
          size="sm"
          aria-label="Next analysis window"
          disabled={selectedWindowIndex === windows.length - 1}
          onClick={() => onSelect(selectedWindowIndex + 1)}
        >
          Next
          <ChevronRight className="h-4 w-4" aria-hidden="true" />
        </Button>
        <span className="text-xs tabular-nums text-muted-foreground">
          Window {selectedWindowIndex + 1} of {windows.length}
        </span>
      </div>

      <div className="mt-5">
        <div className="relative h-2 rounded-full bg-border" aria-label="Selected analysis window on audio timeline">
          <span
            className={cn(
              "absolute inset-y-0 rounded-full bg-foreground",
              overlapsSelectedRegion ? "ring-2 ring-primary/35 ring-offset-1" : null,
            )}
            style={{
              left: `${Math.min(selectedStart, 99)}%`,
              width: `${Math.min(selectedWidth, 100 - selectedStart)}%`,
            }}
          />
        </div>
        <div className="mt-2 flex justify-between text-xs tabular-nums text-muted-foreground">
          <span>{formatSeconds(timelineStart)}</span>
          <span>{formatSeconds(timelineEnd)}</span>
        </div>
      </div>
      <p className="mt-3 text-xs text-muted-foreground">
        {overlapsSelectedRegion
          ? "This window overlaps the selected candidate region."
          : "Use the controls to inspect each overlapping analysis window."}
      </p>
    </div>
  );
}

export function SemanticEvidencePanel({
  semantic,
  status,
  error,
  temporalRegions = [],
  selectedRegionId = null,
}: {
  semantic: SemanticExplanation | null | undefined;
  status: ComponentStatus;
  error?: ExplanationError | null;
  temporalRegions?: RenderedTemporalRegion[];
  selectedRegionId?: number | null;
}) {
  const [selectedRank, setSelectedRank] = useState(0);
  const [semanticWindowsOpen, setSemanticWindowsOpen] = useState(false);
  const [selectedWindowIndex, setSelectedWindowIndex] = useState(0);

  if (!semantic) {
    return (
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <CardTitle>Semantic evidence</CardTitle>
            <ComponentStatusBadge status={status} />
          </div>
        </CardHeader>
        <CardContent>
          <EvidenceUnavailablePanel
            title="No semantic evidence"
            description={
              error?.message ??
              `The semantic component is ${COMPONENT_STATUS_LABELS[
                status
              ].toLowerCase()}. No acoustic feature attributions were produced for this run.`
            }
          />
        </CardContent>
      </Card>
    );
  }

  const maxAbsShap = semantic.feature_importance.reduce(
    (max, item) => Math.max(max, Math.abs(item.shap_value)),
    0,
  );
  const selectedRegion =
    temporalRegions.find((region) => region.region_id === selectedRegionId) ?? null;
  const rankedFeatures = [...semantic.feature_importance].sort(
    (a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value),
  );
  const selectedFeature =
    rankedFeatures.find((feature) => feature.rank === selectedRank) ?? rankedFeatures[0] ?? null;
  const selectedWindow = semantic.windows[selectedWindowIndex] ?? semantic.windows[0] ?? null;

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <CardTitle>Semantic evidence</CardTitle>
            <CardDescription>
              Acoustic features that contributed most to this analysis, with the direction reported
              by the model.
            </CardDescription>
          </div>
          <ComponentStatusBadge status={semantic.status} />
        </div>
      </CardHeader>
      <CardContent className="space-y-5">
        {semantic.status !== "completed" ? (
          <EvidenceUnavailablePanel
            title="Analyzing semantic evidence..."
            description={`The semantic component is ${COMPONENT_STATUS_LABELS[
              semantic.status
            ].toLowerCase()}.`}
          />
        ) : null}

        {rankedFeatures.length === 0 ? (
          <section className="mt-5">
            <h4 className="text-sm font-semibold">Top feature ranking</h4>
            <p className="mt-2 text-sm text-muted-foreground">
              No ranked feature contributions were reported for this run.
            </p>
          </section>
        ) : (
          <>
            <div className="mt-5 grid items-start gap-5 print:hidden lg:grid-cols-[minmax(0,1.15fr)_minmax(20rem,0.85fr)]">
            <ShapRankingPlot
              features={rankedFeatures}
              maxAbsShap={maxAbsShap}
              selectedRank={selectedFeature?.rank ?? 0}
              onSelect={setSelectedRank}
              selectedRegion={selectedRegion}
            />
            <section aria-labelledby="selected-feature-heading" className="lg:sticky lg:top-4">
              <h4 id="selected-feature-heading" className="text-sm font-semibold">
                Selected feature details
              </h4>
              {selectedFeature ? (
                <ul className="mt-3">
                  <ContributionRow
                    contribution={selectedFeature}
                    maxAbsShap={maxAbsShap}
                    selected
                    highlighted={Boolean(selectedRegion && overlapsRegion(selectedFeature, selectedRegion))}
                  />
                </ul>
              ) : null}
            </section>
            </div>
            <SemanticFeaturePrintDetails features={rankedFeatures} maxAbsShap={maxAbsShap} />
          </>
        )}

        <section className="print:hidden">
          <button
            type="button"
            aria-expanded={semanticWindowsOpen}
            aria-controls="semantic-analysis-windows"
            onClick={() => setSemanticWindowsOpen((open) => !open)}
            className="flex w-full items-center justify-between gap-3 rounded-lg text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <span>
              <span className="flex flex-wrap items-center gap-2 text-sm font-semibold">
                Semantic analysis windows
                {semantic.windows.length > 0 ? (
                  <Badge variant="secondary">{semantic.windows.length} windows</Badge>
                ) : null}
              </span>
              <span className="mt-1 block text-xs font-normal text-muted-foreground">
                {semantic.windows.length > 0
                  ? "View the overlapping sliding windows used for the semantic analysis."
                  : "No sliding-window attributions were reported for this run."}
              </span>
            </span>
            {semantic.windows.length > 0 ? (
              <ChevronDown
                className={cn(
                  "h-4 w-4 shrink-0 text-muted-foreground transition-transform",
                  semanticWindowsOpen ? "rotate-180" : null,
                )}
                aria-hidden="true"
              />
            ) : null}
          </button>

          {semantic.windows.length === 0 ? (
            <p className="mt-2 text-sm text-muted-foreground">
              Whole-clip evidence only. This run did not produce sliding-window attributions, so
              semantic findings are not localised in time.
            </p>
          ) : semanticWindowsOpen ? (
            <div id="semantic-analysis-windows" className="mt-4 space-y-4">
              {selectedRegion ? (
                <p className="text-xs text-muted-foreground">
                  Windows overlapping Region {selectedRegion.region_id} are highlighted: {" "}
                  {formatSeconds(selectedRegion.start_seconds)} - {formatSeconds(selectedRegion.end_seconds)}.
                </p>
              ) : null}
              <SemanticWindowNavigator
                windows={semantic.windows}
                selectedWindowIndex={selectedWindowIndex}
                onSelect={setSelectedWindowIndex}
                selectedRegion={selectedRegion}
              />
              {selectedWindow ? (
                <div className="rounded-xl border bg-card p-4">
                  <div className="flex flex-wrap items-baseline justify-between gap-2">
                    <div>
                      <h5 className="text-sm font-semibold">Selected window</h5>
                      <p className="mt-1 text-sm font-medium tabular-nums">
                        {formatWindowInterval(selectedWindow)}
                      </p>
                    </div>
                    <Badge variant="secondary">
                      {selectedWindow.feature_importance.length} feature
                      {selectedWindow.feature_importance.length === 1 ? "" : "s"}
                    </Badge>
                  </div>
                  <ul className="mt-3 grid gap-x-5 divide-y sm:grid-cols-2 sm:divide-y-0">
                    {selectedWindow.feature_importance.map((contribution) => (
                      <li
                        key={`${contribution.rank}-${contribution.feature_name}`}
                        className="flex min-w-0 items-center justify-between gap-3 py-2 text-sm"
                      >
                        <span className="min-w-0 truncate" title={contribution.display_name}>
                          <span className="mr-1 text-muted-foreground">#{contribution.rank}</span>
                          {contribution.display_name}
                        </span>
                        <span
                          className={cn(
                            "shrink-0 text-xs font-medium tabular-nums",
                            contribution.direction === "toward_spoof"
                              ? "text-destructive"
                              : "text-emerald-700",
                          )}
                        >
                          {contribution.shap_value >= 0 ? "+" : ""}
                          {formatDecimal(contribution.shap_value, 4)}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}
            </div>
          ) : null}
        </section>

      </CardContent>
    </Card>
  );
}
