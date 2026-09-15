import type {
  SemanticEvidenceWindow,
  SemanticFeatureContribution,
  TemporalExplanation,
  TemporalRegion,
} from "@/types/api";

export type RenderedTemporalRegion = {
  region_id: number;
  start_seconds: number;
  end_seconds: number;
  duration_seconds: number;
  attention_score: number;
};

export type NormalizedTemporalRegions = {
  regions: RenderedTemporalRegion[];
  malformed: boolean;
  missing: boolean;
};

export type RegionalSemanticFeature = {
  feature_name: string;
  display_name: string;
  direction: "toward_spoof" | "toward_bonafide" | "mixed";
  strength: "Strong" | "Moderate" | "Lower";
  window_count: number;
};

export function finiteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function formatSeconds(value?: number | null) {
  if (!finiteNumber(value)) return "N/A";
  if (Math.abs(value) < 0.0001) return "0.00s";
  return `${value.toFixed(2)}s`;
}

export function formatDecimal(value?: number | null, digits = 3) {
  if (!finiteNumber(value)) return "N/A";
  return value.toFixed(digits);
}

export function formatProbabilityPercent(value?: number | null) {
  if (!finiteNumber(value)) return "N/A";
  return `${(value * 100).toFixed(2)}%`;
}

export function normalizeTemporalRegions(
  temporal?: TemporalExplanation | null,
): NormalizedTemporalRegions {
  if (!temporal) return { regions: [], malformed: false, missing: true };

  // The spectrogram uses the per-clip visualization threshold. Prefer the
  // matching intervals for the region cards and semantic drill-down so every
  // temporal view describes the same highlighted spans. A null value denotes
  // an older backend response, which still uses the evaluation regions.
  const visualizationRegions = temporal.visualization_high_attention_regions;
  return normalizeRegionList(
    Array.isArray(visualizationRegions)
      ? visualizationRegions
      : Array.isArray(temporal.high_attention_regions)
        ? temporal.high_attention_regions
        : null,
  );
}

function normalizeRegionList(
  rawRegions: TemporalRegion[] | null,
): NormalizedTemporalRegions {

  if (!rawRegions) {
    return { regions: [], malformed: false, missing: true };
  }

  const regions: RenderedTemporalRegion[] = [];
  for (const raw of rawRegions) {
    const start = finiteNumber(raw.start_seconds) ? raw.start_seconds : raw.start_sec;
    const end = finiteNumber(raw.end_seconds) ? raw.end_seconds : raw.end_sec;
    const score = finiteNumber(raw.attention_score)
      ? raw.attention_score
      : finiteNumber(raw.peak_score)
        ? raw.peak_score
        : raw.score;

    if (
      !Number.isInteger(raw.region_id) ||
      !finiteNumber(start) ||
      !finiteNumber(end) ||
      !finiteNumber(score) ||
      start < 0 ||
      end <= start
    ) {
      return { regions: [], malformed: true, missing: false };
    }

    const duration = finiteNumber(raw.duration_seconds)
      ? raw.duration_seconds
      : finiteNumber(raw.duration_sec)
        ? raw.duration_sec
        : end - start;

    regions.push({
      region_id: raw.region_id,
      start_seconds: start,
      end_seconds: end,
      duration_seconds: duration,
      attention_score: score,
    });
  }

  return { regions, malformed: false, missing: false };
}

export function temporalRegionCount(temporal: TemporalExplanation, regions: RenderedTemporalRegion[]) {
  return finiteNumber(temporal.visualization_high_attention_region_count)
    ? temporal.visualization_high_attention_region_count
    : finiteNumber(temporal.high_attention_region_count)
      ? temporal.high_attention_region_count
    : regions.length;
}

export function overlapsRegion(
  item: { start_seconds?: number | null; end_seconds?: number | null },
  region: RenderedTemporalRegion,
) {
  return (
    finiteNumber(item.start_seconds) &&
    finiteNumber(item.end_seconds) &&
    item.start_seconds < region.end_seconds &&
    item.end_seconds > region.start_seconds
  );
}

export function featuresForRegion(
  features: SemanticFeatureContribution[],
  region: RenderedTemporalRegion,
) {
  return features
    .filter((feature) => overlapsRegion(feature, region))
    .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));
}

export function windowsForRegion(windows: SemanticEvidenceWindow[], region: RenderedTemporalRegion) {
  return windows.filter((window) => overlapsRegion(window, region));
}

/**
 * Groups the per-window SHAP observations that intersect one temporal region.
 * A feature can rank highly in several sliding windows, but it should appear
 * once in the region summary rather than as visually duplicate rows.
 */
export function regionalSemanticFeatures(
  windows: SemanticEvidenceWindow[],
  region: RenderedTemporalRegion,
): RegionalSemanticFeature[] {
  const grouped = new Map<
    string,
    {
      feature_name: string;
      display_name: string;
      window_count: number;
      absolute_shap_total: number;
      positive_count: number;
      negative_count: number;
    }
  >();

  for (const window of windowsForRegion(windows, region)) {
    for (const feature of window.feature_importance) {
      const summary = grouped.get(feature.feature_name) ?? {
        feature_name: feature.feature_name,
        display_name: feature.display_name,
        window_count: 0,
        absolute_shap_total: 0,
        positive_count: 0,
        negative_count: 0,
      };
      summary.window_count += 1;
      summary.absolute_shap_total += Math.abs(feature.shap_value);
      if (feature.shap_value > 0) summary.positive_count += 1;
      if (feature.shap_value < 0) summary.negative_count += 1;
      grouped.set(feature.feature_name, summary);
    }
  }

  const features = [...grouped.values()]
    .map((feature) => ({
      ...feature,
      mean_absolute_shap: feature.absolute_shap_total / feature.window_count,
    }))
    .sort(
      (left, right) =>
        right.mean_absolute_shap - left.mean_absolute_shap ||
        left.display_name.localeCompare(right.display_name),
    );
  const strongest = features[0]?.mean_absolute_shap ?? 0;

  return features.map((feature) => ({
    feature_name: feature.feature_name,
    display_name: feature.display_name,
    direction:
      feature.positive_count > 0 && feature.negative_count > 0
        ? "mixed"
        : feature.positive_count > 0
          ? "toward_spoof"
          : "toward_bonafide",
    strength:
      strongest > 0 && feature.mean_absolute_shap / strongest >= 2 / 3
        ? "Strong"
        : strongest > 0 && feature.mean_absolute_shap / strongest >= 1 / 3
          ? "Moderate"
          : "Lower",
    window_count: feature.window_count,
  }));
}
