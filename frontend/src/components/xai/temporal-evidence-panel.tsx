import { type MouseEvent, useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { EvidenceUnavailablePanel, NotAvailable } from "@/components/xai/evidence-availability";
import { ProvenancePanel } from "@/components/xai/provenance-panel";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { WarningsPanel } from "@/components/xai/warnings-panel";
import { ComponentStatusBadge } from "@/components/xai/xai-status-badge";
import { getExplanationArtifact } from "@/lib/api/xai";
import { getPredictionAudio } from "@/lib/api/predictions";
import { COMPONENT_STATUS_LABELS } from "@/lib/xai/status";
import {
  finiteNumber,
  formatDecimal,
  formatSeconds,
  normalizeTemporalRegions,
  regionalSemanticFeatures,
  windowsForRegion,
  type RenderedTemporalRegion,
} from "@/lib/xai/regions";
import type {
  ComponentStatus,
  ExplanationComponentErrors,
  ExplanationError,
  ExplanationArtifactReference,
  ExplanationProvenance,
  SemanticExplanation,
  TemporalExplanation,
} from "@/types/api";

function displayRegionId(regionId: number) {
  return regionId.toString().padStart(2, "0");
}

type SpectrogramData = {
  matrix: number[][];
  attention: number[] | null;
  visualizationRegions: RenderedTemporalRegion[] | null;
};

type SpectrogramState =
  | { status: "loading" }
  | { status: "ready"; data: SpectrogramData }
  | { status: "image"; url: string }
  | { status: "error"; message: string };

const MATRIX_KEYS = ["log_mel_db", "mel_spectrogram", "melSpectrogram", "spectrogram", "mel", "matrix"];
const ATTENTION_KEYS = [
  "attention_density",
  "attention",
  "attention_scores",
  "attentionScores",
  "attention_weights",
  "attentionWeights",
  "attention_rollout",
  "attentionRollout",
];

const ATTENTION_LINE_CHART_PADDING = { top: 20, right: 16, bottom: 26, left: 48 };
const MINIMUM_REGION_STRIP_WIDTH = 2;

/**
 * Finds the highlighted waveform region under a pointer position.
 *
 * The chart does not use the entire canvas for time data: the y-axis and
 * right-side margin are outside the plot. Keeping this calculation in lockstep
 * with the red-strip drawing math prevents the visible strip and hover target
 * from drifting apart.
 */
export function waveformRegionAtPointer(
  pointerX: number,
  canvasWidth: number,
  durationSeconds: number,
  regions: RenderedTemporalRegion[],
) {
  if (!Number.isFinite(pointerX) || canvasWidth <= 0 || durationSeconds <= 0) return null;

  const plotWidth = canvasWidth - ATTENTION_LINE_CHART_PADDING.left - ATTENTION_LINE_CHART_PADDING.right;
  if (plotWidth <= 0) return null;

  return (
    regions.find((region) => {
      const left =
        ATTENTION_LINE_CHART_PADDING.left +
        (region.start_seconds / durationSeconds) * plotWidth;
      const right =
        ATTENTION_LINE_CHART_PADDING.left +
        (region.end_seconds / durationSeconds) * plotWidth;
      return pointerX >= left && pointerX <= Math.max(left + MINIMUM_REGION_STRIP_WIDTH, right);
    }) ?? null
  );
}

function numericVector(value: unknown): number[] | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  const numbers = value.map((item) => (typeof item === "number" ? item : Number.NaN));
  return numbers.every(Number.isFinite) ? numbers : null;
}

function numberMatrix(value: unknown): number[][] | null {
  if (!Array.isArray(value) || value.length === 0) return null;
  const rows = value.map(numericVector);
  if (rows.some((row) => !row) || rows.length === 0) return null;
  const matrix = rows as number[][];
  const width = Math.min(...matrix.map((row) => row.length));
  return width > 1 ? matrix.map((row) => row.slice(0, width)) : null;
}

function matrixFromFlatRecord(value: Record<string, unknown>): number[][] | null {
  const flat = numericVector(value.data);
  const shape = numericVector(value.shape);
  if (!flat || !shape || shape.length !== 2) return null;
  const [height, width] = shape;
  if (!Number.isInteger(height) || !Number.isInteger(width) || height * width !== flat.length) return null;
  return Array.from({ length: height }, (_, row) => flat.slice(row * width, (row + 1) * width));
}

function findMatrix(value: unknown, depth = 0): number[][] | null {
  if (depth > 5 || !value || typeof value !== "object") return null;
  if (Array.isArray(value)) return numberMatrix(value);
  const record = value as Record<string, unknown>;
  const flattened = matrixFromFlatRecord(record);
  if (flattened) return flattened;
  for (const key of MATRIX_KEYS) {
    const matrix = numberMatrix(record[key]) ?? findMatrix(record[key], depth + 1);
    if (matrix) return matrix;
  }
  for (const child of Object.values(record)) {
    const matrix = findMatrix(child, depth + 1);
    if (matrix) return matrix;
  }
  return null;
}

function meanColumns(value: unknown): number[] | null {
  const vector = numericVector(value);
  if (vector) return vector;
  const matrix = numberMatrix(value);
  if (!matrix) return null;
  return matrix[0].map((_, column) =>
    matrix.reduce((sum, row) => sum + row[column], 0) / matrix.length,
  );
}

function findAttention(value: unknown, depth = 0): number[] | null {
  if (depth > 5 || !value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  for (const key of ATTENTION_KEYS) {
    const attention = meanColumns(record[key]);
    if (attention) return attention;
  }
  for (const child of Object.values(record)) {
    const attention = findAttention(child, depth + 1);
    if (attention) return attention;
  }
  return null;
}

export function visualizationRegionsFromSpectrogram(
  value: unknown,
): RenderedTemporalRegion[] | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;

  const record = value as Record<string, unknown>;
  if (!Object.prototype.hasOwnProperty.call(record, "high_attention_regions")) {
    return null;
  }

  const rawRegions = record.high_attention_regions;
  if (!Array.isArray(rawRegions)) return null;

  const regions: RenderedTemporalRegion[] = [];
  for (const rawRegion of rawRegions) {
    if (!rawRegion || typeof rawRegion !== "object" || Array.isArray(rawRegion)) {
      return null;
    }

    const region = rawRegion as Record<string, unknown>;
    const regionId = region.region_id;
    const start = region.start_seconds;
    const end = region.end_seconds;
    const score = region.attention_score;

    if (
      typeof regionId !== "number" ||
      !Number.isInteger(regionId) ||
      !finiteNumber(start) ||
      !finiteNumber(end) ||
      !finiteNumber(score) ||
      start < 0 ||
      end <= start
    ) {
      return null;
    }

    regions.push({
      region_id: regionId,
      start_seconds: start,
      end_seconds: end,
      duration_seconds: end - start,
      attention_score: score,
    });
  }

  return regions;
}

function parseSpectrogram(value: unknown): SpectrogramData | null {
  const matrix = findMatrix(value);
  if (!matrix) return null;
  const attention = findAttention(value);
  const visualizationRegions = visualizationRegionsFromSpectrogram(value);
  if (attention && attention.length === matrix.length && attention.length !== matrix[0].length) {
    const transposed = matrix[0].map((_, column) => matrix.map((row) => row[column]));
    return { matrix: transposed, attention, visualizationRegions };
  }
  return { matrix, attention, visualizationRegions };
}

function interpolateAttention(attention: number[], index: number, width: number) {
  if (attention.length === 1) return attention[0];
  const position = (index / Math.max(1, width - 1)) * (attention.length - 1);
  const lower = Math.floor(position);
  const upper = Math.min(attention.length - 1, lower + 1);
  return attention[lower] + (attention[upper] - attention[lower]) * (position - lower);
}

function spectrogramColor(value: number): [number, number, number] {
  const stop = Math.max(0, Math.min(1, value));
  const red = Math.round(22 + 231 * Math.pow(stop, 1.35));
  const green = Math.round(20 + 199 * Math.sin(stop * Math.PI * 0.82));
  const blue = Math.round(58 + 133 * (1 - stop) * (1 - stop));
  return [red, green, blue];
}

function attentionOverlayColor(value: number): [number, number, number] {
  const stop = Math.max(0, Math.min(1, value));
  // Cyan-to-amber keeps density distinct from the spectrogram palette.
  return [
    Math.round(8 + 245 * stop),
    Math.round(173 + 56 * stop),
    Math.round(218 - 184 * stop),
  ];
}

function valueRange(values: Iterable<number>) {
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (const value of values) {
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  return { minimum, maximum };
}

/**
 * This only changes the visual density of the curve. The source attention
 * values remain untouched and are still used for all evidence calculations.
 */
function smoothAttentionForDisplay(values: number[], maximumPoints = 180) {
  const bucketSize = Math.max(1, Math.ceil(values.length / maximumPoints));
  const downsampled = Array.from({ length: Math.ceil(values.length / bucketSize) }, (_, bucket) => {
    const start = bucket * bucketSize;
    const slice = values.slice(start, start + bucketSize);
    return slice.reduce((sum, value) => sum + value, 0) / slice.length;
  });

  return downsampled.map((_, index) => {
    const start = Math.max(0, index - 2);
    const end = Math.min(downsampled.length, index + 3);
    const window = downsampled.slice(start, end);
    return window.reduce((sum, value) => sum + value, 0) / window.length;
  });
}

function AttentionSpectrogram({
  predictionId,
  artifacts,
  highAttentionRegions,
  durationSeconds,
}: {
  predictionId: string;
  artifacts: ExplanationArtifactReference[];
  highAttentionRegions: RenderedTemporalRegion[];
  durationSeconds: number;
}) {
  const artifact = artifacts.find((item) => item.kind === "attention_spectrogram");
  const artifactId = artifact?.artifact_id;
  const [state, setState] = useState<SpectrogramState>(() =>
    artifact ? { status: "loading" } : { status: "error", message: "No attention spectrogram is available for this run." },
  );
  const [hoveredRegionId, setHoveredRegionId] = useState<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRegions =
    state.status === "ready" && state.data.visualizationRegions !== null
      ? state.data.visualizationRegions
      : highAttentionRegions;
  const hoveredRegion = overlayRegions.find((region) => region.region_id === hoveredRegionId) ?? null;

  useEffect(() => {
    if (!artifactId) return;
    let cancelled = false;
    let objectUrl: string | null = null;

    const load = async () => {
      setState({ status: "loading" });
      try {
        const blob = await getExplanationArtifact(predictionId, artifactId);
        if (blob.type.startsWith("image/")) {
          objectUrl = URL.createObjectURL(blob);
          if (!cancelled) setState({ status: "image", url: objectUrl });
          return;
        }
        const parsed = parseSpectrogram(JSON.parse(await blob.text()));
        if (!parsed) {
          throw new Error("The artifact does not include a supported mel-spectrogram matrix.");
        }
        if (!cancelled) setState({ status: "ready", data: parsed });
      } catch (error) {
        if (!cancelled) {
          setState({
            status: "error",
            message:
              error instanceof Error
                ? error.message
                : "The attention spectrogram could not be rendered.",
          });
        }
      }
    };

    void load();
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [artifactId, predictionId]);

  useEffect(() => {
    if (state.status !== "ready" || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const draw = () => {
      const context = canvas.getContext("2d");
      if (!context) return;
      const width = Math.max(320, Math.floor((canvas.parentElement?.clientWidth ?? 720) - 8));
      const height = Math.round(Math.min(420, Math.max(220, width * 0.42)));
      const pixelRatio = window.devicePixelRatio || 1;
      canvas.width = width * pixelRatio;
      canvas.height = height * pixelRatio;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

      const { matrix, attention } = state.data;
      const { minimum, maximum } = valueRange(matrix.flat());
      const range = maximum - minimum || 1;
      const source = document.createElement("canvas");
      source.width = matrix[0].length;
      source.height = matrix.length;
      const sourceContext = source.getContext("2d");
      if (!sourceContext) return;
      const image = sourceContext.createImageData(source.width, source.height);
      matrix.forEach((row, rowIndex) => {
        row.forEach((value, columnIndex) => {
          const [red, green, blue] = spectrogramColor((value - minimum) / range);
          const pixel = ((matrix.length - rowIndex - 1) * source.width + columnIndex) * 4;
          image.data[pixel] = red;
          image.data[pixel + 1] = green;
          image.data[pixel + 2] = blue;
          image.data[pixel + 3] = 255;
        });
      });
      sourceContext.putImageData(image, 0, 0);
      context.imageSmoothingEnabled = true;
      context.drawImage(source, 0, 0, width, height);

      if (attention?.length) {
        const { minimum: attentionMinimum, maximum: attentionMaximum } = valueRange(attention);
        const attentionRange = attentionMaximum - attentionMinimum || 1;
        for (let column = 0; column < width; column += 1) {
          const score = (interpolateAttention(attention, column, width) - attentionMinimum) / attentionRange;
          const [red, green, blue] = attentionOverlayColor(score);
          context.fillStyle = `rgba(${red}, ${green}, ${blue}, ${0.06 + score * 0.38})`;
          context.fillRect(column, 0, 1, height);
        }
      }

      if (durationSeconds > 0) {
        overlayRegions.forEach((region) => {
          const left = Math.max(0, Math.min(width, (region.start_seconds / durationSeconds) * width));
          const right = Math.max(left + 2, Math.min(width, (region.end_seconds / durationSeconds) * width));
          const hovered = region.region_id === hoveredRegionId;
          context.fillStyle = hovered ? "rgba(185, 28, 28, 0.50)" : "rgba(185, 28, 28, 0.28)";
          context.fillRect(left, 0, right - left, height);
          context.strokeStyle = hovered ? "rgba(127, 29, 29, 1)" : "rgba(185, 28, 28, 0.95)";
          context.lineWidth = hovered ? 3 : 1;
          context.shadowColor = hovered ? "rgba(127, 29, 29, 0.95)" : "transparent";
          context.shadowBlur = hovered ? 12 : 0;
          context.strokeRect(left + 0.5, 0.5, Math.max(0, right - left - 1), height - 1);
          context.shadowBlur = 0;
        });
      }
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas.parentElement ?? canvas);
    return () => observer.disconnect();
  }, [durationSeconds, hoveredRegionId, overlayRegions, state]);

  const onSpectrogramMouseMove = (event: MouseEvent<HTMLCanvasElement>) => {
    if (durationSeconds <= 0) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const seconds = ((event.clientX - bounds.left) / bounds.width) * durationSeconds;
    const nextRegion = overlayRegions.find(
      (region) => seconds >= region.start_seconds && seconds <= region.end_seconds,
    );
    setHoveredRegionId((current) => (current === (nextRegion?.region_id ?? null) ? current : nextRegion?.region_id ?? null));
  };

  return (
    <section aria-labelledby="attention-spectrogram-heading">
      <div>
        <h4 id="attention-spectrogram-heading" className="text-sm font-semibold">
          Mel-spectrogram with attention overlay
        </h4>
        <p className="mt-1 text-xs text-muted-foreground">
          Colour intensity shows continuous attention density; red spans use the per-clip visualization threshold.
        </p>
      </div>
      <div className="mt-3 grid grid-cols-[1.25rem_minmax(0,1fr)] gap-2">
        <p className="flex items-center justify-center text-xs font-medium text-slate-500 [writing-mode:vertical-rl] rotate-180">
          Mel frequency bins
        </p>
        <div>
          <div className="relative overflow-hidden rounded-md border bg-slate-950 p-1">
            {state.status === "loading" ? (
              <p className="p-4 text-sm text-slate-200">Loading spectrogram…</p>
            ) : null}
            {state.status === "image" ? (
              <>
                {/* The authenticated artifact is a short-lived Blob URL, so it cannot use Next Image optimisation. */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={state.url}
                  alt="Mel-spectrogram with attention overlay"
                  className="h-auto w-full rounded-sm"
                />
              </>
            ) : null}
            {state.status === "ready" ? (
              <canvas
                ref={canvasRef}
                role="img"
                aria-label="Mel-spectrogram with attention overlay"
                className="block max-w-full cursor-crosshair rounded-sm"
                onMouseMove={onSpectrogramMouseMove}
                onMouseLeave={() => setHoveredRegionId(null)}
              />
            ) : null}
            {state.status === "error" ? (
              <p className="p-4 text-sm text-slate-200">{state.message}</p>
            ) : null}
            {hoveredRegion && durationSeconds > 0 ? (
              <div
                role="tooltip"
                className="pointer-events-none absolute top-4 z-10 rounded-md border border-amber-200 bg-amber-950/95 px-3 py-2 text-xs text-amber-50 shadow-lg"
                style={{
                  left: `${Math.max(
                    8,
                    Math.min(92, ((hoveredRegion.start_seconds + hoveredRegion.end_seconds) / 2 / durationSeconds) * 100),
                  )}%`,
                  transform: "translateX(-50%)",
                }}
              >
                <p className="font-semibold">High attention · Region {displayRegionId(hoveredRegion.region_id)}</p>
                <p className="mt-0.5 text-amber-100">Attention score: {formatDecimal(hoveredRegion.attention_score)}</p>
              </div>
            ) : null}
          </div>
          <RegionTimeAxis durationSeconds={durationSeconds} regions={overlayRegions} />
        </div>
      </div>
    </section>
  );
}

function RegionTimeAxis({
  durationSeconds,
  regions,
  plotPadding = { left: 0, right: 0 },
}: {
  durationSeconds: number;
  regions: RenderedTemporalRegion[];
  plotPadding?: { left: number; right: number };
}) {
  if (durationSeconds <= 0) return null;

  const horizontalPadding = plotPadding.left + plotPadding.right;
  const horizontalPosition = (seconds: number) => {
    const proportion = Math.max(0, Math.min(1, seconds / durationSeconds));
    return `calc(${plotPadding.left}px + ${proportion * 100}% - ${proportion * horizontalPadding}px)`;
  };

  return (
    <div className="mx-2 mt-2 border-t border-border/70 pt-1 text-[11px] tabular-nums text-muted-foreground">
      <div className="relative h-4">
        <span className="absolute left-0 top-0" style={{ left: horizontalPosition(0) }}>
          0.00s
        </span>
        <span
          className="absolute top-0"
          style={{ left: horizontalPosition(durationSeconds), transform: "translateX(-100%)" }}
        >
          {formatSeconds(durationSeconds)}
        </span>
      </div>
      <div className="relative -mt-1 h-6 overflow-visible">
        {regions.map((region) => {
          return (
            <div
              key={region.region_id}
              className="absolute top-0 flex flex-col items-center"
              style={{ left: horizontalPosition(region.end_seconds), transform: "translateX(-50%)" }}
              aria-label={`Region ${displayRegionId(region.region_id)} ends at ${formatSeconds(region.end_seconds)}`}
            >
              <span className="h-1.5 border-l border-muted-foreground" aria-hidden="true" />
              <span className="whitespace-nowrap text-[10px] font-medium text-foreground">
                {formatSeconds(region.end_seconds)}
              </span>
            </div>
          );
        })}
      </div>
      <p className="text-center text-xs font-medium text-muted-foreground">Time (seconds)</p>
    </div>
  );
}

type AttentionSeriesState =
  | { status: "loading" }
  | { status: "ready"; values: number[] }
  | { status: "error"; message: string };

function AttentionLineChart({
  predictionId,
  artifacts,
  highAttentionRegions,
  durationSeconds,
  threshold,
}: {
  predictionId: string;
  artifacts: ExplanationArtifactReference[];
  highAttentionRegions: RenderedTemporalRegion[];
  durationSeconds: number;
  threshold?: number | null;
}) {
  const artifact = artifacts.find((item) => item.kind === "attention_visualization");
  const artifactId = artifact?.artifact_id;
  const [state, setState] = useState<AttentionSeriesState>(() =>
    artifact
      ? { status: "loading" }
      : { status: "error", message: "No attention visualization is available for this run." },
  );
  const [hoveredRegionId, setHoveredRegionId] = useState<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const hoveredRegion = highAttentionRegions.find((region) => region.region_id === hoveredRegionId) ?? null;

  useEffect(() => {
    if (!artifactId) return;
    let cancelled = false;
    const load = async () => {
      setState({ status: "loading" });
      try {
        const blob = await getExplanationArtifact(predictionId, artifactId);
        const parsed = JSON.parse(await blob.text());
        const values = numericVector(parsed) ?? findAttention(parsed);
        if (!values) throw new Error("The artifact does not include a supported attention series.");
        if (!cancelled) setState({ status: "ready", values });
      } catch (error) {
        if (!cancelled) {
          setState({
            status: "error",
            message:
              error instanceof Error
                ? error.message
                : "The attention visualization could not be rendered.",
          });
        }
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [artifactId, predictionId]);

  useEffect(() => {
    if (state.status !== "ready" || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const draw = () => {
      const context = canvas.getContext("2d");
      if (!context) return;
      const width = Math.max(320, Math.floor((canvas.parentElement?.clientWidth ?? 720) - 8));
      const height = 280;
      const pixelRatio = window.devicePixelRatio || 1;
      canvas.width = width * pixelRatio;
      canvas.height = height * pixelRatio;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, width, height);

      const padding = ATTENTION_LINE_CHART_PADDING;
      const plotWidth = width - padding.left - padding.right;
      const plotHeight = height - padding.top - padding.bottom;
      const renderedValues = smoothAttentionForDisplay(state.values);
      const rawValueRange = valueRange(state.values);
      // Keep the plot and threshold in the backend's original score domain.
      // Smoothed values are used only to make the drawn curve easier to read.
      const minimum = finiteNumber(threshold)
        ? Math.min(rawValueRange.minimum, threshold)
        : rawValueRange.minimum;
      const maximum = finiteNumber(threshold)
        ? Math.max(rawValueRange.maximum, threshold)
        : rawValueRange.maximum;
      const range = maximum - minimum || 1;
      context.fillStyle = "rgba(15, 23, 42, 0.96)";
      context.fillRect(0, 0, width, height);
      context.strokeStyle = "rgba(148, 163, 184, 0.16)";
      context.lineWidth = 1;
      context.font = "11px sans-serif";
      context.fillStyle = "#94a3b8";
      context.textAlign = "right";
      context.textBaseline = "middle";
      for (let index = 0; index <= 4; index += 1) {
        const y = padding.top + (plotHeight / 4) * index;
        context.beginPath();
        context.moveTo(padding.left, y);
        context.lineTo(width - padding.right, y);
        context.stroke();
        const tickValue = maximum - (range / 4) * index;
        context.fillText(formatDecimal(tickValue), padding.left - 7, y);
      }
      for (let index = 1; index < 6; index += 1) {
        const x = padding.left + (plotWidth / 6) * index;
        context.beginPath();
        context.moveTo(x, padding.top);
        context.lineTo(x, padding.top + plotHeight);
        context.stroke();
      }
      if (durationSeconds > 0) {
        highAttentionRegions.forEach((region) => {
          const left = padding.left + (region.start_seconds / durationSeconds) * plotWidth;
          const right = padding.left + (region.end_seconds / durationSeconds) * plotWidth;
          const hovered = region.region_id === hoveredRegionId;
          context.fillStyle = hovered ? "rgba(185, 28, 28, 0.50)" : "rgba(185, 28, 28, 0.28)";
          context.fillRect(
            left,
            padding.top,
            Math.max(MINIMUM_REGION_STRIP_WIDTH, right - left),
            plotHeight,
          );
          if (hovered) {
            context.strokeStyle = "rgba(127, 29, 29, 1)";
            context.lineWidth = 2;
            context.shadowColor = "rgba(127, 29, 29, 0.95)";
            context.shadowBlur = 12;
            context.strokeRect(left + 1, padding.top + 1, Math.max(0, right - left - 2), plotHeight - 2);
            context.shadowBlur = 0;
          }
        });
      }

      if (finiteNumber(threshold)) {
        const y = Math.max(
          padding.top,
          Math.min(padding.top + plotHeight, padding.top + plotHeight - ((threshold - minimum) / range) * plotHeight),
        );
        context.setLineDash([6, 5]);
        context.strokeStyle = "rgba(185, 28, 28, 0.95)";
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(padding.left, y);
        context.lineTo(width - padding.right, y);
        context.stroke();
        context.setLineDash([]);
        const label = `Threshold ${formatDecimal(threshold)}`;
        context.font = "12px sans-serif";
        context.textAlign = "left";
        context.textBaseline = "alphabetic";
        const labelWidth = context.measureText(label).width + 12;
        context.fillStyle = "rgba(15, 23, 42, 0.92)";
        context.fillRect(width - padding.right - labelWidth, Math.max(padding.top, y - 18), labelWidth, 16);
        context.fillStyle = "#b91c1c";
        context.fillText(label, width - padding.right - labelWidth + 6, Math.max(padding.top + 12, y - 6));
      }

      const toPoint = (value: number, index: number, valueCount: number) => ({
        x: padding.left + (index / Math.max(1, valueCount - 1)) * plotWidth,
        y: padding.top + plotHeight - ((value - minimum) / range) * plotHeight,
      });

      // Keep the raw density visible so a plotted threshold crossing always
      // corresponds to the interval-selection data. The smoothed curve remains
      // as a more readable trend layer above it.
      context.beginPath();
      state.values.forEach((value, index) => {
        const point = toPoint(value, index, state.values.length);
        if (index === 0) context.moveTo(point.x, point.y);
        else context.lineTo(point.x, point.y);
      });
      context.strokeStyle = "rgba(96, 165, 250, 0.8)";
      context.lineWidth = 1;
      context.lineJoin = "round";
      context.lineCap = "round";
      context.stroke();

      const first = toPoint(renderedValues[0], 0, renderedValues.length);
      const last = toPoint(renderedValues[renderedValues.length - 1], renderedValues.length - 1, renderedValues.length);
      context.beginPath();
      renderedValues.forEach((value, index) => {
        const point = toPoint(value, index, renderedValues.length);
        if (index === 0) context.moveTo(point.x, point.y);
        else context.lineTo(point.x, point.y);
      });
      context.lineTo(last.x, padding.top + plotHeight);
      context.lineTo(first.x, padding.top + plotHeight);
      context.closePath();
      context.fillStyle = "rgba(56, 189, 248, 0.16)";
      context.fill();

      context.beginPath();
      renderedValues.forEach((value, index) => {
        const point = toPoint(value, index, renderedValues.length);
        if (index === 0) context.moveTo(point.x, point.y);
        else context.lineTo(point.x, point.y);
      });
      context.strokeStyle = "#22d3ee";
      context.lineWidth = 1.5;
      context.lineJoin = "round";
      context.lineCap = "round";
      context.stroke();
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas.parentElement ?? canvas);
    return () => observer.disconnect();
  }, [durationSeconds, highAttentionRegions, hoveredRegionId, state, threshold]);

  const onAttentionWaveformMouseMove = (event: MouseEvent<HTMLCanvasElement>) => {
    if (durationSeconds <= 0) return;
    const canvas = event.currentTarget;
    const bounds = canvas.getBoundingClientRect();
    const canvasWidth = canvas.clientWidth || bounds.width;
    const pointerX = ((event.clientX - bounds.left) / bounds.width) * canvasWidth;
    const nextRegion = waveformRegionAtPointer(
      pointerX,
      canvasWidth,
      durationSeconds,
      highAttentionRegions,
    );
    setHoveredRegionId((current) =>
      current === (nextRegion?.region_id ?? null) ? current : (nextRegion?.region_id ?? null),
    );
  };

  return (
    <section aria-labelledby="attention-waveform-heading">
      <div>
        <h4 id="attention-waveform-heading" className="text-sm font-semibold">
          Attention waveform
        </h4>
        <p className="mt-1 text-xs text-muted-foreground">
          Red spans show label-neutral high attention.
        </p>
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted-foreground" aria-label="Waveform legend">
          <span className="inline-flex items-center gap-1.5">
            <span className="h-px w-4 bg-blue-400" aria-hidden="true" />
            Raw attention density
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="h-0.5 w-4 bg-cyan-400" aria-hidden="true" />
            Smoothed attention trend
          </span>
        </div>
      </div>
      <div className="mt-3 grid grid-cols-[1.25rem_minmax(0,1fr)] gap-2">
        <p className="flex items-center justify-center text-xs font-medium text-slate-500 [writing-mode:vertical-rl] rotate-180">
          Attention score
        </p>
        <div>
          <div className="relative overflow-hidden rounded-md border bg-slate-950 p-1">
            {state.status === "loading" ? <p className="p-4 text-sm text-slate-200">Loading attention waveform…</p> : null}
            {state.status === "ready" ? (
              <canvas
                ref={canvasRef}
                role="img"
                aria-label="Attention waveform with highlighted high-attention regions"
                className="block max-w-full cursor-crosshair rounded-sm"
                onMouseMove={onAttentionWaveformMouseMove}
                onMouseLeave={() => setHoveredRegionId(null)}
              />
            ) : null}
            {state.status === "error" ? <p className="p-4 text-sm text-slate-200">{state.message}</p> : null}
            {hoveredRegion && durationSeconds > 0 ? (
              <div
                role="tooltip"
                className="pointer-events-none absolute top-4 z-10 rounded-md border border-amber-200 bg-amber-950/95 px-3 py-2 text-xs text-amber-50 shadow-lg"
                style={{
                  left: `${Math.max(
                    8,
                    Math.min(92, ((hoveredRegion.start_seconds + hoveredRegion.end_seconds) / 2 / durationSeconds) * 100),
                  )}%`,
                  transform: "translateX(-50%)",
                }}
              >
                <p className="font-semibold">High attention · Region {displayRegionId(hoveredRegion.region_id)}</p>
                <p className="mt-0.5 text-amber-100">Attention score: {formatDecimal(hoveredRegion.attention_score)}</p>
              </div>
            ) : null}
          </div>
          <RegionTimeAxis
            durationSeconds={durationSeconds}
            regions={highAttentionRegions}
            plotPadding={{ left: 48, right: 16 }}
          />
        </div>
      </div>
    </section>
  );
}

type AudioClipState =
  | { status: "loading" }
  | { status: "ready"; url: string }
  | { status: "error" };

function AudioClip({ predictionId, available }: { predictionId: string; available: boolean }) {
  const [state, setState] = useState<AudioClipState>(() =>
    available ? { status: "loading" } : { status: "error" },
  );

  useEffect(() => {
    if (!available) return;
    let cancelled = false;
    const load = async () => {
      setState({ status: "loading" });
      try {
        const playback = await getPredictionAudio(predictionId);
        if (!cancelled) setState({ status: "ready", url: playback.playback_url });
      } catch {
        if (!cancelled) setState({ status: "error" });
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [available, predictionId]);

  if (!available) {
    return <p className="text-sm text-muted-foreground">Audio playback is unavailable for this prediction.</p>;
  }
  if (state.status === "error") {
    return <p className="text-sm text-muted-foreground">Audio playback is currently unavailable.</p>;
  }
  if (state.status === "loading") {
    return <p className="text-sm text-muted-foreground">Loading audio clip…</p>;
  }

  return (
    <audio controls src={state.url} className="w-full" aria-label="Audio clip">
      Audio playback is unavailable.
    </audio>
  );
}

function TemporalVisualWorkspace({
  predictionId,
  artifacts,
  highAttentionRegions,
  durationSeconds,
  attentionThreshold,
  audioAvailable,
}: {
  predictionId: string;
  artifacts: ExplanationArtifactReference[];
  highAttentionRegions: RenderedTemporalRegion[];
  durationSeconds: number;
  attentionThreshold?: number | null;
  audioAvailable: boolean;
}) {
  return (
    <section className="space-y-4" aria-label="Temporal attention visualizations">
      <Tabs defaultValue="raw">
        <TabsList aria-label="Attention visualization mode">
          <TabsTrigger value="raw">Raw</TabsTrigger>
          <TabsTrigger value="waveform">Waveform</TabsTrigger>
        </TabsList>
        <TabsContent value="raw">
          <AttentionSpectrogram
            predictionId={predictionId}
            artifacts={artifacts}
            highAttentionRegions={highAttentionRegions}
            durationSeconds={durationSeconds}
          />
        </TabsContent>
        <TabsContent value="waveform">
          <AttentionLineChart
            predictionId={predictionId}
            artifacts={artifacts}
            highAttentionRegions={highAttentionRegions}
            durationSeconds={durationSeconds}
            threshold={attentionThreshold}
          />
        </TabsContent>
      </Tabs>
      <AudioClip predictionId={predictionId} available={audioAvailable} />
    </section>
  );
}

function HighAttentionRegionNavigator({
  regions,
  selectedRegionId,
  onSelect,
}: {
  regions: RenderedTemporalRegion[];
  selectedRegionId?: number | null;
  onSelect: (regionId: number) => void;
}) {
  return (
    <section className="flex h-full min-h-0 flex-col rounded-md border bg-muted/20" aria-labelledby="region-navigator-heading">
      <div className="flex items-center justify-between gap-3 border-b px-3 py-2">
        <div>
          <h5 id="region-navigator-heading" className="text-sm font-semibold">Region navigator</h5>
          <p className="text-xs text-muted-foreground">Select a region to inspect its details.</p>
        </div>
        <span className="shrink-0 rounded-md bg-muted px-2 py-1 text-xs font-medium tabular-nums">
          {regions.length} {regions.length === 1 ? "region" : "regions"}
        </span>
      </div>
      <div
        className="flex-1 space-y-1 overflow-y-auto p-2 [scrollbar-width:thin] [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-muted-foreground/40 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar]:w-1.5"
        aria-label="High-attention region list"
      >
        {regions.map((region) => {
          const selected = region.region_id === selectedRegionId;
          return (
            <button
              key={region.region_id}
              type="button"
              aria-pressed={selected}
              className={`w-full rounded-md border px-3 py-3 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                selected
                  ? "border-red-700 bg-red-50 dark:bg-red-950/30"
                  : "border-border bg-background hover:border-red-400 hover:bg-red-50/50 dark:hover:bg-red-950/20"
              }`}
              onClick={() => onSelect(region.region_id)}
            >
              <span className="flex items-center justify-between gap-2">
                <span className="text-sm font-semibold">Region {displayRegionId(region.region_id)}</span>
                <span className="rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground">
                  {selected ? "Selected" : "View details"}
                </span>
              </span>
              <dl className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
                <div className="rounded-md border bg-background/70 px-2 py-1.5">
                  <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                    Start time
                  </dt>
                  <dd className="mt-0.5 text-sm font-semibold tabular-nums">
                    {formatSeconds(region.start_seconds)}
                  </dd>
                </div>
                <div className="rounded-md border bg-background/70 px-2 py-1.5">
                  <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                    End time
                  </dt>
                  <dd className="mt-0.5 text-sm font-semibold tabular-nums">
                    {formatSeconds(region.end_seconds)}
                  </dd>
                </div>
                <div className="rounded-md border bg-background/70 px-2 py-1.5">
                  <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                    Duration
                  </dt>
                  <dd className="mt-0.5 text-sm font-semibold tabular-nums">
                    {formatSeconds(region.duration_seconds)}
                  </dd>
                </div>
                <div className="rounded-md border bg-background/70 px-2 py-1.5">
                  <dt className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                    Attention score
                  </dt>
                  <dd className="mt-0.5 text-sm font-semibold tabular-nums text-red-700 dark:text-red-400">
                    {formatDecimal(region.attention_score)}
                  </dd>
                </div>
              </dl>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function SelectedRegionSemanticFeatures({
  region,
  semantic,
  title = "Semantic evidence in this region",
  printedDetail = false,
}: {
  region: RenderedTemporalRegion;
  semantic?: SemanticExplanation | null;
  title?: string;
  printedDetail?: boolean;
}) {
  const overlappingWindows = semantic ? windowsForRegion(semantic.windows, region) : [];
  const regionalFeatures = semantic ? regionalSemanticFeatures(semantic.windows, region) : [];

  return (
    <section className="h-full rounded-md border bg-muted/30 p-4">
      <h4 className="text-sm font-semibold">{title}</h4>
      <p className="mt-1 text-xs text-muted-foreground">
        {printedDetail
          ? `High-attention interval ${displayRegionId(region.region_id)}: ${formatSeconds(region.start_seconds)} - ${formatSeconds(region.end_seconds)}`
          : `Region ${displayRegionId(region.region_id)} · ${formatSeconds(region.start_seconds)} - ${formatSeconds(region.end_seconds)}`}
      </p>
      <section className="mt-4">
        {semantic ? (
          <div className="mt-2 space-y-3">
            <p className="text-sm">
              Analysis windows intersecting this region:{" "}
              <span className="font-medium">{overlappingWindows.length}</span>
            </p>
            {regionalFeatures.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No time-localised semantic features were reported for this region.
              </p>
            ) : (
              <>
                <div className="rounded-md border bg-background">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Acoustic feature</TableHead>
                        <TableHead>Evidence in this region</TableHead>
                        <TableHead>Strength</TableHead>
                        <TableHead className="text-right">Appears in</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {regionalFeatures.slice(0, 6).map((feature) => {
                        const evidence =
                          feature.direction === "toward_spoof"
                            ? "Supports spoof"
                            : feature.direction === "toward_bonafide"
                              ? "Supports bonafide"
                              : "Mixed evidence";
                        const variant =
                          feature.direction === "toward_spoof"
                            ? "destructive"
                            : feature.direction === "toward_bonafide"
                              ? "success"
                              : "warning";
                        return (
                          <TableRow key={feature.feature_name}>
                            <TableCell className="font-medium">{feature.display_name}</TableCell>
                            <TableCell><Badge variant={variant}>{evidence}</Badge></TableCell>
                            <TableCell>{feature.strength}</TableCell>
                            <TableCell className="text-right tabular-nums">
                              {feature.window_count} of {overlappingWindows.length} windows
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>
                <p className="text-xs text-muted-foreground">
                  Repeated window-level observations are grouped by feature. Strength reflects the mean absolute SHAP contribution, not a probability.
                </p>
              </>
            )}
          </div>
        ) : (
          <p className="mt-2 text-sm text-muted-foreground">
            Semantic evidence is not available for this analysis.
          </p>
        )}
      </section>
    </section>
  );
}

/**
 * The interactive navigator exposes one selected region at a time. A printed
 * report must be complete and self-contained, so it expands every reported
 * region (including its time interval, attention score, and semantic detail).
 */
function HighAttentionRegionPrintDetails({
  regions,
  semantic,
}: {
  regions: RenderedTemporalRegion[];
  semantic?: SemanticExplanation | null;
}) {
  return (
    <section className="hidden print:block" aria-labelledby="printed-region-details-heading">
      <h4 id="printed-region-details-heading" className="text-sm font-semibold">
        All high-attention region details
      </h4>
      <div className="mt-3 space-y-4">
        {regions.map((region) => (
          <section key={region.region_id} className="print-region-detail rounded-md border bg-muted/20 p-4">
            <h5 className="text-sm font-semibold">
              Region {displayRegionId(region.region_id)} print details
            </h5>
            <dl className="mt-3 grid gap-3 sm:grid-cols-3">
              <div>
                <dt className="text-xs text-muted-foreground">Time interval</dt>
                <dd className="mt-1 text-sm font-medium tabular-nums">
                  {formatSeconds(region.start_seconds)} - {formatSeconds(region.end_seconds)}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Duration</dt>
                <dd className="mt-1 text-sm font-medium tabular-nums">
                  {formatSeconds(region.duration_seconds)}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Attention score</dt>
                <dd className="mt-1 text-sm font-medium tabular-nums">
                  {formatDecimal(region.attention_score)}
                </dd>
              </div>
            </dl>
            <div className="mt-4">
              <SelectedRegionSemanticFeatures
                region={region}
                semantic={semantic}
                title="Printed semantic evidence in this high-attention region"
                printedDetail
              />
            </div>
          </section>
        ))}
      </div>
    </section>
  );
}

export function TemporalEvidencePanel({
  predictionId,
  temporal,
  status,
  error,
  clipDurationSeconds,
  semantic,
  selectedRegionId,
  onSelectedRegionIdChange,
  audioAvailable = false,
  warnings,
  componentErrors,
  runError,
  provenance,
  explanationId,
  requestId,
  createdAt,
  completedAt,
}: {
  predictionId: string;
  temporal: TemporalExplanation | null | undefined;
  status: ComponentStatus;
  error?: ExplanationError | null;
  clipDurationSeconds?: number | null;
  semantic?: SemanticExplanation | null;
  selectedRegionId?: number | null;
  onSelectedRegionIdChange?: (regionId: number | null) => void;
  audioAvailable?: boolean;
  warnings?: string[];
  componentErrors?: ExplanationComponentErrors;
  runError?: ExplanationError | null;
  provenance?: ExplanationProvenance;
  explanationId?: string;
  requestId?: string;
  createdAt?: string;
  completedAt?: string | null;
}) {
  const [locallySelectedRegionId, setLocallySelectedRegionId] = useState<number | null>(null);
  const [selectedDetailHeight, setSelectedDetailHeight] = useState<number | null>(null);
  const selectedDetailRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const detail = selectedDetailRef.current;
    if (!detail) {
      setSelectedDetailHeight(null);
      return;
    }

    const syncHeight = () => {
      setSelectedDetailHeight(Math.ceil(detail.getBoundingClientRect().height));
    };
    syncHeight();
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(syncHeight);
    observer.observe(detail);
    return () => observer.disconnect();
  }, [locallySelectedRegionId, semantic, selectedRegionId, temporal]);

  if (!temporal) {
    return (
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <CardTitle>Temporal evidence</CardTitle>
            <ComponentStatusBadge status={status} />
          </div>
        </CardHeader>
        <CardContent>
          <EvidenceUnavailablePanel
            title="No temporal evidence"
            description={
              error?.message ??
              `The temporal component is ${COMPONENT_STATUS_LABELS[
                status
              ].toLowerCase()}. No attention regions were produced for this run.`
            }
          />
        </CardContent>
      </Card>
    );
  }

  const duration =
    typeof clipDurationSeconds === "number" && clipDurationSeconds > 0 ? clipDurationSeconds : null;
  const {
    regions: highAttentionRegions,
    malformed,
    missing,
  } = normalizeTemporalRegions(temporal);
  const totalDuration =
    duration ??
    Math.max(
      ...highAttentionRegions.map((region) => region.end_seconds),
      temporal.high_attention_combined_duration_seconds ?? 0,
    );
  const peakAttention = finiteNumber(temporal.attention_score_peak)
    ? temporal.attention_score_peak
    : temporal.peak_attention;
  const highAttentionCount = finiteNumber(temporal.visualization_high_attention_region_count)
    ? temporal.visualization_high_attention_region_count
    : finiteNumber(temporal.high_attention_region_count)
      ? temporal.high_attention_region_count
    : highAttentionRegions.length;
  const selectedHighAttentionRegion =
    highAttentionRegions.find(
      (region) => region.region_id === (selectedRegionId ?? locallySelectedRegionId),
    ) ??
    highAttentionRegions[0] ??
    null;
  const onSelectRegion = (regionId: number) => {
    setLocallySelectedRegionId(regionId);
    onSelectedRegionIdChange?.(regionId);
  };

  return (
    <Card>
      <CardContent className="space-y-6">
        <div className="grid gap-6 print:block xl:grid-cols-[minmax(0,1.65fr)_minmax(18rem,0.85fr)] xl:items-start">
          <div className="print:hidden">
            <TemporalVisualWorkspace
              predictionId={predictionId}
              artifacts={temporal.artifacts}
              highAttentionRegions={highAttentionRegions}
              durationSeconds={totalDuration}
              attentionThreshold={
                temporal.visualization_attention_threshold ?? temporal.attention_threshold
              }
              audioAvailable={audioAvailable}
            />
          </div>

          <section className="print-keep-together rounded-lg border bg-muted/30 p-4 print:mt-0 xl:mt-24" aria-labelledby="attention-details-heading">
            <h4 id="attention-details-heading" className="text-sm font-semibold">
              Attention details
            </h4>
            <dl className="mt-4 grid gap-x-5 gap-y-4 sm:grid-cols-2">
              <div>
                <dt className="text-xs text-muted-foreground">Status</dt>
                <dd className="mt-1 text-sm font-medium">{temporal.status}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Score kind</dt>
                <dd className="mt-1 text-sm font-medium">{temporal.score_kind}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Peak attention</dt>
                <dd className="mt-1 text-sm font-semibold tabular-nums text-sky-600 dark:text-sky-400">
                  {finiteNumber(peakAttention) ? peakAttention.toFixed(3) : <NotAvailable />}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">Attention threshold</dt>
                <dd className="mt-1 text-sm font-semibold tabular-nums text-amber-600 dark:text-amber-400">
                  {finiteNumber(temporal.attention_threshold) ? (
                    temporal.attention_threshold.toFixed(3)
                  ) : (
                    <NotAvailable />
                  )}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">High-attention regions</dt>
                <dd className="mt-1 text-sm font-medium">{highAttentionCount}</dd>
              </div>
              <div>
                <dt className="text-xs text-muted-foreground">High-attention duration</dt>
                <dd className="mt-1 text-sm font-medium tabular-nums">
                  {finiteNumber(temporal.high_attention_combined_duration_seconds) ? (
                    formatSeconds(temporal.high_attention_combined_duration_seconds)
                  ) : (
                    <NotAvailable />
                  )}
                </dd>
              </div>
            </dl>
          </section>
        </div>

        {malformed ? (
          <EvidenceUnavailablePanel
            title="Temporal evidence unavailable"
            description="The temporal component returned malformed region data, so the timeline cannot be rendered safely."
          />
        ) : missing && temporal.status === "completed" ? (
          <EvidenceUnavailablePanel
            title="Temporal evidence unavailable"
            description="The temporal component completed, but the region array was not included in the response."
          />
        ) : temporal.status !== "completed" ? (
          <EvidenceUnavailablePanel
            title="Analyzing temporal evidence..."
            description={`The temporal component is ${COMPONENT_STATUS_LABELS[
              temporal.status
            ].toLowerCase()}.`}
          />
        ) : highAttentionCount === 0 && highAttentionRegions.length === 0 ? (
          <EvidenceUnavailablePanel
            title="No high-attention regions"
            description={
              typeof temporal.threshold_crossing_count === "number" &&
              temporal.threshold_crossing_count > 0
                ? "Attention crossed the configured threshold, but all intervals were removed by post-processing filters."
                : "The temporal analysis completed without identifying any interval above the configured attention threshold."
            }
          />
        ) : (
          <>
            <section>
              <h4 className="text-sm font-semibold">High-attention regions</h4>
              <p className="mt-1 text-sm text-muted-foreground">
                Red regions show where the model focused, regardless of whether the local result is spoof or bonafide.
              </p>
              <div className="mt-3 grid gap-4 print:hidden xl:grid-cols-2 xl:items-start">
                <div
                  className="xl:self-start"
                  style={
                    selectedDetailHeight === null
                      ? undefined
                      : { height: `${selectedDetailHeight}px` }
                  }
                >
                  <HighAttentionRegionNavigator
                    regions={highAttentionRegions}
                    selectedRegionId={selectedHighAttentionRegion?.region_id}
                    onSelect={onSelectRegion}
                  />
                </div>
                {selectedHighAttentionRegion ? (
                  <div ref={selectedDetailRef} className="xl:sticky xl:top-4">
                    <SelectedRegionSemanticFeatures
                      region={selectedHighAttentionRegion}
                      semantic={semantic}
                      title="Semantic evidence in this high-attention region"
                    />
                  </div>
                ) : (
                  <NotAvailable label="Region detail unavailable" />
                )}
              </div>
              <HighAttentionRegionPrintDetails
                regions={highAttentionRegions}
                semantic={semantic}
              />
            </section>
          </>
        )}

        {warnings && componentErrors ? (
          <WarningsPanel warnings={warnings} componentErrors={componentErrors} runError={runError} />
        ) : null}

        {provenance && explanationId && requestId && createdAt ? (
          <ProvenancePanel
            provenance={provenance}
            explanationId={explanationId}
            requestId={requestId}
            createdAt={createdAt}
            completedAt={completedAt}
          />
        ) : null}
      </CardContent>
    </Card>
  );
}
