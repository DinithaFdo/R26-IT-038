"use client";

import { useEffect, useState } from "react";
import { toast } from "sonner";
import {
  BarChart2,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  ClipboardPaste,
  FileText,
  Loader2,
  ScanText,
  Shield,
  ShieldAlert,
  ShieldCheck,
  X,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { ErrorState } from "@/components/shared/error-state";
import { ClassificationResultCard } from "@/components/text-classification/classification-result-card";
import { SignalAnalysisPanel } from "@/components/text-classification/signal-analysis-panel";
import {
  HeatmapTabContent,
  InterpretabilityTabContent,
} from "@/components/text-classification/xai-result-panel";
import { useAuditTextExplanation } from "./hooks";
import type {
  ClassificationView,
  TextClassificationResponse,
  TextXaiAuditResponse,
} from "@/types/api";

// ─── Constants ────────────────────────────────────────────────────────────────

const MIN_WORD_COUNT = 150;
const MAX_CHARS = 10_000;

const SAMPLE_TEXT =
  "The implementation of advanced machine learning techniques has demonstrated " +
  "significant improvements across multiple domains of natural language processing. " +
  "These methodologies leverage transformer-based architectures to capture long-range " +
  "dependencies in sequential data, enabling more accurate representations of linguistic " +
  "structures. Furthermore, the utilization of attention mechanisms allows the model to " +
  "weight the relative importance of different tokens dynamically, resulting in enhanced " +
  "contextual understanding. The empirical results indicate that these approaches consistently " +
  "outperform traditional baseline methods on standard benchmark datasets, achieving " +
  "state-of-the-art performance metrics across precision, recall, and F1 evaluation criteria. " +
  "Additionally, the scalability of these architectures facilitates deployment across " +
  "resource-constrained environments without substantial degradation in predictive accuracy.";

const ANALYSIS_PHASES = [
  {
    icon: "Shield",
    text: "Sanitizing input...",
  },
  {
    icon: "ScanText",
    text: "Analysing text...",
  },
  {
    icon: "BarChart2",
    text: "Extracting signals...",
  },
  {
    icon: "FileText",
    text: "Generating report...",
  }
] as const;

// ─── Helpers ──────────────────────────────────────────────────────────────────

function countWords(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

// ─── XAI → classification view adapter ─────────────────────────────────────
// /xai/audit returns predicted_class + a single confidence (0-100, complementary
// to the other class) instead of /classify's label + prob_ai/prob_human split.

function toClassificationView(
  xai: TextXaiAuditResponse
): ClassificationView {
  const isAI = xai.predicted_class === "AI-Generated";
  const primaryProb = xai.confidence / 100;

  return {
    label: isAI ? "AI-Generated" : "Human-Written",
    prob_ai: isAI ? primaryProb : 1 - primaryProb,
    prob_human: isAI ? 1 - primaryProb : primaryProb,
    sanitization: xai.sanitization,
    model_version: xai.model_version,
    processing_time_ms: xai.processing_time_ms,
    signal_analysis: xai.signal_analysis,
    final_label: xai.final_label,
    leans_toward: xai.leans_toward,
  };
}

// ─── Main page ────────────────────────────────────────────────────────────────

export function TextAnalyzePage() {
  const [text, setText] = useState("");
  const [xaiResult, setXaiResult] = useState<TextXaiAuditResponse | null>(null);
  const [isSanitizationExpanded, setIsSanitizationExpanded] = useState(false);
  const [analysisPhase, setAnalysisPhase] = useState(0);

  const xaiMutation = useAuditTextExplanation();

  useEffect(() => {
    if (!xaiMutation.isPending) {
      setAnalysisPhase(0);
      return;
    }

    const interval = setInterval(() => {
      setAnalysisPhase((prev) =>
        prev < ANALYSIS_PHASES.length - 1 ? prev + 1 : prev
      );
    }, 4000);

    return () => clearInterval(interval);
  }, [xaiMutation.isPending]);

  const wordCount = countWords(text);
  const charCount = text.length;
  const isTextReady = wordCount >= MIN_WORD_COUNT && charCount <= MAX_CHARS;
  const isLoading = xaiMutation.isPending;

  const classificationView = xaiResult ? toClassificationView(xaiResult) : null;

  async function handleAnalyze() {
    if (!isTextReady || isLoading) return;

    setXaiResult(null);
    xaiMutation.reset();

    try {
      const xaiRes = await xaiMutation.mutateAsync({ text, metadata: {} });
      setXaiResult(xaiRes);
    } catch {
      toast.error("Analysis failed. Please check your connection and try again.");
    }
  }

  function handleClear() {
    setText("");
    setXaiResult(null);
    xaiMutation.reset();
  }

  function handlePasteSample() {
    setText(SAMPLE_TEXT);
  }

  const showResults =
    xaiMutation.isPending || classificationView !== null || xaiMutation.isError;

  const showXai = xaiResult !== null;

  // Sanitization report details (attack_report is unknown[]; stringify defensively)
  const sanitizationIssues = classificationView
    ? classificationView.sanitization.attack_report.map((item) => String(item))
    : [];
  const sanitizationIssueCount =
    sanitizationIssues.length > 0 ? sanitizationIssues.length : 1;

  return (
    <div className="space-y-8">

      {/* ── Page header ──────────────────────────────────────────────────── */}
      <div className="space-y-1">
        <h1 className="text-2xl font-bold tracking-tight">Text Analysis</h1>
        <p className="text-sm text-muted-foreground">
          AI-generated text detection with semantic and stylometric evidence
        </p>
      </div>

      {/* ── Input card ───────────────────────────────────────────────────── */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
           
            <CardTitle className="text-base font-semibold">Analyse Text</CardTitle>
          </div>
          <p className="text-xs text-muted-foreground">
            Paste or type the text you want to analyse. A minimum of{" "}
            {MIN_WORD_COUNT} words is required for stylometric analysis.
          </p>
        </CardHeader>
        <CardContent className="space-y-4 pt-0">
          <div className="relative">
            <textarea
              id="text-input"
              placeholder="Paste your text here…"
              className="flex min-h-56 w-full resize-y rounded-lg border border-input bg-background px-4 py-3 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 leading-relaxed"
              value={text}
              onChange={(e) => setText(e.target.value)}
              disabled={isLoading}
              maxLength={MAX_CHARS}
              aria-label="Text to analyse"
            />

            {/* Scan overlay — only when pending */}
            {xaiMutation.isPending && (
              <div className="absolute inset-0 rounded-md overflow-hidden pointer-events-none">
                {/* Semi-transparent overlay */}
                <div className="absolute inset-0 bg-background/60" />

                {/* Animated scan line */}
                <div className="absolute inset-x-0 h-[3px] animate-scan bg-gradient-to-r from-transparent via-primary/50 to-transparent" />

                {/* Center label */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="flex flex-col items-center gap-2">
                    {/* Phase icon */}
                    {analysisPhase === 0 && (
                      <Shield className="h-5 w-5 text-muted-foreground animate-pulse" />
                    )}
                    {analysisPhase === 1 && (
                      <ScanText className="h-5 w-5 text-muted-foreground animate-pulse" />
                    )}
                    {analysisPhase === 2 && (
                      <BarChart2 className="h-5 w-5 text-muted-foreground animate-pulse" />
                    )}
                    {analysisPhase === 3 && (
                      <FileText className="h-5 w-5 text-muted-foreground animate-pulse" />
                    )}
                    {analysisPhase === 4 && (
                      <CheckCircle2 className="h-5 w-5 text-muted-foreground animate-pulse" />
                    )}

                    {/* Phase text */}
                    <span className="text-sm font-medium text-muted-foreground tracking-wide">
                      {ANALYSIS_PHASES[analysisPhase].text}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Counters + actions */}
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-3 text-xs">
              <span
                className={cn(
                  "tabular-nums",
                  wordCount >= MIN_WORD_COUNT
                    ? "text-emerald-600 dark:text-emerald-400 font-medium"
                    : "text-muted-foreground"
                )}
              >
                {wordCount} / {MIN_WORD_COUNT} words
              </span>
              <span className="text-muted-foreground tabular-nums">
                {charCount.toLocaleString()} / {MAX_CHARS.toLocaleString()} chars
              </span>
            </div>

            <div className="flex items-center gap-2">
              {text.length > 0 && !isLoading && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleClear}
                  className="h-8 gap-1.5 text-xs text-muted-foreground"
                  aria-label="Clear text"
                >
                 
                  Clear
                </Button>
              )}
              {text.length === 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handlePasteSample}
                  className="h-8 gap-1.5 text-xs"
                  aria-label="Paste sample text"
                >
                  <ClipboardPaste className="h-3.5 w-3.5" aria-hidden />
                  Paste sample
                </Button>
              )}
              <Button
                id="analyze-button"
                onClick={handleAnalyze}
                disabled={!isTextReady || isLoading}
                size="sm"
                className="h-8 gap-1.5"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden />
                    Analysing…
                  </>
                ) : (
                  <>
                    Analyse Text
                  </>
                )}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* ── Classification loading skeleton ──────────────────────────────── */}
      {xaiMutation.isPending && (
        <div className="space-y-4" aria-live="polite" aria-busy="true">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} className="h-20 w-full rounded-lg" />
            ))}
          </div>
          <Skeleton className="h-48 w-full rounded-lg" />
          <Skeleton className="h-48 w-full rounded-lg" />
        </div>
      )}

      {/* ── Classification error ─────────────────────────────────────────── */}
      {xaiMutation.isError && !xaiMutation.isPending && (
        <ErrorState
          error={xaiMutation.error}
          title="Classification failed"
        />
      )}

      {/* ── Results (visible only after classification succeeds) ─────────── */}
      {classificationView !== null && (
        <div className="space-y-6">

          {/* Input sanitization */}
          {/* Input sanitization */}
<section aria-label="Input sanitization">
  {classificationView.sanitization.was_attacked ? (
    <div className="rounded-lg border border-border overflow-hidden">
      <button
        type="button"
        onClick={() => setIsSanitizationExpanded((v) => !v)}
        className="flex w-full items-center gap-3 p-4 text-left bg-amber-50 dark:bg-amber-950/30"
        aria-expanded={isSanitizationExpanded}
      >
        <ShieldAlert
          className="h-5 w-5 flex-shrink-0 text-amber-600 dark:text-amber-500"
          aria-hidden
        />
        <span className="flex-1 text-sm font-medium text-amber-700 dark:text-amber-400">
          We found {sanitizationIssueCount} adversarial attack
          {sanitizationIssueCount === 1 ? "" : "s"} in your text and 
          detected and neutralised before analysis.
        </span>
        <span className="flex flex-shrink-0 items-center gap-1 text-xs font-medium text-amber-600 dark:text-amber-500">
          View report
          {isSanitizationExpanded ? (
            <ChevronUp className="h-3.5 w-3.5" aria-hidden />
          ) : (
            <ChevronDown className="h-3.5 w-3.5" aria-hidden />
          )}
        </span>
      </button>
      <div
        className={cn(
          "overflow-hidden transition-[max-height] duration-200 ease-out",
          isSanitizationExpanded ? "max-h-[600px]" : "max-h-0"
        )}
      >
        <div className="space-y-3 border-t border-border p-4">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium text-foreground">
              Sanitization Report
            </p>
            <span className="text-xs text-muted-foreground">
              {sanitizationIssueCount} issue{sanitizationIssueCount === 1 ? "" : "s"} resolved
            </span>
          </div>
          {sanitizationIssues.length > 0 && (
            <ul className="space-y-1.5">
              {sanitizationIssues.map((issue, i) => (
                <li
                  key={i}
                  className="flex items-start gap-2 text-xs text-muted-foreground"
                >
                  <span className="mt-0.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-muted-foreground/50" />
                  {issue}
                </li>
              ))}
            </ul>
          )}
          <div className="flex gap-6 border-t border-border pt-3 text-xs text-muted-foreground">
            <span>Original length: {text.length} chars</span>
            <span>
              Clean length:{" "}
              {classificationView.sanitization.clean_text.length} chars
            </span>
          </div>
        </div>
      </div>
    </div>
  ) : (
    <div className="flex items-center gap-3 rounded-lg border border-border bg-muted/30 p-4">
      <ShieldCheck
        className="h-4 w-4 flex-shrink-0 text-muted-foreground"
        aria-hidden
      />
      <span className="text-sm text-muted-foreground">
        No adversarial patterns detected
      </span>
    </div>
  )}
</section>

          {/* Classification result + Model probabilities */}
          <div className="flex flex-col gap-4 lg:flex-row lg:items-start">
            {/* Left: Classification Result — wider */}
            <div className="flex-[2] min-w-0">
              {/* ClassificationResultCard is typed against /classify's full response
                  shape but never reads token_highlights; asserting here avoids
                  fabricating that field just to satisfy the prop type. */}
              <ClassificationResultCard
                result={classificationView as TextClassificationResponse}
                conflictMessage={
                  classificationView.signal_analysis?.conflict_level === "HIGH"
                    ? classificationView.label === "AI-Generated"
                      ? "Most of this text looks AI-generated, but we also found some human-like writing patterns. It may have been edited or partly written by a person."
                      : "This text contains some AI-typical signals, but overall patterns are consistent with human-written text."
                    : undefined
                }
                conflictLevel={
                  classificationView.signal_analysis?.conflict_level ?? undefined
                }
                showProbabilities={false}
              />
            </div>

            {/* Right: Model Probabilities — narrower */}
            <div className="flex-[1] min-w-0 rounded-lg border border-border p-4 space-y-3">
              <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                Model Probabilities
              </p>

              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span>AI-Generated</span>
                  <span className="font-medium">
                    {(classificationView.prob_ai * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                  <div
                    className="h-full rounded-full bg-destructive transition-all duration-500"
                    style={{
                      width: `${(classificationView.prob_ai * 100).toFixed(1)}%`,
                    }}
                  />
                </div>
              </div>

              <div className="space-y-1">
                <div className="flex justify-between text-sm">
                  <span>Human-Written</span>
                  <span className="font-medium">
                    {(classificationView.prob_human * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                  <div
                    className="h-full rounded-full bg-emerald-500 transition-all duration-500"
                    style={{
                      width: `${(classificationView.prob_human * 100).toFixed(1)}%`,
                    }}
                  />
                </div>
              </div>

              {/* <div className="flex items-center gap-3 border-t border-border pt-3 text-xs text-muted-foreground">
                <span>{classificationView.model_version}</span>
                <span>{classificationView.processing_time_ms.toFixed(0)} ms</span>
              </div> */}
            </div>
          </div>

          {/* Stylometric signal analysis */}
          {classificationView.signal_analysis !== null &&
            classificationView.signal_analysis !== undefined && (
              <section aria-label="Stylometric signal analysis">
                <SignalAnalysisPanel
                  analysis={classificationView.signal_analysis}
                  label={classificationView.label}
                />
              </section>
            )}
        </div>
      )}

      {/* ── Evidence Explorer (tabbed, XAI results) ──────────────────────── */}
      {showXai && (
        <section aria-label="Evidence explorer">
          <Card>
            <CardHeader className="pb-3">
              <div className="space-y-0.5">
                <CardTitle className="text-base font-semibold">
                  Evidence Explorer
                </CardTitle>
                <p className="text-xs text-muted-foreground">
                  Deep-dive into the model's explainability outputs
                </p>
              </div>
            </CardHeader>
            <CardContent className="pt-0">

              {/* XAI loading skeleton */}
              {xaiMutation.isPending && (
                <div className="space-y-3" aria-live="polite" aria-busy="true">
                  <Skeleton className="h-8 w-64 rounded-md" />
                  <Skeleton className="h-40 w-full rounded-lg" />
                  <Skeleton className="h-24 w-full rounded-lg" />
                </div>
              )}

              {/* XAI error */}
              {xaiMutation.isError && !xaiMutation.isPending && (
                <ErrorState
                  error={xaiMutation.error}
                  title="Explainability analysis failed"
                />
              )}

              {/* XAI tabs */}
              {xaiResult !== null && (
                <Tabs defaultValue="heatmap">
                  <TabsList className="mb-4">
                    <TabsTrigger value="heatmap">Attention Heatmap</TabsTrigger>
                    <TabsTrigger value="report">Interpretability Report</TabsTrigger>
                  </TabsList>

                  {/* Heatmap tab */}
                  <TabsContent value="heatmap">
                    <HeatmapTabContent result={xaiResult} />
                  </TabsContent>

                  {/* Interpretability report tab */}
                  <TabsContent value="report">
                    <InterpretabilityTabContent result={xaiResult} />
                  </TabsContent>

                </Tabs>
              )}
            </CardContent>
          </Card>
        </section>
      )}
    </div>
  );
}
