"use client";

import { AlertTriangle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type {
  TextSignalAnalysis,
  ConflictLevel,
  TextClassificationLabel,
} from "@/types/api";

const CONFLICT_STYLES: Record<
  ConflictLevel,
  { badge: string; dot: string; label: string }
> = {
  LOW: {
    badge: "bg-secondary text-secondary-foreground border-border",
    dot: "bg-slate-400",
    label: "Low Conflict",
  },
  MODERATE: {
    badge:
      "bg-amber-50 text-amber-800 border-amber-200 dark:bg-amber-950 dark:text-amber-300 dark:border-amber-800",
    dot: "bg-amber-500",
    label: "Moderate Conflict",
  },
  HIGH: {
    badge:
      "bg-rose-50 text-rose-700 border-rose-200 dark:bg-rose-950 dark:text-rose-300 dark:border-rose-800",
    dot: "bg-rose-500",
    label: "High Conflict",
  },
  "N/A": {
    badge: "bg-secondary text-secondary-foreground border-border",
    dot: "bg-slate-400",
    label: "N/A",
  },
};

interface SignalAnalysisPanelProps {
  analysis: TextSignalAnalysis;
  label: TextClassificationLabel;
}

type SignalDisplayConfig = {
  showAiSignals: boolean;
  showHumanSignals: boolean;
  aiSignalsLabel: string;
  humanSignalsLabel: string;
  message: string | null;
  messageStyle: "high" | "moderate" | null;
};

function getSignalDisplayConfig(
  conflictLevel: ConflictLevel,
  label: TextClassificationLabel
): SignalDisplayConfig {
  const DEFAULT_AI_LABEL = "AI-Typical Signals";
  const DEFAULT_HUMAN_LABEL = "Human-Style Signals";

  if (conflictLevel === "HIGH") {
    if (label === "AI-Generated") {
      return {
        showAiSignals: false,
        showHumanSignals: true,
        aiSignalsLabel: DEFAULT_AI_LABEL,
        humanSignalsLabel: "Conflicting Human-Style Signals",
        message: null,
        messageStyle: null,
      };
    }
    return {
      showAiSignals: true,
      showHumanSignals: false,
      aiSignalsLabel: "Conflicting AI-Typical Signals",
      humanSignalsLabel: DEFAULT_HUMAN_LABEL,
      message: null,
      messageStyle: null,
    };
  }

  if (conflictLevel === "MODERATE") {
    return {
      showAiSignals: true,
      showHumanSignals: true,
      aiSignalsLabel: DEFAULT_AI_LABEL,
      humanSignalsLabel: DEFAULT_HUMAN_LABEL,
      message:
        "Mixed signals detected — see the full analysis in the Evidence Explorer below.",
      messageStyle: "moderate",
    };
  }

  if (conflictLevel === "LOW") {
    const isAi = label === "AI-Generated";
    return {
      showAiSignals: isAi,
      showHumanSignals: !isAi,
      aiSignalsLabel: DEFAULT_AI_LABEL,
      humanSignalsLabel: DEFAULT_HUMAN_LABEL,
      message: null,
      messageStyle: null,
    };
  }

  // "N/A" — not one of the defined scenarios; preserve prior (show-both) behaviour.
  return {
    showAiSignals: true,
    showHumanSignals: true,
    aiSignalsLabel: DEFAULT_AI_LABEL,
    humanSignalsLabel: DEFAULT_HUMAN_LABEL,
    message: null,
    messageStyle: null,
  };
}

export function SignalAnalysisPanel({
  analysis,
  label,
}: SignalAnalysisPanelProps) {
  const showConflictAlert =
    analysis.conflict_detected &&
    analysis.xgboost_direction !== null &&
    analysis.deberta_direction !== analysis.xgboost_direction;

  const conflictStyle = CONFLICT_STYLES[analysis.conflict_level];
  const signalConfig = getSignalDisplayConfig(analysis.conflict_level, label);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="space-y-1">
          <CardTitle className="text-base font-semibold">
            Stylometric Signal Analysis
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-5 pt-0">
        {/* Model direction chips */}
        {/* <div className="flex flex-wrap items-center gap-2">
     
          <div className="flex items-center gap-1.5 rounded-md border bg-muted/40 px-3 py-1.5">
            <span className="text-xs text-muted-foreground font-medium">
              Semantic
            </span>
            <span className="text-xs text-muted-foreground">·</span>
            <Badge
              variant={
                analysis.deberta_direction === "AI" ? "destructive" : "success"
              }
              className="text-xs px-1.5 py-0"
            >
              {analysis.deberta_direction}
            </Badge>
          </div>

          {/* XGBoost */}
          {/* {analysis.xgboost_direction !== null && (
            <div className="flex items-center gap-1.5 rounded-md border bg-muted/40 px-3 py-1.5">
              <span className="text-xs text-muted-foreground font-medium">
                Stylometric
              </span>
              <span className="text-xs text-muted-foreground">·</span>
              <Badge
                variant={
                  analysis.xgboost_direction === "AI"
                    ? "destructive"
                    : "success"
                }
                className="text-xs px-1.5 py-0"
              >
                {analysis.xgboost_direction}
              </Badge>
            </div>
          )} */}

          {/* Conflict level */}
          {/* {analysis.css_conflict_score !== null && (
            <div
              className={cn(
                "flex items-center gap-1.5 rounded-md border px-3 py-1.5",
                conflictStyle.badge
              )}
            >
              <span
                className={cn(
                  "h-1.5 w-1.5 rounded-full flex-shrink-0",
                  conflictStyle.dot
                )}
                aria-hidden
              />
              <span className="text-xs font-medium">
                {conflictStyle.label}
              </span>
            </div>
          )}
        </div> */} 

        {/* Conflict alert */}
        {/* {showConflictAlert && (
          <Alert variant="warning" className="py-3">
            <AlertTriangle className="h-4 w-4" aria-hidden />
            <AlertTitle className="text-sm">Signal conflict detected</AlertTitle>
            <AlertDescription className="text-xs mt-1">
              DeBERTa and XGBoost disagree on the text&apos;s origin (conflict
              level:{" "}
              <span className="font-medium">{analysis.conflict_level}</span>).
              Treat this result with lower confidence.
            </AlertDescription>
          </Alert>
        )} */}

        {/* SHAP signal columns */}
        {((signalConfig.showAiSignals && analysis.shap_ai_signals.length > 0) ||
          (signalConfig.showHumanSignals &&
            analysis.shap_human_signals.length > 0)) && (
          <div className="space-y-3">
            {signalConfig.message && (
              <p
                className={
                  signalConfig.messageStyle === "high"
                    ? "text-sm text-muted-foreground italic"
                    : "text-xs text-muted-foreground"
                }
              >
                {signalConfig.message}
              </p>
            )}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              {signalConfig.showAiSignals &&
                analysis.shap_ai_signals.length > 0 && (
                  <div className="space-y-2.5">
                    <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                      {signalConfig.aiSignalsLabel}
                    </p>
                    <ul className="space-y-1.5">
                      {analysis.shap_ai_signals.map((signal, index) => (
                        <li
                          key={`ai-${index}-${signal}`}
                          className="flex items-center gap-2.5 rounded-md px-3 py-1.5 text-xs dark:border-rose-900/40 dark:bg-rose-950/20"
                        >
                          <span
                            className="h-1 w-1 flex-shrink-0 rounded-full bg-black/50"
                            aria-hidden
                          />
                          <span className="text-foreground">{signal}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

              {signalConfig.showHumanSignals &&
                analysis.shap_human_signals.length > 0 && (
                  <div className="space-y-2.5">
                    <p className="text-xs font-semibold uppercase tracking-widest text-muted-foreground">
                      {signalConfig.humanSignalsLabel}
                    </p>
                    <ul className="space-y-1.5">
                      {analysis.shap_human_signals.map((signal, index) => (
                        <li
                          key={`human-${index}-${signal}`}
                          className="flex items-center gap-2.5 rounded-md borde  px-3 py-1.5 text-xs dark:border-emerald-900/40 dark:bg-emerald-950/20"
                        >
                          <span
                            className="h-1 w-1 flex-shrink-0 rounded-full bg-black/50"
                            aria-hidden
                          />
                          <span className="text-foreground">{signal}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
            </div>
          </div>
        )}

        {/* Note */}
        {analysis.note !== null && (
          <p className="text-xs text-muted-foreground italic">{analysis.note}</p>
        )}

        {/* Footer metadata */}
        <div className="flex flex-wrap items-center gap-4 border-t pt-3 text-xs text-muted-foreground">
          <span>
            Word count:{" "}
            <span className="font-medium tabular-nums">{analysis.word_count}</span>
          </span>
          {analysis.css_conflict_score !== null && (
            <span>
              Conflict score:{" "}
              <span className="font-medium tabular-nums">
                {analysis.css_conflict_score.toFixed(3)}
              </span>
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
