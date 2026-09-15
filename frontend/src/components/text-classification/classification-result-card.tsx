"use client";

import { AlertTriangle, Clock, Cpu, TrendingUp } from "lucide-react";
import {
  Card,
  CardContent,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import type {
  TextClassificationResponse,
  ConflictLevel,
  TextClassificationLabel,
} from "@/types/api";

interface ClassificationResultCardProps {
  result: TextClassificationResponse;
  conflictMessage?: string;
  conflictLevel?: ConflictLevel;
  showProbabilities?: boolean;
}

function getSubtitle(
  label: TextClassificationLabel,
  conflictLevel?: ConflictLevel
): string | null {
  const isAI = label === "AI-Generated";

  if (conflictLevel === "HIGH") return null;

  if (conflictLevel === "MODERATE") {
    return isAI
      ? "This text is mostly AI-generated, but contains a few sections that feel more naturally written."
      : "This text shows mixed signals, but leans toward human-written content.";
  }

  // LOW, 
  return isAI
    ? "This text was likely written by AI"
    : "This text was likely written by a human";
}

export function ClassificationResultCard({
  result,
  conflictMessage,
  conflictLevel,
  showProbabilities = true,
}: ClassificationResultCardProps) {
  const isAI = result.label === "AI-Generated";
  const isMixed = result.final_label === "Mixed";
  const finalIsAI = result.final_label === "AI-Generated";
  const primaryScore = isAI ? result.prob_ai : result.prob_human;
  const primaryPct = (primaryScore * 100).toFixed(1);
  const subtitle = getSubtitle(result.label, conflictLevel);

  return (
    <div className="space-y-3">
      {/* Sanitization warning */}
      {/* {result.sanitization.was_attacked && (
        <div
          role="note"
          className="flex items-start gap-3 rounded-lg  px-4 py-3 dark:border-amber-800/60 dark:bg-amber-950/40"
        >
         
          <div className="min-w-0 space-y-1">
            <p className="text-sm font-medium text-amber-800 dark:text-amber-200">
              Adversarial input detected and neutralised
            </p>
            {result.sanitization.attack_report.length > 0 && (
              <details className="mt-1">
                <summary className="cursor-pointer text-xs text-amber-700 hover:underline dark:text-amber-300">
                  Show sanitization details
                </summary>
                <ul className="mt-2 space-y-1 pl-1">
                  {result.sanitization.attack_report.map((item, i) => (
                    <li
                      key={i}
                      className="text-xs text-amber-700 dark:text-amber-300"
                    >
                      {String(item)}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        </div>
      )} */}

      {/* Main result card */}
      <Card className="overflow-hidden">
        {/* Verdict strip */}
        <div
          className={cn(
            "px-6 py-5 border-2 bg-background",
            isMixed
              ? "border-amber-500"
              : finalIsAI
                ? "border-destructive"
                : "border-emerald-600"
          )}
        >
          <div className="flex items-start justify-between gap-4">
            <div className="space-y-1">
              <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
                Classification Result
              </p>
              <h3
                className={cn(
                  "text-2xl font-bold tracking-tight",
                  isMixed
                    ? "text-amber-700 dark:text-amber-400"
                    : finalIsAI
                      ? "text-rose-700 dark:text-rose-300"
                      : "text-emerald-700 dark:text-emerald-300"
                )}
              >
                {isMixed
                  ? "Mixed Content Detected"
                  : finalIsAI
                    ? "Likely AI-Generated"
                    : "Likely Human-Written"}
              </h3>
              {subtitle && (
                <p className="text-sm text-muted-foreground">{subtitle}</p>
              )}
              {isMixed && (
                <p className={cn(
                  "mt-1 text-sm italic",
                  "text-amber-700 dark:text-amber-400"
                )}>
                  This text appears to be written by
                  both a human and an AI tool. Parts
                  of it look like natural human writing,
                  while other parts follow AI patterns.
                  
                </p>
              )}
               {!isMixed && conflictMessage && (
                <p className="mt-1 text-sm text-muted-foreground italic">
                  {conflictMessage}
                </p>
              )}
              {isMixed && result.leans_toward && (
                <p className={cn(
                  "mt-1 text-sm font-medium",
                  "text-amber-600 dark:text-amber-400"
                )}>
                  Leans {result.leans_toward}
                </p>
              )}
            </div>
            <div className="flex flex-col items-end gap-2 flex-shrink-0">
              
              <div className="flex items-baseline gap-1">
                <span
                  className={cn(
                    "text-3xl font-bold tabular-nums",
                    isMixed
                      ? "text-amber-700 dark:text-amber-400"
                      : finalIsAI
                        ? "text-rose-700 dark:text-rose-300"
                        : "text-emerald-700 dark:text-emerald-300"
                  )}
                >
                  {primaryPct}%
                </span>
                <span className="text-xs text-muted-foreground">confidence</span>
              </div>
            </div>
          </div>
        </div>

        {/* Probability breakdown */}
        {showProbabilities && (
          <CardContent className="space-y-4 pt-5 pb-5">
            <div className="space-y-1">
              <p className="text-xs font-medium uppercase tracking-widest text-muted-foreground flex items-center gap-1.5">
                <TrendingUp className="h-3 w-3" aria-hidden />
                Model probabilities
              </p>
            </div>

            <div className="space-y-3">
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">AI-Generated</span>
                  <span className="font-semibold tabular-nums">
                    {(result.prob_ai * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress
                  value={result.prob_ai * 100}
                  className="h-1.5 [&>div]:bg-rose-500"
                  aria-label={`AI probability: ${(result.prob_ai * 100).toFixed(1)}%`}
                />
              </div>
              <div className="space-y-1.5">
                <div className="flex items-center justify-between text-sm">
                  <span className="text-muted-foreground">Human-Written</span>
                  <span className="font-semibold tabular-nums">
                    {(result.prob_human * 100).toFixed(1)}%
                  </span>
                </div>
                <Progress
                  value={result.prob_human * 100}
                  className="h-1.5 [&>div]:bg-emerald-500"
                  aria-label={`Human probability: ${(result.prob_human * 100).toFixed(1)}%`}
                />
              </div>
            </div>

            {/* Metadata footer */}
            {/* <div className="flex flex-wrap items-center gap-4 border-t pt-4 text-xs text-muted-foreground">
              <div className="flex items-center gap-1.5">
                <Cpu className="h-3.5 w-3.5" aria-hidden />
                <span>{result.model_version}</span>
              </div>
              <div className="flex items-center gap-1.5">
                <Clock className="h-3.5 w-3.5" aria-hidden />
                <span>{result.processing_time_ms.toFixed(0)} ms</span>
              </div>
            </div> */}
          </CardContent>
        )}
      </Card>
    </div>
  );
}
