"use client";

import { useEffect, useState } from "react";
import { Loader2 } from "lucide-react";

/**
 * Indeterminate loading state for the real backend analysis call, which can
 * legitimately take from several seconds up to a few minutes (Glottal alone
 * measures ~7s on CPU even when nothing fails; SSL can add significant
 * cold-start latency; the backend runs all four branches sequentially, not
 * in parallel).
 *
 * Deliberately no percentage: the backend does not stream per-branch
 * progress to this request, so a percentage here would have to be
 * fabricated. These messages cycle on a timer purely to communicate "still
 * working" -- they do not assert that a specific branch has actually
 * finished.
 */

const STAGE_MESSAGES = [
  "Preparing audio…",
  "Running detection models…",
  "Analyzing voice-source characteristics…",
  "Combining model evidence…",
] as const;

const STAGE_INTERVAL_MS = 4000;

export function AnalysisWaitState() {
  const [messageIndex, setMessageIndex] = useState(0);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setMessageIndex((index) => (index + 1) % STAGE_MESSAGES.length);
    }, STAGE_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <div className="rounded-lg border bg-card p-4" aria-live="polite">
      <div className="flex items-center gap-3">
        <Loader2 className="h-4 w-4 shrink-0 animate-spin text-primary" aria-hidden="true" />
        <span className="text-sm">{STAGE_MESSAGES[messageIndex]}</span>
      </div>
      <p className="mt-2 text-xs text-muted-foreground">
        The full four-branch analysis can take up to a couple of minutes, especially on a cold
        start. This page will move on automatically when the result is ready.
      </p>
    </div>
  );
}
