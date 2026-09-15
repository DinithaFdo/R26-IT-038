"use client";

import { cn } from "@/lib/utils";
import type { TokenHighlight, TokenHighlightLevel } from "@/types/api";

const TOKEN_LEVEL_CLASSES: Record<TokenHighlightLevel, string> = {
  high: "bg-rose-50 border-rose-200 text-rose-900 dark:bg-rose-950/40 dark:border-rose-800/60 dark:text-rose-200",
  mid: "bg-amber-50 border-amber-200 text-amber-900 dark:bg-amber-950/40 dark:border-amber-800/60 dark:text-amber-200",
  low: "bg-muted border-transparent text-muted-foreground",
};

interface TokenHighlightViewProps {
  tokens: TokenHighlight[];
}

export function TokenHighlightView({ tokens }: TokenHighlightViewProps) {
  const highTokens = tokens.filter((t) => t.level === "high");
  const midTokens = tokens.filter((t) => t.level === "mid");
  const lowTokens = tokens.filter((t) => t.level === "low");

  const groups: { label: string; items: TokenHighlight[]; level: TokenHighlightLevel }[] = (
    [
      { label: "High attention", items: highTokens, level: "high" as TokenHighlightLevel },
      { label: "Medium attention", items: midTokens, level: "mid" as TokenHighlightLevel },
      { label: "Low attention", items: lowTokens, level: "low" as TokenHighlightLevel },
    ] as { label: string; items: TokenHighlight[]; level: TokenHighlightLevel }[]
  ).filter((g) => g.items.length > 0);

  if (tokens.length === 0) {
    return (
      <p className="text-sm text-muted-foreground">No token highlights available.</p>
    );
  }

  return (
    <div className="space-y-3" role="list" aria-label="Token highlights by attention level">
      {groups.map((group) => (
        <div key={group.level} className="space-y-1.5">
          <p className="text-xs font-medium text-muted-foreground">{group.label}</p>
          <div className="flex flex-wrap gap-1.5">
            {group.items.map((t, i) => (
              <span
                key={i}
                role="listitem"
                title={`Score: ${(t.score * 100).toFixed(0)}%`}
                className={cn(
                  "rounded-md px-2 py-0.5 text-sm border cursor-default font-medium transition-colors",
                  TOKEN_LEVEL_CLASSES[t.level]
                )}
              >
                {t.word}
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
