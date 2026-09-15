import { Progress } from "@/components/ui/progress";
import { formatProbability } from "@/lib/formatters";
import type { ProbabilityScores } from "@/types/api";

export function ProbabilityBar({ probabilities }: { probabilities?: ProbabilityScores | null }) {
  if (!probabilities) {
    return <p className="text-sm text-muted-foreground">Branch score unavailable.</p>;
  }
  const spoof = probabilities.spoof * 100;
  const bonafide = probabilities.bonafide * 100;
  return (
    <div className="space-y-3" aria-label={`Spoof ${formatProbability(probabilities.spoof)}, bonafide ${formatProbability(probabilities.bonafide)}`}>
      <div>
        <div className="mb-1 flex justify-between text-xs">
          <span>Spoof</span>
          <span>{formatProbability(probabilities.spoof)}</span>
        </div>
        <Progress value={spoof} className="[&>div]:bg-destructive" />
      </div>
      <div>
        <div className="mb-1 flex justify-between text-xs">
          <span>Bonafide</span>
          <span>{formatProbability(probabilities.bonafide)}</span>
        </div>
        <Progress value={bonafide} className="[&>div]:bg-emerald-500" />
      </div>
    </div>
  );
}
