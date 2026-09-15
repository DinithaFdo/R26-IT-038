import { getDetectorDisplayName } from "@/lib/copy/detector-names";
import { formatWeightPercent } from "@/lib/formatters";

/**
 * Horizontal contribution visualization for the fusion weights Fusion V3 was
 * trained with. Deliberately separate from any branch's own spoof
 * probability -- see the caption below the bars -- a weight is how much a
 * branch counts toward the fused score, not how confident that branch was.
 */
export function FusionWeightBars({ weights }: { weights: Record<string, number> }) {
  const entries = Object.entries(weights).sort(([, a], [, b]) => b - a);
  if (!entries.length) {
    return <p className="text-sm text-muted-foreground">No fusion weights were reported.</p>;
  }
  return (
    <div className="space-y-3">
      {entries.map(([branch, weight]) => (
        <div key={branch} className="space-y-1">
          <div className="flex items-center justify-between text-sm">
            <span>{getDetectorDisplayName(branch)}</span>
            <span className="font-medium tabular-nums">{formatWeightPercent(weight)}</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-muted" role="presentation">
            <div
              className="h-full rounded-full bg-primary"
              style={{ width: `${Math.max(0, Math.min(1, weight)) * 100}%` }}
            />
          </div>
        </div>
      ))}
      <p className="text-xs text-muted-foreground">
        These are fixed fusion weights, not branch confidence -- how much each detector counts
        toward the combined score, not how sure that detector was.
      </p>
    </div>
  );
}
