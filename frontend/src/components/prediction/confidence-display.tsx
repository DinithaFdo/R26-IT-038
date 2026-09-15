import { ShieldCheck } from "lucide-react";
import { formatProbability } from "@/lib/formatters";

export function ConfidenceDisplay({ value }: { value?: number | null }) {
  return (
    <div className="flex items-center gap-3 rounded-lg border bg-muted/40 p-4">
      <ShieldCheck className="h-5 w-5 text-primary" aria-hidden="true" />
      <div>
        <p className="text-xs text-muted-foreground">Confidence</p>
        <p className="text-xl font-semibold">{formatProbability(value)}</p>
      </div>
    </div>
  );
}
