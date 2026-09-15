import { CheckCircle2, Loader2 } from "lucide-react";
import { StatusBadge } from "@/components/prediction/status-badge";
import type { PredictionStatus } from "@/types/api";

const stages = ["queued", "validating", "storing", "processing", "completed"] as const;

export function ProcessingTimeline({ status }: { status?: PredictionStatus | null }) {
  const activeIndex = stages.findIndex((stage) => stage === status);
  return (
    <div className="rounded-lg border bg-card p-4" aria-live="polite">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h2 className="font-semibold">Processing status</h2>
        <StatusBadge status={status} />
      </div>
      <div className="grid gap-3 sm:grid-cols-5">
        {stages.map((stage, index) => {
          const complete = activeIndex > index || status === "completed";
          const active = activeIndex === index;
          return (
            <div key={stage} className="rounded-md border p-3 text-sm">
              {complete ? <CheckCircle2 className="h-4 w-4 text-emerald-400" /> : active ? <Loader2 className="h-4 w-4 animate-spin text-primary" /> : <div className="h-4 w-4 rounded-full border" />}
              <p className="mt-2 capitalize">{stage}</p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
