import { Badge } from "@/components/ui/badge";
import { statusText } from "@/lib/formatters";
import type { BranchStatus, PredictionLabel, PredictionStatus } from "@/types/api";

export function StatusBadge({ status }: { status?: PredictionStatus | BranchStatus | null }) {
  const variant =
    status === "completed" || status === "success"
      ? "success"
      : status === "failed"
        ? "destructive"
        : status === "queued" || status === "processing" || status === "validating" || status === "storing"
          ? "info"
          : "secondary";
  return <Badge variant={variant}>{statusText(status)}</Badge>;
}

/** Plain-language verdict. Backend values (bonafide/spoof) are unchanged -- only the label shown here changes. */
export function LabelBadge({ label }: { label?: PredictionLabel | null }) {
  if (label === "spoof") return <Badge variant="destructive">Likely AI-generated</Badge>;
  if (label === "bonafide") return <Badge variant="success">Likely authentic</Badge>;
  return <Badge variant="secondary">No decision</Badge>;
}
