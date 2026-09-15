import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  CircleDashed,
  CircleSlash,
  Loader2,
  XCircle,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  COMPONENT_STATUS_LABELS,
  COMPONENT_STATUS_TONES,
  EXPLANATION_STATUS_LABELS,
  EXPLANATION_STATUS_TONES,
  type StatusTone,
} from "@/lib/xai/status";
import type { ComponentStatus, ExplanationStatus } from "@/types/api";

/**
 * Status is conveyed by icon + text, never by colour alone (WCAG 1.4.1).
 */
const TONE_VARIANT: Record<
  StatusTone,
  "secondary" | "info" | "success" | "warning" | "destructive"
> = {
  neutral: "secondary",
  progress: "info",
  positive: "success",
  warning: "warning",
  critical: "destructive",
};

const TONE_ICON: Record<StatusTone, LucideIcon> = {
  neutral: CircleDashed,
  progress: Loader2,
  positive: CheckCircle2,
  warning: AlertTriangle,
  critical: XCircle,
};

export function ExplanationStatusBadge({ status }: { status: ExplanationStatus }) {
  const tone = EXPLANATION_STATUS_TONES[status];
  const Icon = TONE_ICON[tone];
  return (
    <Badge variant={TONE_VARIANT[tone]} className="gap-1.5">
      <Icon
        className={tone === "progress" ? "h-3.5 w-3.5 animate-spin" : "h-3.5 w-3.5"}
        aria-hidden="true"
      />
      {EXPLANATION_STATUS_LABELS[status]}
    </Badge>
  );
}

export function ComponentStatusBadge({ status }: { status: ComponentStatus }) {
  const tone = COMPONENT_STATUS_TONES[status];
  const Icon =
    status === "not_available" ? CircleSlash : status === "blocked" ? Ban : TONE_ICON[tone];
  return (
    <Badge variant={TONE_VARIANT[tone]} className="gap-1.5">
      <Icon
        className={tone === "progress" ? "h-3.5 w-3.5 animate-spin" : "h-3.5 w-3.5"}
        aria-hidden="true"
      />
      {COMPONENT_STATUS_LABELS[status]}
    </Badge>
  );
}
