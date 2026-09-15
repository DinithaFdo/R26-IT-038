import { CheckCircle2, CircleSlash, FlaskConical, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import type { BranchState } from "@/lib/models/branch-state";

/**
 * Status is carried by an icon and a word, never by colour alone (WCAG 1.4.1).
 *
 * `real` and `real_unverified` share the same non-alarming presentation: a
 * normal reader must not see a warning triangle on a model that is actually
 * running correctly. The scientific-verification distinction between the two
 * lives only in `BranchStateDescriptor.technicalNote`, for an explicit
 * "technical details" disclosure, per describeBranchState.
 */
const BRANCH_STATE_PRESENTATION: Record<
  BranchState,
  { variant: "success" | "warning" | "destructive" | "secondary"; Icon: typeof CheckCircle2 }
> = {
  real: { variant: "success", Icon: CheckCircle2 },
  real_unverified: { variant: "success", Icon: CheckCircle2 },
  dummy: { variant: "warning", Icon: FlaskConical },
  disabled: { variant: "secondary", Icon: CircleSlash },
  failed: { variant: "destructive", Icon: XCircle },
};

export function BranchStateBadge({ state, label }: { state: BranchState; label: string }) {
  const { variant, Icon } = BRANCH_STATE_PRESENTATION[state];
  return (
    <Badge variant={variant} className="gap-1.5">
      <Icon className="h-3.5 w-3.5" aria-hidden="true" />
      {label}
    </Badge>
  );
}
