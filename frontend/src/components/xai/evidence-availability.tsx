import { CircleSlash } from "lucide-react";
import { cn } from "@/lib/utils";

/**
 * Explicit "this value does not exist" rendering.
 *
 * Used everywhere a backend field is optional. Rendering `0`, a bare dash, or
 * an inferred default in place of a missing forensic/evaluation value would
 * misrepresent the state of the pipeline, so the absence is always stated in
 * words along with the reason when the backend supplies one.
 */
export function NotAvailable({
  reason,
  className,
  label = "Not available",
}: {
  reason?: string | null;
  className?: string;
  label?: string;
}) {
  return (
    <span
      className={cn("inline-flex items-center gap-1.5 text-sm text-muted-foreground", className)}
    >
      <CircleSlash className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
      <span>{reason ? `${label} — ${reason}` : label}</span>
    </span>
  );
}

/** Empty-state block for a whole XAI component that produced no evidence. */
export function EvidenceUnavailablePanel({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="rounded-lg border border-dashed bg-card p-8 text-center">
      <CircleSlash className="mx-auto h-8 w-8 text-muted-foreground" aria-hidden="true" />
      <h3 className="mt-4 text-base font-semibold">{title}</h3>
      <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">{description}</p>
    </div>
  );
}
