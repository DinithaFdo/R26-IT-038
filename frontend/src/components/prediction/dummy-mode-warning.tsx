import { AlertTriangle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";

export function DummyModeWarning({ show, reason }: { show: boolean; reason?: string | null }) {
  if (!show) return null;
  return (
    <Alert variant="warning">
      <AlertTriangle className="mb-2 h-4 w-4" aria-hidden="true" />
      <AlertTitle>Development-mode output</AlertTitle>
      <AlertDescription>
        This result includes development-mode model outputs and is not eligible for research evaluation.
        {reason ? <span className="mt-1 block">{reason}</span> : null}
      </AlertDescription>
    </Alert>
  );
}
