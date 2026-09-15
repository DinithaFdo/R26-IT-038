import { AlertTriangle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { getUserFriendlyErrorMessage, normalizeApiError } from "@/lib/api/errors";

export function ErrorState({ error, title = "Something went wrong" }: { error: unknown; title?: string }) {
  const normalized = normalizeApiError(error);
  return (
    <Alert variant="destructive">
      <AlertTriangle className="mb-2 h-4 w-4" aria-hidden="true" />
      <AlertTitle>{title}</AlertTitle>
      <AlertDescription>
        {getUserFriendlyErrorMessage(error)}
        {normalized.retryAfterSeconds !== undefined ? (
          <span className="mt-1 block text-xs opacity-80">
            Retry in {normalized.retryAfterSeconds}s.
          </span>
        ) : null}
        {normalized.requestId ? (
          <span className="mt-1 block text-xs opacity-80">Request ID: {normalized.requestId}</span>
        ) : null}
      </AlertDescription>
    </Alert>
  );
}
