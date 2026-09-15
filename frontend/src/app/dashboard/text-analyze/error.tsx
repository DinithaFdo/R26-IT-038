"use client";
import { useEffect } from "react";

export default function TextAnalyzeError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Text analyze page error:", error);
  }, [error]);

  return (
    <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
      <h2 className="text-lg font-semibold text-destructive">
        Something went wrong
      </h2>
      <p className="text-sm text-muted-foreground">
        The text analysis page encountered an unexpected error.
      </p>
      <button
        onClick={reset}
        className="px-4 py-2 text-sm rounded-md bg-primary text-primary-foreground hover:bg-primary/90"
      >
        Try again
      </button>
    </div>
  );
}
