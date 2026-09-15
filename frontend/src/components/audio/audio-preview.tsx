"use client";

import { useEffect, useState } from "react";
import { formatBytes, formatDuration } from "@/lib/formatters";

export function AudioPreview({
  file,
  url,
  duration,
}: {
  file?: File | null;
  url?: string | null;
  duration?: number | null;
}) {
  if (!file && !url) return null;
  return (
    <div className="rounded-lg border bg-muted/30 p-4">
      {file ? (
        <div className="mb-3">
          <p className="break-all text-sm font-medium">{file.name}</p>
          <p className="text-xs text-muted-foreground">
            {file.type || "Unknown MIME"} · {formatBytes(file.size)} · {formatDuration(duration)}
          </p>
        </div>
      ) : null}
      {url ? <audio controls className="w-full" src={url}>Audio preview unavailable.</audio> : null}
    </div>
  );
}

export function useObjectUrl(file: Blob | null) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!file) {
      setUrl(null);
      return;
    }
    const nextUrl = URL.createObjectURL(file);
    setUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [file]);
  return url;
}
