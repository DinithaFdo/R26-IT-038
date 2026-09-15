"use client";

import { useCallback, useRef, useState } from "react";
import { UploadCloud, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { FILE_ACCEPT } from "@/lib/constants/audio";
import { validateAudioFile, validateKnownDuration } from "@/lib/validation/audio";
import { cn } from "@/lib/utils";

export function AudioDropzone({
  file,
  onFile,
  onDuration,
  error,
}: {
  file: File | null;
  onFile: (file: File | null) => void;
  onDuration: (duration: number | null) => void;
  error?: string | null;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const acceptFile = useCallback((nextFile: File | null) => {
    if (!nextFile) return;
    onFile(nextFile);
    onDuration(null);
    const audio = document.createElement("audio");
    const url = URL.createObjectURL(nextFile);
    audio.preload = "metadata";
    audio.onloadedmetadata = () => {
      onDuration(Number.isFinite(audio.duration) ? audio.duration : null);
      URL.revokeObjectURL(url);
    };
    audio.onerror = () => URL.revokeObjectURL(url);
    audio.src = url;
  }, [onDuration, onFile]);

  const validation = file ? validateAudioFile(file) : { valid: true };
  const shownError = error || (!validation.valid ? validation.message : null);

  return (
    <div className="space-y-3">
      <div
        className={cn(
          "flex min-h-64 flex-col items-center justify-center rounded-lg border border-dashed bg-card p-6 text-center transition-colors",
          dragging && "border-primary bg-accent/30",
          shownError && "border-destructive",
        )}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          acceptFile(event.dataTransfer.files?.[0] || null);
        }}
      >
        <UploadCloud className="h-10 w-10 text-primary" aria-hidden="true" />
        <h2 className="mt-4 text-base font-semibold">Drop audio here</h2>
        <p className="mt-2 max-w-md text-sm text-muted-foreground">
          WAV, FLAC, MP3, M4A, AAC, Opus, OGG, and audio-only WebM are accepted by the backend.
        </p>
        <div className="mt-5 flex flex-wrap justify-center gap-2">
          <Button type="button" onClick={() => inputRef.current?.click()}>Choose file</Button>
          {file ? (
            <Button type="button" variant="outline" onClick={() => { onFile(null); onDuration(null); }}>
              <X className="h-4 w-4" /> Remove
            </Button>
          ) : null}
        </div>
        <input
          ref={inputRef}
          type="file"
          accept={FILE_ACCEPT}
          className="sr-only"
          onChange={(event) => acceptFile(event.target.files?.[0] || null)}
          aria-label="Choose audio file"
        />
      </div>
      {shownError ? <p className="text-sm text-destructive">{shownError}</p> : null}
    </div>
  );
}

export function durationValidationMessage(duration: number | null) {
  const validation = validateKnownDuration(duration);
  return validation.valid ? null : validation.message || null;
}
