"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Mic, Pause, Play, Square, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { env } from "@/lib/config/env";
import { formatDuration } from "@/lib/formatters";
import { selectRecordingMimeType } from "@/features/audio-recorder/mime";

export function AudioRecorder({
  onRecording,
}: {
  onRecording: (file: File | null) => void;
}) {
  const recorderRef = useRef<MediaRecorder | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const intervalRef = useRef<number | null>(null);
  const [status, setStatus] = useState<"idle" | "recording" | "paused" | "stopped">("idle");
  const [seconds, setSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [supportsPause, setSupportsPause] = useState(false);

  const stopTimer = () => {
    if (intervalRef.current) window.clearInterval(intervalRef.current);
    intervalRef.current = null;
  };

  const stopTracks = useCallback(() => {
    stopTimer();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }, []);

  useEffect(() => () => stopTracks(), [stopTracks]);

  const start = async () => {
    setError(null);
    if (!env.enableAudioRecording) {
      setError("Browser recording is disabled by frontend configuration.");
      return;
    }
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setError("This browser does not expose microphone recording APIs.");
      return;
    }
    if (typeof MediaRecorder === "undefined") {
      setError("MediaRecorder is not supported in this browser.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = selectRecordingMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      chunksRef.current = [];
      streamRef.current = stream;
      recorderRef.current = recorder;
      setSupportsPause(typeof recorder.pause === "function" && typeof recorder.resume === "function");
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        stopTimer();
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || "audio/webm" });
        if (blob.size === 0) {
          setError("Recording stopped without audio data.");
          onRecording(null);
          return;
        }
        const extension = recorder.mimeType.includes("ogg") ? "ogg" : recorder.mimeType.includes("mp4") ? "m4a" : "webm";
        onRecording(new File([blob], `browser-recording-${Date.now()}.${extension}`, { type: blob.type }));
        stopTracks();
        setStatus("stopped");
      };
      recorder.onerror = () => {
        setError("Recording stopped unexpectedly.");
        stopTracks();
      };
      recorder.start();
      setSeconds(0);
      setStatus("recording");
      intervalRef.current = window.setInterval(() => {
        setSeconds((current) => {
          const next = current + 1;
          if (next >= env.maxAudioDurationSeconds) recorder.stop();
          return next;
        });
      }, 1000);
    } catch (err) {
      setError(err instanceof DOMException && err.name === "NotAllowedError" ? "Microphone permission was denied." : "Microphone recording could not be started.");
      stopTracks();
    }
  };

  const pause = () => {
    recorderRef.current?.pause();
    stopTimer();
    setStatus("paused");
  };
  const resume = () => {
    recorderRef.current?.resume();
    intervalRef.current = window.setInterval(() => setSeconds((current) => current + 1), 1000);
    setStatus("recording");
  };
  const stop = () => recorderRef.current?.state !== "inactive" && recorderRef.current?.stop();
  const discard = () => {
    if (recorderRef.current?.state === "recording" || recorderRef.current?.state === "paused") recorderRef.current.stop();
    chunksRef.current = [];
    onRecording(null);
    stopTracks();
    setStatus("idle");
    setSeconds(0);
  };

  return (
    <div className="rounded-lg border bg-card p-5">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="font-semibold">Browser recording</h2>
          <p className="mt-1 text-sm text-muted-foreground">Record, stop, preview, then submit. This is not live-stream detection.</p>
        </div>
        <div className="text-2xl font-semibold tabular-nums">{formatDuration(seconds)}</div>
      </div>
      <div className="mt-5 flex flex-wrap gap-2">
        {status === "idle" || status === "stopped" ? <Button type="button" onClick={start}><Mic className="h-4 w-4" /> Start</Button> : null}
        {status === "recording" && supportsPause ? <Button type="button" variant="outline" onClick={pause}><Pause className="h-4 w-4" /> Pause</Button> : null}
        {status === "paused" ? <Button type="button" variant="outline" onClick={resume}><Play className="h-4 w-4" /> Resume</Button> : null}
        {status === "recording" || status === "paused" ? <Button type="button" variant="secondary" onClick={stop}><Square className="h-4 w-4" /> Stop</Button> : null}
        <Button type="button" variant="outline" onClick={discard}><Trash2 className="h-4 w-4" /> Discard</Button>
      </div>
      {error ? <Alert variant="destructive" className="mt-4"><AlertDescription>{error}</AlertDescription></Alert> : null}
    </div>
  );
}
