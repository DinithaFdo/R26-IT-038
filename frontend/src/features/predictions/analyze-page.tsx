"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { toast } from "sonner";
import { z } from "zod";
import { PageHeader } from "@/components/shared/page-header";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AudioDropzone, durationValidationMessage } from "@/components/audio/audio-dropzone";
import { AudioPreview, useObjectUrl } from "@/components/audio/audio-preview";
import { AudioRecorder } from "@/components/audio/audio-recorder";
import { UploadProgress } from "@/components/audio/upload-progress";
import { AnalysisWaitState } from "@/components/audio/analysis-wait-state";
import { ProcessingTimeline } from "@/components/audio/processing-timeline";
import { useCreatePrediction, usePredictionStatus } from "@/features/predictions/hooks";
import { isTerminalStatus } from "@/lib/formatters";
import { predictionSubmitSchema, validateAudioFile } from "@/lib/validation/audio";
import { getUserFriendlyErrorMessage, normalizeApiError } from "@/lib/api/errors";

type FormValues = z.infer<typeof predictionSubmitSchema>;

export function AnalyzePage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [submittedPredictionId, setSubmittedPredictionId] = useState<string | null>(null);
  const previewUrl = useObjectUrl(file);
  const createPrediction = useCreatePrediction();
  const status = usePredictionStatus(submittedPredictionId || "", Boolean(submittedPredictionId));
  const form = useForm<FormValues>({
    resolver: zodResolver(predictionSubmitSchema),
    defaultValues: { sourceType: "dashboard_upload" },
  });

  const durationMessage = useMemo(() => durationValidationMessage(duration), [duration]);

  useEffect(() => {
    if (status.data && isTerminalStatus(status.data.status) && status.data.status === "completed") {
      router.replace(`/dashboard/predictions/${status.data.prediction_id}`);
    }
  }, [router, status.data]);

  const submit = async (values: FormValues) => {
    const fileValidation = validateAudioFile(file);
    if (!fileValidation.valid) {
      toast.error(fileValidation.message);
      return;
    }
    if (durationMessage) {
      toast.error(durationMessage);
      return;
    }
    if (!file) return;
    try {
      setUploadProgress(0);
      const response = await createPrediction.mutateAsync({
        file,
        sourceType: values.sourceType,
        clientFilename: values.clientFilename || file.name,
        idempotencyKey: crypto.randomUUID(),
        onUploadProgress: setUploadProgress,
      });
      setSubmittedPredictionId(response.prediction_id);
      if (isTerminalStatus(response.status)) {
        router.push(`/dashboard/predictions/${response.prediction_id}`);
      }
    } catch (error) {
      const normalized = normalizeApiError(error);
      const descriptionParts = [
        normalized.retryAfterSeconds !== undefined ? `Retry in ${normalized.retryAfterSeconds}s.` : undefined,
        normalized.requestId ? `Request ID: ${normalized.requestId}` : undefined,
      ].filter(Boolean);
      toast.error(getUserFriendlyErrorMessage(error), {
        description: descriptionParts.length ? descriptionParts.join(" ") : undefined,
      });
    }
  };

  return (
    <div className="space-y-6">
      <PageHeader
        title="Analyze audio"
        description="Upload a supported audio file or record a completed browser clip, then submit it to the backend analysis pipeline."
      />
      <Alert>
        <AlertTitle>Backend validation is authoritative</AlertTitle>
        <AlertDescription>
          The frontend checks size, extension, and browser-known duration for usability. The backend still inspects container, codec, duration, and ownership.
        </AlertDescription>
      </Alert>
      <form onSubmit={form.handleSubmit(submit)} className="space-y-6">
        <Tabs defaultValue="upload" onValueChange={(value) => form.setValue("sourceType", value === "record" ? "live_recording" : "dashboard_upload")}>
          <TabsList>
            <TabsTrigger value="upload">File upload</TabsTrigger>
            <TabsTrigger value="record">Browser recording</TabsTrigger>
          </TabsList>
          <TabsContent value="upload">
            <Card>
              <CardHeader><CardTitle>Audio file</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <AudioDropzone file={file} onFile={setFile} onDuration={setDuration} error={durationMessage} />
                <AudioPreview file={file} url={previewUrl} duration={duration} />
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="record">
            <div className="space-y-4">
              <AudioRecorder onRecording={(recording) => {
                setFile(recording);
                setDuration(null);
                if (recording) form.setValue("sourceType", "live_recording");
              }} />
              <AudioPreview file={file} url={previewUrl} duration={duration} />
            </div>
          </TabsContent>
        </Tabs>
        {createPrediction.isPending ? (
          uploadProgress < 100 ? (
            <UploadProgress value={uploadProgress} stage="Uploading audio" />
          ) : (
            <AnalysisWaitState />
          )
        ) : null}
        {status.data ? <ProcessingTimeline status={status.data.status} /> : null}
        <Button type="submit" disabled={createPrediction.isPending || !file}>
          Submit for analysis
        </Button>
      </form>
    </div>
  );
}
