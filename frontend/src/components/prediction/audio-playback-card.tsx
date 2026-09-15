"use client";

import { Volume2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { usePredictionAudio } from "@/features/predictions/hooks";
import { ErrorState } from "@/components/shared/error-state";

export function AudioPlaybackCard({
  predictionId,
  available,
}: {
  predictionId: string;
  available: boolean;
}) {
  const audio = usePredictionAudio(predictionId, available);
  return (
    <Card>
      <CardHeader><CardTitle className="flex items-center gap-2"><Volume2 className="h-5 w-5 text-primary" /> Audio</CardTitle></CardHeader>
      <CardContent>
        {!available ? <p className="text-sm text-muted-foreground">Audio playback is unavailable for this prediction.</p> : null}
        {audio.isError ? <ErrorState error={audio.error} title="Audio playback unavailable" /> : null}
        {audio.data?.playback_url ? (
          <div className="space-y-3">
            <audio controls src={audio.data.playback_url} className="w-full">Audio playback is unavailable.</audio>
            <p className="text-xs text-muted-foreground">Signed URL expires in {audio.data.expires_in_seconds} seconds and is not stored persistently.</p>
            <Button size="sm" variant="outline" onClick={() => void audio.refetch()}>Refresh playback URL</Button>
          </div>
        ) : available && audio.isLoading ? <p className="text-sm text-muted-foreground">Requesting signed playback URL...</p> : null}
      </CardContent>
    </Card>
  );
}
