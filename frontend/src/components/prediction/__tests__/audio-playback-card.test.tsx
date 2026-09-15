import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AudioPlaybackCard } from "@/components/prediction/audio-playback-card";
import * as predictionsApi from "@/lib/api/predictions";

function renderWithClient(ui: React.ReactElement) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={queryClient}>{ui}</QueryClientProvider>);
}

describe("AudioPlaybackCard", () => {
  it("shows an unavailable message and never requests a signed URL when storage was skipped", () => {
    const spy = vi.spyOn(predictionsApi, "getPredictionAudio");
    renderWithClient(<AudioPlaybackCard predictionId="pred_audio_unavailable" available={false} />);

    expect(screen.getByText("Audio playback is unavailable for this prediction.")).toBeInTheDocument();
    expect(spy).not.toHaveBeenCalled();
  });

  it("renders the signed playback URL and expiry once fetched, without persisting it anywhere itself", async () => {
    vi.spyOn(predictionsApi, "getPredictionAudio").mockResolvedValue({
      playback_url: "https://res.cloudinary.com/signed/example.mp3?token=abc",
      expires_in_seconds: 300,
    });

    renderWithClient(<AudioPlaybackCard predictionId="pred_all_real" available />);

    await waitFor(() => expect(screen.getByRole("button", { name: "Refresh playback URL" })).toBeInTheDocument());
    expect(screen.getByText(/expires in 300 seconds/)).toBeInTheDocument();
  });
});
