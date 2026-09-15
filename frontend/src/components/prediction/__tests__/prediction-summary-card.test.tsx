import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PredictionSummaryCard } from "@/components/prediction/prediction-summary-card";
import { mockPredictionScenarios } from "@/mocks/predictions";

async function openTechnicalDetails() {
  await userEvent.click(screen.getByRole("button", { name: "Technical details" }));
}

describe("PredictionSummaryCard", () => {
  it("shows a plain-language verdict headline for a spoof result", () => {
    render(<PredictionSummaryCard prediction={mockPredictionScenarios.allDummyCompleted} />);
    expect(screen.getByText("Likely AI-Generated or Manipulated Voice")).toBeInTheDocument();
  });

  it("marks an all-real completed prediction as research eligible", () => {
    render(<PredictionSummaryCard prediction={mockPredictionScenarios.allRealCompleted} />);
    expect(screen.getByText("Research eligible")).toBeInTheDocument();
  });

  it("marks a dummy-containing prediction as research validation in progress, using the backend field as the source of truth", () => {
    render(<PredictionSummaryCard prediction={mockPredictionScenarios.allDummyCompleted} />);
    expect(screen.getByText("Research validation in progress")).toBeInTheDocument();
  });

  it("renders fusion spoof/bonafide probabilities directly from the backend without recalculating them, behind technical details", async () => {
    const prediction = mockPredictionScenarios.allDummyCompleted;
    render(<PredictionSummaryCard prediction={prediction} />);
    await openTechnicalDetails();

    const spoofPercent = `${(prediction.fusion!.probabilities!.spoof * 100).toFixed(2)}%`;
    const bonafidePercent = `${(prediction.fusion!.probabilities!.bonafide * 100).toFixed(2)}%`;
    expect(screen.getAllByText(spoofPercent).length).toBeGreaterThan(0);
    expect(screen.getAllByText(bonafidePercent).length).toBeGreaterThan(0);
  });

  it("falls back to an explicit 'Backend did not report' for the unverified decision threshold field, never a silent zero", async () => {
    render(<PredictionSummaryCard prediction={mockPredictionScenarios.allRealCompleted} />);
    await openTechnicalDetails();
    expect(screen.getByText("Backend did not report")).toBeInTheDocument();
  });

  it("renders 'No decision' and a no-result headline for a failed prediction with no fusion result, not a crash or blank value", () => {
    render(<PredictionSummaryCard prediction={mockPredictionScenarios.failed} />);
    expect(screen.getByText("No Result Available")).toBeInTheDocument();
  });
});
