import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { FusionDetailsCard } from "@/components/prediction/fusion-details-card";
import { fusionResult } from "@/mocks/predictions";

async function openTechnicalDetails() {
  await userEvent.click(screen.getByRole("button", { name: "Technical details" }));
}

describe("FusionDetailsCard", () => {
  it("shows an unavailable message when fusion is null", () => {
    render(<FusionDetailsCard fusion={null} />);
    expect(screen.getByText("Fusion details are unavailable for this prediction.")).toBeInTheDocument();
  });

  it("names the detection methods included using plain-language names", () => {
    render(<FusionDetailsCard fusion={fusionResult(false, 0.28)} />);

    expect(screen.getByText("Detection methods included")).toBeInTheDocument();
    expect(
      screen.getByText(
        "Acoustic Pattern Analysis, Voice Structure Analysis, Temporal Voice Analysis, Vocal Source Analysis",
      ),
    ).toBeInTheDocument();
  });

  it("renders method, config version, and configured weights from the backend payload directly, behind technical details", async () => {
    render(<FusionDetailsCard fusion={fusionResult(false, 0.28)} />);

    await openTechnicalDetails();

    expect(screen.getByText("weighted_average")).toBeInTheDocument();
    expect(screen.getByText("fusion-v1")).toBeInTheDocument();
    expect(screen.getByText(/Included:/)).toHaveTextContent(
      "Included: lfcc_cnn_tcn, aasist, ssl_sequence, glottal",
    );
  });

  it("shows the backend-provided dummy warning only when contains_dummy_branches is true", () => {
    const { rerender } = render(<FusionDetailsCard fusion={fusionResult(true, 0.6)} />);
    expect(
      screen.getByText("Dummy mode is for system development only; not a research result."),
    ).toBeInTheDocument();

    rerender(<FusionDetailsCard fusion={fusionResult(false, 0.28)} />);
    expect(
      screen.queryByText("Dummy mode is for system development only; not a research result."),
    ).not.toBeInTheDocument();
  });

  it("lists excluded branches with their backend-reported reason codes, behind technical details", async () => {
    render(
      <FusionDetailsCard
        fusion={{
          ...fusionResult(false, 0.57),
          excluded_branches: { aasist: "branch_failed" },
        }}
      />,
    );

    await openTechnicalDetails();
    expect(screen.getByText("branch_failed")).toBeInTheDocument();
  });

  it("shows a horizontal contribution bar for all four Fusion V3 weights, with a caption distinguishing weight from probability", () => {
    render(<FusionDetailsCard fusion={fusionV3Result()} />);

    expect(screen.getByText("Fusion contribution")).toBeInTheDocument();
    expect(screen.getByText("34.89%")).toBeInTheDocument();
    expect(screen.getByText("28.98%")).toBeInTheDocument();
    expect(screen.getByText("22.52%")).toBeInTheDocument();
    expect(screen.getByText("13.61%")).toBeInTheDocument();
    expect(
      screen.getByText(/not branch confidence/i),
    ).toBeInTheDocument();
  });

  it("exposes fusion_version, fusion_mode, decision_threshold, and fallback_used behind technical details", async () => {
    render(<FusionDetailsCard fusion={fusionV3Result()} />);

    await openTechnicalDetails();
    expect(screen.getAllByText("fusion-convex-4branch-v3").length).toBeGreaterThan(0);
    expect(screen.getByText("learned_constrained")).toBeInTheDocument();
    expect(screen.getByText("0.5")).toBeInTheDocument();
    expect(screen.getByText("No")).toBeInTheDocument();
  });

  it("renders the legacy fallback identity distinctly from Fusion V3", async () => {
    render(<FusionDetailsCard fusion={fusionLegacyFallbackResult()} />);

    expect(screen.getByText(/legacy detector's combined score/i)).toBeInTheDocument();
    await openTechnicalDetails();
    expect(screen.getByText("legacy-3branch-frozen-v1")).toBeInTheDocument();
    expect(screen.getAllByText("Yes").length).toBeGreaterThan(0);
  });
});
