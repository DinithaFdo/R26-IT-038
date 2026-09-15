import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { PartialSystemNotice } from "@/components/prediction/partial-system-notice";
import {
  allRealBranches,
  currentStageBranches,
  currentStageFusion,
  fusionResult,
} from "@/mocks/predictions";

describe("PartialSystemNotice", () => {
  it("names the detection methods used and the ones still in development, in plain language", () => {
    render(
      <PartialSystemNotice branches={currentStageBranches} fusion={currentStageFusion} />,
    );

    expect(screen.getByText("Based on the methods currently available")).toBeInTheDocument();
    expect(
      screen.getByText(/combines Acoustic Pattern Analysis and Voice Structure Analysis/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Temporal Voice Analysis and Vocal Source Analysis are still in development/i),
    ).toBeInTheDocument();
  });

  it("explains every research blocker the backend reported behind technical details", () => {
    render(
      <PartialSystemNotice branches={currentStageBranches} fusion={currentStageFusion} />,
    );

    expect(screen.getByText(/Only part of the designed branch set contributed/i)).toBeInTheDocument();
    expect(
      screen.getByText(/feature pipeline or class mapping is unverified/i),
    ).toBeInTheDocument();
  });

  it("states that the result is not full-system performance", () => {
    render(
      <PartialSystemNotice branches={currentStageBranches} fusion={currentStageFusion} />,
    );

    expect(
      screen.getByText(/must not be interpreted as full MULTI-SCOPE system performance/i),
    ).toBeInTheDocument();
  });

  it("stays hidden when the full detection-method set contributed", () => {
    const { container } = render(
      <PartialSystemNotice branches={allRealBranches} fusion={fusionResult(false, 0.28)} />,
    );

    expect(container).toBeEmptyDOMElement();
  });

  it("falls back to the blocker list when system_stage is absent", () => {
    const legacyFusion = { ...currentStageFusion, system_stage: null };
    render(<PartialSystemNotice branches={currentStageBranches} fusion={legacyFusion} />);

    expect(screen.getByText("Based on the methods currently available")).toBeInTheDocument();
  });
});
