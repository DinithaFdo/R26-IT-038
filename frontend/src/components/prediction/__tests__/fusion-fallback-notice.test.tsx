import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { FusionFallbackNotice } from "@/components/prediction/fusion-fallback-notice";
import { fusionLegacyFallbackResult, fusionV3Result } from "@/mocks/predictions";

describe("FusionFallbackNotice", () => {
  it("renders nothing when fusion is null", () => {
    const { container } = render(<FusionFallbackNotice fusion={null} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders nothing when Fusion V3 succeeded (fallback_used is false)", () => {
    const { container } = render(<FusionFallbackNotice fusion={fusionV3Result()} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("shows the fallback notice and the legacy fusion_version, never labelled as Fusion V3", () => {
    render(<FusionFallbackNotice fusion={fusionLegacyFallbackResult()} />);

    expect(screen.getByText("Fallback detector used")).toBeInTheDocument();
    expect(
      screen.getByText(
        /The complete four-branch detector was unavailable for this request, so the legacy three-branch detector produced the final decision\./i,
      ),
    ).toBeInTheDocument();
    expect(screen.getByText(/legacy-3branch-frozen-v1/)).toBeInTheDocument();
    expect(screen.queryByText(/Fusion V3/)).not.toBeInTheDocument();
  });
});
