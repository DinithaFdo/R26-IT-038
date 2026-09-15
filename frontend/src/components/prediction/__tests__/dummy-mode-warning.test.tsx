import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { DummyModeWarning } from "@/components/prediction/dummy-mode-warning";

describe("DummyModeWarning", () => {
  it("renders nothing when show is false", () => {
    const { container } = render(<DummyModeWarning show={false} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the fixed research-eligibility warning copy when show is true", () => {
    render(<DummyModeWarning show />);
    expect(
      screen.getByText(
        "This result includes development-mode model outputs and is not eligible for research evaluation.",
      ),
    ).toBeInTheDocument();
  });

  it("appends the backend-provided reason text when given one", () => {
    render(<DummyModeWarning show reason="Fusion includes deterministic dummy branch outputs." />);
    expect(screen.getByText("Fusion includes deterministic dummy branch outputs.")).toBeInTheDocument();
  });
});
