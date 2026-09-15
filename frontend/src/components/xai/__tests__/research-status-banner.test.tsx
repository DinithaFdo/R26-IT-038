import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  ResearchStatusBadges,
  ResearchStatusBanner,
} from "@/components/xai/research-status-banner";

describe("ResearchStatusBanner", () => {
  it("shows the development-evidence warning when the backend flags a placeholder", () => {
    render(<ResearchStatusBanner developmentPlaceholder researchEligible={false} />);
    expect(screen.getByText("Development evidence")).toBeInTheDocument();
    expect(screen.getByText("Research validation pending")).toBeInTheDocument();
    expect(
      screen.getByText(/must not be used as evidence in a research result/i),
    ).toBeInTheDocument();
  });

  it("always states the decision-support requirement", () => {
    render(<ResearchStatusBanner developmentPlaceholder researchEligible={false} />);
    expect(
      screen.getByText("Decision-support evidence. Requires qualified human review."),
    ).toBeInTheDocument();
  });

  it("calls out a dummy classifier branch explicitly", () => {
    render(
      <ResearchStatusBanner
        developmentPlaceholder={false}
        researchEligible={false}
        containsDummyBranches
      />,
    );
    expect(screen.getByText("Contains dummy branch")).toBeInTheDocument();
    expect(screen.getByText(/ran in dummy mode/i)).toBeInTheDocument();
  });

  it("hides itself only when the run is genuinely clean on every flag", () => {
    const { container } = render(
      <ResearchStatusBanner
        developmentPlaceholder={false}
        researchEligible
        containsDummyBranches={false}
      />,
    );
    expect(container).toBeEmptyDOMElement();
  });
});

describe("ResearchStatusBadges", () => {
  it("labels a non-eligible run as research validation pending", () => {
    render(<ResearchStatusBadges developmentPlaceholder researchEligible={false} />);
    expect(screen.getByText("Development evidence")).toBeInTheDocument();
    expect(screen.getByText("Research validation pending")).toBeInTheDocument();
  });

  it("labels a fully eligible run as model-backed and research eligible", () => {
    render(<ResearchStatusBadges developmentPlaceholder={false} researchEligible />);
    expect(screen.getByText("Model-backed evidence")).toBeInTheDocument();
    expect(screen.getByText("Research eligible")).toBeInTheDocument();
  });
});
