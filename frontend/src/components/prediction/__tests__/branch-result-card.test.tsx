import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { BranchResultCard } from "@/components/prediction/branch-result-card";
import {
  allDummyBranches,
  allRealBranches,
  currentStageBranches,
  partialBranchFailureBranches,
} from "@/mocks/predictions";

async function openTechnicalDetails() {
  await userEvent.click(screen.getByRole("button", { name: "Technical details" }));
}

describe("BranchResultCard", () => {
  it("labels a dummy branch as a development placeholder, not a model result, using a plain-language name", () => {
    const [lfcc] = allDummyBranches;
    render(<BranchResultCard branch={lfcc} />);

    expect(screen.getByRole("heading", { name: "Acoustic Pattern Analysis" })).toBeInTheDocument();
    expect(screen.getByText("Technical name: LFCC CNN/TCN")).toBeInTheDocument();
    expect(screen.getByText("Development only")).toBeInTheDocument();
    expect(screen.getByText(/Placeholder output used while building the system/i)).toBeInTheDocument();
    expect(screen.getAllByText(/AI-generated/i).length).toBeGreaterThan(0);
  });

  it("labels a verified real branch as included in the result, not flagged unverified", () => {
    const [lfcc] = allRealBranches;
    render(<BranchResultCard branch={lfcc} />);

    expect(screen.getByText("Included in this result")).toBeInTheDocument();
    expect(screen.getAllByText(/authentic/i).length).toBeGreaterThan(0);
  });

  it("renders a real SSL (XLS-R + Mamba) branch with its plain-language name, not as dummy/unavailable", () => {
    const ssl = allRealBranches[2];
    expect(ssl.model_name).toBe("ssl_sequence");
    render(<BranchResultCard branch={ssl} />);

    expect(screen.getByRole("heading", { name: "Temporal Voice Analysis" })).toBeInTheDocument();
    expect(screen.getByText("Included in this result")).toBeInTheDocument();
    expect(screen.queryByText("Development only")).not.toBeInTheDocument();
    expect(screen.queryByText("Not currently available")).not.toBeInTheDocument();
  });

  it("shows the same plain label for a real branch whose pipeline is unverified, and still shows its score", async () => {
    const [cnn] = currentStageBranches;
    render(<BranchResultCard branch={cnn} />);

    // Normal-user layer must not read differently from a verified branch.
    expect(screen.getByText("Included in this result")).toBeInTheDocument();
    expect(screen.queryByText(/unverified/i)).not.toBeInTheDocument();
    // The score is real model output, so it is shown -- just never as validated.
    expect(screen.getAllByText(/AI-generated/i).length).toBeGreaterThan(0);

    // The scientific-verification caveat is still available, just tucked away.
    await openTechnicalDetails();
    expect(screen.getByText(/Training configuration verification pending/i)).toBeInTheDocument();
  });

  it("renders a disabled branch as unavailable with no invented scores", () => {
    const ssl = currentStageBranches[2];
    render(<BranchResultCard branch={ssl} />);

    expect(screen.getByText("Not currently available")).toBeInTheDocument();
    expect(screen.getByText(/isn't part of the system yet/i)).toBeInTheDocument();
    expect(screen.getByText("No scores were produced by this branch.")).toBeInTheDocument();
    expect(screen.queryByText(/AI-generated/i)).not.toBeInTheDocument();
  });

  it("distinguishes a failed branch from an unavailable one", () => {
    const failed = partialBranchFailureBranches[1];
    render(<BranchResultCard branch={failed} />);

    expect(screen.getByText("Temporarily unavailable")).toBeInTheDocument();
    expect(screen.getByText(/couldn't complete for this recording/i)).toBeInTheDocument();
    expect(screen.getByText("MODEL_BRANCH_FAILED")).toBeInTheDocument();
    expect(screen.queryByText(/isn't part of the system yet/i)).not.toBeInTheDocument();
  });

  it("falls back to 'Not reported' instead of a raw undefined for missing metadata", async () => {
    const branchWithoutMetadata = {
      ...allDummyBranches[0],
      metadata: {},
    };
    render(<BranchResultCard branch={branchWithoutMetadata} />);

    await openTechnicalDetails();
    expect(screen.getAllByText("Not reported")).toHaveLength(2);
  });
});
