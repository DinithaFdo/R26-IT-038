import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { PredictionSummaryCard } from "@/components/prediction/prediction-summary-card";
import { FusionDetailsCard } from "@/components/prediction/fusion-details-card";
import { BranchResultCard } from "@/components/prediction/branch-result-card";
import { FusionFallbackNotice } from "@/components/prediction/fusion-fallback-notice";
import type { PredictionDetailResponse, PredictionSubmissionResponse } from "@/types/api";
import realResponse from "@/mocks/fixtures/real-fusion-v3-response.json";

/**
 * `real-fusion-v3-response.json` is a byte-for-byte capture of an actual
 * `POST /api/v1/predictions` response from the real backend (real CNN,
 * AASIST, SSL, and Glottal branches, real Fusion V3), taken while verifying
 * this integration end-to-end. It is not hand-written or shaped to fit the
 * frontend -- this test exists to prove `types/api.ts` and the result
 * components actually match what the backend sends today, not an assumption
 * about what it might send.
 */
const submission = realResponse as unknown as PredictionSubmissionResponse;
const asDetail: PredictionDetailResponse = {
  ...submission,
  preprocessing: {},
  total_processing_time_ms: null,
  warnings: [],
  updated_at: submission.created_at,
  completed_at: submission.created_at,
};

describe("real captured Fusion V3 backend response", () => {
  it("is a genuine four-branch Fusion V3 success, not a fallback", () => {
    expect(submission.status).toBe("completed");
    expect(submission.branches).toHaveLength(4);
    expect(submission.branches.every((branch) => branch.status === "success")).toBe(true);
    expect(submission.branches.every((branch) => branch.mode === "real")).toBe(true);
    expect(submission.fusion?.fusion_version).toBe("fusion-convex-4branch-v3");
    expect(submission.fusion?.fallback_used).toBe(false);
    expect(submission.fusion?.decision_threshold).toBe(0.5);
  });

  it("renders the real verdict and Fusion V3 badge in PredictionSummaryCard without recalculating the fusion score", () => {
    render(<PredictionSummaryCard prediction={asDetail} />);

    expect(screen.getByText(/Fusion V3/)).toBeInTheDocument();
    expect(screen.getByText(/threshold 0\.5/)).toBeInTheDocument();
    // The verdict shown must be driven by the backend's own prediction, not
    // a frontend recomputation of the spoof probability against a threshold.
    expect(
      submission.fusion?.prediction === "bonafide"
        ? screen.getByText("Likely Authentic Voice")
        : screen.getByText("Likely AI-Generated or Manipulated Voice"),
    ).toBeInTheDocument();
  });

  it("renders all four real branch cards from the real response, including Glottal", () => {
    for (const branch of submission.branches) {
      const { unmount } = render(<BranchResultCard branch={branch} />);
      expect(screen.getByText("Included in this result")).toBeInTheDocument();
      unmount();
    }
    expect(submission.branches.map((branch) => branch.model_name)).toEqual([
      "cnn_acoustic",
      "aasist",
      "ssl_wavlm_xlsr",
      "glottal_features",
    ]);
  });

  it("renders the real fusion contribution weights, summing to 100%", () => {
    render(<FusionDetailsCard fusion={submission.fusion} />);

    expect(screen.getByText("34.89%")).toBeInTheDocument();
    expect(screen.getByText("28.98%")).toBeInTheDocument();
    expect(screen.getByText("22.52%")).toBeInTheDocument();
    expect(screen.getByText("13.61%")).toBeInTheDocument();

    const total = Object.values(submission.fusion!.branch_weights).reduce((sum, w) => sum + w, 0);
    expect(total).toBeCloseTo(1, 9);
  });

  it("shows no fallback notice for a real Fusion V3 success", () => {
    const { container } = render(<FusionFallbackNotice fusion={submission.fusion} />);
    expect(container).toBeEmptyDOMElement();
  });
});
