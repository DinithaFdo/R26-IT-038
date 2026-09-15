import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ClassifierSnapshotPanel } from "@/components/xai/classifier-snapshot-panel";
import { NarrativePanel } from "@/components/xai/narrative-panel";
import { explanationFixture } from "@/mocks/xai";

describe("ClassifierSnapshotPanel", () => {
  it("shows when Glottal evidence is auxiliary only and not used for the primary decision", () => {
    const fixture = explanationFixture({
      classifier_snapshot: {
        ...explanationFixture().classifier_snapshot,
        branches: [
          {
            branch_name: "lfcc_cnn_tcn",
            model_name: "cnn_acoustic",
            status: "success",
            mode: "real",
            spoof_probability: 0.8,
          },
        ],
      },
    });

    render(<ClassifierSnapshotPanel snapshot={fixture.classifier_snapshot} />);
    expect(screen.getByText("Not used for primary decision")).toBeInTheDocument();
    expect(screen.getByText("Auxiliary only")).toBeInTheDocument();
    expect(screen.getByText("Glottal spoof probability: 61.00%")).toBeInTheDocument();
    expect(screen.getByText("Confidence")).toBeInTheDocument();
    expect(screen.getByText("50.00%")).toBeInTheDocument();
    expect(screen.getAllByText("Spoof probability")).toHaveLength(2);
  });

  it("shows when Glottal evidence was used for the primary decision", () => {
    const fixture = explanationFixture({
      classifier_snapshot: {
        ...explanationFixture().classifier_snapshot,
        auxiliary_evidence: {
          glottal_spoof_probability: 0.43,
          used_for_primary_decision: true,
        },
      },
    });

    render(<ClassifierSnapshotPanel snapshot={fixture.classifier_snapshot} />);
    expect(screen.getByText("Used for primary decision")).toBeInTheDocument();
    expect(screen.getByText("Decision input")).toBeInTheDocument();
    expect(screen.getByText("Glottal spoof probability: 43.00%")).toBeInTheDocument();
  });
});

describe("NarrativePanel", () => {
  it("renders narrative not_available as an explicit unavailable state", () => {
    const fixture = explanationFixture();
    render(
      <NarrativePanel
        narrative={fixture.narrative}
        status={fixture.component_statuses.narrative}
        error={fixture.component_errors.narrative}
      />,
    );
    expect(screen.getByText("Narrative not available")).toBeInTheDocument();
    expect(screen.getByText(/Narrative generation is disabled/i)).toBeInTheDocument();
  });

  it("renders a completed narrative as non-authoritative", () => {
    render(
      <NarrativePanel
        narrative={{
          status: "completed",
          provider: "alibaba_model_studio",
          model_id: "gpt-test",
          generated_at: "2026-08-09T10:10:00Z",
          prompt_version: "narrative-v1",
          input_sha256: "0".repeat(64),
          evidence_references: ["combined_report", "temporal"],
          summary: "Backend evidence suggests manual review is required.",
          detailed_explanation: "The deterministic report remains authoritative.",
        }}
        status="completed"
      />,
    );

    expect(screen.getByText("Not authoritative")).toBeInTheDocument();
    expect(screen.getByText(/manual review is required/i)).toBeInTheDocument();
    expect(screen.getByText("gpt-test")).toBeInTheDocument();
  });

  it("offers a retry only for a failed optional narrative", async () => {
    const user = userEvent.setup();
    let retries = 0;
    render(
      <NarrativePanel
        narrative={null}
        status="failed"
        error={{
          component: "narrative",
          code: "narrative_upstream_timeout",
          message: "The AI narrative service timed out.",
        }}
        onRetry={() => { retries += 1; }}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Retry AI narrative" }));
    expect(retries).toBe(1);
  });
});
