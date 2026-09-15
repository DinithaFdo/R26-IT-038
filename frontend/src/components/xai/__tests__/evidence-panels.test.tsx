import { describe, expect, it } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import { QualityEvidencePanel } from "@/components/xai/quality-evidence-panel";
import { SemanticEvidencePanel } from "@/components/xai/semantic-evidence-panel";
import {
  TemporalEvidencePanel,
  visualizationRegionsFromSpectrogram,
  waveformRegionAtPointer,
} from "@/components/xai/temporal-evidence-panel";
import { WarningsPanel } from "@/components/xai/warnings-panel";
import { explanationFixture, semanticFixture, temporalFixture } from "@/mocks/xai";

describe("TemporalEvidencePanel", () => {
  it("uses visualization regions embedded in the spectrogram artifact", () => {
    const regions = visualizationRegionsFromSpectrogram({
      high_attention_regions: [
        {
          region_id: 1,
          start_seconds: 1.25,
          end_seconds: 1.75,
          attention_score: 1.42,
        },
      ],
    });

    expect(regions).toEqual([
      {
        region_id: 1,
        start_seconds: 1.25,
        end_seconds: 1.75,
        duration_seconds: 0.5,
        attention_score: 1.42,
      },
    ]);
  });

  it("does not treat malformed visualization regions as valid overlays", () => {
    expect(
      visualizationRegionsFromSpectrogram({
        high_attention_regions: [
          {
            region_id: 1,
            start_seconds: 2,
            end_seconds: 1,
            attention_score: 1.42,
          },
        ],
      }),
    ).toBeNull();
  });

  it("uses the waveform plot bounds when finding a hovered high-attention region", () => {
    const region = {
      region_id: 1,
      start_seconds: 2,
      end_seconds: 3,
      duration_seconds: 1,
      attention_score: 1.42,
    };

    // On an 800px canvas, the plot starts after the 48px y-axis gutter. The
    // red strip starts at roughly 195px, not at 160px as a full-canvas time mapping
    // would assume.
    expect(waveformRegionAtPointer(196, 800, 10, [region])).toEqual(region);
    expect(waveformRegionAtPointer(160, 800, 10, [region])).toBeNull();
  });

  it("renders backend regions with their reported times and scores", () => {
    render(
      <TemporalEvidencePanel
        predictionId="prediction-1"
        temporal={temporalFixture}
        status="completed"
        clipDurationSeconds={3}
      />,
    );
    expect(screen.getAllByText("High-attention regions").length).toBeGreaterThan(0);
    const region = screen.getByRole("button", { name: /Region 01/i });
    expect(region).toHaveTextContent(/Start time\s*0.40s/);
    expect(region).toHaveTextContent(/End time\s*0.90s/);
    expect(region).toHaveTextContent(/Duration\s*0.50s/);
    expect(region).toHaveTextContent(/Attention score\s*0.810/);
  });

  it("renders high_attention_regions from the backend", () => {
    render(
      <TemporalEvidencePanel
        predictionId="prediction-1"
        temporal={temporalFixture}
        status="completed"
        clipDurationSeconds={3}
      />,
    );
    expect(screen.getByRole("button", { name: /Region 01/i })).toHaveTextContent(
      /Start time\s*0.40s/,
    );
  });

  it("renders complete print details for every reported high-attention region", () => {
    render(
      <TemporalEvidencePanel
        predictionId="prediction-1"
        temporal={temporalFixture}
        semantic={semanticFixture}
        status="completed"
        clipDurationSeconds={3}
      />,
    );

    expect(screen.getByRole("heading", { name: "All high-attention region details" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Region 01 print details" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Region 02 print details" })).toBeInTheDocument();
  });

  it("uses the visualization regions for temporal cards and semantic evidence", () => {
    render(
      <TemporalEvidencePanel
        predictionId="prediction-1"
        temporal={{
          ...temporalFixture,
          high_attention_region_count: 1,
          high_attention_regions: [
            {
              region_id: 1,
              start_seconds: 0,
              end_seconds: 7.42,
              attention_score: 2.256,
            },
          ],
          visualization_high_attention_region_count: 2,
          visualization_high_attention_regions: [
            {
              region_id: 1,
              start_seconds: 1.44,
              end_seconds: 2.08,
              attention_score: 2.256,
            },
            {
              region_id: 2,
              start_seconds: 2.66,
              end_seconds: 3.22,
              attention_score: 2.1,
            },
          ],
        }}
        status="completed"
        clipDurationSeconds={7.42}
      />,
    );

    expect(screen.getByRole("button", { name: /Region 01/i })).toHaveTextContent(
      /Start time\s*1.44s/,
    );
    expect(screen.getByRole("button", { name: /Region 02/i })).toHaveTextContent(
      /End time\s*3.22s/,
    );
    expect(screen.queryByRole("button", { name: /0.00s.*7.42s/i })).not.toBeInTheDocument();
  });

  it("renders a valid empty high_attention_regions array as no high-attention regions", () => {
    render(
      <TemporalEvidencePanel
        predictionId="prediction-1"
        temporal={{
          ...temporalFixture,
          threshold_crossing_count: 0,
          high_attention_region_count: 0,
          high_attention_regions: [],
        }}
        status="completed"
        clipDurationSeconds={3}
      />,
    );
    expect(screen.getByText("No high-attention regions")).toBeInTheDocument();
  });

  it("does not render the removed temporal limitation warning", () => {
    render(
      <TemporalEvidencePanel
        predictionId="prediction-1"
        temporal={temporalFixture}
        status="completed"
        clipDurationSeconds={3}
      />,
    );
    expect(screen.queryByText(/not confirmed spoof artifacts/i)).not.toBeInTheDocument();
  });

  it("hides internal temporal region-count diagnostics", () => {
    render(
      <TemporalEvidencePanel
        predictionId="prediction-1"
        temporal={temporalFixture}
        status="completed"
        clipDurationSeconds={3}
      />,
    );

    expect(screen.queryByText("Threshold crossings")).not.toBeInTheDocument();
    expect(screen.queryByText("Raw region count")).not.toBeInTheDocument();
    expect(screen.queryByText("Post-filter region count")).not.toBeInTheDocument();
  });

  it("shows the visual workspace when clip duration is unknown", () => {
    render(
      <TemporalEvidencePanel
        predictionId="prediction-1"
        temporal={temporalFixture}
        status="completed"
        clipDurationSeconds={null}
      />,
    );
    expect(screen.getByText("Mel-spectrogram with attention overlay")).toBeInTheDocument();
    expect(screen.getByText(/Audio playback is unavailable/i)).toBeInTheDocument();
  });

  it("renders queued temporal analysis as loading rather than empty evidence", () => {
    render(
      <TemporalEvidencePanel
        predictionId="prediction-1"
        temporal={{ ...temporalFixture, status: "queued", high_attention_regions: undefined }}
        status="queued"
      />,
    );
    expect(screen.getByText("Analyzing temporal evidence...")).toBeInTheDocument();
    expect(screen.queryByText("No high-attention regions")).not.toBeInTheDocument();
  });

  it("renders high-attention regions in a compact navigator", () => {
    render(
      <TemporalEvidencePanel
        predictionId="prediction-1"
        temporal={temporalFixture}
        status="completed"
        clipDurationSeconds={3}
      />,
    );
    expect(screen.getByRole("heading", { name: "Region navigator" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Region 02/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Region 02/i })).toHaveTextContent(
      /Start time\s*1.60s/,
    );
  });

  it("updates the selected detail when a navigator region is clicked", () => {
    render(
      <TemporalEvidencePanel
        predictionId="prediction-1"
        temporal={temporalFixture}
        status="completed"
        clipDurationSeconds={3}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /Region 02/i }));

    expect(screen.getByText(/Region 02 · 1.60s - 2.05s/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Region 02/i })).toHaveAttribute("aria-pressed", "true");
  });

  it("groups repeated semantic feature observations within the selected region", () => {
    const [spoofFeature, bonafideFeature] = semanticFixture.feature_importance;
    render(
      <TemporalEvidencePanel
        predictionId="prediction-1"
        temporal={temporalFixture}
        status="completed"
        clipDurationSeconds={3}
        semantic={{
          ...semanticFixture,
          windows: [
            {
              start_seconds: 0.4,
              end_seconds: 0.9,
              feature_importance: [
                { ...spoofFeature, shap_value: 0.2, start_seconds: 0.4, end_seconds: 0.9 },
                { ...bonafideFeature, shap_value: -0.1, start_seconds: 0.4, end_seconds: 0.9 },
              ],
            },
            {
              start_seconds: 0.6,
              end_seconds: 1.1,
              feature_importance: [
                { ...spoofFeature, shap_value: 0.4, start_seconds: 0.6, end_seconds: 1.1 },
              ],
            },
          ],
        }}
      />,
    );

    expect(screen.getByText("Semantic evidence in this high-attention region")).toBeInTheDocument();
    expect(screen.getAllByText("Acoustic feature").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Pitch-period jitter").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Supports spoof").length).toBeGreaterThan(0);
    expect(screen.getAllByText("2 of 2 windows").length).toBeGreaterThan(0);
  });

  it("formats near-zero region starts as zero seconds", () => {
    render(
      <TemporalEvidencePanel
        predictionId="prediction-1"
        temporal={{
          ...temporalFixture,
          high_attention_region_count: 1,
          high_attention_regions: [
            {
              region_id: 1,
              start_seconds: 9.313225746154785e-9,
              end_seconds: 3.9764375,
              duration_seconds: 3.97643749,
              attention_score: 1.595384836,
            },
          ],
        }}
        status="completed"
        clipDurationSeconds={3.9764375}
      />,
    );
    expect(screen.getAllByText("0.00s").length).toBeGreaterThan(0);
  });

  it("shows an unavailable state, not an error, when there is no temporal result", () => {
    render(
      <TemporalEvidencePanel predictionId="prediction-1" temporal={null} status="not_available" />,
    );
    expect(screen.getByText("No temporal evidence")).toBeInTheDocument();
  });

  it("does not treat malformed temporal regions as empty evidence", () => {
    render(
      <TemporalEvidencePanel
        predictionId="prediction-1"
        temporal={{
          ...temporalFixture,
          high_attention_regions: [
            {
              region_id: 1,
              start_seconds: 1,
              end_seconds: 0.5,
              attention_score: 1.2,
            },
          ],
        }}
        status="completed"
        clipDurationSeconds={3}
      />,
    );
    expect(screen.getByText("Temporal evidence unavailable")).toBeInTheDocument();
    expect(screen.queryByText("No high-attention regions")).not.toBeInTheDocument();
  });

  it("states when an audio clip cannot be shown because playback was not retained", () => {
    render(
      <TemporalEvidencePanel
        predictionId="prediction-1"
        temporal={temporalFixture}
        status="completed"
        clipDurationSeconds={3}
      />,
    );
    expect(screen.getByText(/Audio playback is unavailable for this prediction/i)).toBeInTheDocument();
  });
});

describe("SemanticEvidencePanel", () => {
  it("uses the backend direction field rather than inferring from SHAP sign", () => {
    render(<SemanticEvidencePanel semantic={semanticFixture} status="completed" />);
    expect(screen.getByRole("button", { name: /Rank 1: .*Toward spoof/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Rank 2: .*Toward bonafide/i })).toBeInTheDocument();
  });

  it("shows a whole-clip marker when a contribution has no time interval", () => {
    render(<SemanticEvidencePanel semantic={semanticFixture} status="completed" />);
    expect(screen.getAllByText(/Whole clip/).length).toBeGreaterThan(0);
  });

  it("states plainly when only whole-clip evidence exists", () => {
    render(<SemanticEvidencePanel semantic={semanticFixture} status="completed" />);
    expect(screen.getByText(/not localised in time/i)).toBeInTheDocument();
  });

  it("does not render the removed semantic callouts", () => {
    render(<SemanticEvidencePanel semantic={semanticFixture} status="completed" />);
    expect(screen.queryByText("Semantic limitation")).not.toBeInTheDocument();
    expect(screen.queryByText(/do not explain the classifier/i)).not.toBeInTheDocument();
  });

  it("does not show unavailable window-level base and predicted values in the header", () => {
    render(<SemanticEvidencePanel semantic={semanticFixture} status="completed" />);
    expect(screen.queryByText("Base value")).not.toBeInTheDocument();
    expect(screen.queryByText("Predicted value")).not.toBeInTheDocument();
  });

  it("shows one selected feature detail card and updates it from the SHAP ranking plot", () => {
    render(<SemanticEvidencePanel semantic={semanticFixture} status="completed" />);

    expect(screen.getByText("Selected feature details")).toBeInTheDocument();
    expect(screen.getAllByText("Pitch-period jitter").length).toBeGreaterThan(1);

    fireEvent.click(
      screen.getByRole("button", {
        name: /Rank 2: Harmonic-to-noise ratio, Toward bonafide, Whole clip/i,
      }),
    );

    expect(screen.getAllByText("Harmonic-to-noise ratio").length).toBeGreaterThan(1);
    expect(screen.getAllByText("21.40").length).toBeGreaterThan(0);
  });

  it("renders complete details for every ranked semantic feature in print", () => {
    render(<SemanticEvidencePanel semantic={semanticFixture} status="completed" />);

    expect(screen.getByRole("heading", { name: "All semantic feature details" })).toBeInTheDocument();
    expect(screen.getAllByText("Pitch-period jitter").length).toBeGreaterThan(1);
    expect(screen.getAllByText("Harmonic-to-noise ratio").length).toBeGreaterThan(1);
  });

  it("shows an unavailable state when there is no semantic result", () => {
    render(<SemanticEvidencePanel semantic={null} status="failed" />);
    expect(screen.getByText("No semantic evidence")).toBeInTheDocument();
  });

  it("keeps semantic windows collapsed until requested, then renders a compact window navigator", () => {
    render(
      <SemanticEvidencePanel
        semantic={{
          ...semanticFixture,
          windows: [
            {
              start_seconds: 0.5,
              end_seconds: 1.5,
              feature_importance: [
                {
                  ...semanticFixture.feature_importance[0],
                  rank: 1,
                  start_seconds: 0.5,
                  end_seconds: 1.5,
                },
              ],
            },
            {
              start_seconds: 2.5,
              end_seconds: 3.5,
              feature_importance: [],
            },
          ],
        }}
        status="completed"
        temporalRegions={[
          {
            region_id: 1,
            start_seconds: 0.4,
            end_seconds: 0.9,
            duration_seconds: 0.5,
            attention_score: 0.81,
          },
        ]}
        selectedRegionId={1}
      />,
    );
    const windowToggle = screen.getByRole("button", { name: /Semantic analysis windows/i });
    expect(windowToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Selected window")).not.toBeInTheDocument();

    fireEvent.click(windowToggle);

    expect(windowToggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/Windows overlapping Region 1 are highlighted/i)).toBeInTheDocument();
    expect(screen.getByText("Selected window")).toBeInTheDocument();
    expect(screen.getAllByText("0.50s - 1.50s").length).toBeGreaterThan(0);

    fireEvent.click(screen.getByRole("button", { name: "Next analysis window" }));
    expect(screen.getByText("2.50s - 3.50s")).toBeInTheDocument();
  });
});

describe("QualityEvidencePanel", () => {
  it("renders the two backend-supported metric values", () => {
    render(<QualityEvidencePanel quality={explanationFixture().quality} />);
    expect(screen.getByText("0.420")).toBeInTheDocument();
    expect(screen.getByText("0.634")).toBeInTheDocument();
    expect(screen.getByText(/Dataset: ASVspoof2019_LA/)).toBeInTheDocument();
    expect(screen.getByText(/2 of 2 quality metrics/)).toBeInTheDocument();
  });

  it("renders uncomputed metrics as 'Not computed', never as zero", () => {
    render(
      <QualityEvidencePanel
        quality={{
          ...explanationFixture().quality,
          temporal_semantic_iou: {
            status: "not_computed",
            value: null,
            scope: "per_analysis",
            reason: "Temporal or semantic explanation is unavailable.",
          },
        }}
      />,
    );
    expect(screen.getByText(/Not computed/)).toBeInTheDocument();
    expect(screen.queryByText("0.000")).not.toBeInTheDocument();
  });

  it("renders metrics the backend omits entirely as not available", () => {
    render(<QualityEvidencePanel quality={{}} />);
    expect(screen.getAllByText(/Not available/)).toHaveLength(2);
  });

  it("groups an unavailable supported metric in the compact layout", () => {
    render(
      <QualityEvidencePanel
        quality={{
          ...explanationFixture().quality,
          temporal_semantic_iou: {
            status: "not_computed",
            value: null,
            scope: "per_analysis",
            reason: "Temporal or semantic explanation is unavailable.",
          },
        }}
        compact
      />,
    );

    expect(screen.getByText("Research quality")).toBeInTheDocument();
    expect(screen.getByText("0.634")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Not computed (1)" })).toHaveAttribute(
      "data-state",
      "closed",
    );

    fireEvent.click(screen.getByRole("button", { name: "Not computed (1)" }));
    expect(screen.getByText("Temporal / semantic agreement (IoU)")).toBeVisible();
  });

  it("does not advertise unsupported quality metrics", () => {
    render(<QualityEvidencePanel quality={explanationFixture().quality} />);

    expect(screen.queryByText("Temporal localisation (IoU)")).not.toBeInTheDocument();
    expect(screen.queryByText("AOPC")).not.toBeInTheDocument();
    expect(screen.queryByText("Robustness / stability")).not.toBeInTheDocument();
  });
});

describe("WarningsPanel", () => {
  it("always explains the limitation of temporal attention regions", () => {
    render(
      <WarningsPanel
        warnings={[]}
        componentErrors={{ temporal: null, semantic: null, report: null }}
      />,
    );
    expect(screen.getByText(/attention-rollout tensors captured during the XLS-R transformer/i)).toBeInTheDocument();
    expect(screen.queryByText("Temporal-analysis limitation")).not.toBeInTheDocument();
  });

  it("uses the semantic limitation when rendered for semantic evidence", () => {
    render(
      <WarningsPanel
        warnings={[]}
        componentErrors={{ temporal: null, semantic: null, report: null }}
        limitationFor="semantic"
      />,
    );
    expect(screen.getByText(/independent acoustic model from sliding-window SHAP attributions/i)).toBeInTheDocument();
    expect(screen.queryByText(/attention-rollout tensors/i)).not.toBeInTheDocument();
  });

  it("shows both analysis limitations in the report", () => {
    render(
      <WarningsPanel
        warnings={[]}
        componentErrors={{ temporal: null, semantic: null, report: null }}
        limitationFor="both"
      />,
    );
    expect(screen.getByText(/attention-rollout tensors captured during the XLS-R transformer/i)).toBeInTheDocument();
    expect(screen.getByText(/independent acoustic model from sliding-window SHAP attributions/i)).toBeInTheDocument();
  });

  it("shows the decision-support limitation when rendered for the combined finding", () => {
    render(
      <WarningsPanel
        warnings={[]}
        componentErrors={{ temporal: null, semantic: null, report: null }}
        limitationFor="report"
      />,
    );
    expect(screen.getByText("Decision support evidence only. Human review required.")).toBeInTheDocument();
    expect(screen.queryByText(/sliding-window SHAP attributions/i)).not.toBeInTheDocument();
  });

  it("renders backend warnings", () => {
    render(
      <WarningsPanel
        warnings={["Development placeholder generated from deterministic fixture SHAP values."]}
        componentErrors={{ temporal: null, semantic: null, report: null }}
      />,
    );
    expect(screen.getByText(/deterministic fixture SHAP values/)).toBeInTheDocument();
  });

  it("renders only the sanitised code and message for a component error", () => {
    render(
      <WarningsPanel
        warnings={[]}
        componentErrors={{
          temporal: null,
          semantic: {
            component: "semantic",
            code: "semantic_analysis_failed",
            message: "The XAI component could not be completed.",
          },
          report: null,
        }}
      />,
    );
    expect(screen.getByText("semantic_analysis_failed")).toBeInTheDocument();
    expect(screen.getByText("The XAI component could not be completed.")).toBeInTheDocument();
  });

  it("confirms explicitly when nothing was reported", () => {
    render(
      <WarningsPanel
        warnings={[]}
        componentErrors={{ temporal: null, semantic: null, report: null }}
      />,
    );
    expect(screen.getByText(/No warnings were reported/i)).toBeInTheDocument();
  });
});
