import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ModelHealthCard } from "@/components/status/model-health-card";
import { modelHealthFixture } from "@/mocks/predictions";

describe("ModelHealthCard", () => {
  it("renders one entry per branch using the backend-provided display name", () => {
    render(<ModelHealthCard models={modelHealthFixture} />);
    expect(screen.getByText("LFCC CNN/TCN")).toBeInTheDocument();
    expect(screen.getByText("AASIST")).toBeInTheDocument();
    expect(screen.getByText("SSL Sequence")).toBeInTheDocument();
    expect(screen.getByText("Glottal")).toBeInTheDocument();
  });

  it("surfaces the backend's dummy-mode warning text per branch", () => {
    render(<ModelHealthCard models={modelHealthFixture} />);
    expect(
      screen.getAllByText("Dummy mode is for system development only; not a research result.").length,
    ).toBe(modelHealthFixture.length);
  });

  it("falls back to 'Not reported' for an unresolved device rather than showing undefined", () => {
    const withoutDevice = modelHealthFixture.map((model) => ({
      ...model,
      resolved_device: null,
      requested_device: null,
    }));
    render(<ModelHealthCard models={withoutDevice} />);
    expect(screen.getAllByText("Not reported").length).toBe(withoutDevice.length);
  });
});
