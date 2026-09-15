import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { SystemReadinessCard } from "@/components/status/system-readiness-card";
import { backendNotReadyFixture } from "@/mocks/predictions";

describe("SystemReadinessCard", () => {
  it("renders 'Not ready' for every readiness flag the backend reports as false", () => {
    render(<SystemReadinessCard readiness={backendNotReadyFixture} />);
    expect(screen.getAllByText("Not ready").length).toBeGreaterThan(0);
  });

  it("renders 'Ready' for a fully ready backend", () => {
    render(
      <SystemReadinessCard
        readiness={{
          ...backendNotReadyFixture,
          status: "ready",
          prediction_ready: true,
          research_ready: true,
          mongodb_available: true,
          storage_available: true,
        }}
      />,
    );
    expect(screen.queryByText("Not ready")).not.toBeInTheDocument();
    expect(screen.getAllByText("Ready").length).toBe(5);
  });

  it("treats disabled storage as ready rather than reporting a false alarm", () => {
    render(
      <SystemReadinessCard
        readiness={{ ...backendNotReadyFixture, storage_enabled: false, storage_available: false }}
      />,
    );
    const storageCard = screen.getByText("Storage").closest("div");
    expect(storageCard).toHaveTextContent("Ready");
  });
});
