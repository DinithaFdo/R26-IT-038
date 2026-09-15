import { describe, expect, it, vi } from "vitest";
import { act, render, screen } from "@testing-library/react";
import { AnalysisWaitState } from "@/components/audio/analysis-wait-state";

describe("AnalysisWaitState", () => {
  it("shows an indeterminate staged message and never a percentage", () => {
    render(<AnalysisWaitState />);
    expect(screen.getByText("Preparing audio…")).toBeInTheDocument();
    expect(screen.queryByText(/%/)).not.toBeInTheDocument();
  });

  it("rotates through the staged messages over time without claiming a specific branch completed", () => {
    vi.useFakeTimers();
    try {
      render(<AnalysisWaitState />);
      expect(screen.getByText("Preparing audio…")).toBeInTheDocument();

      act(() => {
        vi.advanceTimersByTime(4000);
      });
      expect(screen.getByText("Running detection models…")).toBeInTheDocument();

      act(() => {
        vi.advanceTimersByTime(4000);
      });
      expect(screen.getByText("Analyzing voice-source characteristics…")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });
});
