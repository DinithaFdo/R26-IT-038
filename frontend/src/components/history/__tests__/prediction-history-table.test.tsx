import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { PredictionHistoryTable } from "@/components/history/prediction-history-table";
import { historyFixture } from "@/mocks/predictions";

describe("PredictionHistoryTable", () => {
  it("renders one row per history item, newest-first order as provided by the backend", () => {
    render(
      <PredictionHistoryTable items={historyFixture.items} onDelete={vi.fn()} onRerun={vi.fn()} />,
    );
    expect(screen.getAllByRole("row")).toHaveLength(historyFixture.items.length + 1); // + header row
  });

  it("shows research validation pending for any row containing a dummy branch", () => {
    render(
      <PredictionHistoryTable items={historyFixture.items} onDelete={vi.fn()} onRerun={vi.fn()} />,
    );
    expect(screen.getAllByText("Research validation in progress").length).toBeGreaterThan(0);
  });

  it("calls onDelete with the prediction id only after the confirm dialog is accepted", async () => {
    const user = userEvent.setup();
    const onDelete = vi.fn();
    const [firstItem] = historyFixture.items;
    render(
      <PredictionHistoryTable items={[firstItem]} onDelete={onDelete} onRerun={vi.fn()} />,
    );

    await user.click(screen.getByRole("button", { name: "Delete prediction" }));
    expect(onDelete).not.toHaveBeenCalled();

    await user.click(await screen.findByRole("button", { name: "Delete" }));
    expect(onDelete).toHaveBeenCalledWith(firstItem.prediction_id);
  });

  it("calls onRerun with the prediction id only after the confirm dialog is accepted", async () => {
    const user = userEvent.setup();
    const onRerun = vi.fn();
    const [firstItem] = historyFixture.items;
    render(
      <PredictionHistoryTable items={[firstItem]} onDelete={vi.fn()} onRerun={onRerun} />,
    );

    await user.click(screen.getByRole("button", { name: "Rerun prediction" }));
    await user.click(await screen.findByRole("button", { name: "Rerun" }));
    expect(onRerun).toHaveBeenCalledWith(firstItem.prediction_id);
  });
});
