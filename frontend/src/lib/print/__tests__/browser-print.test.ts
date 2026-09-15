import { describe, expect, it, vi } from "vitest";
import { openBrowserPrintDialog } from "@/lib/print/browser-print";

describe("openBrowserPrintDialog", () => {
  it("notifies once after the browser closes the print dialog", () => {
    const complete = vi.fn();
    const print = vi.spyOn(window, "print").mockImplementation(() => {
      window.dispatchEvent(new Event("afterprint"));
    });

    openBrowserPrintDialog(complete);

    expect(print).toHaveBeenCalledOnce();
    expect(complete).toHaveBeenCalledOnce();
    print.mockRestore();
  });

  it("does not notify while the browser print dialog remains open", () => {
    const complete = vi.fn();
    const print = vi.spyOn(window, "print").mockImplementation(() => undefined);

    openBrowserPrintDialog(complete);

    expect(complete).not.toHaveBeenCalled();
    print.mockRestore();
  });
});
