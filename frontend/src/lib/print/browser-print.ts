/**
 * Opens the browser's print dialog and notifies the caller only after it has
 * closed. The browser owns whether a physical print or "Save as PDF" succeeds,
 * so callers should not claim a file was created.
 */
export function openBrowserPrintDialog(onComplete: () => void) {
  let completed = false;
  const complete = () => {
    if (completed) return;
    completed = true;
    onComplete();
  };

  const onAfterPrint = () => complete();
  window.addEventListener("afterprint", onAfterPrint, { once: true });
  window.print();
}
