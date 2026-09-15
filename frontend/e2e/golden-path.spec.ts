import path from "node:path";
import { test, expect } from "@playwright/test";
import { clerk, setupClerkTestingToken } from "@clerk/testing/playwright";

/**
 * Authenticated golden-path smoke test: sign in -> upload -> dummy result ->
 * history -> delete.
 *
 * This requires a real user to already exist in the configured Clerk
 * *development* instance (pk_test_/sk_test_ keys), identified by
 * E2E_CLERK_TEST_EMAIL. Clerk's ticket-based test sign-in
 * (clerk.signIn({ emailAddress })) issues a session for that user via the
 * Clerk Backend API without needing a password. It also requires the FastAPI
 * backend to be running and reachable at NEXT_PUBLIC_API_BASE_URL, with
 * MongoDB configured, since the analyze flow is a real network round trip.
 *
 * If no test user email is configured, every test in this file is skipped
 * with an explicit reason rather than silently reported as passing.
 */

const testEmail = process.env.E2E_CLERK_TEST_EMAIL;
const hasCredentials = Boolean(testEmail && process.env.CLERK_SECRET_KEY);

test.describe("authenticated golden path", () => {
  test.skip(
    !hasCredentials,
    "Set E2E_CLERK_TEST_EMAIL to an existing user's email in the Clerk development instance to run this spec.",
  );

  test.beforeEach(async ({ page }) => {
    await setupClerkTestingToken({ page });
    await page.goto("/");
    await clerk.signIn({ page, emailAddress: testEmail! });
  });

  test("uploads audio, sees a completed dummy result with four branch cards and the research warning, then deletes it from history", async ({
    page,
  }) => {
    // Submission is a real synchronous backend call today (validate, store,
    // run all four branches, fuse) and can take a while, so give this
    // whole test more room than Playwright's 30s default.
    test.setTimeout(120_000);

    await page.goto("/dashboard/analyze");
    await expect(page.getByRole("heading", { name: "Analyze audio" })).toBeVisible();

    await page
      .getByLabel("Choose audio file")
      .setInputFiles(path.join(__dirname, "fixtures", "sample-voice.wav"));

    await page.getByRole("button", { name: "Submit for analysis" }).click();

    await expect(page.getByRole("heading", { name: "Result summary" })).toBeVisible({
      timeout: 60_000,
    });

    // Backend runs today with all four branches in dummy mode.
    for (const branch of ["LFCC CNN/TCN", "AASIST", "SSL Sequence", "Glottal"]) {
      await expect(page.getByRole("heading", { name: branch })).toBeVisible();
    }
    await expect(page.getByText("Development-mode output")).toBeVisible();
    await expect(page.getByText("Not research eligible").first()).toBeVisible();

    const predictionId = page.url().split("/predictions/")[1];
    expect(predictionId).toBeTruthy();

    await page.goto("/dashboard/history");
    await expect(page.getByRole("heading", { name: "History" })).toBeVisible();

    const row = page.getByRole("row").filter({ hasText: "sample-voice.wav" }).first();
    await expect(row).toBeVisible();
    await row.getByRole("button", { name: "Delete prediction" }).click();
    await page.getByRole("button", { name: "Delete" }).click();

    await expect(page.getByText("sample-voice.wav")).not.toBeVisible();
  });

  test("rejects an unsupported file client-side before ever calling the backend", async ({ page }) => {
    await page.goto("/dashboard/analyze");

    const textFile = path.join(__dirname, "fixtures", "not-audio.txt");
    await page.getByLabel("Choose audio file").setInputFiles(textFile);

    await expect(page.getByText(/unsupported|not a supported/i)).toBeVisible();
  });
});
