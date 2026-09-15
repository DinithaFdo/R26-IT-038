import { test, expect } from "@playwright/test";

/**
 * Unauthenticated / backend-independent smoke tests. These do not need a
 * Clerk test user and do not need the FastAPI backend to be reachable -
 * some of them deliberately simulate it being unreachable via route
 * interception.
 */

test("unauthenticated users are redirected away from the dashboard", async ({ page }) => {
  await page.goto("/dashboard");
  // Next dev-mode cold compiles the sign-in route on first hit, which can
  // comfortably exceed Playwright's 5s default assertion timeout.
  await expect(page).toHaveURL(/sign-in/, { timeout: 15_000 });
});

test("public system status page reports a readiness error instead of hanging when the backend is unreachable", async ({
  page,
}) => {
  await page.route("**/ready", (route) => route.abort("connectionrefused"));
  await page.route("**/api/v1/voice/models/health", (route) => route.abort("connectionrefused"));

  await page.goto("/system", { waitUntil: "domcontentloaded" });
  await expect(page.getByRole("heading", { name: "System status" })).toBeVisible();
  // TanStack Query retries network failures a couple of times with backoff
  // before surfacing isError, so give this more room than the default 5s.
  await expect(page.getByText("Readiness unavailable")).toBeVisible({ timeout: 15_000 });
  await expect(page.getByText("Model health unavailable")).toBeVisible({ timeout: 15_000 });
});

test("public system status page renders live readiness and model health when the backend responds", async ({
  page,
}) => {
  await page.route("**/ready", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        status: "ready",
        prediction_ready: true,
        research_ready: false,
        ffmpeg_available: true,
        ffprobe_available: true,
        mongodb_configured: true,
        mongodb_available: true,
        storage_enabled: true,
        storage_available: true,
        components: {},
      }),
    }),
  );
  await page.route("**/api/v1/voice/models/health", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify([
        {
          branch_name: "lfcc_cnn_tcn",
          model_name: "cnn_acoustic",
          display_name: "LFCC CNN/TCN",
          mode: "dummy",
          is_loaded: true,
          uses_dummy_mode: true,
          warning: "Dummy mode is for system development only; not a research result.",
        },
      ]),
    }),
  );

  await page.goto("/system", { waitUntil: "domcontentloaded" });
  await expect(page.getByText("LFCC CNN/TCN")).toBeVisible();
  await expect(page.getByText("Ready").first()).toBeVisible();
});
