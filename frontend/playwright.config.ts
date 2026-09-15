import path from "node:path";
import fs from "node:fs";
import { defineConfig, devices } from "@playwright/test";

// Playwright's Node process does not go through Next.js's env loading, but
// @clerk/testing needs CLERK_SECRET_KEY / NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
// in process.env, so load the same .env file the dev server uses.
const envFile = path.resolve(__dirname, ".env");
if (fs.existsSync(envFile)) {
  process.loadEnvFile(envFile);
}

const baseURL = process.env.E2E_BASE_URL || "http://localhost:3000";

export default defineConfig({
  testDir: "./e2e",
  globalSetup: "./e2e/global-setup.ts",
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: [["list"]],
  use: {
    baseURL,
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
  webServer: {
    command: "npm run dev",
    url: baseURL,
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
