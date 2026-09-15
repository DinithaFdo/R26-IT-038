import path from "node:path";
import fs from "node:fs";
import { clerkSetup } from "@clerk/testing/playwright";

const envFile = path.resolve(__dirname, "..", ".env");
if (fs.existsSync(envFile)) {
  process.loadEnvFile(envFile);
}

export default async function globalSetup() {
  if (!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY || !process.env.CLERK_SECRET_KEY) {
    // Clerk is not configured in this environment; authenticated specs
    // self-skip (see e2e/golden-path.spec.ts), so there is nothing to set up.
    return;
  }
  await clerkSetup();
}
