import type { NextConfig } from "next";
import { createHash } from "node:crypto";
import { join } from "node:path";
import { tmpdir } from "node:os";

const projectCacheKey = createHash("sha1")
  .update(process.cwd())
  .digest("hex")
  .slice(0, 12);

const nextConfig: NextConfig = {
  reactStrictMode: true,
  allowedDevOrigins: ["127.0.0.1"],
  // Produces a minimal .next/standalone server bundle for the Docker image
  // instead of copying the full node_modules tree into the runtime layer.
  output: "standalone",
  webpack: (config, { dev, isServer }) => {
    if (dev && config.cache && typeof config.cache === "object") {
      config.cache = {
        ...config.cache,
        cacheDirectory: join(
          tmpdir(),
          "multi-scope-next-webpack-cache",
          projectCacheKey,
          isServer ? "server-development" : "client-development",
        ),
      };
    }

    return config;
  },
};

export default nextConfig;
