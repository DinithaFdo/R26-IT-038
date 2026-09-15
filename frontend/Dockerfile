FROM node:20-slim AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:20-slim AS builder
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1

# NEXT_PUBLIC_* variables are inlined into the client bundle at build time,
# not read at container runtime, so they must arrive as build args rather
# than as compose `environment:` on the final image.
ARG NEXT_PUBLIC_API_BASE_URL
ARG NEXT_PUBLIC_APP_NAME
ARG NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY
ARG NEXT_PUBLIC_CLERK_SIGN_IN_URL
ARG NEXT_PUBLIC_CLERK_SIGN_UP_URL
ARG NEXT_PUBLIC_CLERK_SIGN_IN_FALLBACK_REDIRECT_URL
ARG NEXT_PUBLIC_CLERK_SIGN_UP_FALLBACK_REDIRECT_URL
ENV NEXT_PUBLIC_API_BASE_URL=${NEXT_PUBLIC_API_BASE_URL} \
    NEXT_PUBLIC_APP_NAME=${NEXT_PUBLIC_APP_NAME} \
    NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=${NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY} \
    NEXT_PUBLIC_CLERK_SIGN_IN_URL=${NEXT_PUBLIC_CLERK_SIGN_IN_URL} \
    NEXT_PUBLIC_CLERK_SIGN_UP_URL=${NEXT_PUBLIC_CLERK_SIGN_UP_URL} \
    NEXT_PUBLIC_CLERK_SIGN_IN_FALLBACK_REDIRECT_URL=${NEXT_PUBLIC_CLERK_SIGN_IN_FALLBACK_REDIRECT_URL} \
    NEXT_PUBLIC_CLERK_SIGN_UP_FALLBACK_REDIRECT_URL=${NEXT_PUBLIC_CLERK_SIGN_UP_FALLBACK_REDIRECT_URL}

COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:20-slim AS runtime
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOST=0.0.0.0 \
    PORT=3000

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system multiscope \
    && useradd --system --gid multiscope --home-dir /app --shell /usr/sbin/nologin multiscope

# next.config.ts sets output: "standalone", so the runtime layer only needs
# the pruned server bundle plus the static/public assets next builds
# separately from it - no full node_modules tree is copied in.
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

RUN chown -R multiscope:multiscope /app

USER multiscope

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://127.0.0.1:3000/ || exit 1

CMD ["node", "server.js"]
