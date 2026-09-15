import type { Metadata } from "next";
import { ClerkProvider } from "@clerk/nextjs";
import "./globals.css";
import { AppProviders } from "@/providers/app-providers";
import { env, isClerkConfigured } from "@/lib/config/env";

export const metadata: Metadata = {
  title: {
    default: "MULTI-SCOPE — Multi-Branch Deepfake Voice Analysis",
    template: "%s | MULTI-SCOPE",
  },
  description:
    "Research-oriented deepfake voice classification with branch-level evidence, score fusion, and explainable temporal and acoustic evidence.",
  applicationName: "MULTI-SCOPE",
  // `src/app/icon.svg` is picked up automatically by Next.js's file-based
  // metadata convention and emitted as the favicon link.
  openGraph: {
    type: "website",
    siteName: "MULTI-SCOPE",
    title: "MULTI-SCOPE — Explainable Multi-Branch Deepfake Voice Classification",
    description:
      "Analyse suspicious audio using multiple voice-classification branches, score-level fusion, and explainable temporal and acoustic evidence.",
  },
  twitter: {
    card: "summary",
    title: "MULTI-SCOPE — Explainable Multi-Branch Deepfake Voice Classification",
    description:
      "Decision-support evidence for deepfake voice analysis. Requires qualified human review.",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  const body = (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>
        <a href="#main-content" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-foreground focus:px-3 focus:py-2 focus:text-background">
          Skip to content
        </a>
        <AppProviders>{children}</AppProviders>
      </body>
    </html>
  );

  if (!isClerkConfigured) return body;

  return (
    <ClerkProvider
      publishableKey={env.clerkPublishableKey}
      signInUrl="/sign-in"
      signUpUrl="/sign-up"
      signInFallbackRedirectUrl="/dashboard"
      signUpFallbackRedirectUrl="/dashboard"
      appearance={{
        variables: {
          colorPrimary: "#000000",
          borderRadius: "8px",
        },
      }}
    >
      {body}
    </ClerkProvider>
  );
}
