import Link from "next/link";
import {
  ArrowRight,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { auth } from "@clerk/nextjs/server";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import {
  HeroReveal,
  ScrollReveal,
  ClassifierFlowVisual,
  HeroInteractivePreview,
  StartFromAnywhereVisual,
  EvidenceReportMockup,
  ScoreFusionMockup,
  DecisionBoundaryChart,
  EvidenceNodeNetwork,
  EvaluatorsCarousel,
} from "@/components/landing/landing-animations";
import { HeroBackground } from "@/components/landing/hero-background";
import {
  landingNav,
  heroTrustBadges,
  heroStatsBar,
  credibilityItems,
  startFromAnywhereLines,
  seriousResearchCards,
  seriousResearchCheckmarks,
  specializedBranchesStats,
  footerColumns,
} from "@/components/landing/landing-data";

export const metadata = {
  title: "MULTI-SCOPE — Multi-Branch Deepfake Voice Analysis",
  description:
    "Research-oriented deepfake voice classification with branch-level evidence, score fusion, and transparent model readiness in a high-contrast monochrome console.",
};

export default async function LandingPage() {
  const { userId } = await auth();

  return (
    <div className="min-h-screen bg-background text-foreground selection:bg-foreground selection:text-background">
      {/* =========================================================
          SECTION 1: HEADER & NAVIGATION (Screenshot Top Bar)
      ========================================================= */}
      <header className="sticky top-0 z-50 border-b border-border/80 bg-background/85 backdrop-blur-md">
        <nav className="container flex min-h-16 items-center justify-between gap-4" aria-label="Main navigation">
          <Link href="/" className="flex items-center gap-2 font-mono text-sm font-bold">
            <span className="flex h-6 w-6 items-center justify-center rounded bg-primary text-primary-foreground font-mono text-xs">
              ◈
            </span>
            <span>MULTI-SCOPE</span>
          </Link>

          <div className="hidden items-center gap-7 lg:flex">
            {landingNav.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="text-sm font-medium text-muted-foreground transition-colors hover:text-foreground"
              >
                {item.label}
              </Link>
            ))}
          </div>

          <div className="flex items-center gap-3">
            {!userId ? (
              <>
                <Link
                  href="/sign-in"
                  className="hidden sm:inline-block text-sm font-medium text-muted-foreground hover:text-foreground transition-colors px-3 py-1.5"
                >
                  Log In
                </Link>
                <Button
                  asChild
                  className="rounded-full bg-brand text-brand-foreground hover:bg-brand/90 px-5 py-2 text-sm font-medium transition-all"
                >
                  <Link href="/dashboard/analyze">Analyze Now</Link>
                </Button>
              </>
            ) : (
              <>
                <Link
                  href="/dashboard"
                  className="hidden sm:inline-block text-sm font-medium text-muted-foreground hover:text-foreground transition-colors px-3 py-1.5"
                >
                  Dashboard
                </Link>
                <Button
                  asChild
                  className="rounded-full bg-brand text-brand-foreground hover:bg-brand/90 px-5 py-2 text-sm font-medium transition-all"
                >
                  <Link href="/dashboard/analyze">Analyze Now</Link>
                </Button>
              </>
            )}
          </div>
        </nav>
      </header>

      <main id="main-content">
        {/* =========================================================
            SECTION 2: HERO SECTION WITH OVERLAPPING MOCKUP WINDOW
            "Credible voice analysis at the speed of AI."
        ========================================================= */}
        <section id="product" className="border-b border-border/80 relative overflow-hidden">
          <HeroBackground />
          <div className="container grid min-h-[calc(100vh-4.5rem)] items-center gap-12 py-16 lg:grid-cols-[1.02fr_1.18fr] lg:py-24 relative z-10">
            <HeroReveal>
              <div className="space-y-6">
                <Badge
                  data-hero-fade
                  variant="outline"
                  className="rounded-full border-border bg-background px-3.5 py-1 font-mono text-xs font-semibold uppercase text-foreground"
                >
                  ◈ MULTI-BRANCH DEEPFAKE VOICE ANALYSIS
                </Badge>

                <h1
                  data-split
                  className="max-w-xl text-5xl font-semibold sm:text-6xl lg:text-7xl leading-[1.06] text-foreground"
                >
                  Credible voice analysis at the speed of AI.
                </h1>

                <p
                  data-split
                  className="max-w-lg text-lg leading-relaxed text-muted-foreground"
                >
                  MULTI-SCOPE combines complementary acoustic, graph-based, self-supervised, and glottal analysis branches into a single transparent, explainable authenticity assessment.
                </p>

                <div data-hero-fade className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 pt-2">
                  {!userId ? (
                    <>
                      <Button
                        asChild
                        size="lg"
                        className="rounded-full bg-brand text-brand-foreground hover:bg-brand/90 px-7 py-6 text-base font-medium transition-all"
                      >
                        <Link href="/dashboard/analyze">
                          Analyze Now <ArrowRight className="ml-1 h-4 w-4" />
                        </Link>
                      </Button>
                      <Button
                        asChild
                        size="lg"
                        variant="outline"
                        className="rounded-full border-border bg-background hover:bg-muted px-6 py-6 text-base font-medium transition-all"
                      >
                        <Link href="/sign-in">Log In</Link>
                      </Button>
                    </>
                  ) : (
                    <>
                      <Button
                        asChild
                        size="lg"
                        className="rounded-full bg-brand text-brand-foreground hover:bg-brand/90 px-7 py-6 text-base font-medium transition-all"
                      >
                        <Link href="/dashboard/analyze">
                          Analyze Now <ArrowRight className="ml-1 h-4 w-4" />
                        </Link>
                      </Button>
                      <Button
                        asChild
                        size="lg"
                        variant="outline"
                        className="rounded-full border-border bg-background hover:bg-muted px-6 py-6 text-base font-medium transition-all"
                      >
                        <Link href="/dashboard">Dashboard</Link>
                      </Button>
                    </>
                  )}
                </div>

                <div
                  data-hero-fade
                  className="grid grid-cols-2 sm:grid-cols-4 gap-4 border-t border-border/80 pt-6 mt-8"
                >
                  {heroStatsBar.map((stat) => (
                    <div key={stat.label}>
                      <p className="font-mono text-xl font-bold text-foreground">{stat.value}</p>
                      <p className="text-xs text-muted-foreground mt-0.5">{stat.label}</p>
                    </div>
                  ))}
                </div>
              </div>
            </HeroReveal>

            <div className="relative">
              <HeroInteractivePreview />
            </div>
          </div>

          {/* Horizontal Bar: Trust Badges & Credibility Checklist below Hero */}
          <div className="border-t border-border/80 bg-muted/30 py-6">
            <div className="container space-y-6">
              <div className="flex flex-wrap items-center justify-center gap-4 sm:gap-8">
                {heroTrustBadges.map((badge) => (
                  <div
                    key={badge.label}
                    className="flex items-center gap-2 rounded-full border border-border/60 bg-background px-4 py-1.5 text-xs font-mono font-medium text-muted-foreground"
                  >
                    <badge.icon className="h-3.5 w-3.5 text-foreground" />
                    <span>{badge.label}</span>
                  </div>
                ))}
              </div>

              <div className="grid gap-4 sm:grid-cols-2 md:grid-cols-5 pt-2">
                {credibilityItems.map((item) => (
                  <div
                    key={item}
                    className="flex items-center gap-2.5 rounded-xl border border-border/70 bg-background p-3.5"
                  >
                    <CheckCircle2 className="h-4 w-4 shrink-0 text-foreground" />
                    <p className="text-xs font-medium text-foreground">{item}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* =========================================================
            SECTION 3: "Start from anywhere" Wide Feature Showcase Card
            (Matching Screenshot's light gray #f4f4f5 full-width container)
        ========================================================= */}
        <section id="start-from-anywhere" className="container py-20 lg:py-28">
          <ScrollReveal>
            <div className="rounded-3xl border border-border/90 bg-zinc-100 dark:bg-zinc-900/50 p-8 sm:p-12 lg:p-16">
              <div className="grid gap-12 lg:grid-cols-[1.1fr_0.9fr] items-center">
                <div className="order-2 lg:order-1">
                  <StartFromAnywhereVisual />
                </div>

                <div className="space-y-6 order-1 lg:order-2">
                  <Badge
                    variant="outline"
                    className="rounded-full border-border bg-background px-3 py-1 font-mono text-xs uppercase"
                  >
                    ◈ FLEXIBLE AUDIO ANALYSIS
                  </Badge>

                  <h2 className="text-4xl sm:text-5xl font-semibold text-foreground">
                    Start from anywhere
                  </h2>

                  <div className="space-y-3">
                    {startFromAnywhereLines.map((line) => (
                      <p
                        key={line}
                        className="text-xl sm:text-2xl font-semibold text-muted-foreground"
                      >
                        {line}
                      </p>
                    ))}
                  </div>

                  <Separator className="my-4" />

                  <p className="text-sm leading-relaxed text-muted-foreground">
                    Standardized 16kHz audio preprocessing pipeline across all branches. Supports WAV, FLAC, and MP3 files with instant dummy/real model readiness verification.
                  </p>

                  <div className="pt-2">
                    <Button
                      asChild
                      variant="outline"
                      className="rounded-full border-border bg-background hover:bg-muted px-6 py-5 text-sm font-medium"
                    >
                      <Link href="/dashboard/analyze">Upload Voice Recording →</Link>
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </ScrollReveal>
        </section>

        {/* =========================================================
            SECTION 4: "Built for serious research."
            2 Large Cards + 4 Column Checkmarks below
        ========================================================= */}
        <section id="serious-research" className="border-y border-border/80 bg-muted/30 py-20 lg:py-28">
          <div className="container space-y-12">
            <div className="grid gap-6 lg:grid-cols-[1fr_0.8fr] items-end justify-between">
              <div>
                <Badge
                  variant="outline"
                  className="rounded-full border-border bg-background px-3.5 py-1 font-mono text-xs uppercase"
                >
                  ◈ ARCHITECTURAL EXCELLENCE
                </Badge>
                <h2 className="mt-4 text-4xl sm:text-5xl font-semibold text-foreground">
                  Built for serious research.
                </h2>
              </div>
              <p className="text-base text-muted-foreground lg:text-right">
                Designed for security teams, audio forensics, and academic researchers requiring verifiable AI transparency and zero hidden dummy bias.
              </p>
            </div>

            {/* 2-Column Showcase Cards */}
            <ScrollReveal className="grid gap-8 md:grid-cols-2">
              {seriousResearchCards.map((card, index) => (
                <div
                  key={card.number}
                  className="rounded-3xl border border-border/80 bg-background p-8 flex flex-col justify-between space-y-6 hover:border-foreground/30 transition-colors"
                >
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs font-semibold uppercase text-muted-foreground">
                        {card.badge}
                      </span>
                      <span className="rounded-full border border-border bg-muted px-2.5 py-0.5 font-mono text-xs font-bold">
                        0{index + 1}
                      </span>
                    </div>
                    <h3 className="text-2xl font-semibold text-foreground">
                      {card.title}
                    </h3>
                    <p className="text-sm leading-relaxed text-muted-foreground">
                      {card.description}
                    </p>
                  </div>

                  <div className="pt-2">
                    {index === 0 ? <EvidenceReportMockup /> : <ScoreFusionMockup />}
                  </div>
                </div>
              ))}
            </ScrollReveal>

            {/* 4-Column Checkmark Statistics List */}
            <ScrollReveal className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 pt-6 border-t border-border/80">
              {seriousResearchCheckmarks.map((item) => (
                <div key={item.title} className="space-y-2">
                  <p className="font-semibold text-foreground text-sm flex items-center gap-1.5 font-mono">
                    {item.title}
                  </p>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {item.description}
                  </p>
                </div>
              ))}
            </ScrollReveal>
          </div>
        </section>

        {/* =========================================================
            SECTION 5: "Four specialized branches for complete audio coverage."
            Horizontal Cards Carousel + Architecture Flow
        ========================================================= */}
        <section id="branches" className="container py-20 lg:py-28 space-y-12">
          <div className="grid gap-6 lg:grid-cols-[1fr_0.9fr] items-end justify-between">
            <div>
              <Badge
                variant="outline"
                className="rounded-full border-border bg-background px-3.5 py-1 font-mono text-xs uppercase"
              >
                ◈ FOUR AI BRANCHES
              </Badge>
              <h2 className="mt-4 text-4xl sm:text-5xl font-semibold text-foreground">
                Four specialized branches for complete audio coverage.
              </h2>
            </div>
            <p className="text-base text-muted-foreground lg:text-right">
              MULTI-SCOPE evaluates spectral cepstrum, spectro-temporal graphs, self-supervised representations, and vocal source glottal flows in parallel.
            </p>
          </div>

          <ScrollReveal className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
            {/* White Showcase Card: Real research outcomes, fast. */}
            <div className="rounded-3xl border border-border bg-card p-8 lg:p-10 flex flex-col justify-between space-y-8">
              <div className="space-y-4">
                <span className="font-mono text-xs uppercase text-muted-foreground">
                  RESEARCH BENCHMARKS
                </span>
                <h3 className="text-3xl font-semibold text-foreground">
                  Real research outcomes, fast.
                </h3>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  Every prediction generates an inspectable breakdown of branch evidence, timestamped model latency, and audio preprocessing metadata.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-6 pt-4 border-t border-border/80">
                {specializedBranchesStats.map((stat) => (
                  <div key={stat.label} className="space-y-1">
                    <stat.icon className="h-4 w-4 text-muted-foreground mb-1" />
                    <p className="font-mono text-3xl font-bold text-foreground">{stat.value}</p>
                    <p className="text-xs text-muted-foreground">{stat.label}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Architecture Pipeline Visual Card */}
            <div className="rounded-3xl border border-border bg-card p-8 flex flex-col justify-between space-y-6">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs uppercase text-muted-foreground">
                  STANDARD PROCESSING CONTRACT
                </span>
                <span className="font-mono text-xs text-muted-foreground">16 kHz · Resampled</span>
              </div>

              <div className="py-2">
                <ClassifierFlowVisual />
              </div>

              <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border pt-4 text-xs text-muted-foreground">
                <span>Parallel model execution with score-level fusion</span>
                <Link
                  href="/developers/api"
                  className="font-mono text-foreground underline underline-offset-4 hover:opacity-80"
                >
                  View REST & MCP Schema →
                </Link>
              </div>
            </div>
          </ScrollReveal>

          {/* Carousel Navigation Footer matching screenshot */}
          <div className="flex items-center justify-between border-t border-border/80 pt-6">
            <span className="font-mono text-xs text-muted-foreground">
              See all 4 analysis branches · Research documentation & OpenAPI spec
            </span>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="icon" className="h-9 w-9 rounded-full border-border" aria-label="Previous branch card">
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <Button variant="outline" size="icon" className="h-9 w-9 rounded-full border-border" aria-label="Next branch card">
                <ChevronRight className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </section>

        {/* =========================================================
            SECTION 6: "Your reputation is on the evidence."
            Split Visual Card (Left: Dark Chart | Right: Light Network)
        ========================================================= */}
        <section id="transparency" className="border-y border-border/80 bg-muted/20 py-20 lg:py-28">
          <div className="container space-y-12">
            <div className="grid gap-6 lg:grid-cols-[1fr_0.9fr] items-end justify-between">
              <div>
                <Badge
                  variant="outline"
                  className="rounded-full border-border bg-background px-3.5 py-1 font-mono text-xs uppercase"
                >
                  ◈ EXPLAINABLE AI
                </Badge>
                <h2 className="mt-4 text-4xl sm:text-5xl lg:text-6xl font-semibold text-foreground">
                  Your reputation is on the evidence.
                </h2>
              </div>
              <p className="text-base text-muted-foreground lg:text-right">
                When presenting audio authenticity findings to stakeholders, you need transparent branch-level proof—not a black-box percentage.
              </p>
            </div>

            {/* Split Visual Card (50% Dark Graphite / 50% Silver Light) */}
            <ScrollReveal>
              <div className="rounded-3xl border border-border/90 overflow-hidden grid md:grid-cols-2">
                {/* Left Half: High-Contrast Dark Graphite Theme */}
                <div className="bg-zinc-950 text-white p-6 lg:p-10 flex flex-col justify-between border-b md:border-b-0 md:border-r border-zinc-800">
                  <DecisionBoundaryChart />
                </div>

                {/* Right Half: Clean Light Silver Theme */}
                <div className="bg-zinc-100 dark:bg-zinc-900/80 text-foreground p-6 lg:p-10 flex flex-col justify-between">
                  <EvidenceNodeNetwork />
                </div>
              </div>
            </ScrollReveal>
          </div>
        </section>

        {/* =========================================================
            SECTION 7: "Hear it from our research evaluators"
            Testimonials Carousel matching Screenshot
        ========================================================= */}
        <section id="evaluators" className="container py-20 lg:py-28 space-y-12">
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-6 border-b border-border/80 pb-6">
            <div>
              <Badge
                variant="outline"
                className="rounded-full border-border bg-background px-3.5 py-1 font-mono text-xs uppercase"
              >
                ◈ ACADEMIC & FORENSIC FEEDBACK
              </Badge>
              <h2 className="mt-4 text-3xl sm:text-4xl lg:text-5xl font-semibold text-foreground">
                Hear it from our research evaluators
              </h2>
            </div>
            <p className="text-sm text-muted-foreground font-mono">
              Independent evaluations across audio processing and security
            </p>
          </div>

          <ScrollReveal>
            <EvaluatorsCarousel />
          </ScrollReveal>
        </section>

        {/* =========================================================
            SECTION 8: BOTTOM CTA BANNER
            "Conduct voice analysis you can trust."
        ========================================================= */}
        <section id="cta" className="border-t border-border/80 bg-zinc-100 dark:bg-zinc-900/50 py-24 lg:py-32">
          <div className="container max-w-3xl text-center space-y-6">
            <Badge
              variant="outline"
              className="rounded-full border-border bg-background px-3.5 py-1 font-mono text-xs uppercase"
            >
              ◈ START INSPECTING AUDIO
            </Badge>

            <h2 className="text-4xl sm:text-5xl lg:text-6xl font-semibold text-foreground leading-tight">
              Conduct voice analysis you can trust.
            </h2>

            <p className="text-lg text-muted-foreground max-w-xl mx-auto leading-relaxed">
              Deploy explainable multi-branch deepfake voice classification with research-grade transparency, score-level fusion, and zero dummy bias.
            </p>

            <div className="pt-4 flex flex-col sm:flex-row items-center justify-center gap-4">
              <Button
                asChild
                size="lg"
                className="rounded-full bg-brand text-brand-foreground hover:bg-brand/90 px-8 py-6 text-base font-medium transition-all"
              >
                <Link href="/dashboard/analyze">
                  Start Analysis <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
              <Button
                asChild
                size="lg"
                variant="outline"
                className="rounded-full border-border bg-background hover:bg-muted px-7 py-6 text-base font-medium transition-all"
              >
                <Link href="/dashboard/developer">API & MCP Integration</Link>
              </Button>
            </div>

            <div className="pt-6 flex flex-wrap items-center justify-center gap-6 font-mono text-xs text-muted-foreground">
              <span>✓ No live-stream claims</span>
              <span>✓ Standalone or MCP server</span>
              <span>✓ Standardized 16kHz</span>
            </div>
          </div>
        </section>
      </main>

      {/* =========================================================
          SECTION 9: MINIMALIST BLACK & WHITE FOOTER
      ========================================================= */}
      <footer className="border-t border-border bg-background py-14">
        <div className="container grid gap-10 lg:grid-cols-[1.3fr_1.7fr]">
          <div className="space-y-4">
            <Link href="/" className="flex items-center gap-2 font-mono text-sm font-bold">
              <span className="flex h-6 w-6 items-center justify-center rounded bg-primary text-primary-foreground font-mono text-xs">
                ◈
              </span>
              <span>MULTI-SCOPE</span>
            </Link>
            <p className="max-w-sm text-sm text-muted-foreground leading-relaxed">
              Explainable multi-branch deepfake voice classification research system developed to investigate reliable and interpretable synthetic speech detection.
            </p>
            <div className="flex items-center gap-3 font-mono text-xs text-muted-foreground pt-2">
              <span>SLIIT Research</span>
              <span>·</span>
              <span>ID: IT22183668</span>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-8">
            {footerColumns.map((column) => (
              <div key={column.title} className="space-y-3">
                <h3 className="font-mono text-xs font-semibold uppercase text-foreground">
                  {column.title}
                </h3>
                <ul className="space-y-2.5">
                  {column.links.map((link) => (
                    <li key={link.href}>
                      <Link
                        href={link.href}
                        className="text-sm text-muted-foreground hover:text-foreground transition-colors"
                      >
                        {link.label}
                      </Link>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>

        <Separator className="container my-10" />

        <div className="container flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between text-xs text-muted-foreground font-mono">
          <span>© 2026 MULTI-SCOPE Research Project. All rights reserved.</span>
          <span>Experimental academic system. Not conclusive forensic, legal, financial, or identity-verification evidence.</span>
        </div>
      </footer>
    </div>
  );
}
