"use client";

import { useRef, useState } from "react";
import { useGSAP } from "@gsap/react";
import gsap from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { Observer } from "gsap/Observer";
import { Draggable } from "gsap/Draggable";
import { SplitText } from "gsap/SplitText";
import { DrawSVGPlugin } from "gsap/DrawSVGPlugin";
import { InertiaPlugin } from "gsap/InertiaPlugin";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { branchCards, evaluatorTestimonials } from "@/components/landing/landing-data";

const hasReducedMotion = () =>
  typeof window !== "undefined" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

let pluginsRegistered = false;

function registerLandingPlugins() {
  if (pluginsRegistered || typeof window === "undefined") return;
  gsap.registerPlugin(
    ScrollTrigger,
    Observer,
    Draggable,
    SplitText,
    DrawSVGPlugin,
    InertiaPlugin,
  );
  pluginsRegistered = true;
}

export function HeroReveal({ children }: { children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement | null>(null);

  useGSAP(
    () => {
      registerLandingPlugins();
      if (hasReducedMotion() || !ref.current) return;
      const split = new SplitText(ref.current.querySelectorAll("[data-split]"), {
        type: "lines,words",
        linesClass: "overflow-hidden",
      });
      gsap.from(split.words, {
        yPercent: 110,
        opacity: 0,
        duration: 0.7,
        stagger: 0.025,
        ease: "power3.out",
      });
      gsap.from(ref.current.querySelectorAll("[data-hero-fade]"), {
        y: 16,
        opacity: 0,
        duration: 0.55,
        stagger: 0.08,
        delay: 0.2,
        ease: "power2.out",
      });
      return () => split.revert();
    },
    { scope: ref },
  );

  return <div ref={ref}>{children}</div>;
}

export function ScrollReveal({ children, className }: { children: React.ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement | null>(null);
  useGSAP(
    () => {
      registerLandingPlugins();
      if (hasReducedMotion() || !ref.current) return;
      gsap.from(ref.current.children, {
        scrollTrigger: {
          trigger: ref.current,
          start: "top 82%",
        },
        y: 22,
        opacity: 0,
        duration: 0.55,
        stagger: 0.08,
        ease: "power2.out",
      });
    },
    { scope: ref },
  );
  return (
    <div ref={ref} className={className}>
      {children}
    </div>
  );
}

export function ClassifierFlowVisual() {
  const ref = useRef<HTMLDivElement | null>(null);
  useGSAP(
    () => {
      registerLandingPlugins();
      if (hasReducedMotion() || !ref.current) return;
      const paths = ref.current.querySelectorAll("[data-flow-path]");
      gsap.set(paths, { drawSVG: "0%" });
      gsap.timeline({ delay: 0.15 })
        .to(paths, { drawSVG: "100%", duration: 1.1, stagger: 0.08, ease: "power2.out" })
        .from(ref.current.querySelectorAll("[data-flow-node]"), {
          scale: 0.96,
          opacity: 0,
          duration: 0.35,
          stagger: 0.04,
          ease: "power2.out",
        }, "<0.15");
    },
    { scope: ref },
  );

  const nodes = [
    { x: 22, y: 88, label: "Audio" },
    { x: 180, y: 88, label: "Preprocess" },
    { x: 362, y: 32, label: "LFCC" },
    { x: 362, y: 72, label: "AASIST" },
    { x: 362, y: 112, label: "SSL" },
    { x: 362, y: 152, label: "Glottal" },
    { x: 558, y: 88, label: "Fusion" },
    { x: 724, y: 88, label: "Result" },
  ];

  return (
    <div ref={ref} className="rounded-lg border bg-card p-4" aria-label="Classifier flow diagram">
      <svg viewBox="0 0 850 210" role="img" className="h-auto w-full">
        <title>Audio flows through preprocessing, four branches, fusion, and result</title>
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
            <path d="M0,0 L8,4 L0,8 Z" fill="currentColor" />
          </marker>
        </defs>
        <g fill="none" stroke="currentColor" strokeWidth="1.5" markerEnd="url(#arrow)" opacity="0.75">
          <path data-flow-path d="M122 106 H180" />
          <path data-flow-path d="M282 106 C320 106 318 50 362 50" />
          <path data-flow-path d="M282 106 C320 106 318 90 362 90" />
          <path data-flow-path d="M282 106 C320 106 318 130 362 130" />
          <path data-flow-path d="M282 106 C320 106 318 170 362 170" />
          <path data-flow-path d="M482 50 C520 50 520 106 558 106" />
          <path data-flow-path d="M482 90 C520 90 520 106 558 106" />
          <path data-flow-path d="M482 130 C520 130 520 106 558 106" />
          <path data-flow-path d="M482 170 C520 170 520 106 558 106" />
          <path data-flow-path d="M658 106 H724" />
        </g>
        {nodes.map((node) => (
          <g key={node.label} data-flow-node>
            <rect x={node.x} y={node.y - 18} width="104" height="36" rx="8" fill="hsl(var(--background))" stroke="currentColor" />
            <text x={node.x + 52} y={node.y + 4} textAnchor="middle" className="fill-current font-mono text-[12px]">
              {node.label}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

export function BranchEvidenceExplorer() {
  const [index, setIndex] = useState(0);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const trackRef = useRef<HTMLDivElement | null>(null);

  const moveTo = (nextIndex: number) => {
    const bounded = Math.max(0, Math.min(nextIndex, branchCards.length - 1));
    setIndex(bounded);
    if (trackRef.current) {
      gsap.to(trackRef.current, {
        xPercent: -bounded * 100,
        duration: hasReducedMotion() ? 0 : 0.45,
        ease: "power2.out",
      });
    }
  };

  useGSAP(
    () => {
      registerLandingPlugins();
      if (hasReducedMotion() || !rootRef.current || !trackRef.current) return;
      const draggable = Draggable.create(trackRef.current, {
        type: "x",
        inertia: true,
        bounds: rootRef.current,
        onDragEnd() {
          const width = rootRef.current?.clientWidth || 1;
          moveTo(Math.round(Math.abs(this.x) / width));
        },
      });
      const observer = Observer.create({
        target: rootRef.current,
        type: "wheel,touch,pointer",
        tolerance: 18,
        preventDefault: false,
        onLeft: () => moveTo(index + 1),
        onRight: () => moveTo(index - 1),
      });
      return () => {
        draggable.forEach((item) => item.kill());
        observer.kill();
      };
    },
    { scope: rootRef, dependencies: [index] },
  );

  return (
    <div className="space-y-4">
      <div ref={rootRef} className="overflow-hidden rounded-lg border bg-card" aria-label="Branch evidence explorer">
        <div ref={trackRef} className="flex touch-pan-y">
          {branchCards.map((branch) => (
            <article key={branch.title} className="min-w-full p-6">
              <p className="font-mono text-xs uppercase text-muted-foreground">{branch.label}</p>
              <h3 className="mt-4 text-2xl font-semibold">{branch.title}</h3>
              <p className="mt-3 max-w-xl text-sm text-muted-foreground">{branch.description}</p>
            </article>
          ))}
        </div>
      </div>
      <div className="flex items-center justify-between">
        <Button type="button" variant="outline" onClick={() => moveTo(index - 1)} disabled={index === 0}>
          <ChevronLeft className="h-4 w-4" /> Previous
        </Button>
        <p className="font-mono text-xs text-muted-foreground">{index + 1} / {branchCards.length}</p>
        <Button type="button" variant="outline" onClick={() => moveTo(index + 1)} disabled={index === branchCards.length - 1}>
          Next <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}

export function CodeReveal({ children }: { children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement | null>(null);
  useGSAP(
    () => {
      registerLandingPlugins();
      if (hasReducedMotion() || !ref.current) return;
      gsap.from(ref.current.querySelectorAll("[data-code-line]"), {
        scrollTrigger: { trigger: ref.current, start: "top 88%" },
        opacity: 0,
        y: 8,
        duration: 0.35,
        stagger: 0.04,
      });
    },
    { scope: ref },
  );
  return <div ref={ref}>{children}</div>;
}

export function HeroInteractivePreview() {
  const [activeStep, setActiveStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  const steps = [
    { id: 1, label: "1. Upload Audio", tag: "WAV / FLAC / MP3", desc: "Audio clip #019 verified and queued." },
    { id: 2, label: "2. 16kHz Preprocess", tag: "16kHz Resampled", desc: "Shared preprocessing contract ready." },
    { id: 3, label: "3. 4 AI Branches", tag: "LFCC / SSL / AASIST", desc: "Parallel spectral and graph evaluation." },
    { id: 4, label: "4. Score Fusion", tag: "64.0% Synthetic", desc: "Score-level fusion with real model status." },
  ];

  useGSAP(
    () => {
      registerLandingPlugins();
      if (hasReducedMotion() || !rootRef.current) return;
      gsap.from(rootRef.current.querySelectorAll("[data-preview-item]"), {
        y: 12,
        opacity: 0,
        duration: 0.45,
        stagger: 0.08,
        ease: "power2.out",
      });
    },
    { scope: rootRef, dependencies: [activeStep] },
  );

  return (
    <div
      ref={rootRef}
      className="relative mx-auto max-w-2xl rounded-2xl border border-border/80 bg-card overflow-hidden"
      aria-label="Interactive Hero Preview Window"
    >
      <div className="flex items-center justify-between border-b border-border bg-muted/50 px-4 py-3">
        <div className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-zinc-400 dark:bg-zinc-600 inline-block" />
          <span className="h-2.5 w-2.5 rounded-full bg-zinc-300 dark:bg-zinc-700 inline-block" />
          <span className="h-2.5 w-2.5 rounded-full bg-zinc-300 dark:bg-zinc-700 inline-block" />
        </div>
        <span className="font-mono text-xs font-medium text-muted-foreground">
          How it works — MULTI-SCOPE Console
        </span>
        <span className="font-mono text-[11px] uppercase text-muted-foreground bg-background border px-2 py-0.5 rounded">
          {steps[activeStep].tag}
        </span>
      </div>

      <div className="p-6 lg:p-8 relative min-h-[360px] flex flex-col justify-between bg-background">
        <div className="grid gap-6 md:grid-cols-[1.2fr_1fr] items-center">
          <div data-preview-item className="space-y-5">
            <div className="flex items-center justify-between">
              <div>
                <span className="font-mono text-xs uppercase text-muted-foreground">Input Recording</span>
                <p className="font-mono text-sm font-semibold mt-0.5">voice_sample_019.wav</p>
              </div>
              <button
                type="button"
                onClick={() => setIsPlaying(!isPlaying)}
                className="flex h-9 w-9 items-center justify-center rounded-full bg-primary text-primary-foreground hover:bg-primary/90 transition-transform active:scale-95"
                aria-label={isPlaying ? "Pause sample" : "Play sample"}
              >
                {isPlaying ? <span className="font-mono text-xs font-bold">❚❚</span> : <span className="font-mono text-xs font-bold">►</span>}
              </button>
            </div>

            <div className="rounded-xl border border-border bg-background p-4 space-y-3">
              <div className="flex items-end justify-between gap-1 h-14 px-2">
                {[42, 68, 85, 30, 95, 75, 40, 90, 60, 35, 80, 50, 70, 45, 88, 65, 38, 72].map((height, i) => (
                  <span
                    key={i}
                    className="w-1.5 rounded-full bg-primary/80 transition-all duration-300"
                    style={{
                      height: isPlaying ? `${Math.max(15, (height * ((i % 3) + 1)) % 100)}%` : `${height}%`,
                      opacity: activeStep >= 1 ? 1 : 0.4,
                    }}
                  />
                ))}
              </div>
              <div className="flex items-center justify-between font-mono text-[11px] text-muted-foreground border-t pt-2">
                <span>00:04 / 00:12</span>
                <span>16 kHz · Single Channel</span>
              </div>
            </div>

            <div className="rounded-lg border bg-muted/40 px-3.5 py-2.5 flex items-center justify-between text-xs">
              <span className="font-medium text-muted-foreground">{steps[activeStep].desc}</span>
              <span className="font-mono text-[11px] font-semibold underline underline-offset-2">Step {activeStep + 1}/4</span>
            </div>
          </div>

          <div
            data-preview-item
            className="rounded-2xl border border-border/90 bg-background/95 backdrop-blur-md p-5 space-y-4 relative z-10 hover:border-foreground/30 transition-colors"
          >
            <div className="flex items-center justify-between border-b pb-3">
              <div>
                <span className="font-mono text-[10px] uppercase text-muted-foreground">FUSED OUTPUT</span>
                <p className="text-sm font-semibold">Voice Authenticity</p>
              </div>
              <span className="rounded-full bg-brand text-brand-foreground font-mono text-xs font-bold px-2.5 py-1">
                64.0%
              </span>
            </div>

            <div className="space-y-2.5 text-xs">
              <div className="flex items-center justify-between rounded-lg border bg-muted/30 px-3 py-2">
                <span className="font-medium">LFCC CNN/TCN</span>
                <span className="font-mono text-[11px] text-muted-foreground">0.68 · Real</span>
              </div>
              <div className="flex items-center justify-between rounded-lg border bg-muted/30 px-3 py-2">
                <span className="font-medium">AASIST Graph</span>
                <span className="font-mono text-[11px] text-muted-foreground">0.56 · Dummy</span>
              </div>
              <div className="flex items-center justify-between rounded-lg border bg-muted/30 px-3 py-2">
                <span className="font-medium">SSL Sequence</span>
                <span className="font-mono text-[11px] text-muted-foreground">0.63 · Real</span>
              </div>
              <div className="flex items-center justify-between rounded-lg border bg-muted/30 px-3 py-2 opacity-60">
                <span className="font-medium">Glottal Source</span>
                <span className="font-mono text-[11px] text-muted-foreground">Skipped</span>
              </div>
            </div>

            <div className="border-t pt-3 flex items-center justify-between text-[11px] text-muted-foreground">
              <span>Mode: Score-Level Fusion</span>
              <span className="font-mono font-medium text-foreground">Verified</span>
            </div>
          </div>
        </div>

        <div className="mt-8 flex flex-wrap items-center justify-between gap-2 border-t pt-4">
          <div className="flex items-center gap-1.5 overflow-x-auto">
            {steps.map((step, idx) => (
              <button
                key={step.id}
                type="button"
                onClick={() => setActiveStep(idx)}
                className={`px-3 py-1.5 text-xs font-mono font-medium rounded-md transition-colors ${
                  activeStep === idx
                    ? "bg-international-orange-600 text-white"
                    : "bg-muted text-muted-foreground hover:text-foreground"
                }`}
              >
                {step.label}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="sm"
              className="h-8 w-8 p-0"
              onClick={() => setActiveStep((prev) => (prev > 0 ? prev - 1 : steps.length - 1))}
              aria-label="Previous preview step"
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-8 w-8 p-0"
              onClick={() => setActiveStep((prev) => (prev < steps.length - 1 ? prev + 1 : 0))}
              aria-label="Next preview step"
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

export function StartFromAnywhereVisual() {
  return (
    <div className="relative overflow-hidden rounded-2xl border border-border/80 bg-background p-6 lg:p-8">
      <div className="space-y-4">
        <div className="flex items-center justify-between border-b pb-3">
          <div className="flex items-center gap-2">
            <span className="h-3 w-3 rounded-full bg-primary inline-block" />
            <span className="font-mono text-xs font-semibold uppercase">Multi-Branch Sample Vault</span>
          </div>
          <span className="font-mono text-xs text-muted-foreground">Standardized 16kHz</span>
        </div>

        <div className="space-y-3">
          <div className="rounded-xl border border-border bg-card p-4 transition-transform hover:-translate-y-0.5">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-mono text-xs text-muted-foreground">UPLOADED FILE</p>
                <p className="font-semibold text-sm mt-0.5">recording_04_eval.wav</p>
              </div>
              <span className="rounded-full border bg-muted px-3 py-1 font-mono text-xs font-semibold">
                64.0% Synthetic
              </span>
            </div>
            <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground border-t pt-2.5">
              <span>LFCC (0.68) + SSL (0.63) + AASIST (0.56)</span>
              <span className="font-mono">16 kHz · 4.2s</span>
            </div>
          </div>

          <div className="rounded-xl border border-border bg-card p-4 transition-transform hover:-translate-y-0.5">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-mono text-xs text-muted-foreground">BROWSER RECORDING</p>
                <p className="font-semibold text-sm mt-0.5">browser_clip_02.flac</p>
              </div>
              <span className="rounded-full border border-foreground/20 bg-background px-3 py-1 font-mono text-xs font-semibold text-foreground">
                12.4% Bonafide
              </span>
            </div>
            <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground border-t pt-2.5">
              <span>Score-Level Fusion · Real Model</span>
              <span className="font-mono">16 kHz · 6.1s</span>
            </div>
          </div>
        </div>

        <div className="rounded-lg border bg-muted/40 p-3 text-center font-mono text-xs text-muted-foreground">
          ✓ Standardized audio preprocessing contract verified across all branches
        </div>
      </div>
    </div>
  );
}

export function EvidenceReportMockup() {
  return (
    <div className="rounded-2xl border border-border bg-card p-5 space-y-4">
      <div className="flex items-center justify-between border-b pb-3">
        <span className="font-mono text-xs font-semibold uppercase text-muted-foreground">EVIDENCE REPORT #1042</span>
        <span className="rounded bg-brand/10 text-brand font-mono text-xs font-bold px-2 py-0.5">
          FUSED SCORE
        </span>
      </div>

      <div className="space-y-3">
        <div>
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="font-medium">LFCC CNN/TCN (Spectral)</span>
            <span className="font-mono font-semibold">68.0%</span>
          </div>
          <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
            <div className="h-full rounded-full bg-international-orange-500 w-[68%]" />
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="font-medium">AASIST (Graph Branch)</span>
            <span className="font-mono font-semibold">56.0%</span>
          </div>
          <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
            <div className="h-full rounded-full bg-international-orange-400 w-[56%]" />
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="font-medium">SSL Sequence (Representation)</span>
            <span className="font-mono font-semibold">63.0%</span>
          </div>
          <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
            <div className="h-full rounded-full bg-international-orange-600 w-[63%]" />
          </div>
        </div>
      </div>

      <div className="rounded-lg border bg-muted/40 px-3.5 py-2.5 flex items-center justify-between text-xs font-mono">
        <span>Branch Weight: 0.35 / 0.35 / 0.30</span>
        <span className="font-semibold underline">Real Model Status</span>
      </div>
    </div>
  );
}

export function ScoreFusionMockup() {
  return (
    <div className="rounded-2xl border border-border bg-card p-5 space-y-4">
      <div className="flex items-center justify-between border-b pb-3">
        <span className="font-mono text-xs font-semibold uppercase text-muted-foreground">SCORE-LEVEL FUSION MATRIX</span>
        <span className="rounded-full bg-foreground text-background font-mono text-[11px] font-bold px-2.5 py-0.5">
          ACTIVE
        </span>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center font-mono text-xs">
        <div className="rounded-xl border bg-muted/30 p-3 space-y-1">
          <p className="text-[10px] text-muted-foreground">LFCC</p>
          <p className="text-lg font-bold">0.35</p>
          <p className="text-[10px] uppercase text-muted-foreground">Weight</p>
        </div>
        <div className="rounded-xl border bg-muted/30 p-3 space-y-1">
          <p className="text-[10px] text-muted-foreground">AASIST</p>
          <p className="text-lg font-bold">0.35</p>
          <p className="text-[10px] uppercase text-muted-foreground">Weight</p>
        </div>
        <div className="rounded-xl border bg-muted/30 p-3 space-y-1">
          <p className="text-[10px] text-muted-foreground">SSL</p>
          <p className="text-lg font-bold">0.30</p>
          <p className="text-[10px] uppercase text-muted-foreground">Weight</p>
        </div>
      </div>

      <div className="rounded-lg border border-border bg-background p-3.5 flex items-center justify-between text-xs">
        <div>
          <p className="font-semibold">Dummy Mode Isolation</p>
          <p className="text-muted-foreground text-[11px] mt-0.5">Placeholder models flagged in metadata</p>
        </div>
        <span className="rounded border bg-muted px-2 py-1 font-mono text-[11px] font-bold">
          0% Hidden Bias
        </span>
      </div>
    </div>
  );
}

export function DecisionBoundaryChart() {
  return (
    <div className="w-full h-full min-h-[340px] flex flex-col justify-between p-2">
      <div className="flex items-center justify-between border-b border-zinc-800 pb-3">
        <div>
          <span className="font-mono text-[10px] uppercase text-zinc-400">RESEARCH BENCHMARK</span>
          <h4 className="text-sm font-semibold text-white mt-0.5">Spoof vs. Bonafide Decision Boundary</h4>
        </div>
        <span className="rounded border border-zinc-700 bg-zinc-900 px-2.5 py-1 font-mono text-xs text-zinc-300">
          AUROC 98.4%
        </span>
      </div>

      <div className="relative my-4 h-48 w-full rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
        <div className="absolute inset-4 grid grid-cols-4 grid-rows-3 gap-0 border-l border-b border-zinc-700/60">
          {[...Array(12)].map((_, i) => (
            <div key={i} className="border-r border-t border-zinc-800/40" />
          ))}
        </div>

        <div className="absolute inset-0 flex items-center justify-center">
          <div className="relative w-[78%] h-[74%]">
            <span className="absolute top-2 left-6 h-3 w-3 rounded-full bg-white" />
            <span className="absolute top-8 left-16 h-2.5 w-2.5 rounded-full bg-zinc-400" />
            <span className="absolute top-12 left-28 h-3 w-3 rounded-full bg-white" />
            <span className="absolute bottom-6 right-10 h-3 w-3 rounded-full bg-zinc-500 border border-white/60" />
            <span className="absolute bottom-10 right-20 h-2.5 w-2.5 rounded-full bg-zinc-400" />
            <span className="absolute bottom-14 right-8 h-3 w-3 rounded-full bg-zinc-600" />
            <div className="absolute inset-0 border-b-2 border-dashed border-white/40 -rotate-12 transform origin-center" />
          </div>
        </div>

        <div className="absolute bottom-1 right-3 font-mono text-[10px] text-zinc-400">
          Spectral Feature Dimension →
        </div>
      </div>

      <div className="flex items-center justify-between text-xs text-zinc-400 font-mono">
        <span>● Synthetic Cluster</span>
        <span>○ Bonafide Speech</span>
        <span>--- Decision Threshold (0.50)</span>
      </div>
    </div>
  );
}

export function EvidenceNodeNetwork() {
  return (
    <div className="w-full h-full min-h-[340px] flex flex-col justify-between p-2">
      <div className="flex items-center justify-between border-b border-border pb-3">
        <div>
          <span className="font-mono text-[10px] uppercase text-muted-foreground">MULTI-BRANCH TOPOLOGY</span>
          <h4 className="text-sm font-semibold text-foreground mt-0.5">Parallel Branch Score Synthesis</h4>
        </div>
        <span className="rounded-full bg-brand text-brand-foreground px-2.5 py-1 font-mono text-xs font-bold">
          4 Branches
        </span>
      </div>

      <div className="relative my-4 h-48 w-full flex items-center justify-center">
        <div className="relative flex items-center justify-center w-full max-w-sm">
          <div className="absolute h-px w-36 bg-border -rotate-45" />
          <div className="absolute h-px w-36 bg-border rotate-45" />
          <div className="absolute h-px w-44 bg-border" />

          <div className="z-10 flex flex-col items-center justify-center rounded-2xl border-2 border-primary bg-background px-4 py-3">
            <span className="font-mono text-[10px] text-muted-foreground uppercase">FUSED LABEL</span>
            <span className="text-sm font-bold mt-0.5">64.0% Synthetic</span>
          </div>

          <div className="absolute -left-2 -top-2 rounded-xl border bg-card px-2.5 py-1.5 text-center">
            <span className="font-mono text-[10px] text-muted-foreground block">LFCC Spectral</span>
            <span className="text-xs font-semibold">0.68</span>
          </div>

          <div className="absolute -right-2 -top-2 rounded-xl border bg-card px-2.5 py-1.5 text-center">
            <span className="font-mono text-[10px] text-muted-foreground block">SSL Sequence</span>
            <span className="text-xs font-semibold">0.63</span>
          </div>

          <div className="absolute -left-2 -bottom-2 rounded-xl border bg-card px-2.5 py-1.5 text-center">
            <span className="font-mono text-[10px] text-muted-foreground block">AASIST Graph</span>
            <span className="text-xs font-semibold">0.56</span>
          </div>

          <div className="absolute -right-2 -bottom-2 rounded-xl border bg-muted px-2.5 py-1.5 text-center opacity-60">
            <span className="font-mono text-[10px] text-muted-foreground block">Glottal Source</span>
            <span className="text-xs font-semibold">Skipped</span>
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between text-xs text-muted-foreground font-mono">
        <span>● Active Model Modes</span>
        <span>○ Skipped/Unavailable</span>
        <span>Score-Level Fusion Weighting</span>
      </div>
    </div>
  );
}

export function EvaluatorsCarousel() {
  const [current, setCurrent] = useState(0);
  const total = evaluatorTestimonials.length;

  const handlePrev = () => setCurrent((prev) => (prev > 0 ? prev - 1 : total - 1));
  const handleNext = () => setCurrent((prev) => (prev < total - 1 ? prev + 1 : 0));

  return (
    <div className="space-y-6">
      <div className="grid gap-6 md:grid-cols-3">
        {evaluatorTestimonials.map((item, idx) => (
          <div
            key={item.name}
            className={`rounded-2xl border border-border/90 bg-card p-6 flex flex-col justify-between transition-all duration-300 hover:border-foreground/40 ${
              idx === current ? "ring-2 ring-primary/20" : ""
            }`}
          >
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1">
                  {[...Array(5)].map((_, i) => (
                    <span key={i} className="text-foreground text-xs font-bold">★</span>
                  ))}
                </div>
                <span className="font-mono text-[11px] uppercase text-muted-foreground border px-2 py-0.5 rounded">
                  {item.rating}
                </span>
              </div>
              <p className="text-sm leading-relaxed text-foreground/90 font-sans">
                “{item.quote}”
              </p>
            </div>

            <div className="mt-6 flex items-center gap-3 border-t pt-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-brand text-brand-foreground font-mono text-xs font-bold">
                {item.avatar}
              </div>
              <div>
                <p className="text-sm font-semibold text-foreground">{item.name}</p>
                <p className="text-xs text-muted-foreground">{item.role}</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="flex items-center justify-end gap-2">
        <span className="font-mono text-xs text-muted-foreground mr-2">
          {current + 1} / {total} Evaluators
        </span>
        <Button
          type="button"
          variant="outline"
          size="icon"
          onClick={handlePrev}
          aria-label="Previous evaluator review"
          className="h-9 w-9 rounded-full border-border hover:bg-muted"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
        <Button
          type="button"
          variant="outline"
          size="icon"
          onClick={handleNext}
          aria-label="Next evaluator review"
          className="h-9 w-9 rounded-full border-border hover:bg-muted"
        >
          <ChevronRight className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
