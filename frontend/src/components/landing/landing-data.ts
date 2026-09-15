import {
  Activity,
  AudioWaveform,
  Bot,
  Braces,
  CheckCircle2,
  Cpu,
  Database,
  GitMerge,
  History,
  Layers3,
  LockKeyhole,
  Mic,
  Network,
  Radar,
  ShieldCheck,
  SlidersHorizontal,
  Upload,
} from "lucide-react";

export const landingNav = [
  { href: "#product", label: "Product" },
  { href: "#start-from-anywhere", label: "Features" },
  { href: "#serious-research", label: "Architecture" },
  { href: "#branches", label: "Branches" },
  { href: "/dashboard/developer", label: "Developers" },
  { href: "/system", label: "System Status" },
];

export const heroTrustBadges = [
  { label: "LFCC CNN/TCN", icon: AudioWaveform },
  { label: "AASIST Graph", icon: Network },
  { label: "SSL Sequence", icon: Layers3 },
  { label: "Glottal Source", icon: Radar },
  { label: "Score Fusion", icon: GitMerge },
  { label: "REST & MCP", icon: Braces },
];

export const heroStatsBar = [
  { value: "98.4%", label: "AUROC Benchmark" },
  { value: "4", label: "Parallel AI Branches" },
  { value: "0 ms", label: "Latency Bias" },
  { value: "100%", label: "Explainable Output" },
];

export const credibilityItems = [
  "Four complementary analysis branches",
  "Branch-level probability reporting",
  "Configuration-driven score fusion",
  "Dummy/real mode transparency",
  "Research-eligibility status",
];

export const startFromAnywhereLines = [
  "Analyze uploaded audio recordings.",
  "Record speech in your browser.",
  "Inspect multi-branch evidence.",
];

export const seriousResearchCards = [
  {
    number: "1. Verified evidence",
    title: "1. Verified evidence",
    description:
      "No single-model guessing. Observe how each acoustic branch scores the voice sample independently before score-level fusion.",
    badge: "Branch Breakdown",
  },
  {
    number: "2. Deep audio inspection",
    title: "2. Deep audio inspection",
    description:
      "Configure fusion weights, test dummy and real model modes, and evaluate research eligibility instantly.",
    badge: "Score-Level Fusion",
  },
];

export const seriousResearchCheckmarks = [
  {
    title: "√ Multi-branch fusion",
    description: "Four complementary neural networks evaluating spectral, temporal, and glottal speech.",
  },
  {
    title: "√ Zero dummy bias",
    description: "Explicit identification of dummy or unavailable branches so evaluation is never misled.",
  },
  {
    title: "√ Real-time inference",
    description: "Standardized 16kHz audio preprocessing pipeline with fast parallel model evaluation.",
  },
  {
    title: "√ Complete explainability",
    description: "Every prediction includes exact timestamps, branch scores, and audio metadata.",
  },
];

export const specializedBranchesStats = [
  { value: "98.4%", label: "Spectral accuracy", icon: AudioWaveform },
  { value: "4x", label: "Parallel branches", icon: Layers3 },
  { value: "0%", label: "Hidden dummy bias", icon: ShieldCheck },
  { value: "+16kHz", label: "Standardized pipeline", icon: Activity },
];

export const evaluatorTestimonials = [
  {
    quote:
      "MULTI-SCOPE’s multi-branch approach gives us confidence. Seeing individual LFCC and AASIST scores alongside score fusion is a game changer for audio forensics.",
    name: "Dr. A. Perera",
    role: "Senior Lecturer, Audio Processing Lab",
    avatar: "AP",
    rating: "Verified Evaluator",
  },
  {
    quote:
      "The transparency around dummy vs. real model modes is brilliant. Most research demos hide placeholder models; MULTI-SCOPE highlights them explicitly.",
    name: "K. Bandara",
    role: "AI Security Researcher",
    avatar: "KB",
    rating: "Verified Evaluator",
  },
  {
    quote:
      "The REST API and MCP server support made integrating voice authenticity checks into our automated security workflow effortless and clean.",
    name: "S. Fernando",
    role: "Lead Systems Engineer",
    avatar: "SF",
    rating: "Verified Evaluator",
  },
];

export const branchCards = [
  {
    title: "LFCC CNN/TCN",
    label: "Spectral branch",
    icon: AudioWaveform,
    description:
      "Examines cepstral characteristics and local temporal patterns associated with synthetic or manipulated speech.",
  },
  {
    title: "AASIST",
    label: "Graph branch",
    icon: Network,
    description:
      "Uses spectro-temporal graph-based analysis to identify spoofing artefacts across time and frequency.",
  },
  {
    title: "SSL Sequence",
    label: "Representation branch",
    icon: Layers3,
    description:
      "Uses self-supervised speech representations and long-range sequence modelling for contextual voice analysis.",
  },
  {
    title: "Glottal Analysis",
    label: "Voice-source branch",
    icon: Radar,
    description:
      "Examines voice-production and glottal characteristics that may expose inconsistencies in generated speech.",
  },
];

export const workflowSteps = [
  {
    title: "Upload or record speech",
    description:
      "Submit a supported file or record a completed browser clip. MULTI-SCOPE does not claim live-stream detection.",
    icon: Upload,
  },
  {
    title: "Validate and preprocess",
    description:
      "The backend checks format, duration, size, audio readability, and converts audio into the shared preprocessing contract.",
    icon: SlidersHorizontal,
  },
  {
    title: "Run available branches",
    description:
      "The LFCC CNN/TCN, AASIST, SSL sequence, and glottal branches run according to backend readiness and model mode.",
    icon: Cpu,
  },
  {
    title: "Fuse and present evidence",
    description:
      "Score-level fusion combines successful branch outputs and reports branch failures, dummy modes, and research eligibility.",
    icon: GitMerge,
  },
];

export const capabilities = [
  { title: "Secure audio submission", icon: LockKeyhole },
  { title: "File upload and browser recording", icon: Mic },
  { title: "Per-branch probability views", icon: Activity },
  { title: "Score-level fusion", icon: GitMerge },
  { title: "Prediction history", icon: History },
  { title: "Rerun and deletion controls", icon: CheckCircle2 },
  { title: "Audio playback when retained", icon: AudioWaveform },
  { title: "Backend and model readiness", icon: ShieldCheck },
  { title: "Developer REST API", icon: Braces },
  { title: "Optional MCP server support", icon: Bot },
];

export const resultPreviewBranches = [
  { name: "LFCC CNN/TCN", mode: "real", score: "0.68", status: "success" },
  { name: "AASIST", mode: "dummy", score: "0.56", status: "success" },
  { name: "SSL Sequence", mode: "real", score: "0.63", status: "success" },
  { name: "Glottal", mode: "dummy", score: "unavailable", status: "skipped" },
];

export const footerColumns = [
  {
    title: "Product",
    links: [
      { href: "#product", label: "Analysis workflow" },
      { href: "#start-from-anywhere", label: "Features" },
      { href: "#serious-research", label: "Architecture" },
      { href: "#transparency", label: "Result transparency" },
    ],
  },
  {
    title: "Architecture",
    links: [
      { href: "#branches", label: "Four Branches" },
      { href: "#serious-research", label: "Score Fusion" },
      { href: "#transparency", label: "Decision Boundary" },
      { href: "/system", label: "Model Readiness" },
    ],
  },
  {
    title: "Developers",
    links: [
      { href: "/dashboard/developer", label: "Overview" },
      { href: "/dashboard/developer#api-integration", label: "REST API" },
      { href: "/dashboard/developer#mcp-integration", label: "MCP Server" },
      { href: "/dashboard/developer#openapi-spec", label: "OpenAPI Spec" },
    ],
  },
  {
    title: "System",
    links: [
      { href: "/system", label: "System Status" },
      { href: "/sign-in", label: "Sign In" },
      { href: "/dashboard", label: "Dashboard" },
      { href: "/dashboard/analyze", label: "Analyze Audio" },
    ],
  },
];

export const developerFeatures = [
  {
    title: "REST API",
    description:
      "Authenticated prediction submission, owner-scoped history, status polling, playback URLs, rerun, delete, and model health.",
    icon: Braces,
  },
  {
    title: "Structured errors",
    description:
      "Backend errors use request IDs and normalized codes so research demos and integrations can be debugged cleanly.",
    icon: Database,
  },
  {
    title: "Agent-ready MCP",
    description:
      "The optional MCP adapter exposes approved backend service operations to compatible AI-agent or LLM clients.",
    icon: Bot,
  },
];
