# Frontend Voice Classification Update — 2026-08-09

## 1. Objective

Integrate the Voice XAI component into the existing MULTI-SCOPE frontend and
apply the requested design-system changes, without inventing backend behaviour
or presenting fixture/mock evidence as validated research output.

The frontend already had a working classifier workflow (upload, prediction,
history, readiness, docs). The dominant gap was Voice XAI: there was **no XAI
API client, no XAI types, no XAI components, and no XAI routes** anywhere in
`src/`. That is the bulk of this change.

## 2. Starting Frontend Status

Branch `feat/frontend-aweesha`, working tree clean at `2226d27`.

| Area | Starting state |
|---|---|
| Stack | Next.js 15 App Router, React 19, TS, Tailwind, shadcn/ui, TanStack Query, Clerk, Axios, GSAP — all preserved, nothing replaced |
| Classifier workflow | Working: analyze, submit, detail, history, readiness |
| Voice XAI | **Entirely absent** |
| Typecheck | Passing |
| Tests | 48 passing, but 2 files errored — Vitest was collecting Playwright e2e specs |
| Design | 35 shadow utilities, 29 `tracking-*`, hero grid overlay, 1 gradient, no favicon |

## 3. Backend Contract Reviewed

The backend OpenAPI document was generated directly from the running app
(`create_app(Settings(xai_enabled=True, xai_mode="mock")).openapi()`) and used
as the source of truth, rather than trusting any prose documentation. All XAI
schemas (`XaiExplanationResponse`, `TemporalExplanation`, `SemanticExplanation`,
`CombinedExplanationReport`, `ExplanationQuality`, `MetricEvidence`,
`SemanticFeatureContribution`, `ExplanationProvenance`, …) and every enum
(`ExplanationStatus`, `ComponentStatus`, `MetricStatus`, `ShapDirection`,
`ReportDisposition`, `SemanticTargetType`) were transcribed field-by-field into
`src/types/api.ts`.

Two contract facts worth recording:

- **XAI auth is not declared in OpenAPI.** `security` is `null` on the XAI
  routes because Clerk auth is a FastAPI dependency (`Depends(require_clerk_user)`)
  rather than a declared OpenAPI security scheme. Auth *is* enforced server-side;
  the frontend sends the Clerk bearer token via the existing Axios interceptor.
- **`GET .../explanation` returns 404 when no run exists.** That is a normal
  state, not an error, and is handled as `null` rather than an error state.

## 4. Backend Endpoints Used

| Method | Endpoint | Frontend Usage | Auth | Status |
|---|---|---|---|---|
| POST | `/api/v1/predictions` | Submit audio | Clerk | Pre-existing |
| GET | `/api/v1/predictions/{id}/status` | Poll job status | Clerk | Pre-existing |
| GET | `/api/v1/me/predictions` | History list | Clerk | Pre-existing |
| GET | `/api/v1/me/predictions/{id}` | Prediction detail | Clerk | Pre-existing |
| GET | `/api/v1/me/predictions/{id}/audio` | Playback | Clerk | Pre-existing |
| POST | `/api/v1/me/predictions/{id}/rerun` | Rerun | Clerk | Pre-existing |
| DELETE | `/api/v1/me/predictions/{id}` | Delete | Clerk | Pre-existing |
| GET | `/ready` | Readiness incl. `components.voice_xai` | none | Pre-existing |
| GET | `/api/v1/voice/models/health` | Model health | Clerk | Pre-existing |
| **POST** | **`/api/v1/me/predictions/{id}/explanation`** | **Trigger explanation** | **Clerk** | **New** |
| **GET** | **`/api/v1/me/predictions/{id}/explanation`** | **Read latest run (404 ⇒ none yet)** | **Clerk** | **New** |
| **POST** | **`/api/v1/me/predictions/{id}/explanation/retry`** | **Create a new run** | **Clerk** | **New** |
| **GET** | **`/api/v1/me/predictions/{id}/explanation/artifacts/{artifact_id}`** | **Download private artifact (blob)** | **Clerk** | **New** |

`/explanation/temporal`, `/explanation/semantic` and `/explanation/report` are
implemented in `src/lib/api/xai.ts` but are not called by the UI: the aggregate
`GET .../explanation` already embeds `temporal`, `semantic` and
`combined_report`, so using the sub-routes would triple the request count for
identical data. They are kept for callers that want a single component.

## 5. Pages Added / Updated

| Route | Change |
|---|---|
| `/dashboard/predictions/[predictionId]/explanation` | **New.** Full XAI view: lifecycle + tabbed temporal / semantic / quality / finding, warnings, reproducibility |
| `/dashboard/predictions/[predictionId]/report` | **New.** Print-ready report: decision → evidence identity → temporal → semantic → combined finding → quality → warnings → reproducibility → validation context → human review |
| `/dashboard/predictions/[predictionId]` | Updated: XAI summary card + evidence identity panel |
| `/dashboard` (layout) | Auth hardened to fail closed in production |

## 6. Components Added / Updated

All new, under `src/components/xai/`:

`xai-status-badge` (icon+text status), `research-status-banner` (+ badges),
`evidence-availability` (`NotAvailable`, `EvidenceUnavailablePanel`),
`temporal-evidence-panel` (region timeline + table + artifacts),
`semantic-evidence-panel` (ranked contributions, direction, per-window),
`quality-evidence-panel`, `combined-finding-panel`, `provenance-panel`,
`warnings-panel`, `evidence-identity-panel`, `backend-gap-panels`
(human review / validation context), `explanation-lifecycle-card`,
`artifact-download-button`, `explanation-summary-card`.

Supporting: `src/lib/xai/status.ts` (all backend→UI vocabulary mapping),
`src/lib/api/xai.ts`, `src/features/xai/hooks.ts`,
`src/features/xai/{explanation,report}-page.tsx`, `src/mocks/xai.ts` (tests only).

## 7. Landing Page Improvements

Scope was deliberately limited to the requested design changes — the existing
content already covered the required sections and rewriting it wholesale was not
justified. Applied: hero grid overlay removed, all `shadow-*` removed, all
`tracking-*` removed, the one decorative gradient replaced with a solid surface,
and the decorative glow (`shadow-[0_0_8px_white]`) removed. Separation now comes
from borders, spacing and surface tone.

## 8. Authentication & Authorization

Clerk preserved; no second auth provider introduced.

**Security fix — dashboard failed open.** `src/app/dashboard/layout.tsx`
previously only ran the `auth()` check when *both* Clerk env vars were present.
With `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` unset, the entire check was skipped and
owner-scoped dashboard routes rendered unauthenticated. It now fails closed:
any incomplete Clerk configuration under `NODE_ENV=production` redirects to
`/sign-in`. The dev-convenience path is retained only outside production.

Authorization remains server-side. The frontend never sends an owner ID; the
backend resolves it from the auth principal and returns 404 (not 403) for
another user's prediction, so editing `predictionId` in the URL cannot confirm
that a foreign prediction exists.

## 9. Prediction Workflow

Unchanged. Submission, validation, polling and duplicate-submit protection were
already correct and were not modified.

One correctness fix: `.env.example` advertised
`NEXT_PUBLIC_MAX_AUDIO_DURATION_SECONDS=300` while the backend enforces **180**
(`max_audio_duration_seconds: int = 180`). Clients configured from the example
would have accepted 5-minute audio client-side only to have it rejected by the
backend. Corrected to 180 with a comment noting the backend is authoritative.

## 10. Prediction Results

The detail page now additionally shows the XAI summary card (status, per-component
badges, links) and the evidence identity panel. Existing decision / branch /
fusion cards are unchanged.

## 11. Voice XAI Integration

### Temporal Evidence
Region timeline + table (start, end, duration, attention score), peak attention,
threshold percentile, combined region duration, artifacts. **Region positions are
computed only from backend `start_seconds`/`end_seconds` against clip duration —
the UI never infers which parts of the audio are suspicious.** When clip duration
is unknown the timeline is omitted rather than guessed. The backend's temporal
warning (currently: fixture-backed attention) is rendered verbatim.

### Semantic Evidence
Ranked contributions with measured value, unit, reference summary, SHAP value,
time interval, and a magnitude bar. **Direction is taken from the backend's
`direction` field, never inferred from the sign or magnitude of `shap_value`** —
duplicating that mapping client-side risks silently inverting the evidence.
Per-window evidence renders when present; otherwise the whole-clip limitation is
stated explicitly. When `target_type` is `independent_acoustic_evidence_model`
the UI states that the attributions do not explain the classifier directly.

### Combined Finding
Rendered verbatim from the backend's deterministic report composer
(`finding`, `primary_evidence`, `quality_checks`, `recommendation`, `limitation`,
`disclaimer`). No scientific conclusion is synthesised client-side. Backed by the
`ReportDisposition` enum, including `inconclusive → "Inconclusive — manual review
required"`.

### Explanation Quality
All five metrics are laid out (temporal/semantic IoU, temporal localisation IoU,
surrogate fidelity R², AOPC, robustness/stability). **A metric that is absent or
`not_computed` renders as "Not available"/"Not computed" — never `0`**, since
zero is itself a legitimate measured value. The backend's `reason`, `scope` and
`dataset_version` are shown when present.

### Warnings and Limits
Backend `warnings[]`, per-component errors, and run-level error. Errors show only
the sanitised `{code, message}` the backend produces; no traceback, filesystem
path or raw exception can reach the UI.

### Reproducibility
Full provenance record. Fields the backend leaves null render as "Not recorded".

### Human Review
**Not implemented as a form.** No backend API exists to persist analyst review, so
an explicit "Human-review persistence is not yet available" panel is shown rather
than a form that would silently discard input.

## 12. Prediction History

Unchanged. An XAI status column was **deliberately not added** — see BACKEND GAP
in §22.

## 13. API Documentation UI

Not redesigned in this pass — see §25.

## 14. MCP Documentation UI

Not redesigned in this pass — see §25.

## 15. Responsive Improvements

New XAI components are responsive by construction: `sm:`/`lg:` grids for metric
rows, `flex-wrap` on all header/action rows, `min-w-0` + `break-all`/`break-words`
on IDs and hashes, and the temporal region table wrapped in `overflow-x-auto`
with `min-w-[34rem]` so it scrolls inside its own container rather than forcing
page-level horizontal overflow.

## 16. Accessibility Improvements

- Status is conveyed by **icon + text**, never colour alone (`ExplanationStatusBadge`,
  `ComponentStatusBadge`, direction badges, research badges).
- Region timeline is `role="img"` with a descriptive `aria-label`.
- Temporal table has a `<caption class="sr-only">` and `scope="col"` headers.
- Report sections use `aria-labelledby` against real `<h2>` ids.
- Artifact download buttons have explicit `aria-label`s.
- Decorative icons are `aria-hidden="true"`.
- No focus outlines were removed.

## 17. Security Improvements

- Dashboard auth now fails closed in production (§8).
- **Artifact IDs are treated as opaque.** Artifacts are fetched only through the
  authenticated `/explanation/artifacts/{artifact_id}` route. The API's
  `download_path` field is deliberately ignored and never used to build a URL.
- Artifact object URLs are revoked immediately after download so bytes are not
  retained.
- Report/finding content is rendered as **plain text**; no `dangerouslySetInnerHTML`
  is introduced anywhere.
- Explanation query cache is bounded (`gcTime: 5min`) rather than retained
  indefinitely.
- No secrets added to `NEXT_PUBLIC_*`; no tokens stored manually (Clerk owns them).

## 18. Files Changed

| File | Change | Reason |
|---|---|---|
| `src/types/api.ts` | +~200 lines of XAI types | Mirror backend schema exactly |
| `src/lib/api/xai.ts` | New | Typed XAI client |
| `src/lib/xai/status.ts` | New | Central backend→UI vocabulary |
| `src/features/xai/hooks.ts` | New | Lifecycle-aware polling, trigger, retry |
| `src/features/xai/explanation-page.tsx` | New | XAI view |
| `src/features/xai/report-page.tsx` | New | Report view |
| `src/components/xai/*` (14 files) | New | XAI component set |
| `src/app/dashboard/predictions/[id]/explanation/page.tsx` | New | Route |
| `src/app/dashboard/predictions/[id]/report/page.tsx` | New | Route |
| `src/lib/query/keys.ts` | +explanation keys | Cache keying |
| `src/features/predictions/prediction-detail-page.tsx` | +XAI summary, +identity panel | Surface XAI |
| `src/app/dashboard/layout.tsx` | Fail-closed auth | Security fix |
| `src/app/layout.tsx` | Favicon + OG/Twitter metadata | Branding |
| `src/app/icon.svg` | New | Favicon |
| `src/app/globals.css` | Removed `.technical-grid` | Hero grid removal |
| `src/app/page.tsx` | Removed grid class, shadows, tracking | Design |
| `src/components/landing/landing-animations.tsx` | Removed shadows/tracking/gradient/glow | Design |
| `src/components/ui/{card,dialog,alert-dialog,select,tabs,tooltip}.tsx` | Removed shadows | Design |
| `src/components/{layout/dashboard-header,shared/page-header}.tsx` | Removed shadow/tracking | Design |
| `src/app/developers/{,api/,mcp/}page.tsx` | Removed `tracking-normal` | Design |
| `vitest.config.ts` | `include`/`exclude` | Stop Vitest collecting Playwright specs |
| `.env.example` | 300 → 180 | Match backend limit |
| `src/mocks/xai.ts` | New | Test fixtures only |
| `src/lib/xai/__tests__/status.test.ts` | New | 10 tests |
| `src/components/xai/__tests__/*.test.tsx` | New | 22 tests |

## 19. Environment Variables

No new variables. `NEXT_PUBLIC_MAX_AUDIO_DURATION_SECONDS` corrected to `180`.

## 20. Tests Added / Updated

32 new tests (48 → 80):

- `status.test.ts` (10): queued/running/partial are polled; completed/failed/**blocked**
  are not (blocked only leaves that state via explicit retry, so polling it would
  spin forever); unknown status is not polled; absent vs `not_computed` vs
  `not_applicable` vs **a genuine measured `0`**; label and direction mapping.
- `research-status-banner.test.tsx` (6): development/not-eligible/dummy-branch
  flags visible; decision-support notice always present; banner hides only when
  genuinely clean.
- `evidence-panels.test.tsx` (16): backend regions/scores rendered; temporal
  warning verbatim; timeline omitted when duration unknown; unavailable (not
  error) states; region-export unavailability stated; **both** SHAP directions
  from the backend field; whole-clip marker; independent-model caveat; computed
  metric shown; uncomputed metrics never render as `0.000`; sanitised error
  code/message only.

## 21. Test Results

```text
Typecheck (tsc --noEmit):  PASS
Lint (eslint .):           PASS  (no warnings)
Unit/component (vitest):   PASS  17 files, 80 tests, 0 failures
Production build:          PASS  17 routes, 0 errors, 0 warnings
```

Build output confirms both new routes and the favicon:
`/dashboard/predictions/[predictionId]/explanation` (2.77 kB),
`/dashboard/predictions/[predictionId]/report` (4.29 kB), `/icon.svg`.

Baseline before this work was 48 tests with 2 **erroring** files (Vitest was
collecting Playwright e2e specs) — fixed via `vitest.config.ts`.

## 22. Backend Features Still Missing

```text
BACKEND GAP — XAI status is absent from prediction history items.
  `PredictionHistoryItem` carries no explanation field and there is no bulk
  explanation-status endpoint, so a history XAI column would require one request
  per row (N+1). Deliberately not implemented. Needs either an `xai_status` field
  on history items or a bulk status endpoint.

BACKEND GAP — analyst/human-review persistence API unavailable.
  No endpoint stores analyst decision, annotations, or sign-off. UI shows an
  explicit unavailable panel instead of a form that would discard input.

BACKEND GAP — original-audio content hash not persisted.
  No SHA-256 of the original upload is returned, so the evidence identity panel
  reports it as "Not recorded" rather than claiming hash preservation.

BACKEND GAP — chain-of-custody reference not implemented.

BACKEND GAP — validation/evaluation metadata not exposed.
  No accuracy, FPR/FNR, calibration, per-attack-type or subgroup metrics, so the
  validation context section reports "Unavailable for this deployment".

BACKEND GAP — no report export artifact.
  No server-generated PDF/report artifact exists, so the report page is a
  browser-printable layout. No fake forensic export workflow was built.

BACKEND GAP — region audio export unavailable.
  No endpoint returns a clipped audio segment for a temporal region, so that
  action is stated as unavailable rather than reconstructed client-side.

BACKEND GAP — AOPC, surrogate fidelity R², robustness/stability not computed.
  Rendered as "Not computed" with the backend's own reason where supplied.
```

## 23. XAI Features Still Mock / Research-Ineligible

Per the backend audit of 2026-08-08, unchanged by this frontend work:

- **Temporal XAI is fixture-backed** (`SavedTensorAttentionProvider`); timestamps
  are duration-scaled, not model-derived. `development_placeholder=true`.
- **Semantic XAI is mock by default**; real mode needs XGBoost/SHAP artifacts that
  are not present. Even in real mode the backend hardcodes
  `research_eligible=false`.
- **`research_eligible` is currently unreachable as `true`** anywhere in the
  backend by design.

The UI reports all of this rather than smoothing over it: the research-status
banner, per-component badges, verbatim backend warnings, and the quality panel's
"not computed" states all remain visible.

## 24. Final Frontend Readiness

```text
Landing Page:          READY  (design changes applied; content unchanged)
Authentication:        READY  (fail-closed fix applied)
Dashboard:             READY
Prediction Integration:READY
History:               READY  (XAI column blocked by backend gap)
Voice XAI UI:          READY
Temporal XAI UI:       READY_FOR_BACKEND  (renders real regions; backend is fixture-backed)
Semantic XAI UI:       READY_FOR_BACKEND  (renders real SHAP; backend artifacts absent)
XAI Report:            PARTIAL  (print layout; no backend export, no review persistence)
API Documentation:     NEEDS_FIXES  (not redesigned this pass)
MCP Documentation:     NEEDS_FIXES  (not redesigned this pass)
Responsive Design:     PASS   (new components; not re-audited at all 4 breakpoints)
Accessibility:         PASS   (icon+text status, labelled tables/timelines, a11y landmarks)
Security Review:       PASS   (auth fail-closed, opaque artifacts, no raw HTML, bounded cache)
Typecheck:             PASS
Lint:                  PASS
Tests:                 PASS   (80)
Production Build:      PASS
```

### Backend Capability Matrix

| Frontend Feature | Backend Support | Frontend Status | Notes |
|---|---|---|---|
| Prediction | Available | Integrated | Pre-existing |
| Prediction History | Available | Integrated | Paginated server-side |
| Branch Results | Available | Integrated | Canonical names + aliases preserved |
| Fusion | Available | Integrated | Visually distinct from branch output |
| XAI Trigger | Available | Integrated | Idempotent; returns existing run |
| XAI Status | Available | Integrated | Polled only while queued/running/partial |
| Temporal Evidence | Available (fixture-backed) | Integrated | Flagged as development evidence |
| Semantic Evidence | Available (mock by default) | Integrated | Flagged as development evidence |
| Combined Report | Available | Integrated | Rendered verbatim |
| Artifact Retrieval | Available | Integrated | Opaque ID, authenticated, hash shown |
| Quality Metrics | Partial (1 of 5 computed) | Integrated | Others explicitly "not computed" |
| Reproducibility | Available | Integrated | Null fields → "Not recorded" |
| Human Review | **Unavailable** | Gap panel | BACKEND GAP |
| Chain of Custody | **Unavailable** | "Not recorded" | BACKEND GAP |
| Validation Context | **Unavailable** | Gap panel | BACKEND GAP |
| History XAI column | **Unavailable** | Not built | BACKEND GAP (N+1) |

## 25. Follow-up Tasks

1. **API + MCP documentation redesign** — not done in this pass. The requested
   shared docs system (`DocsShell`, `DocsSidebar`, `EndpointBlock`, `CodeBlock`,
   `TableOfContents`, OpenAPI-derived content) remains outstanding; the existing
   `src/components/docs/*` pages still work but were not restructured.
2. **Dashboard "Recent XAI explanations" panel** — blocked by the same missing
   bulk XAI status endpoint as the history column.
3. **Responsive re-audit** at 375/768/1024/1440 px across all pages, including
   the pre-existing ones this pass did not touch.
4. **E2E coverage** for the XAI lifecycle via Playwright (Vitest now correctly
   excludes `e2e/`, but no new spec was added).
5. Close the BACKEND GAPs in §22, prioritising history XAI status (cheapest, and
   unblocks two UI features).
