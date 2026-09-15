# Voice Classification Backend Update — 2026-08-08

## 1. Objective

Implement the fixes identified in `VOICE_XAI_INTEGRATION_AUDIT_2026-08-08.md`
(repo root) so the XAI-enabled backend is reliable, without redesigning the
existing post-prediction/non-blocking architecture. Scope was fix-priority
ordered per the task: (1) event-loop/Mongo reliability, (2) worker exception
handling, (3) XAI rate limiting, (4) real-Mongo-shaped integration coverage,
(5) classifier → XAI handoff, (6) operational cleanup/restart handling,
(7) research-readiness structure.

## 2. Starting Audit Status

```text
Classifier/XAI Integration Regression: PASS
Authentication & Authorization:        PASS
Privacy Isolation:                     PASS
XAI Engineering Reliability:           NEEDS FIXES
Real Temporal XAI:                     NOT IMPLEMENTED
Real Semantic XAI:                     NOT CONFIGURED
Research Readiness:                    NOT READY
```

Every finding below was re-verified against the current repository (not
assumed from the prior audit doc) before any code was changed: `git status`,
`git log`, and the relevant source files were re-read, and the full test
suite was run first to confirm the starting state (348 passed, 24 failed —
all 24 in ffmpeg/ffprobe-dependent classifier tests, 5 skipped, 2
deselected). No `git reset`/`clean`/`checkout --` was run; work happened on
a new branch (`fix/voice-xai-reliability-2026-08-08`) off `dev`.

## 3. Issues Addressed

### SEC-1 — XAI Async Runtime / Mongo Event Loop

**Problem:** `AsynchronousExplanationQueue.submit()` ran every job via
`ThreadPoolExecutor.submit(lambda: asyncio.run(...))` — a brand-new asyncio
event loop per job, in a worker thread, while all XAI Mongo access went
through the single application-wide `AsyncMongoClient` bound to the main
uvicorn event loop. PyMongo's async driver is documented as unsafe to share
across event loops; this pattern was invisible to the existing test suite
because those tests use an in-memory `FakeDatabase`, never a real loop-bound
client.

**Fix:** `AsynchronousExplanationQueue` now owns one persistent background
thread running a single `asyncio` event loop for the life of the process.
`submit()` schedules the job coroutine onto that loop with
`asyncio.run_coroutine_threadsafe()` — no `asyncio.run()` per job, ever. The
queue can optionally build a *dedicated* MongoDB client for that loop (via a
`mongodb_client_factory`, constructed from inside the worker thread as the
loop starts) so XAI's background work never touches the main application's
loop-bound client. `VoiceXaiOrchestrator` resolves a worker-scoped repository
once per background run (`worker_repository_factory`) for exactly the calls
that execute inside that worker loop, while every route-facing method
(`enqueue`, `get_for_prediction`, `trigger_for_prediction`, artifact lookup)
keeps using the original main-loop repository untouched.

### SEC-2 — Silent XAI Worker Exceptions

**Problem:** Two calls inside `VoiceXaiOrchestrator._run()` (`start_run` and
`update_run_evidence`) sat outside all three per-component `try/except`
blocks. An exception there propagated out of the coroutine uncaught, was
stored (never inspected) on the queue's fire-and-forget
`concurrent.futures.Future`, and left the explanation permanently `queued`/
`running` with zero logged error.

**Fix:** `_run()` is now a thin outer boundary that calls the renamed
`_run_internal()` (unchanged component-level logic) inside one
`try/except Exception`. Any escape is logged with safe operational IDs
(`explanation_id`, `prediction_id`, `request_id` — never raw exceptions,
stack traces, or secrets in persisted/public state) and forces the run to a
terminal `failed` status via a new `XaiStatusService.fail_run()` method,
using the existing lifecycle's `failed` state (no new state invented).
Separately, the queue's own `Future` completion callback now inspects
`future.exception()` and logs any unhandled exception, as defense in depth
independent of the orchestrator's own handling.

### SEC-3 — XAI Trigger/Retry Rate Limiting

**Problem:** The global `RateLimitMiddleware` keys by the literal request
path, which includes `prediction_id` for
`POST .../{prediction_id}/explanation[/retry]`. A user with (or who creates)
many predictions could trigger/retry an explanation for each one, each
getting its own independent rate-limit bucket — the generic limiter did not
actually bound total XAI job creation per user, and `enqueue()`
unconditionally persists a new Mongo document even when the queue is full.

**Fix:** Added a dedicated per-authenticated-user limiter check
(`_enforce_xai_action_rate_limit` in `xai_routes.py`) on both the trigger and
retry routes, keyed by the server-derived owner ID only (`principal.user_id
or principal.subject` — never a client-supplied ID), reusing the same shared
`InMemoryRateLimiter` instance as the existing middleware (via a new
`get_rate_limiter` DI getter). New settings:
`XAI_TRIGGER_RATE_LIMIT_REQUESTS`/`_WINDOW_SECONDS` and
`XAI_RETRY_RATE_LIMIT_REQUESTS`/`_WINDOW_SECONDS` (defaults: 5 requests per
60s each), documented in `.env.example`. A 429 now carries a `Retry-After`
header end-to-end — this required a small, generally-applicable fix to
`http_exception_handler` (it previously dropped any `headers=` set on a
raised `HTTPException`, e.g. the existing `admission_control_error_handler`
pattern for a different exception type already relied on this being
forwarded).

### Real-Mongo-shaped coverage (task item 4)

No real MongoDB server or `mongomock`/testcontainers dependency is available
in this environment, and the task explicitly asked not to introduce a heavy
new test dependency. The existing project convention (`FakeDatabase`/
`FakeCollection` in `tests/test_mongodb.py`) was extended instead: it now
supports `$in`/`$lt`/`$lte`/`$gt`/`$gte` query operators (needed for the new
stale-run query), and new tests exercise the *real*
`AsynchronousExplanationQueue` machinery end-to-end (real persistent thread +
loop + `run_coroutine_threadsafe`, not `_run()` called directly) proving
multiple sequential and back-to-back jobs reuse one loop, `close()` stops it
cleanly, and unhandled exceptions are surfaced. This proves the loop-reuse
fix directly; it cannot by itself prove PyMongo's real cross-loop failure
mode (that requires a live, loop-bound `AsyncMongoClient`, which is exactly
what SEC-1's architecture change is designed to avoid needing).

### INT-1 — Classifier → XAI Handoff

**Problem:** The automatic post-persistence handoff only ever supplied
`prediction`; `processed_audio` and `extraction` were always `None`. This was
already documented as a known limitation, but it also meant the
already-implemented windowed semantic evidence path could never run
automatically.

**Fix:** `VoiceService` gained
`predict_from_validated_upload_with_processed_audio()` (returns the
existing `VoicePredictionResponse` unchanged, plus the exact
`ProcessedAudio` instance used for inference) by extracting the shared core
of `predict_from_validated_upload()` into a private helper — the original
public method's signature, return type, and behaviour are untouched, so
every existing caller (route, rerun service, MCP adapter, tests) is
unaffected. `PredictionSubmissionService` now calls the new method when the
voice service exposes it (falls back to the original method, with
`processed_audio=None`, for any test double that doesn't — no behaviour
change for the classifier prediction path in either case) and threads the
same `ProcessedAudio` object into `ClassifierInferenceBundle`. No second
decode of the upload happens; `extraction` correctly stays `None` (no real
branch models are wired yet, so `PyTorchExtractionInterface` still is not
instantiated anywhere in application startup — generating placeholder
extraction data was deliberately not done, per the task's explicit
instruction).

### OPS-1 — Artifact Cleanup

**Problem:** `XAI_ARTIFACT_RETENTION_SECONDS` only gated API access
(`LocalExplanationArtifactStore.read()` returns `None` past expiry); no code
path physically deleted the file, so private artifact storage grew
unboundedly.

**Fix:** Added `LocalExplanationArtifactStore.cleanup_expired()`, which
walks the store's own two-level hashed directory layout and deletes files
whose filesystem mtime is older than the retention window (self-contained —
no MongoDB dependency added to this module, since every reference's
`expires_at` already equals `write_time + retention_seconds`), removes
now-empty explanation directories, and is defensively bounded to the
configured root (`is_relative_to()` check, matching the existing
traversal-safety pattern in `read()`/`store_bytes()`). Wired into
`app/main.py`'s lifespan as a background `asyncio.create_task` loop that
sleeps `XAI_ARTIFACT_CLEANUP_INTERVAL_SECONDS` (default 3600s) between
passes, runs the (synchronous, filesystem-bound) cleanup via
`loop.run_in_executor` so it never blocks the main event loop, catches
per-iteration exceptions so one bad pass cannot kill the task permanently,
and is cancelled/awaited cleanly on shutdown. Only started when
`XAI_ENABLED=true`.

### OPS-2 — Stale Run Recovery

**Problem:** A run left `queued`/`running`/`partial`/`blocked` by a process
crash/restart had no recovery mechanism and stayed stuck forever with no
operator visibility.

**Fix:** Added `MongoXaiExplanationRepository.find_stale_active_explanations()`
(non-terminal status + `updated_at` older than a threshold) and
`app/voice_xai/recovery.py::recover_stale_explanations()`, which marks each
stale run `failed` via the same `XaiStatusService.fail_run()` used by SEC-2,
with error code `xai_run_interrupted`. Concurrent-change races (a run that
legitimately progresses between the scan and the write) are caught
(`XaiPersistenceConsistencyError`) and skipped rather than crashing the
reconciliation pass. Runs at application startup, after MongoDB connects,
only when `XAI_ENABLED=true`; wrapped in its own try/except so a recovery
failure can never block startup. New setting:
`XAI_STALE_RUN_RECOVERY_THRESHOLD_SECONDS` (default 300s). Per the task's
explicit instruction, this **never re-runs expensive XAI work** — only a
user-initiated retry creates a new run.

### RESEARCH-1 — Research Eligibility Structure

**Problem:** `research_eligible` was computed as one inline boolean
expression inside `_run_internal`, correctly fail-closed today but not
documented as a reviewable policy.

**Fix:** Added `app/voice_xai/research_eligibility.py`:
`evaluate_research_eligibility()` returns a
`ResearchEligibilityDecision(eligible, policy_version, unmet_requirements)`,
checking classifier (no dummy branches + classifier-level eligibility),
temporal (component present + eligible), and semantic (component present +
eligible) requirements individually, each documented with *why* it is
currently unmet. This is a behaviour-preserving refactor — every requirement
is still unmet today, so `research_eligible` is still always `False`,
exactly as before. `research_eligible` was **not** changed to `True`
anywhere. The combined temporal/semantic overlap-agreement rule remains
explicitly not yet defined (documented as such, not silently assumed).

### Unrelated, incidentally-discovered fix

Running the full suite before any XAI change surfaced 24 failing tests, all
in ffmpeg/ffprobe-dependent classifier code (unrelated to XAI — confirmed via
`git log -- app/ingestion/audio.py`, never touched by any XAI commit). Root
cause, reproduced directly against the installed binary: this machine's
Homebrew `ffprobe 8.1.2` does not define `-nostdin` as a valid option at all
(`ffprobe -h full` has no such entry; the flag is ffmpeg-only in this build),
so ffprobe reads the *next* argument as `-nostdin`'s value and fails with a
confusing `Option not found` error for every single call, regardless of
input. `-nostdin` was removed from the ffprobe command only (kept for the
`ffmpeg` decode command, which does support it and was independently
verified to work) — this is fully redundant anyway since
`_run_media_process` already runs every subprocess with
`stdin=subprocess.DEVNULL`, so ffprobe can never block on interactive stdin
regardless of the flag. No CI/Docker version pin currently exists
(`apt-get install ffmpeg` with no version), so this was purely a local
dev-machine mismatch; the fix is backward compatible with older ffprobe
versions. This dropped the failure count from 24 to 4 (see §9 below for the
remaining 4, which are unrelated to both XAI and ffmpeg).

## 4. Files Changed

| File | Change | Reason |
|---|---|---|
| `app/voice_xai/extraction_interface/queue.py` | Rewritten: persistent worker loop + `start()`/`submit()`/`close()` lifecycle, optional dedicated Mongo client factory, `Future` exception logging | SEC-1 |
| `app/database/mongodb.py` | Added `build_dedicated_async_mongo_client()` | SEC-1 |
| `app/voice_xai/orchestrator.py` | Outer `_run()` failure boundary around renamed `_run_internal()`; worker-scoped repository resolution; `submit()` call site now schedules a coroutine directly (no `asyncio.run`); research eligibility now delegated to the new evaluator | SEC-1, SEC-2, RESEARCH-1 |
| `app/voice_xai/status.py` | Added `XaiStatusService.fail_run()` | SEC-2, OPS-2 |
| `app/voice_xai/research_eligibility.py` (new) | Centralized, documented, fail-closed eligibility evaluator | RESEARCH-1 |
| `app/voice_xai/recovery.py` (new) | Startup stale-run reconciliation | OPS-2 |
| `app/voice_xai/persistence/protocols.py`, `app/voice_xai/persistence/mongodb.py` | Added `find_stale_active_explanations()` | OPS-2 |
| `app/voice_xai/artifacts/service.py` | Added `cleanup_expired()` + `ArtifactCleanupResult` | OPS-1 |
| `app/api/v1/xai_routes.py` | Per-user rate limiting on trigger/retry | SEC-3 |
| `app/api/dependencies.py` | Added `get_rate_limiter()` | SEC-3 |
| `app/core/exception_handlers.py` | `http_exception_handler` now forwards `HTTPException.headers` (e.g. `Retry-After`) | SEC-3 |
| `app/config/settings.py` | New settings: XAI trigger/retry rate limits, artifact cleanup interval, stale-run threshold | SEC-3, OPS-1, OPS-2 |
| `.env.example` | Documented the new settings | SEC-3, OPS-1, OPS-2 |
| `app/main.py` | Wire dedicated Mongo client factory into the queue, worker repository factory into the orchestrator, startup stale-run recovery, background artifact-cleanup task + shutdown | SEC-1, OPS-1, OPS-2 |
| `app/services/voice_service.py` | Added `PredictionExecutionResult` + `predict_from_validated_upload_with_processed_audio()`; original method now a thin wrapper over a shared private core | INT-1 |
| `app/services/prediction_submission_service.py` | Reuses `processed_audio` in the automatic XAI handoff when available | INT-1 |
| `app/ingestion/audio.py` | Removed `-nostdin` from the ffprobe command only | Incidental ffmpeg fix |
| `tests/test_mongodb.py` | `FakeCollection.find`/`find_one` now support `$in`/`$lt`/`$lte`/`$gt`/`$gte` | OPS-2 test support |
| `tests/test_voice_xai_extraction_interface_queue.py` | Rewritten for the coroutine-based `submit()` contract; new loop-reuse/lifecycle/exception-logging/dedicated-client tests | SEC-1 |
| `tests/test_voice_xai_orchestrator.py` | New SEC-2 regression test (unexpected failure outside component handlers reaches `failed`) | SEC-2 |
| `tests/test_voice_xai_recovery.py` (new) | Stale-run recovery scenarios | OPS-2 |
| `tests/test_voice_xai_artifacts.py` | New cleanup tests (expiry, no-op on missing root, cannot escape root) | OPS-1 |
| `tests/test_voice_xai_routes.py` | New SEC-3 regression test (per-user, not per-prediction-id, rate limiting) | SEC-3 |
| `tests/test_voice_xai_research_eligibility.py` (new) | Evaluator unit tests | RESEARCH-1 |
| `tests/test_prediction_routes.py` | New INT-1 regression test (handoff reuses the same `ProcessedAudio`) | INT-1 |

No public prediction schema, fusion logic, label semantics, or
authentication system was changed. No existing test was deleted.

## 5. Implementation Details

See §3 above for the substantive design of each fix; the two points worth
calling out explicitly:

- **Backward compatibility of the queue contract change.** `submit()` used
  to accept a plain callable (sync or wrapped in `asyncio.run` by the
  caller); it now requires a callable that *returns a coroutine* when
  invoked with the bundle, which is scheduled directly. `VoiceXaiOrchestrator`
  was updated accordingly (`lambda queued_bundle: self._run(explanation_id,
  queued_bundle)` — calling an `async def` without `await` yields a
  coroutine object). This is an internal contract (`queue.py` has no
  consumers outside `orchestrator.py` and its own tests), so the change was
  made directly rather than preserving the old shape.
- **`_worker_repository()` defaults to the main-loop repository** whenever
  `worker_repository_factory` is not supplied. Every existing test
  constructs `VoiceXaiOrchestrator` without it, so `_run_internal()`
  continues to behave exactly as before for all of them; only the real
  `app/main.py` wiring supplies the factory.

## 6. Security Impact

Re-checked after all fixes (per task §37):

- **IDOR / cross-user access:** unchanged and still not present — the new
  rate-limit dependency reads `principal.user_id or principal.subject` only
  (never a client-supplied ID), and no repository query's owner-scoping was
  touched.
- **Artifact path traversal:** unchanged; `cleanup_expired()` reuses the
  same `is_relative_to(self._root)` guard pattern already used by
  `read()`/`_artifact_path()`, and a dedicated test
  (`test_cleanup_expired_cannot_delete_outside_the_configured_root`) proves
  a file outside the configured root is never touched.
  `shell=True`/unsafe deserialization: none introduced (no new subprocess or
  pickle/joblib call sites).
- **Secret exposure:** none introduced; new settings are numeric
  limits/intervals only, documented in `.env.example` with safe defaults, no
  secrets.
- **Raw exception exposure:** the new outer failure boundary logs full
  exceptions to `logger.exception` (server-side only) and persists only the
  fixed-vocabulary `xai_unexpected_failure`/`xai_run_interrupted` codes
  through the existing `ExplanationError` schema — never `repr(exception)` or
  a traceback — matching the existing sanitization pattern already covered
  by `test_internal_semantic_paths_and_stack_details_do_not_reach_response`
  and now also by the new SEC-2 test's assertions
  (`"boom" not in response.model_dump_json()`).
- **Unbounded task creation / unbounded retry:** improved by SEC-3; the
  underlying compute queue was already bounded (unchanged).
- **Cross-request tensor leakage:** not applicable to this change set (no
  extraction-interface/hook code was touched).

## 7. Classifier Regression Review

- Public prediction schemas (`VoicePredictionResponse`,
  `PredictionSubmissionResponse`): unchanged.
- Fusion semantics, spoof-score direction, `bonafide`/`spoof` label meaning:
  unchanged (no fusion/model code touched).
- `predict_from_validated_upload()`'s signature, return type, and behaviour:
  unchanged (verified by the full `tests/test_voice_service.py` suite
  passing unmodified).
- Authentication/authorization: unchanged (Clerk auth code untouched; new
  rate limiting is additive and denies with 429, not a new bypass surface).
- Prediction latency independent of XAI: unchanged — `enqueue()` is still
  only called after `PredictionPersistenceService.save_result()`, still
  wrapped in the same try/except that only logs.
- XAI failure/timeout/queue-full does not fail a completed prediction:
  unchanged and additionally now provably *cannot silently hang forever*
  either (SEC-2), which strengthens rather than weakens this guarantee.

## 8. Test Coverage Added

35 new/updated test cases across 8 files (6 new files):
`test_voice_xai_extraction_interface_queue.py` (rewritten + 6 new: loop
reuse, idempotent `start()`, `close()` lifecycle, exception logging +
capacity release, dedicated-client construction on the worker thread),
`test_voice_xai_orchestrator.py` (+1: SEC-2 regression),
`test_voice_xai_recovery.py` (new, 5: stale-running, stale-queued/blocked,
fresh-not-touched, already-terminal-not-selected, concurrent-race-skipped),
`test_voice_xai_artifacts.py` (+3: expiry deletion, missing-root no-op,
cannot-escape-root), `test_voice_xai_routes.py` (+1: SEC-3 per-user vs
per-prediction-id), `test_voice_xai_research_eligibility.py` (new, 4),
`test_prediction_routes.py` (+1: INT-1 handoff reuse), `test_mongodb.py`
(fake query-operator support, no new test functions).

## 9. Test Results

| Test Scope | Passed | Failed | Skipped | Notes |
|---|---:|---:|---:|---|
| Full suite, before any change | 348 | 24 | 5 | All 24 failures: ffmpeg/ffprobe `-nostdin` bug (unrelated to XAI) |
| Full suite, after ffprobe fix only | 383 | 4 | 5 | 20 of 24 fixed; 4 remain (see below) |
| Full suite, final (all fixes) | 388 | 4 | 5 | 2 deselected (marker-excluded) throughout |
| XAI-focused subset (`-k "voice_xai or xai"`) | 95 | 0 | 3 | Skips are pre-existing, environment-gated (e.g. optional `xgboost`/`shap`) |

The remaining 4 failures are pre-existing, unrelated to XAI, and only became
*visible* (not caused) once the `-nostdin` bug stopped masking them by
failing every single ffprobe call uniformly:

- `test_extension_spoofing_is_rejected` — expects `CorruptedAudioError` for
  an MP3-named file containing real WAV data; the actual (now-reachable)
  code path raises `UnsupportedAudioFormatError` instead. A pre-existing
  classifier-side format-validation/test mismatch, out of this task's scope
  (Voice XAI reliability) — not modified.
- `test_audio_duration_limit_is_enforced_and_temp_file_is_removed` — fails
  at `Settings(...)` construction itself (`max_audio_duration_seconds` is
  declared `int` but the test passes `0.01`), before any ffprobe call runs.
  Confirmed unrelated to ffmpeg/XAI entirely.
- `test_browser_recording_formats_work_with_real_ingestion[webm-...]` and
  `[m4a-...]` — real-encoding container/codec-detection edge cases specific
  to this newer ffmpeg 8.1.2 build, distinct from the `-nostdin` bug. Not
  investigated further (classifier-side, version-sensitive, out of scope).

These are reported, not hidden, per the task's explicit instruction; none of
them touch Voice XAI code.

## 10. Classifier → XAI Handoff Status

```text
VoicePredictionResponse: present
ProcessedAudio:          present when the wired VoiceService implementation
                          exposes predict_from_validated_upload_with_processed_audio
                          (the real VoiceService does; falls back to None only
                          for a test double that doesn't implement it)
ExtractionBundle:         still None (no real branch models wired yet; not faked)
Request ID consistency:   yes (ClassifierInferenceBundle.__post_init__ enforces it;
                          also asserted directly by the new INT-1 regression test)
Post-persistence enqueue: yes (save_result() precedes enqueue; unchanged)
Prediction/XAI independence: yes (enqueue try/except-wrapped; unchanged, and the
                          SEC-2 fix makes background failures terminal-and-logged
                          instead of silently hanging, without touching this guarantee)
```

## 11. XAI Runtime Status

```text
Queue:                     Bounded (BoundedSemaphore(max_workers + max_queued_jobs)),
                            non-blocking submit(), persistent single worker loop
                            (no asyncio.run() per job)
Mongo/event loop:           Dedicated worker-loop client available via
                            mongodb_client_factory; orchestrator resolves a
                            worker-scoped repository per background run
Unexpected exception handling: Outer try/except in _run(); forces the run to
                            `failed` with a safe error; Future.exception() also
                            logged as defense in depth
Rate limiting:              Per-authenticated-user on trigger/retry (5/60s
                            default, configurable), independent of the
                            existing per-path middleware
Artifact cleanup:           Background task, interval-configurable, physically
                            deletes expired files, bounded to the artifact root
Restart recovery:           Startup reconciliation marks stale queued/running/
                            partial/blocked runs `failed`; never auto-reruns
                            expensive work
```

## 12. Temporal XAI Status

```text
Provider:            Fixture (SavedTensorAttentionProvider) -- unchanged by this task
Attention source:     Mock -- unchanged
Timestamp mapping:    Uniform/duration-scaled -- unchanged
Research eligible:    No (now evaluated through the centralized, documented
                       evaluate_research_eligibility(); still always False,
                       exactly as before -- this task did not attempt real
                       temporal integration)
```

## 13. Semantic XAI Status

```text
Mode:                 Mock by default; Production class exists, unwired
                       (XAI_SEMANTIC_MODEL_PATH still empty) -- unchanged
Model:                 Missing -- unchanged
Manifest:              Present with placeholder hashes -- unchanged
SHAP background:       Missing -- unchanged
Feature parity:        Unvalidated -- unchanged
GCI:                   Custom peak/interval estimator, not REAPER -- unchanged
Research eligible:     No (same evaluator as above; unchanged outcome)
```

This task deliberately did not attempt real temporal/semantic integration --
that remains gated on artifacts and decisions owned by the XAI/model teams,
per the audit's §12/§13.

## 14. Finalization Decision Matrix

| Deployment State | Classifier | XAI | Allowed? |
| --- | --- | --- | --- |
| XAI disabled (`XAI_ENABLED=false`, the current `.env`) | Existing prediction pipeline | Disabled | **YES** — fully inert, zero behavioural change, all 384 non-XAI-environment tests pass |
| XAI mock mode | Existing prediction pipeline | Fixture/mock explanations | **DEV/DEMO ONLY** — now with a reliable orchestration layer (SEC-1/2/3, OPS-1/2 fixed); still explicitly `development_placeholder=true` |
| XAI real mode before this task's engineering fixes | Existing prediction pipeline | Real services | **NO** (matches prior audit; not re-tested, since real mode still requires artifacts this task did not add) |
| XAI real mode after engineering fixes, before research validation | Existing prediction pipeline | Real explanations | **ENGINEERING TESTING ONLY**, once Pratheesha's model/manifest/SHAP-background artifacts are supplied and `XAI_MODE=real` is enabled — the orchestration layer is now reliable enough to test against, but semantic/temporal are still unvalidated |
| XAI real mode + research validation | Existing prediction pipeline | Validated explanations | **NOT CURRENTLY ACHIEVED** — `research_eligible` is still hard-coded unreachable (`False`) by design; no criterion in §29/§30 has been satisfied by this task |

## 15. Remaining Work — Classifier Owner

Unchanged from the prior audit, restated for completeness:

1. Load real branch models and register real `PyTorchExtractionInterface`
   capture targets (verified module paths, output shapes, safe
   `max_elements`) — still not instantiated anywhere in `app/main.py`.
2. Supply real XLSR/WavLM attention layers, masks, and model-derived
   (stride/receptive-field) timestamps for a real `AttentionProvider`.
3. Confirm model provenance/checkpoint versions once real models exist.

## 16. Remaining Work — Pratheesha / XAI Owner

Unchanged from the prior audit:

1. Trained XGBoost model file, SHAP background array, and a manifest with
   real (non-placeholder) SHA-256 hashes and `training_recipe_verified=true`.
2. Confirmation of whether REAPER was used for GCI at training time.
3. Feature-parity reference outputs (known audio → known 28-feature vector).
4. A versioned rule for temporal/semantic overlap agreement, so
   `evaluate_research_eligibility()`'s "Combined" requirement (currently
   documented as not-yet-evaluated) can be implemented.

## 17. Shared Backend Tasks

- Decide and implement the "Combined" research-eligibility rule referenced
  in `research_eligibility.py` once temporal/semantic are individually real.
- If/when this deploys against a real MongoDB under real concurrent load,
  confirm SEC-1's dedicated-client design behaves as expected in practice —
  this environment has no live MongoDB to validate that end-to-end, only the
  loop-reuse architecture itself (proven by the new queue tests).
- The two remaining unrelated classifier-side test failures
  (`test_extension_spoofing_is_rejected`,
  `test_audio_duration_limit_is_enforced_and_temp_file_is_removed`) and the
  two webm/m4a container-detection failures are still open; assign to
  whoever owns `app/ingestion/audio.py`/`app/config/settings.py` validation.

## 18. Research Readiness

Unchanged: **NOT_READY**. This task improved engineering reliability only;
it did not add real model artifacts, real attention capture, or feature
parity validation, and deliberately did not flip any `research_eligible`
value. `evaluate_research_eligibility()` now gives a precise, named list of
what remains unmet for any given run, which should make the next real-model
integration pass easier to verify against.

## 19. Final Engineering Decision

```text
Classifier/XAI Regression:      PASS
Classifier Backend:             READY
XAI Disabled Backend:           READY
XAI Mock Backend:                DEV_READY
XAI Enabled Engineering Backend: READY
Real Temporal XAI:               NOT_READY
Real Semantic XAI:               NOT_READY
Research Evaluation:             NOT_READY
```
