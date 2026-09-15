# Idempotency Key Fix

Date: 2026-08-19
Scope: `POST /api/v1/predictions` idempotency handling (backend only).

## Root Cause

The 409 was **correct backend behaviour reacting to a reused client key**, not a
backend defect.

`POST /api/v1/predictions` accepts an **optional multipart form field**
`idempotency_key` (there is no `Idempotency-Key` HTTP header anywhere in this
codebase). When a key is supplied, the submission service reserves an
owner-scoped prediction document keyed by `(owner_user_id, idempotency_key)`.
On a hit, the stored *logical request* fingerprint is compared with the incoming
one; a mismatch raised `HTTPException(409, "Idempotency key was already used for
a different request.")`, which the centralised handler rendered with the generic
code `http_error` — the exact payload reported.

Three factors made this easy to trigger during manual testing and hard to read:

1. **Swagger UI keeps typed form values between `Execute` presses.** Once a key
   is typed into the `idempotency_key` box, every subsequent upload in that page
   session resubmits the same key. Changing only the file changes
   `submitted_filename`, so the second upload conflicts.
2. **`DEV_AUTH_BYPASS=true` collapses every local caller into one synthetic
   owner** (`dev-local-user`, `app/auth/clerk_auth.py:243`). Because idempotency
   is owner-scoped, keys claimed during any earlier local session stay claimed in
   MongoDB across restarts — a key used weeks ago still conflicts.
3. **Copy-paste sample keys.** The frontend developer/docs pages ship static
   sample keys (`demo-001`, `ext-001` in `frontend/src/components/docs/docs-data.ts`
   and `frontend/src/app/dashboard/developer/page.tsx`). Pasting one of those
   curl lines twice with different audio reproduces the 409 immediately.
   Frontend is out of scope for this task and was **not** modified.

The generic `http_error` code plus a message that did not say *what to do next*
turned an intentional guard into a mystery.

## Current Idempotency Flow

`app/api/v1/prediction_routes.py:94` reads the optional `idempotency_key` form
field →
`PredictionSubmissionService.submit()` (`app/services/prediction_submission_service.py:73`):

1. `_owner_user_id(principal)` — owner scope (Clerk `user_id`, or API-key
   subject on the external route).
2. `_logical_request(...)` — the fingerprint (see below).
3. `_sanitize_idempotency_key(...)` — `None`/blank/whitespace ⇒ `None`; longer
   than 120 chars ⇒ HTTP 422.
4. `_reserve_or_create_prediction(...)`:
   - **no key** → `persistence.create_queued(...)` → always a brand-new
     prediction (`idempotency_key: null` stored).
   - **key present** → `repository.reserve_prediction(...)`
     (`app/repositories/mongodb.py:114`) does a
     `find_one_and_update({owner_user_id, idempotency_key}, {$setOnInsert: doc},
     upsert=True)`. `created` is `True` only when the returned document carries
     this request's `request_id`.
5. `created == False` → `_validate_replay_logical_request(...)`: identical
   fingerprint ⇒ replay the stored document via `_response_from_document(...)`
   (no re-validation, no storage upload, no inference); different fingerprint ⇒
   409.

Storage: `predictions` collection, fields `idempotency_key` and
`idempotency_logical_request`, with the unique partial index
`uniq_predictions_owner_idempotency_key` on `(owner_user_id, idempotency_key)`
(`app/database/indexes.py:46`). Nothing about this storage changed.

The same service backs `POST /api/v1/external/predictions` (API-key principals),
so the semantics are shared.

## Swagger Behavior

Audited `app/main.py:_add_openapi_examples` and the generated `/openapi.json`:

- **No static default was ever submitted.** The generated schema for
  `idempotency_key` had **no `default`, no `example`, no `examples`** — only a
  description. Swagger UI therefore never pre-filled or auto-sent a key.
- The existing multipart examples (`humanVoice`, `syntheticVoice`, `repoSample`)
  are `summary`/`description` only, with no `value`, so they cannot inject a key
  either.
- The real Swagger trap is **value stickiness**: a key typed once is resubmitted
  on every later `Execute` in the same page session.

Because no static default existed, nothing had to be removed. The field
documentation was rewritten instead, and a UUID4 is mentioned **only inside the
description text** — deliberately *not* as `examples=[...]`, since Swagger UI
pre-fills form inputs from schema examples and that would have created exactly
the failure mode this task warns about. A code comment in
`app/api/v1/prediction_routes.py` records that reasoning so the example is not
"helpfully" promoted to a schema example later.

## Request Fingerprint

`_logical_request()` (`app/services/prediction_submission_service.py:616`)
returns:

| Field | Included | Note |
| --- | --- | --- |
| `source_type` | yes | `dashboard_upload` / `live_recording` / `public_api` |
| `client_filename` | yes | trimmed, `None` when blank |
| `submitted_filename` | yes | `_safe_logical_filename(file.filename)`; path-bearing or unsafe names collapse to `None` |
| owner identity | implicit | the reservation query is scoped by `owner_user_id`, so the key is owner-scoped by construction |
| audio bytes / SHA-256 | **no** | see below |
| multipart boundary, temp path, `request_id` | no | correctly excluded |

**Finding (documented, not redesigned):** the fingerprint is *filename-based,
not content-based*. Reservation happens **before** the upload stream is read, so
no content hash is available at that point. Consequences:

- Same key + same filenames + **different audio bytes** ⇒ treated as a replay
  and the earlier prediction is returned.
- Same audio + different filename ⇒ treated as a different logical request.

This is the existing, tested behaviour (`tests/test_prediction_routes.py:268`,
`:321`). Making it content-based would mean buffering/hashing the whole upload
before reservation, changing the streaming-validation design and the
persistence/reservation ordering — out of scope here. The behaviour is now
pinned by an explicit test
(`test_replay_matching_is_filename_based_not_content_based`) so the trade-off is
visible rather than accidental, and a fix can be scoped deliberately later.

## Fix Implemented

1. **Dedicated error type** — `IdempotencyConflictError`
   (`app/core/exceptions.py`): `status_code = 409`,
   `error_code = "idempotency_conflict"`, actionable public message.
2. **Centralised handler** — `idempotency_conflict_handler`
   (`app/core/exception_handlers.py`) registered with the existing
   `register_exception_handlers()`; reuses `_error_response`, so the error
   envelope (`request_id` + `error.{code,message,details}`) is unchanged.
   The 409s raised elsewhere (XAI routes, rerun service, job runner) still map
   to `http_error` — no status-code-wide remapping was done.
3. **Service raises the typed error** —
   `_validate_replay_logical_request(...)` now raises
   `IdempotencyConflictError` instead of a bare `HTTPException(409)`.
4. **Development-only diagnostics** — `_idempotency_conflict_details(...)`
   populates `error.details` **only** when
   `APP_ENV=development AND DEV_AUTH_BYPASS=true`, with `changed_fields`,
   `existing_prediction_id`, and both logical requests. Production keeps
   `details: null`.
5. **Swagger/OpenAPI documentation** — rewritten `idempotency_key` field
   description; 409 response description now names `idempotency_conflict`; an
   `idempotency_conflict` response example and an "Idempotency" paragraph were
   added to the operation description in `app/main.py`. No default value, no
   schema example.
6. **Manual-testing docs** — `docs/setup/swagger-real-world-testing.md` now
   explains: leave the field blank for normal testing; one fresh `uuidgen` per
   new upload; reuse only to retry the exact same request; Swagger's sticky form
   values and the shared `dev-local-user` owner as the usual causes of a stale
   key.

**Deliberately not done** (and why):

- Idempotency was **not** disabled, relaxed, or made non-conflicting anywhere.
- **No dev-only server-side key generation** (task option C). Option A already
  applies: the field is optional and a keyless request always creates a new
  prediction, so generating a UUID server-side would only write a random key no
  client could ever retry with — added risk, zero benefit.
- **A client-supplied key is never replaced or regenerated**, in any
  environment.
- No `Idempotency-Key` *header* alias was introduced: adding one would silently
  give idempotency semantics (and new 409s) to existing integrations whose HTTP
  clients set that header by default.
- Frontend sample keys were left alone (frontend is explicitly out of scope);
  the live frontend flow already sends `crypto.randomUUID()` per submission
  (`frontend/src/features/predictions/analyze-page.tsx:65`).

## Development Behavior

`APP_ENV=development` + `DEV_AUTH_BYPASS=true`:

- `idempotency_key` omitted or blank ⇒ normal new prediction every time; nothing
  is generated or stored (`idempotency_key: null`). This is the recommended
  manual-testing mode.
- A reused key with a different logical request still returns **409
  `idempotency_conflict`** — dev mode does not weaken the guard.
- The 409 body additionally carries `error.details` naming the changed fields
  and the prediction the key is already bound to, so a stale key is obvious.
- All local requests share owner `dev-local-user`, so keys stay claimed across
  restarts. Documented in the Swagger testing guide.

## Production Behavior

- Unchanged reservation, replay, and conflict semantics.
- `dev_auth_bypass_enabled()` is hard-gated on `app_env == "development"`, so
  `DEV_AUTH_BYPASS=true` in a production settings object has no effect —
  verified by a test that constructs production settings with the flag on.
- `error.details` stays `null`; no other request's stored shape is exposed.
- Conflicting requests never create a second prediction, never run inference,
  and never upload to storage.

## Error Response

Before:

```json
{"request_id":"...","error":{"code":"http_error","message":"Idempotency key was already used for a different request.","details":null}}
```

After (production):

```json
{"request_id":"...","error":{"code":"idempotency_conflict","message":"Idempotency key was already used for a different request. Use a new unique idempotency_key for a new submission, or resend the exact original request to replay its result.","details":null}}
```

After (development + `DEV_AUTH_BYPASS=true`) — same code/message, plus:

```json
"details":{"development_hint":"...","changed_fields":["submitted_filename"],"existing_prediction_id":"...","existing_logical_request":{...},"submitted_logical_request":{...}}
```

HTTP status stays **409**. The envelope shape is unchanged; only `error.code`
became specific. This matches what the frontend already expects
(`frontend/src/lib/api/__tests__/errors.test.ts` asserts an
`idempotency_conflict` code), so the change closes a client/server mismatch
rather than creating one.

## Tests

New: `backend/tests/test_prediction_idempotency.py` — 16 tests, all passing.

| # | Requirement | Test |
| --- | --- | --- |
| 1 | Missing key follows the contract | `test_missing_or_blank_key_creates_a_new_prediction_every_time` (`None`, `""`, `"   "`) |
| 2 | New key + request A ⇒ success | `test_new_key_processes_request_normally` |
| 3 | Same key + identical A ⇒ replay | `test_same_key_and_identical_request_replays_without_running_twice` |
| 4 | Same key + different request ⇒ 409 | `test_same_key_with_a_different_request_returns_409_conflict` (file, `client_filename`, `source_type`) |
| 5 | Different key + request B ⇒ success | `test_different_key_for_a_different_request_succeeds` |
| 6 | Same audio + different key ⇒ new request | `test_same_audio_with_a_different_key_creates_a_new_prediction` |
| 7 | OpenAPI forces no static key default | `test_openapi_does_not_force_a_static_idempotency_key_default`, `test_openapi_documents_when_to_reuse_an_idempotency_key` |
| 8 | Dev mode does not weaken protection | `test_development_bypass_still_rejects_a_reused_key` |
| 9 | Production rejects same key + different request | `test_production_rejects_a_reused_key_and_hides_diagnostics` |
| 10 | No server-side key generation / no key replacement | `test_development_bypass_does_not_generate_a_key_for_keyless_requests` + the assertions in #8 |
| — | Fingerprint semantics documented | `test_replay_matching_is_filename_based_not_content_based` |

The module resets the shared per-process prediction rate limiter between tests
(`app.state.rate_limiter`), otherwise the 20-requests-per-window admission limit
masks the assertions.

## Regression Checks

Full backend suite: `570 passed, 4 failed, 36 skipped, 3 deselected`.

The 4 failures are **pre-existing and unrelated** to idempotency — all are real
local-FFmpeg ingestion tests:

- `tests/test_audio_preprocessing.py::test_extension_spoofing_is_rejected` —
  `UnsupportedAudioFormatError: File extension does not match the detected audio
  container` under local ffmpeg 8.1.2.
- `tests/test_audio_preprocessing.py::test_audio_duration_limit_is_enforced_and_temp_file_is_removed`
  — the test passes `max_audio_duration_seconds=0.01` to an `int` settings field
  (pydantic `int_from_float`).
- `tests/test_prediction_routes.py::test_browser_recording_formats_work_with_real_ingestion[webm…]`
  and `[m4a…]` — 400 from the audio-validation stage, before any idempotency
  logic runs.

These touch only `app/ingestion/audio.py` / local tooling, which this change does
not modify; `test_audio_preprocessing.py` does not import any changed module.

Not modified: AASIST / AASIST-Light V2, CNN, SSL, glottal, fusion weights,
inference behaviour, production auth behaviour, persistence schema, XAI, and
frontend.

## Verdict

The backend guard was working as designed; the defects were **observability and
documentation**, plus a manual-testing workflow (Swagger's sticky form field +
one shared dev-bypass owner + copy-pasted sample keys) that made a reused key
easy to send and hard to diagnose.

Idempotency protection is fully preserved: same key + same request still
replays, same key + different request still returns 409 in every environment,
and no key is ever generated or substituted on the client's behalf. What changed
is that the rejection now carries the dedicated `idempotency_conflict` code, an
actionable message, development-only diagnostics, and Swagger documentation that
tells callers to leave the field blank or send a fresh UUID per submission.
