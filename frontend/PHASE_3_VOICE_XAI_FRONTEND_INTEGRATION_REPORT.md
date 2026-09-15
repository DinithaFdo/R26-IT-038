# Phase 3 Voice XAI Frontend Integration Report

Date: 2026-08-23

## Verdict

Phase 3 is complete. The frontend now matches the stabilized Phase 2 Voice XAI backend contract, exposes explanation evidence in the dashboard and report views, avoids duplicate explanation triggers/retries, and validates the requested edge states with passing frontend checks.

## Backend Contract Verified

Verified against `backend/app/api/v1/xai_routes.py` and `backend/app/schemas/xai.py`.

- `POST /api/v1/me/predictions/{prediction_id}/explanation`
- `GET /api/v1/me/predictions/{prediction_id}/explanation`
- `GET /api/v1/me/predictions/{prediction_id}/explanation/temporal`
- `GET /api/v1/me/predictions/{prediction_id}/explanation/semantic`
- `GET /api/v1/me/predictions/{prediction_id}/explanation/report`
- `GET /api/v1/me/predictions/{prediction_id}/explanation/artifacts/{artifact_id}`
- `POST /api/v1/me/predictions/{prediction_id}/explanation/retry`

## Frontend Integration

- Updated Voice XAI TypeScript models to include Phase 2 schema fields:
  - `narrative`
  - `classifier_snapshot.auxiliary_evidence`
  - extra artifact kinds
  - local temporal region probabilities
  - temporal threshold/count/label metadata
  - report `authoritative`
  - provenance `configuration_hash_inputs`
- Added a classifier snapshot panel that clearly shows whether Glottal evidence was used for the primary decision or remained auxiliary-only.
- Added a non-authoritative AI narrative panel, including explicit `not_available` rendering when narrative generation is disabled or absent.
- Extended the temporal evidence panel with candidate-region metadata, attention threshold, and local spoof/bonafide probabilities without fabricating missing values.
- Extended reproducibility output with configuration hash inputs.
- Hardened explanation trigger and retry handlers with local in-flight guards to prevent accidental duplicate POST calls before React Query pending state re-renders.
- Bounded 408/429 retry behavior to the same limited retry budget used for transient service errors.
- Cleaned `postcss.config.mjs`, which contained appended executable JavaScript after the valid config and blocked Vitest/PostCSS loading.

## Evidence Semantics Preserved

- No frontend fixture is used as a runtime fallback.
- Missing quality metrics render as unavailable/not computed, never as zero.
- SHAP direction remains sourced from the backend `direction` field.
- Combined findings remain backend-composed; the UI does not synthesize scientific conclusions.
- Narrative content is presented only as a non-authoritative reading aid.
- Artifact downloads continue through the authenticated owner-scoped backend route.

## Test Coverage Added or Verified

- Lifecycle status mapping: queued/running/partial polling, completed/failed/blocked terminal states.
- Glottal role true/false rendering.
- Dummy/development evidence and disabled/unavailable XAI messaging.
- Failed or unavailable temporal/semantic/report component rendering.
- `temporal_localization_iou` null/not-computed handling.
- Narrative `not_available` handling.
- 429 `Retry-After` parsing and bounded retry behavior.

## Validation

All required frontend commands were run from `frontend/`.

- `npm run typecheck` - passed
- `npm run lint` - passed
- `npm run test` - passed, 24 files / 136 tests
- `npm run build` - passed

## Notes

- The root repository currently reports `frontend/` as untracked because the frontend is its own nested git working tree.
- `frontend/.env` was already untracked and was not modified.
