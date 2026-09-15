# Phase 2 Backend XAI Stabilization Report

## A. Executive Summary

Backend: READY for local import and route registration.
Prediction: READY for the existing single-instance backend path; full real inference smoke is environment-blocked by missing `ffprobe`.
Voice XAI: READY FOR FRONTEND INTEGRATION for the stabilized response contract and deterministic evidence pipeline.
Frontend Contract: STABLE for classifier snapshot, temporal/semantic/report/provenance, and non-authoritative narrative fields.
Real Mode: STARTUP ENVIRONMENT BLOCKED by missing optional `shap`; failure is closed and does not fall back to mock.
Tests: Focused XAI/fusion suites pass; full suite remains environment-blocked by missing `ffprobe` and `resampy`.

## B. Fixed Issues

| Audit ID | Fix | Files Changed | Validation |
| -------- | --- | ------------- | ---------- |
| CRIT-1 | Imported `ConstrainedFourBranchFusion` from `app.utils.fusion` so `app.main` imports. | `app/services/voice_service.py` | `.venv/bin/python -c "import app.main"` PASS |
| CRIT-2 | Glottal `used_for_primary_decision` is derived from actual `FusionResult.contributing_branches`, not branch-name assumptions. | `app/schemas/xai.py`, `app/voice_xai/orchestrator.py` | Glottal V3/legacy/failed/disabled regression tests PASS |
| CRIT-3 contract decision | Added stable `narrative` object with `status=not_available`, `text=null`, and `authoritative=false`; deterministic report remains authoritative. | `app/schemas/xai.py`, `app/voice_xai/persistence/mongodb.py` | XAI schema and orchestrator tests PASS |
| HIGH-4 | Active XAI run creation is idempotent through Mongo-style atomic upsert and partial unique index for queued/running/partial runs. | `app/voice_xai/persistence/mongodb.py`, `app/database/indexes.py` | Repository idempotency tests PASS |
| HIGH-5 | Temporal smoothing now consumes `xai_temporal_smoothing_ms` and converts ms to seconds. | `app/voice_xai/temporal/xlsr_attention_service.py` | Smoothing regression test PASS |
| HIGH-6 | Temporal calibration can fail closed on model hash mismatch; checked-in calibration now records the local SSL artifact SHA-256. | `app/voice_xai/temporal/calibration.py`, calibration JSON | Model-hash drift tests PASS |
| MED-1 | XAI configuration hash now includes behaviorally significant values and content hashes, not absolute paths. | `app/voice_xai/orchestrator.py` | Stability/change/relocation hash tests PASS |
| HIGH-8/HIGH-9 | Existing dummy and class-mapping tests were retained; no silent inversion changes were made. | tests retained | Existing focused suites PASS |
| HIGH-10 | Public artifact endpoint refuses `intermediate_representations`; normal JSON/PNG artifacts remain available. | `app/voice_xai/orchestrator.py` | Covered by artifact boundary behavior |

## C. Remaining Issues

Frontend blockers: none identified for the stabilized XAI response shape.

Research blockers: real XAI startup requires optional `shap`; semantic resampling test requires `resampy`; full real prediction validation requires local `ffprobe`.

Production-only blockers: distributed queue, shared rate limiter, shared/object artifact storage, hard compute cancellation, multi-worker GPU scheduling, and production metadata retention policy.

## D. Test Results

| Command | Passed | Failed | Skipped | Blocked |
| ------- | ------ | ------ | ------- | ------- |
| `.venv/bin/python -c "import app.main"` | PASS | 0 | 0 | 0 |
| `.venv/bin/python -m pytest --collect-only -q` | 734 collected | 0 | 7 deselected | 0 |
| `.venv/bin/python -m pytest tests/test_fusion.py tests/test_fusion_v3.py tests/test_voice_xai_routes.py tests/test_voice_xai_artifacts.py tests/test_voice_xai_jobs_queue.py tests/test_prediction_persistence.py -q` | 71 | 0 | 1 deselected | 0 |
| `.venv/bin/python -m pytest tests/test_voice_xai_orchestrator.py tests/test_voice_xai_repository.py tests/test_voice_xai_real_temporal.py tests/test_voice_xai_recovery.py -q` | 35 | 0 | 0 | 0 |
| `.venv/bin/python -m pytest -q` | 683 | 5 | 36 skipped, 7 deselected | 10 errors |
| `.venv/bin/python -m ruff check .` | 0 | 494 lint findings | 0 | Existing repo-wide lint debt |

Full-suite blockers: `ffprobe is not available` causes AASIST standalone failures/errors; `resampy` missing causes one semantic feature resampling failure.

## E. Real Mode Validation

Semantic artifact integrity: blocked before completion because optional `shap` is not installed.
Temporal calibration: validates pipeline parameters, smoothing, and SSL artifact SHA-256 when the model artifact is present.
Startup: normal `app.main` import passes; real-mode XAI service startup fails closed on missing `shap`.
Real inference: not executed because `ffprobe`/real XAI optional dependencies are missing.
Fallback behavior: no real-to-mock silent fallback observed; missing real dependency raises `SemanticArtifactCompatibilityError`.

## F. Frontend Contract

Representative response shape:

```json
{
  "schema_version": "voice-xai-api-v1",
  "explanation_id": "explanation-123",
  "prediction_id": "prediction-123",
  "request_id": "request-123",
  "status": "queued",
  "component_statuses": {"temporal": "queued", "semantic": "queued", "report": "queued"},
  "classifier_snapshot": {
    "verdict": "spoof",
    "spoof_probability": 0.8,
    "bonafide_probability": 0.2,
    "confidence": 0.8,
    "decision_threshold": 0.5,
    "contains_dummy_branches": false,
    "research_eligible": false,
    "branches": [
      {"branch_name": "glottal", "status": "success", "mode": "real", "spoof_probability": 0.7}
    ],
    "auxiliary_evidence": {
      "glottal_spoof_probability": 0.7,
      "used_for_primary_decision": true
    }
  },
  "combined_report": {
    "status": "completed",
    "authoritative": true,
    "finding": "Deterministic finding text",
    "primary_evidence": "Evidence summary",
    "quality_checks": "Quality summary",
    "limitation": "Known limitation",
    "recommendation": "Recommended action",
    "disclaimer": "Decision-support evidence only."
  },
  "narrative": {
    "status": "not_available",
    "authoritative": false,
    "model_id": null,
    "generated_at": null,
    "prompt_version": null,
    "evidence_references": [],
    "text": null,
    "failure_reason": null
  },
  "quality": {
    "temporal_localization_iou": {
      "status": "not_applicable",
      "value": null,
      "scope": "per_analysis",
      "reason": "Not measured for this user submission."
    }
  },
  "provenance": {
    "pipeline_version": "voice-xai-pipeline-v1",
    "configuration_hash": "<sha256>",
    "configuration_hash_inputs": ["mode", "pipeline_version", "temporal_smoothing_ms"]
  }
}
```

## G. Production Deferred Work

- Distributed queue
- Shared rate limiter
- Shared/object artifact storage
- Hard compute cancellation
- Multi-worker GPU strategy
- Metadata retention strategy for historical explanation documents

## Phase 2 Final Verdict

Backend: READY
Prediction: READY, with real smoke blocked by local `ffprobe`
Voice XAI: READY FOR FRONTEND INTEGRATION
Real XAI: ENVIRONMENT BLOCKED
Frontend Contract: STABLE
Research Reliability: IMPROVED, still blocked by optional dependency/runtime validation
Tests: FOCUSED PASS, FULL SUITE ENVIRONMENT BLOCKED
Production: NOT YET REQUIRED

Frontend integration can start: YES

Remaining blockers:
1. Install/verify `ffprobe` for real audio inspection and AASIST standalone validation.
2. Install optional XAI dependencies, including `shap` and `resampy`, for real semantic startup/inference.
3. Run end-to-end real prediction plus XAI once local model/audio infrastructure is complete.

Next recommended phase: Phase 3 — Voice XAI Frontend Integration
