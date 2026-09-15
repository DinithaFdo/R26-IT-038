# AASIST-Light V2 Phase 6 API Regression

## Scope

Phase 6 validates the finalized AASIST-Light V2 backend integration through the existing public prediction, persistence, model-health/readiness, storage/upload, and XAI handoff surfaces.

No frontend files, checkpoint files, model artifacts, fusion weights, public API schemas, DB schemas, XAI algorithms, queues, deployment infrastructure, AASIST architecture, AASIST preprocessing, or class mappings were intentionally changed in this phase.

## Public Prediction Runtime Path

Traced existing `POST /api/v1/predictions` flow:

1. Public route: `backend/app/api/v1/prediction_routes.py::create_prediction`
2. Auth: `backend/app/auth/clerk_auth.py::require_clerk_user`
3. Multipart/form fields: `file`, `source_type`, optional `client_filename`, optional `idempotency_key`
4. Service dependency: `backend/app/api/dependencies.py::get_prediction_submission_service`
5. Submission orchestration: `backend/app/services/prediction_submission_service.py::PredictionSubmissionService.submit`
6. Reservation/persistence: `PredictionSubmissionService._reserve_or_create_prediction` -> `PredictionPersistenceService.create_queued`
7. Inline job boundary: `backend/app/services/prediction_job_runner.py::InlinePredictionJobRunner.run`
8. Validation/upload: `PredictionSubmissionService.execute_prediction` -> `save_validated_audio_upload_blocking`
9. Audio ingestion: `backend/app/ingestion/audio.py::save_validated_audio_upload`, `inspect_audio_file`, `preprocess_audio_file`
10. Shared audio object: `backend/app/ingestion/audio.py::ProcessedAudio`
11. Classifier service: `backend/app/services/voice_service.py::VoiceService.predict_from_validated_upload_with_processed_audio`
12. Branch registry: `backend/app/models/registry.py::ModelRegistry.models`
13. Runtime factory: `backend/app/models/factory.py::ModelFactory.create_all`
14. AASIST V2 branch: `backend/app/models/real/inference.py::build_aasist_loader`, `load_aasist_light_v2_checkpoint_strict`, `predict`
15. Other branches: CNN `lfcc_cnn_tcn`, SSL `ssl_sequence`, glottal `glottal`
16. Fusion: `backend/app/utils/fusion.py::FusionEngine.fuse`
17. Result persistence: `PredictionPersistenceService.save_result` -> `MongoPredictionRepository.save_prediction_result` in real DB mode
18. Response serialization: `PredictionSubmissionService._response_from_prediction`
19. Optional XAI handoff: `PredictionSubmissionService._enqueue_xai_without_affecting_prediction` -> `VoiceXaiOrchestrator.enqueue`

## API Contract Validation

Validated by `backend/tests/test_aasist_light_v2_api_regression.py::test_phase6_public_prediction_openapi_contract_has_no_aasist_v2_fields` and existing `backend/tests/test_openapi_contract.py`.

Result:
- Route remains `/api/v1/predictions`.
- Method remains `POST`.
- Request remains multipart/form-data.
- Public response model remains `PredictionSubmissionResponse`.
- Response fields remain `prediction_id`, `request_id`, `status`, `source_type`, `audio`, `branches`, `fusion`, `research_eligible`, `created_at`.
- Status-code surface remains `200`, `400`, `401`, `409`, `413`, `415`, `422`, `500`, `503`.
- No AASIST V2 internal public field was added.
- Error envelope remains `{request_id, error}`.

## Real AASIST V2 Participation

Real public workflow smoke test passed through FastAPI `TestClient`, test auth override, multipart upload, real ffprobe/ffmpeg validation, fake storage, real `VoiceService`, real canonical `aasist` branch, AASIST-Light V2 checkpoint, fake persistence, and XAI enqueue.

Observed AASIST result:
- Branch key: `aasist`
- Mode: `real`
- Status: `success`
- Final branch label: `spoof`
- Branch P(spoof): `0.9999964237213135`
- Fused P(spoof): `0.9999964237213135`
- Final fusion label: `spoof`
- Checkpoint-backed internal architecture: `aasist-light-v2-finalized-baseline`

## Response Validation

Successful public response validation:
- Existing schema matched.
- Final label remains under `fusion.prediction`.
- Fused score remains under `fusion.probabilities.spoof`.
- Branch result remains compatible with public `BranchPrediction`.
- No `aasist_v2`, `aasist_light_v2`, checkpoint path, raw tensor, or private implementation field leaked into the public response.
- Probability pairs were finite and approximately summed to 1.0.
- No NaN or Infinity serialization was observed.

## Persistence Regression

Persistence path inspected:
- `backend/app/services/prediction_persistence_service.py::PredictionPersistenceService`
- `backend/app/repositories/mongodb.py::MongoPredictionRepository`
- `MongoPredictionRepository._new_prediction_document`
- `MongoPredictionRepository.save_prediction_result`
- `backend/app/services/prediction_rerun_service.py`

Focused persistence regression verified with repository test doubles because local full Mongo tests are blocked by missing `pymongo`.

Result:
- Branch data remains stored in `branches`.
- Fusion data remains stored in `fusion`.
- AASIST branch remains stored as `model_name: "aasist"`.
- Fusion branch weights remain keyed by `aasist`.
- Model version/hash continues to live in metadata/provenance patterns, not as a public branch rename.
- Rerun document fields (`parent_prediction_id`, `rerun_reason`, `preprocessing_version`, `model_versions`) were inspected and not changed in Phase 6.

Real Mongo E2E: BLOCKED by missing `pymongo`.

## Model Health / Readiness

Validated with focused tests:
- Valid AASIST V2 checkpoint loads and reports `ready: true` after real inference.
- `checkpoint_configured: true`
- `checkpoint_valid: true`
- `model_version: checkpoint-c944c69f05a0`
- `architecture: aasist-light-v2-finalized-baseline`
- `research_ready: false`

Missing checkpoint behavior:
- Controlled not-ready state.
- `ready: false`
- `research_ready: false`
- `checkpoint_valid: false`
- No public schema change.

## XAI Handoff Regression

Inspected:
- `backend/app/services/prediction_submission_service.py::_enqueue_xai_without_affecting_prediction`
- `backend/app/voice_xai/contracts.py::ClassifierInferenceBundle`
- `backend/app/voice_xai/orchestrator.py::VoiceXaiOrchestrator.enqueue`
- `backend/app/voice_xai/CLASSIFIER_XAI_CONTRACT.md`

Validated:
- Prediction result is persisted before XAI enqueue.
- Handoff includes `prediction_id`, `request_id`, `owner_user_id`, `source_type`, prediction, optional extraction, and processed audio.
- The processed audio handoff reuses the exact classifier `ProcessedAudio` object when available.
- XAI enqueue failure does not change completed prediction status.
- No new AASIST-specific XAI algorithm or public XAI field was added.

## Storage / Upload Regression

Validated by the real endpoint smoke test and existing audio route tests:
- Multipart upload path remains unchanged.
- ffprobe/ffmpeg validation remains in `backend/app/ingestion/audio.py`.
- Temp validated upload cleanup remains in `PredictionSubmissionService.execute_prediction`.
- Storage upload remains via `AudioStorage.upload_audio`.
- Fake/no-op storage and Cloudinary-shaped metadata contracts remain compatible.
- AASIST V2's `unnormalised_waveform` addition is internal to `ProcessedAudio` and does not alter stored/uploaded media metadata.

Two existing browser-recording route tests are currently failing outside AASIST because the generated 0.25s fixtures are rejected before prediction. They were excluded from the clean Phase 6 regression pass and recorded as environment/test-contract issues.

## Error Handling Regression

Validated or inspected controlled cases:
- Unauthorized request: unchanged `401`, error code `authentication_failed`.
- Unsupported/corrupted/near-silent audio: existing audio exception handlers remain unchanged.
- Missing AASIST checkpoint: controlled branch/readiness failure, no schema change.
- AASIST inference failure: existing `predict_safe`/fusion fallback policy preserved.
- No valid branch result: unchanged `503`, error code `model_unavailable`.
- Error envelope remains `{request_id, error}`.

## Other Branch Regression

Verified identifiers and weights:
- CNN canonical branch remains `lfcc_cnn_tcn`, public model name `cnn_acoustic`.
- AASIST canonical branch remains `aasist`, public model name `aasist`.
- SSL canonical branch remains `ssl_sequence`, public model name `ssl_wavlm_xlsr`.
- Glottal canonical branch remains `glottal`, public model name `glottal_features`.
- Configured weights remain 0.25 each.
- Default development weights remain 0.25 each.
- No CNN, SSL, or glottal architecture/preprocessing/adapters were changed in Phase 6.

## Real Endpoint Smoke Test

REAL ENDPOINT SMOKE TEST: PASS

- Endpoint: `/api/v1/predictions`
- Method: `POST`
- Transport: FastAPI `TestClient`
- Auth: existing test Clerk dependency override; no real token or secret used
- Audio: `frontend/e2e/fixtures/sample-voice.wav`
- Response status: `200`
- Final label: `spoof`
- Fused score: `fusion.probabilities.spoof = 0.9999964237213135`
- AASIST participation: `aasist`, real, success
- Request ID: `30964864-fd34-4c82-b2ff-a86f51bd0592`
- Persistence result: fake repository document status `completed`; branch persisted as `aasist`
- XAI status: enqueue invoked; bundle captured with prediction id, request id, and processed audio
- Total request latency: `476.414 ms`

## Tests Added

Added `backend/tests/test_aasist_light_v2_api_regression.py`.

Coverage added:
1. Public prediction OpenAPI route/method/status/schema contract.
2. Unauthorized request regression.
3. Real AASIST V2 public endpoint smoke test.
4. Response schema and no internal V2 field leakage.
5. Persistence key/shape regression with test doubles.
6. Valid checkpoint model health/readiness.
7. Missing checkpoint not-ready behavior.
8. XAI handoff processed-audio and failure handling.
9. Branch-failure fusion policy.
10. CNN/SSL/glottal identifier and weight regression.
11. No-valid-branch public error envelope.

## Test Results

Commands run:

```text
pytest backend/tests/test_aasist_light_v2_api_regression.py -q
10 passed, 13 warnings
```

```text
pytest backend/tests/test_aasist_light_v2_api_regression.py::test_phase6_real_aasist_v2_public_endpoint_smoke_exercises_existing_path -q -s
1 passed, 13 warnings
```

```text
pytest backend/tests/test_aasist_light_v2_api_regression.py backend/tests/test_aasist_light_v2_fusion_validation.py backend/tests/test_aasist_light_v2_standalone_inference.py backend/tests/test_aasist_light_v2_checkpoint_integration.py backend/tests/test_real_model_architectures.py backend/tests/test_real_model_preprocessing.py -q
100 passed, 9 skipped, 13 warnings
```

```text
pytest backend/tests/test_aasist_light_v2_api_regression.py backend/tests/test_prediction_routes.py backend/tests/test_prediction_persistence.py backend/tests/test_prediction_job_runner.py backend/tests/test_voice_service.py backend/tests/test_health.py backend/tests/test_openapi_contract.py backend/tests/test_model_runtime_phase3.py backend/tests/test_real_model_adapters.py backend/tests/test_fusion.py backend/tests/test_audio_preprocessing.py backend/tests/test_ffmpeg_audio_formats.py backend/tests/test_voice_xai_contract.py backend/tests/test_voice_xai_research_eligibility.py backend/tests/test_voice_xai_schemas.py backend/tests/test_voice_xai_extraction_interface_contracts.py backend/tests/test_voice_xai_extraction_interface_queue.py backend/tests/test_voice_xai_report_service.py backend/tests/test_voice_xai_semantic.py backend/tests/test_voice_xai_semantic_windows.py -k 'not checkpoint_paths_cannot_escape_the_model_root and not extension_spoofing_is_rejected and not audio_duration_limit_is_enforced_and_temp_file_is_removed and not browser_recording_formats_work_with_real_ingestion and not trio' -q
200 passed, 15 skipped, 17 deselected, 13 warnings
```

```text
python -m compileall backend/app backend/tests/test_aasist_light_v2_api_regression.py
PASS
```

Full backend pytest:

```text
pytest backend/tests -q
BLOCKED during collection: 7 errors from missing pymongo
```

Ruff:

```text
python -m ruff check backend/tests/test_aasist_light_v2_api_regression.py
BLOCKED: No module named ruff
```

## Environment Blockers

- `pymongo` is not installed, blocking collection for `tests/test_mongodb.py` and Mongo-dependent XAI tests:
  - `tests/test_voice_xai_artifacts.py`
  - `tests/test_voice_xai_orchestrator.py`
  - `tests/test_voice_xai_recovery.py`
  - `tests/test_voice_xai_repository.py`
  - `tests/test_voice_xai_routes.py`
  - `tests/test_voice_xai_status.py`
- `trio` is not installed, so AnyIO trio parametrizations fail in existing tests unless deselected or forced to asyncio.
- `ruff` is not installed.
- Existing unrelated audio/browser-recording tests reject generated 0.25s fixtures before prediction.
- Existing unrelated tests remain documented from earlier phases:
  - `test_extension_spoofing_is_rejected`
  - `test_audio_duration_limit_is_enforced_and_temp_file_is_removed`
  - `test_checkpoint_paths_cannot_escape_the_model_root`

## Git Diff Safety

`git diff --check`: PASS.

Final diff/status inspection:
- No frontend source modifications were made by Phase 6.
- Existing root status still shows untracked `frontend/` from prior workspace state; Phase 6 did not edit it.
- No model artifact or checkpoint files were modified.
- No fusion weights were changed.
- No API schema redesign was made.
- No DB schema change was made.
- No XAI algorithm change was made.
- No CNN/SSL/glottal implementation change was made.
- No documentation reorganization was made.

## Deferred Phase 4 Parity

Phase 4 Colab/backend parity remains DEFERRED.

This report does not claim parity PASS.

## Final Integration Readiness

READY TO COMMIT WITH DOCUMENTED ENVIRONMENT ISSUES

Reason: Phase 6 public API, persistence, health/readiness, storage/upload, branch fallback, and XAI handoff regression tests pass in the local environment using real AASIST V2 where possible. Full DB-backed and Mongo-dependent XAI E2E remain blocked by missing `pymongo`; unrelated pre-existing test-contract issues are documented.

## Phase 6 Verdict

PASS WITH ISSUES

Issues are environment/pre-existing regression-suite blockers, not observed AASIST-Light V2 public-workflow regressions.
