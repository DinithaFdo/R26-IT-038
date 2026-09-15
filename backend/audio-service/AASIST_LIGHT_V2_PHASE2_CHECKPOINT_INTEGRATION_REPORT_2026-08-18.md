# AASIST-Light V2 Phase 2 Checkpoint Integration Report

Date: 2026-08-18

## Scope

Phase 2 integrates the finalized AASIST-Light V2 checkpoint into the existing backend runtime under the canonical public branch identifier `aasist`.

This phase intentionally does not add a fifth public branch, change fusion policy, modify API contracts, touch frontend code, alter authentication, MongoDB, Cloudinary, or XAI behavior, or mutate/resave the checkpoint artifact.

## Artifact Locations

- Checkpoint: `model_artifacts/aasist/aasist_light_v2_best.pt`
- Final summary: `model_artifacts/aasist/aasist_light_v2_final_summary.json`
- Backend default model root: `../model_artifacts`, resolved from `backend`

Resolved checkpoint path during verification:

`/Volumes/Awee NVME/Aweesha OS/01_University/Research/deepfake-voice-classification-system/model_artifacts/aasist/aasist_light_v2_best.pt`

## Files Created

- `backend/tests/test_aasist_light_v2_checkpoint_integration.py`
- `backend/AASIST_LIGHT_V2_PHASE2_CHECKPOINT_INTEGRATION_REPORT_2026-08-18.md`

## Files Modified

- `backend/app/config/settings.py`
- `backend/app/models/runtime.py`
- `backend/app/models/real/inference.py`
- `backend/app/models/architectures/__init__.py`
- `backend/app/models/architectures/aasist_light_v2.py`

Phase 1 files remain in place and are used by this phase:

- `backend/app/models/preprocessing/aasist_light_v2.py`
- `backend/app/ingestion/audio.py`
- `backend/tests/test_real_model_architectures.py`
- `backend/tests/test_real_model_preprocessing.py`

## Canonical AASIST Wiring

- Public branch identifier remains `aasist`.
- Internal architecture version is `aasist-light-v2-finalized-baseline`.
- Default AASIST checkpoint path now resolves to `aasist/aasist_light_v2_best.pt`.
- Runtime registry maps `aasist` to the finalized AASIST-Light V2 architecture version.
- The loader keeps the existing real-model branch lifecycle and lazy loading behavior.

## Checkpoint Validation

- Filename: `aasist_light_v2_best.pt`
- SHA-256: `c944c69f05a0135ad9dfdaf86fb8a31815339f862d780c5f2c5d834f181e64ec`
- Epoch: `13`
- Best dev EER: `0.039245587694380565`
- Class mapping: `{"bonafide": 0, "spoof": 1}`
- Architecture parameter count: `641795`
- Strict load result: pass
- Missing keys: `[]`
- Unexpected keys: `[]`

The checkpoint loader uses `torch.load(..., weights_only=True)` with explicit safe globals for the existing checkpoint metadata types. It rejects malformed payloads, absent state dictionaries, missing keys, unexpected keys, reversed or changed class mappings, epoch mismatches, best-EER mismatches, and parameter-count mismatches.

## Final Summary Metadata

The adjacent finalized summary is read as metadata and validated when present. The loader verifies:

- `model_name == "AASIST-Light V2"`
- `status == "FINALIZED_BASELINE"`
- `checkpoint.best_epoch == 13`
- `checkpoint.best_dev_eer == 0.039245587694380565`
- `label_mapping == {"bonafide": 0, "spoof": 1}`
- `preprocessing.sample_rate == 16000`
- `preprocessing.max_audio_samples == 64600`

Known summary weaknesses are retained as internal branch metadata and are not converted into product confidence claims.

## Device / Lifecycle

- Verified selected device: `cpu`
- Verified model training mode after load: `False`
- The model is moved to the selected device, set to eval mode, cached by the branch loader, and reused across predictions.

## Model Health / Readiness

The AASIST branch reports ready after the finalized checkpoint is loaded through the canonical runtime path. Missing-checkpoint behavior remains a controlled branch failure.

## Tests Added

`backend/tests/test_aasist_light_v2_checkpoint_integration.py` covers:

- Canonical `aasist` branch loading AASIST-Light V2.
- Model-root checkpoint resolution.
- Checkpoint SHA-256 identity.
- Controlled missing-checkpoint failure.
- Invalid payload rejection.
- Missing state-dict key rejection.
- Unexpected state-dict key rejection.
- Reversed class mapping rejection.
- Eval/device lifecycle.
- Loader reuse.
- Optional logits smoke test.
- Model health readiness.
- Fusion defaults unchanged.
- Non-AASIST branch configuration unchanged.
- Final summary metadata loading.

## Test Results

- `pytest tests/test_aasist_light_v2_checkpoint_integration.py`: 13 passed.
- Focused related suite excluding one known pre-existing path-escape assertion: 92 passed, 23 skipped, 1 deselected.
- `python -m compileall app tests/test_aasist_light_v2_checkpoint_integration.py tests/test_real_model_architectures.py tests/test_real_model_preprocessing.py`: passed.
- `python -m ruff check ...`: not run successfully because `ruff` is not installed in the active Python environment.
- `pytest -q`: blocked during collection by pre-existing missing `pymongo` dependency in Mongo/XAI tests.

## Regression Checks

- Fusion configuration defaults remain unchanged.
- Non-AASIST real-model branch configuration remains unchanged.
- Public branch naming remains unchanged.
- No endpoint, frontend, auth, database, Cloudinary, or XAI changes were introduced.
- The checkpoint artifact was read but not modified.

## Remaining Known Issues

- Full test collection requires `pymongo`; it is missing from the active environment.
- `tests/test_real_model_adapters.py::test_checkpoint_paths_cannot_escape_the_model_root` has a known unrelated assertion mismatch around escaped checkpoint paths.
- `ruff` is unavailable in the active environment.

## Phase 2 Verdict

PASS WITH ISSUES.

The finalized AASIST-Light V2 checkpoint is integrated under the canonical `aasist` branch, loads strictly with zero missing or unexpected keys, validates finalized checkpoint metadata, uses the finalized preprocessing contract, reports ready through model health, and preserves existing branch/fusion/API boundaries. Remaining issues are environment or pre-existing test-suite blockers outside the Phase 2 implementation scope.
