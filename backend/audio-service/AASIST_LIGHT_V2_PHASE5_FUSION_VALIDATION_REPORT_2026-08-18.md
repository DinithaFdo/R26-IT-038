# AASIST-Light V2 Phase 5 Fusion Validation

## Scope

Phase 5 validates that the finalized AASIST-Light V2 backend implementation participates correctly in the existing MULTI-SCOPE fusion pipeline under the canonical public branch key `aasist`.

This phase does not retrain any model, alter AASIST architecture/preprocessing/checkpoints, add branches, rename public identifiers, tune fusion weights or thresholds, add API routes, change response schemas, modify frontend, change XAI behavior, change MongoDB persistence schemas, add infrastructure, claim Colab parity, or claim research accuracy improvements.

Phase 4 status: Colab/backend probability parity remains DEFERRED.

Phase 6 status: API / persistence / XAI regression remains NEXT.

## Existing Fusion Architecture

Traced path:

1. `VoiceService._predict_from_validated_upload()` receives validated audio metadata.
2. `VoiceService` preprocesses once through `preprocess_audio_file()` into `ProcessedAudio`.
3. `ModelRegistry.models` are iterated in canonical order.
4. `_predict_branch_with_timeout()` calls each branch adapter through `model.predict_safe(processed_audio)`.
5. Each successful branch returns `BranchPrediction` with `probabilities.bonafide` and `probabilities.spoof`.
6. `VoiceService` adds model provenance with `_branch_with_model_provenance()`.
7. `FusionEngine.fuse(branch_predictions)` filters successful predictions with probabilities.
8. Failed/skipped/missing-probability branches are recorded in `excluded_branches`.
9. Required branches are checked using `canonical_branch_name()`.
10. `minimum_successful_branches` is enforced.
11. Weighted fusion calls `_weighted_average()`.
12. Branch weights are looked up by public branch aliases and normalized over successful branches.
13. `_average_probabilities()` computes weighted bonafide and spoof probabilities from branch probabilities.
14. Final fusion label is `spoof` if fused spoof probability is greater than or equal to `spoof_threshold`.
15. Research blockers are added for dummy contributors, incomplete branch sets, and unverified real branches.

Fusion uses probabilities only. It does not consume logits, argmax labels, strings, percentages, or thresholded 0/1 branch outputs.

## Canonical Branch Contract

- Public fusion branch key: `aasist`
- Internal AASIST implementation: `aasist-light-v2-finalized-baseline`
- Public branch key remains distinct from the internal version string.
- Fusion branch weights contain `aasist`; no `aasist_v2`, `aasist_light_v2`, or `aasist-light-v2` key is introduced.

## AASIST Score Semantics

The real AASIST branch returns a `BranchPrediction` through `real_prediction_from_spoof_probability()`.

For AASIST-Light V2:

- logits shape is `[1, 2]`
- softmax index `0` is bonafide
- softmax index `1` is spoof
- value sent into fusion is `BranchPrediction.probabilities.spoof`
- this equals `P(spoof) = softmax(logits)[1]`

Phase 3 reference sample:

- Logits: `[[-6.6685566902160645, 5.85805082321167]]`
- P(bonafide): `0.0000036287908642407274`
- P(spoof): `0.9999964237213135`

Phase 5 confirms fusion receives `0.9999964237213135` for `aasist`, not bonafide probability, raw logit, argmax, percentage text, or a thresholded value.

## Current Fusion Weights

Configured settings weights:

- `lfcc_cnn_tcn`: `0.25`
- `aasist`: `0.25`
- `ssl_sequence`: `0.25`
- `glottal`: `0.25`

Development default alias weights:

- `cnn`: `0.25`
- `aasist`: `0.25`
- `ssl`: `0.25`
- `glottal`: `0.25`

Normalization behavior:

- Weighted-average fusion first looks up configured weights for successful branch model names through aliases.
- Effective weights are normalized across the successful branches only.
- If a successful branch has zero configured weight, it contributes zero.
- If all effective weights are zero and equal-weight fallback is enabled, successful branches receive equal weights.
- If all effective weights are zero and equal-weight fallback is disabled, fusion fails with `No positive fusion weights are available.`

No fusion weights were tuned in this phase.

## Real/Dummy Runtime Behavior

- Real mode canonical `aasist` resolves to AASIST-Light V2.
- Dummy mode canonical `aasist` remains the existing dummy adapter behavior.
- Fusion does not select dummy AASIST when `aasist_model_mode="real"` and the finalized checkpoint is valid.
- Missing real checkpoint remains a controlled branch failure through the existing adapter path and does not become a fake production success.

## Fusion Failure / Fallback Behavior

Existing behavior is preserved:

- All configured successful branches: fusion succeeds.
- AASIST succeeds while another branch fails: failed branch is excluded; successful branch weights are renormalized.
- AASIST fails while other branches succeed: AASIST is excluded; successful branch weights are renormalized.
- One valid branch: succeeds only if `minimum_successful_branches=1`; otherwise fails.
- No valid branches: controlled failed `FusionResult` with no probabilities.
- Missing required branch: controlled failed `FusionResult` with `required_branch_missing`.

## Real AASIST V2 Fusion Participation

Real smoke path used:

- Real AASIST-Light V2 branch.
- Existing backend `VoiceService` path.
- Production audio ingestion/preprocessing path.
- Deterministic dummy/test branch doubles for CNN, SSL, and glottal.
- No fabricated real non-AASIST results.

Input audio:

- `frontend/e2e/fixtures/sample-voice.wav`

Branch scores:

- `cnn_acoustic`: `0.2` TEST/DETERMINISTIC STUB RESULT
- `aasist`: `0.9999964237213135` REAL MODEL RESULT
- `ssl_wavlm_xlsr`: `0.4` TEST/DETERMINISTIC STUB RESULT
- `glottal_features`: `0.6` TEST/DETERMINISTIC STUB RESULT

Branch modes:

- `cnn_acoustic`: `dummy`
- `aasist`: `real`
- `ssl_wavlm_xlsr`: `dummy`
- `glottal_features`: `dummy`

Effective fusion weights:

- `cnn_acoustic`: `0.25`
- `aasist`: `0.25`
- `ssl_wavlm_xlsr`: `0.25`
- `glottal_features`: `0.25`

Fused output:

- Fused spoof score: `0.5499991059303284`
- Prediction: `spoof`
- Contains dummy branches: `true`
- Research eligible: `false`
- Warning: deterministic dummy branch outputs are development-only

## Manual Fusion Calculation

Manual weighted calculation:

```text
(0.2 * 0.25)
+ (0.9999964237213135 * 0.25)
+ (0.4 * 0.25)
+ (0.6 * 0.25)
= 0.5499991059303284
```

Actual fusion spoof score:

```text
0.5499991059303284
```

Absolute difference:

```text
0.0
```

This proves the AASIST V2 score enters fusion unchanged under the expected `aasist` key.

## Label Direction Validation

Explicit tests confirm:

- High AASIST P(spoof) increases the fused spoof probability.
- Low AASIST P(spoof) lowers the fused spoof probability.
- AASIST score is not thresholded before fusion.
- Raw AASIST logits are not passed into fusion.
- P(bonafide) is not accidentally used as the AASIST spoof contribution.

Result: no label inversion detected.

## Health / Readiness

`ModelRegistry.readiness()` consumes branch health from each model. A loaded real AASIST V2 branch reports:

- ready: `true`
- mode summary real branches: `["aasist"]`
- unverified real branches: `["aasist"]`

This matches existing policy: the model can be operationally ready while still not research-ready until verification flags are explicitly attested.

## Tests Added

Added `backend/tests/test_aasist_light_v2_fusion_validation.py`.

Coverage includes:

- Fusion branch key remains `aasist`.
- AASIST fusion input equals P(spoof), not P(bonafide).
- Internal V2 version does not change public fusion key.
- Current AASIST fusion weight is reused.
- Existing fusion weights remain unchanged.
- Manual weighted fusion calculation matches actual fusion.
- High/low AASIST spoof probability direction is correct.
- AASIST score is not thresholded before fusion.
- Raw logits are not passed to fusion.
- All-branch success path works.
- AASIST-success / other-branch-failure fallback behavior is preserved.
- AASIST-failure / other-branches-success fallback behavior is preserved.
- No-valid-branches behavior remains controlled.
- Real mode resolves canonical `aasist` to V2.
- Dummy mode remains unchanged.
- Model registry/factory behavior remains compatible.
- Fusion API/public schema remains unchanged.
- CNN, SSL, and glottal architecture identifiers remain unchanged.
- Phase 3 standalone inference remains deterministic.
- Real AASIST participates in the VoiceService fusion path.
- AASIST readiness feeds ModelRegistry readiness.

## Test Results

- `pytest tests/test_aasist_light_v2_fusion_validation.py`: `24 passed, 13 warnings`.
- Main Phase 5 regression set: `152 passed, 23 skipped, 1 deselected, 13 warnings`.
- Expanded scoped regression set including audio ingestion/ffmpeg with known unrelated failures excluded: `195 passed, 24 skipped, 3 deselected, 13 warnings`.
- `python -m compileall app scripts/validate_aasist_light_v2_inference.py tests/test_aasist_light_v2_fusion_validation.py tests/test_aasist_light_v2_standalone_inference.py tests/test_aasist_light_v2_checkpoint_integration.py`: passed.
- Full `pytest -q`: blocked during collection by missing `pymongo`, causing 7 Mongo/XAI collection errors.
- `python -m ruff check ...`: not run successfully because `ruff` is not installed in the active environment.

Known unrelated excluded failures:

- `tests/test_real_model_adapters.py::test_checkpoint_paths_cannot_escape_the_model_root`
- `tests/test_audio_preprocessing.py::test_extension_spoofing_is_rejected`
- `tests/test_audio_preprocessing.py::test_audio_duration_limit_is_enforced_and_temp_file_is_removed`

## Regression Checks

- Fusion configuration changed: no.
- API schemas changed: no.
- Persistence schemas changed: no.
- XAI behavior changed: no.
- CNN changed: no.
- SSL changed: no.
- Glottal changed: no.
- AASIST architecture/preprocessing/checkpoint changed: no.
- Frontend changed: no.

## Performance Snapshot

Real AASIST + deterministic stub VoiceService fusion snapshot:

- AASIST branch processing time: `93.87675000471063 ms`
- Total voice classification time: `301.8054999993183 ms`
- Wall-clock script time: `302.0249589899322 ms`

Fusion computation itself is a small in-process weighted average and was not optimized or separately changed in this phase.

## Known Issues

- Full test collection still requires `pymongo` in this environment.
- `ruff` is not installed in the active Python environment.
- Three known unrelated tests are excluded from the clean scoped regression run, listed above.
- The real fusion snapshot uses deterministic dummy/test outputs for CNN, SSL, and glottal; only AASIST is a real model result.
- Phase 4 Colab/backend probability parity is still deferred.

## Deferred Work

- Phase 4: run Colab/backend probability parity on the same audio and checkpoint.
- Phase 6: perform API / persistence / XAI regression validation through the public prediction path.
- Future research work: fusion calibration, weight tuning, threshold tuning, stacking/meta-classifier experiments, or EER optimization. None of that belongs in Phase 5.

## Phase 5 Verdict

PASS WITH ISSUES.

The canonical public branch remains `aasist`, real `aasist` resolves to AASIST-Light V2, fusion receives P(spoof), no label inversion was detected, the current AASIST weight is preserved, fusion math matches manual calculation exactly, branch failure behavior remains controlled, real/dummy behavior remains correct, API/persistence/XAI contracts are unchanged, CNN/SSL/glottal are unchanged, and Phase 1-3 AASIST tests still pass. Remaining issues are unrelated environment/test-suite blockers and deferred Colab/API parity phases.
