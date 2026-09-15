# AASIST-Light V2 Phase 3 Standalone Inference

## Scope

Phase 3 adds backend-only standalone inference verification for the finalized AASIST-Light V2 checkpoint under the existing canonical public branch `aasist`.

This phase does not change fusion weights, thresholds, final classification logic, API schemas, user-facing routes, frontend code, XAI, MongoDB persistence, CNN, SSL, glottal, checkpoint artifacts, model architecture, or AASIST V2 preprocessing.

## Runtime Inference Path

Actual traced path:

1. Local audio file enters `scripts/validate_aasist_light_v2_inference.py`.
2. `app.ingestion.audio.inspect_audio_file()` validates with `ffprobe`.
3. `app.ingestion.audio.decode_audio_file()` decodes with `ffmpeg`.
4. FFmpeg emits mono float32 PCM at `Settings.target_sample_rate == 16000`.
5. `app.ingestion.audio.preprocess_audio_file()` calls `preprocess_waveform()`.
6. `ProcessedAudio.unnormalised_waveform` is captured before shared peak normalization.
7. `ModelFactory(settings).create("aasist")` creates the canonical AASIST real adapter.
8. `RealModelAdapter.load()` reuses the existing model lifecycle.
9. `build_aasist_loader()` creates `AasistLightV2WaveformFrontEnd`.
10. `_prepare_input()` detects `requires_unnormalized_waveform == True` and passes `ProcessedAudio.unnormalised_waveform`.
11. `AasistLightV2WaveformFrontEnd.prepare()` applies first-crop/right-zero-pad to 64,600 samples.
12. `preprocess_aasist_light_v2_waveform()` applies per-waveform z-score normalization with std guard `< 1e-7`.
13. Final tensor shape is `[1, 1, 64600]`.
14. `runtime_model.module(model_input)` runs the finalized checkpoint.
15. `torch.softmax(logits, dim=-1)` maps index 0 to bonafide and index 1 to spoof.
16. `_spoof_probability_from_logits()` validates finite logits/probabilities and probability sum.

Confirmed V2 path properties:

- Mono audio: yes, FFmpeg uses `-ac 1`.
- 16000 Hz: yes, FFmpeg uses `-ar 16000`.
- Pre-shared-normalization waveform: yes, `unnormalised_waveform` is used.
- First 64600 samples when long: yes, V2 length policy is `first_crop_right_zero_pad`.
- Right-zero-pad when short: yes.
- Z-score normalization: yes, `per_waveform_zscore`.
- Std guard `< 1e-7`: yes.
- No peak normalization in V2 input: yes, the AASIST front end uses the unnormalised waveform.

## Files Created

- `backend/scripts/validate_aasist_light_v2_inference.py`
- `backend/tests/test_aasist_light_v2_standalone_inference.py`
- `backend/AASIST_LIGHT_V2_PHASE3_STANDALONE_INFERENCE_REPORT_2026-08-18.md`

## Files Modified

- None outside the new standalone validation script, new focused test file, and this report.

## Real Model Smoke Test

Command:

```bash
python scripts/validate_aasist_light_v2_inference.py ../frontend/e2e/fixtures/sample-voice.wav --json
```

Input file:

- `../frontend/e2e/fixtures/sample-voice.wav`

Checkpoint:

- Filename: `aasist_light_v2_best.pt`
- SHA-256: `c944c69f05a0135ad9dfdaf86fb8a31815339f862d780c5f2c5d834f181e64ec`
- Epoch: `13`
- Best EER metadata: `0.039245587694380565`
- Class mapping: `{"bonafide": 0, "spoof": 1}`

Runtime:

- Public branch: `aasist`
- Model: `aasist-light-v2-finalized-baseline`
- Model version: `checkpoint-c944c69f05a0`
- Device: `cpu`
- `model.training`: `False`

## Input Preprocessing Diagnostics

- Decoded sample rate: `16000`
- Decoded sample count: `48000`
- Original sample rate: `16000`
- Original channels: `1`
- Shared normalization applied: `true`
- AASIST source uses unnormalised waveform: `true`
- AASIST source samples: `48000`
- AASIST source peak: `0.244140625`
- Shared waveform peak after shared normalization: `0.949999988079071`
- V2 crop occurred: `false`
- V2 padding occurred: `true`
- V2 model input shape: `[1, 1, 64600]`
- V2 model input samples: `64600`
- V2 model input mean: `2.8049243483430075e-10`
- V2 model input std: `0.9999932646751404`
- V2 model input finite: `true`

## Probability Output

- Logits: `[[-6.6685566902160645, 5.85805082321167]]`
- Logits shape: `[1, 2]`
- P(bonafide): `0.0000036287908642407274`
- P(spoof): `0.9999964237213135`
- Probability sum: `1.0000000525121777`
- Predicted label: `spoof`

The reported prediction is diagnostic-only and uses the existing branch convention: spoof if P(spoof) >= P(bonafide).

## Determinism Check

Same file, same CPU device, same loaded adapter instance:

- P(spoof) run 1: `0.9999964237213135`
- P(spoof) run 2: `0.9999964237213135`
- Absolute difference: `0.0`
- Runtime model reused: `true`

## Performance Snapshot

Single standalone script run:

- Model load time: `37.37195899884682 ms`
- Preprocessing time: `287.84999999334104 ms`
- Inference time: `79.41237500926945 ms`
- Total time: `597.7609999972628 ms`

Second deterministic run with already loaded adapter:

- Preprocessing time: `97.68824999628123 ms`
- Inference time: `40.113334005582146 ms`
- Total time: `138.45474999106955 ms`

These are measurement-only diagnostics; no optimization was performed.

## Tests Added

`backend/tests/test_aasist_light_v2_standalone_inference.py` covers:

- Standalone harness uses canonical `aasist`.
- Actual real adapter produces logits shape `[1, 2]`.
- Softmax index 0 is bonafide.
- Softmax index 1 is spoof.
- Probabilities are finite.
- Probabilities sum approximately to 1.
- V2 input is exactly 64,600 samples.
- V2 input mean is approximately 0.
- V2 input std is approximately 1.
- Shared peak-normalized waveform is not accidentally used.
- Known valid audio runs without error.
- Corrupted audio fails through controlled ingestion exception.
- Near-silent audio fails through controlled V2 preprocessing error.
- Repeated inference is deterministic.
- Model is reused rather than reloaded per call.
- Public branch remains `aasist`.
- CNN remains unchanged.
- SSL remains unchanged.
- Glottal remains unchanged.
- Fusion remains unchanged.
- API schemas remain unchanged.

## Test Results

- `pytest tests/test_aasist_light_v2_standalone_inference.py`: `21 passed, 13 warnings`.
- Related requested suite including known unrelated issues: `149 passed, 2 failed, 24 skipped, 1 deselected, 13 warnings`.
- Related requested suite excluding known unrelated issues: `149 passed, 24 skipped, 3 deselected, 13 warnings`.
- `python -m compileall app scripts/validate_aasist_light_v2_inference.py tests/test_aasist_light_v2_standalone_inference.py`: passed.
- Full `pytest -q`: blocked during collection by missing `pymongo`, causing 7 Mongo/XAI collection errors.

Known unrelated failures in the related suite:

- `tests/test_audio_preprocessing.py::test_extension_spoofing_is_rejected`: expected `CorruptedAudioError`, current code raises `UnsupportedAudioFormatError`.
- `tests/test_audio_preprocessing.py::test_audio_duration_limit_is_enforced_and_temp_file_is_removed`: active Python/Pydantic environment rejects fractional `max_audio_duration_seconds` for an integer setting.
- `tests/test_real_model_adapters.py::test_checkpoint_paths_cannot_escape_the_model_root`: pre-existing path-escape assertion mismatch, excluded from the clean regression run.

## Regression Checks

- Fusion settings unchanged.
- API routes/schemas unchanged; no user-facing AASIST validation route was added.
- CNN architecture identifier unchanged.
- SSL architecture identifier unchanged.
- Glottal architecture identifier unchanged.
- Checkpoint artifact was loaded read-only and not modified.
- Frontend was not modified.
- XAI was not modified.
- MongoDB persistence schema was not modified.

## Colab Parity Handoff

Do not claim parity PASS yet. Parity requires running the same audio and checkpoint path in Colab and comparing probabilities.

Use this local fixture:

- `frontend/e2e/fixtures/sample-voice.wav`

Upload that exact file to Colab, then run the finalized AASIST-Light V2 Colab inference using:

- same checkpoint SHA-256: `c944c69f05a0135ad9dfdaf86fb8a31815339f862d780c5f2c5d834f181e64ec`
- same preprocessing: mono 16 kHz, first 64,600 samples or right-zero-pad, per-waveform z-score, std guard `< 1e-7`, no peak normalization
- same class mapping: bonafide index `0`, spoof index `1`

Backend values to compare:

- P(bonafide): `0.0000036287908642407274`
- P(spoof): `0.9999964237213135`
- V2 input mean: `2.8049243483430075e-10`
- V2 input std: `0.9999932646751404`
- Logits: `[[-6.6685566902160645, 5.85805082321167]]`

## Known Issues

- Full test collection still requires `pymongo` in this environment.
- Two existing audio-ingestion tests fail for unrelated expectations/type validation as listed above.
- One existing real-adapter path-escape test has a known unrelated assertion mismatch.
- The smoke audio is a frontend E2E fixture, not a labelled ASVspoof sample; it proves mechanics, not research accuracy.

## Phase 3 Verdict

PASS WITH ISSUES.

The finalized checkpoint loads, real audio reaches the model through backend ingestion, the V2 tensor has exactly 64,600 samples, z-score preprocessing is preserved, the shared peak-normalized waveform is not used, logits are `[1, 2]`, probabilities are finite and sum approximately to 1, class mapping remains bonafide=0/spoof=1, repeated inference is deterministic on CPU, and the existing model lifecycle is reused. The issues are unrelated test-suite/environment blockers and the absence of a Colab parity comparison run.
