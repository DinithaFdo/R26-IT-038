# AASIST-Light V2 Phase 1 Implementation

## Scope

Phase 1 added the finalized AASIST-Light V2 architecture and deterministic
waveform preprocessing contract without loading the finalized checkpoint,
changing fusion, changing API schemas, adding a fifth branch, or touching the
frontend.

The canonical public branch identifier remains `aasist`.

## Files Changed

### `app/models/architectures/aasist_light_v2.py`

- Added the finalized AASIST-Light V2 PyTorch architecture.
- Implements Conv1d/BatchNorm/SiLU frontend with kernels 15/11/7 and strides
  5/4/4.
- Implements a 2-layer bidirectional GRU with hidden size 128 and dropout 0.2.
- Implements attention pooling and the 256 -> 128 -> 2 classifier.
- Returns raw logits only.
- Exposes `LABEL_MAPPING = {"bonafide": 0, "spoof": 1}`.

### `app/models/preprocessing/aasist_light_v2.py`

- Added a dedicated AASIST-Light V2 waveform preprocessor.
- Enforces 16 kHz / 64,600-sample contract.
- Uses first-sample crop for long clips.
- Uses right zero-padding for short clips.
- Rejects silent/near-silent waveforms when std is below `1e-7`.
- Applies per-waveform z-score normalization with epsilon `1e-6`.
- Does not peak normalize or RMS normalize.
- Defines a front end that opts into the unnormalised decoded waveform.

### `app/models/architectures/__init__.py`

- Exported `build_aasist_light_v2_net`.
- Existing `build_aasist_light_net` export remains unchanged.

### `app/ingestion/audio.py`

- Added `ProcessedAudio.unnormalised_waveform`.
- Captures the cleaned, mono, resampled waveform immediately before existing
  shared peak normalization.
- Existing `ProcessedAudio.waveform` remains the same normalized waveform used
  by current CNN, AASIST V1/recovered, SSL, and glottal paths.

### `app/models/real/inference.py`

- Added an opt-in source selection hook for front ends that declare
  `requires_unnormalized_waveform = True`.
- Existing front ends continue receiving `ProcessedAudio.waveform`.
- If an opt-in front end is used without an available raw waveform, inference
  fails with controlled `ModelInferenceError`.

### `tests/test_real_model_architectures.py`

- Added AASIST-Light V2 architecture contract tests:
  - instantiation,
  - exact trainable parameter count,
  - `[2, 64600] -> [2, 2]` output,
  - raw logits, not model-internal softmax,
  - label mapping.

### `tests/test_real_model_preprocessing.py`

- Added AASIST-Light V2 preprocessing contract tests:
  - first-sample crop,
  - right zero-padding,
  - z-score normalization,
  - near-silent rejection,
  - no peak normalization,
  - opt-in use of unnormalised waveform,
  - existing CNN/AASIST V1/SSL front ends do not opt in,
  - canonical `aasist` branch identifier remains unchanged.

## Architecture Contract

AASIST-Light V2 is implemented as:

- Conv1d 1 -> 32, kernel 15, stride 5, padding 7
- BatchNorm1d(32)
- SiLU
- Conv1d 32 -> 64, kernel 11, stride 4, padding 5
- BatchNorm1d(64)
- SiLU
- Conv1d 64 -> 128, kernel 7, stride 4, padding 3
- BatchNorm1d(128)
- SiLU
- GRU input 128, hidden 128, 2 layers, batch-first, bidirectional, dropout 0.2
- attention Linear 256 -> 128, Tanh, Linear 128 -> 1
- softmax over time
- weighted temporal sum
- classifier Linear 256 -> 128, ReLU, Dropout 0.3, Linear 128 -> 2

Parameter count: exactly `641,795` trainable parameters.

The model accepts `[B, T]` or `[B, 1, T]` and returns raw logits `[B, 2]`.

## Preprocessing Contract

The AASIST-Light V2 preprocessor:

- expects a 1-D mono waveform already decoded/resampled by ingestion,
- requires 16 kHz,
- outputs exactly 64,600 samples,
- crops long waveforms as `waveform[:64600]`,
- right zero-pads short waveforms,
- computes std after crop/pad,
- rejects std `< 1e-7`,
- returns `(waveform - mean) / (std + 1e-6)`,
- never peak normalizes,
- never RMS normalizes,
- never center-crops or randomly crops.

## Shared Peak-Normalization Issue

Previous behavior:

- `app/ingestion/audio.py` decoded audio to mono 16 kHz float32.
- It then performed shared peak normalization before constructing
  `ProcessedAudio.waveform`.
- All existing model front ends received that normalized waveform.

Implemented safe solution:

- `ProcessedAudio` now preserves `unnormalised_waveform`, captured after
  numeric cleanup, mono conversion, and resampling, but before shared peak
  normalization.
- The generic real inference path still defaults to `ProcessedAudio.waveform`.
- Only front ends that explicitly declare
  `requires_unnormalized_waveform = True` receive `unnormalised_waveform`.
- The AASIST-Light V2 front end declares that opt-in.

Why other branches are unaffected:

- Current CNN, AASIST V1/recovered, SSL, and glottal paths do not declare the
  opt-in flag.
- Their source waveform remains `ProcessedAudio.waveform`.
- No fusion weights, branch names, route schemas, persistence models, XAI
  contracts, or checkpoint settings were changed.

## Tests Added

Added 13 focused Phase 1 tests covering:

- architecture instantiation,
- exact parameter count,
- output tensor shape,
- absence of model-internal softmax,
- long waveform first-crop behavior,
- short waveform right-padding behavior,
- z-score normalization,
- near-silent rejection,
- no peak normalization,
- label contract,
- preserved canonical branch identifier,
- unnormalised waveform opt-in,
- existing front ends remaining on their previous source waveform.

## Test Results

Passed:

```text
pytest tests/test_real_model_architectures.py::test_aasist_light_v2_instantiates_with_finalized_parameter_count \
  tests/test_real_model_architectures.py::test_aasist_light_v2_accepts_batched_waveforms_and_returns_logits \
  tests/test_real_model_architectures.py::test_aasist_light_v2_forward_does_not_apply_softmax \
  tests/test_real_model_architectures.py::test_aasist_light_v2_label_contract_is_bonafide_then_spoof \
  tests/test_real_model_preprocessing.py::test_aasist_light_v2_long_waveform_uses_first_samples_not_centre_crop \
  tests/test_real_model_preprocessing.py::test_aasist_light_v2_short_waveform_is_right_zero_padded \
  tests/test_real_model_preprocessing.py::test_aasist_light_v2_zscore_normalises_typical_waveform \
  tests/test_real_model_preprocessing.py::test_aasist_light_v2_rejects_near_silent_waveform \
  tests/test_real_model_preprocessing.py::test_aasist_light_v2_preprocessing_does_not_peak_normalise \
  tests/test_real_model_preprocessing.py::test_aasist_light_v2_frontend_declares_unnormalised_waveform_requirement \
  tests/test_real_model_preprocessing.py::test_real_inference_uses_unnormalised_waveform_for_opt_in_frontends \
  tests/test_real_model_preprocessing.py::test_existing_cnn_and_aasist_v1_frontends_do_not_opt_into_raw_waveform \
  tests/test_real_model_preprocessing.py::test_aasist_public_branch_identifier_remains_canonical
```

Result: `13 passed`.

Syntax compile:

```text
python -m compileall app tests/test_real_model_architectures.py tests/test_real_model_preprocessing.py
```

Result: passed.

Relevant existing suite attempted:

```text
pytest tests/test_real_model_architectures.py tests/test_real_model_preprocessing.py tests/test_audio_preprocessing.py tests/test_model_runtime_phase3.py tests/test_real_model_adapters.py tests/test_fusion.py
```

Result: `88 passed, 23 skipped, 3 failed`.

The three failures were pre-existing/environment-shaped and not caused by the
new AASIST-Light V2 contract tests:

- `test_extension_spoofing_is_rejected` expected `CorruptedAudioError`, but the
  current backend raises `UnsupportedAudioFormatError` for extension/container
  mismatch.
- `test_audio_duration_limit_is_enforced_and_temp_file_is_removed` constructs
  `Settings(max_audio_duration_seconds=0.01)`, but the current setting type is
  `int`.
- `test_checkpoint_paths_cannot_escape_the_model_root` observed
  `checkpoint_not_found` rather than the expected escape/extension code in this
  environment.

Full backend pytest attempted:

```text
pytest
```

Result: collection failed because `pymongo` is not installed in the current
Python environment. No packages were installed because Phase 1 scope forbids
installation.

Lint:

```text
python -m ruff check app tests/test_real_model_architectures.py tests/test_real_model_preprocessing.py
```

Result: not run because `ruff` is not installed in the current Python
environment.

## Regression Checks

- Frontend untouched.
- No API routes added.
- No API schemas changed.
- No fusion behavior changed.
- No checkpoint path configuration changed.
- No finalized checkpoint copied into the repository.
- No MongoDB/auth/XAI/persistence/Cloudinary/API-key behavior changed.
- Current canonical `aasist` branch name preserved.
- Existing CNN/AASIST V1/SSL front ends remain on the existing normalized
  waveform source.

## Known Remaining Work

Phase 2:

- checkpoint artifact placement,
- checkpoint configuration,
- strict checkpoint loading,
- state-dict validation.

Phase 3:

- standalone AASIST backend inference.

Phase 4:

- Colab/backend probability parity testing.

Phase 5:

- fusion validation.

Phase 6:

- API / persistence / XAI regression.

## Phase 1 Verdict

PASS WITH ISSUES

The Phase 1 implementation itself passes its focused contract tests and compile
check. Repository-wide validation is blocked by current environment dependency
gaps and pre-existing unrelated test failures.
