# CNN Verification & Real Model Guard Repair

Date: 2026-08-19
Type: verification infrastructure and diagnostics only. No CNN/AASIST
architecture, preprocessing, checkpoint, fusion weight, fusion threshold,
branch threshold, `CNN_CLASS_ORDER`, frontend, API schema, persistence, or
XAI code was changed. Production inference behaviour is unchanged.

## Scope

Follow-up to `VOICE_BACKEND_FRONTEND_INTEGRATION_AND_LABEL_MAPPING_AUDIT_2026-08-19.md`.
That audit found the AASIST integration correctly wired and proven by its own
checkpoint attestation, but the CNN branch's class mapping and feature
pipeline unverifiable from any repository artifact, and the guard tests meant
to catch exactly that silently skipping. This phase repairs the test
infrastructure and adds verification tooling; it does not draw a conclusion
about which CNN class order is correct, because no labelled evidence exists
yet to draw one from.

## Audit Findings Addressed

| Finding | Status |
|---|---|
| Real-model discrimination tests silently skip (stale checkpoint paths) | Fixed -- see *Runtime Checkpoint Resolution* |
| `ProcessedAudio` test helper cannot drive real AASIST V2 (`unnormalised_waveform` missing) | Fixed -- see *Repaired Real-Model Test Helpers* |
| CNN checkpoint has no class-mapping metadata; mapping unverifiable | Confirmed, unchanged; tooling added -- see *CNN Label Mapping Status* |
| No CNN label-order verification tooling exists | Added -- see *Label-Order Validation Tool* |
| AASIST spoof index not cross-checked against checkpoint attestation | Confirmed, documented with a characterization test, deliberately not fixed (out of scope) |

## Runtime Checkpoint Resolution

`backend/tests/real_model_helpers.py` previously hardcoded:

```
CNN_CHECKPOINT    = model_artifacts/best_cnn_full_weighted.pth      (never existed)
AASIST_CHECKPOINT = model_artifacts/best_aasist_light_full_weighted.pth  (never existed;
                     also the superseded V1 filename)
```

against a deployment that actually configures (`backend/.env`):

```
MODEL_ROOT_DIR=..
CNN_MODEL_PATH=models/best_cnn_full_weighted.pth
AASIST_MODEL_PATH=model_artifacts/aasist/aasist_light_v2_best.pt
```

The repair does not reimplement path resolution. It calls
`app/models/runtime.py::checkpoint_identity_for_path` -- the literal function
`ModelFactory.branch_config()` uses in production -- against a `Settings`
instance built the same way a locally started server would build one
(`backend/.env` when present, read via an absolute path computed from this
file's own location so it is independent of pytest's invocation directory;
pure environment/defaults otherwise, e.g. on a CI runner with no `.env`).

Result: test path resolution and production path resolution are now the same
code path, not two independently-maintained guesses. Proven by
`tests/test_cnn_verification_guard.py`:

- `test_cnn_checkpoint_resolves_through_runtime_settings_not_a_guessed_path`
  / the AASIST equivalent -- compares the helper's resolved identity against
  a freshly-built `ModelFactory(...).branch_config(...)`.checkpoint`.
- `test_cnn_checkpoint_path_matches_an_independent_raw_dotenv_parse` / the
  AASIST equivalent -- cross-checks against a deliberately separate,
  regex-only parse of `backend/.env`, so a future regression to a
  hardcoded/guessed path fails this test even if it does not touch
  `checkpoint_identity_for_path` at all.

`AASIST_CHECKPOINT` now resolves to `model_artifacts/aasist/aasist_light_v2_best.pt`
(the deployed V2 checkpoint). A new `AASIST_LIGHT_V1_CHECKPOINT` constant
(`models/best_aasist_light_full_weighted.pth`) was added alongside it,
deliberately kept separate -- see *AASIST Regression Status*.

## Repaired Real-Model Test Helpers

`tests/real_model_helpers.py::processed_audio()` previously omitted
`unnormalised_waveform`, so any real AASIST V2 branch (which sets
`requires_unnormalized_waveform = True`) raised `model_input_invalid` before a
single real test could run through it -- confirmed by execution before the
fix.

Fix: `unnormalised_waveform` is now populated from the same input samples used
for `waveform`. This helper deliberately does **not** route through the
production `preprocess_waveform` pipeline (which would apply real peak
normalisation and NaN-sanitisation): one existing test
(`test_non_finite_audio_is_rejected_rather_than_fed_to_the_model`)
intentionally injects a corrupted/non-finite waveform to prove the model
adapter's own input-validation layer is a genuine second line of defence, not
a restatement of ingestion's NaN sanitisation. Routing through
`preprocess_waveform` would have silently sanitised that NaN away before the
assertion ever ran, breaking a legitimate defence-in-depth test. Because of
this, `waveform` and `unnormalised_waveform` are identical arrays in this
helper -- adequate for shape, determinism, and label-mapping assertions, not a
stand-in for testing real amplitude-normalisation behaviour (that belongs to,
and is already covered by, `tests/test_audio_preprocessing.py`). This
simplification is documented in the helper's own docstring.

Regression test:
`test_cnn_verification_guard.py::test_processed_audio_helper_includes_unnormalised_waveform`
and `test_real_aasist_v2_can_run_through_the_repaired_helper`.

`real_model_settings()` was also changed from hardcoded test-only
`model_root_dir`/`cnn_model_path`/`aasist_model_path` values to reading the
same three fields from the resolved deployment `Settings`, so every test
built on it now exercises the actually-deployed checkpoints. Explicit
`overrides` (e.g. a test that deliberately passes a bad path) still take
precedence.

## CNN Checkpoint Identity

| Field | Value |
|---|---|
| Configured path | `models/best_cnn_full_weighted.pth` |
| Resolved path | `<repo_root>/models/best_cnn_full_weighted.pth` |
| SHA-256 | `4bf86dd15108a8cff43569ef907a9f3852978353af6d6e184001480a0fa2e6f8` |
| Size | 104,273 bytes |
| Format | Bare `collections.OrderedDict` state dict, 23 tensor keys, **zero** non-tensor entries |
| Class mapping metadata | **None** -- there is nothing to pin |

Recorded through the same `CheckpointIdentity`/provenance mechanism AASIST
uses (`app/models/runtime.py::checkpoint_identity_for_path`), pinned in
`test_cnn_checkpoint_identity_is_recorded_through_the_provenance_mechanism`.
Unlike AASIST, there is no `class_mapping`/`spoof_class_index`/epoch/EER field
to assert -- the checkpoint genuinely does not carry one, confirmed again by
direct read-only inspection in this phase.

## AASIST V2 Checkpoint Identity

| Field | Value | Source |
|---|---|---|
| Filename | `aasist_light_v2_best.pt` | checkpoint identity |
| SHA-256 | `c944c69f05a0135ad9dfdaf86fb8a31815339f862d780c5f2c5d834f181e64ec` | independently recomputed via `shasum -a 256` and via `hashlib.sha256` in-test; matches |
| Architecture | `aasist-light-v2-finalized-baseline` | `runtime_model.architecture_version` |
| Epoch | 13 | checkpoint's own `epoch` field, strict-load-verified |
| Parameter count | 641,795 | checkpoint's own attestation, strict-load-verified against the reconstructed architecture |
| Class mapping | `{bonafide: 0, spoof: 1}` | checkpoint's own `class_mapping`, rejected at load if it ever disagrees |

Pinned as an explicit regression assertion in
`test_aasist_v2_checkpoint_identity_is_pinned` -- if this checkpoint file is
ever silently swapped, this test fails loudly instead of the system quietly
running a different model under the same branch name.
`test_aasist_checkpoint_resolution_is_not_the_superseded_v1_file` separately
proves `AASIST_CHECKPOINT` is not the old V1 artifact (by filename and by
hash).

## CNN Label Mapping Status

**Unchanged, and still correctly reported as unverified.** No evidence
gathered in this phase changes that; none was intended to. `CNN_CLASS_ORDER=bonafide_spoof`
remains an operator assertion the checkpoint cannot corroborate (it carries no
metadata at all). Tooling to gather that evidence was built (see *Label-Order
Validation Tool*), but no labelled audio was available in this environment to
run it against, so no mapping conclusion is drawn here. See *Real-World
Sanity Validation Support* for how to supply it.

One additional piece of mechanical (not evidentiary) confirmation: flipping
`CNN_CLASS_ORDER` between `bonafide_spoof` and `spoof_bonafide` on the real
loaded checkpoint inverts every score exactly
(`spoof_prob(forward) == 1 - spoof_prob(reversed)`, `abs` tolerance 1e-6) --
`test_cnn_class_order_setting_mechanically_inverts_the_real_branchs_output`,
restating the existing `test_class_order_selects_the_opposite_logit`. This
proves the setting does what it claims to mechanically; it says nothing about
which direction is correct.

## CNN Preprocessing Status

Exact current backend front end (`app/models/preprocessing/spectral.py`,
`SpectralFeatureConfig` as configured by `.env`):

| Parameter | Value | Status |
|---|---|---|
| Sample rate | 16,000 Hz | PROVEN FROM TRAINING ARTIFACT (indirectly -- BatchNorm running stats are consistent with ~zero-mean, unit-variance input; sample rate itself is a declared assumption, not directly provable) |
| Input waveform source | Shared peak-normalised `ProcessedAudio.waveform` (peak scaled to 0.95 when the original peak is >= 1e-3) | INFERRED / DECLARED BY BACKEND |
| Feature type | log-mel (`CNN_FEATURE_TYPE=log_mel`) | INFERRED / DECLARED BY BACKEND |
| Filter count | 20 mel filters, `fmin=0`, `fmax=8000` | INFERRED / DECLARED BY BACKEND |
| FFT | `n_fft=512` | INFERRED / DECLARED BY BACKEND |
| Hop length | 160 samples (10 ms at 16 kHz) | INFERRED / DECLARED BY BACKEND |
| Window | 400 samples (25 ms), Hann | INFERRED / DECLARED BY BACKEND |
| Crop policy | Deterministic centre crop to 400 frames (~4.0 s at this hop) | INFERRED / DECLARED BY BACKEND |
| Normalisation | `global_zscore` over the full feature map after cropping | INFERRED / DECLARED BY BACKEND, partially supported -- the checkpoint's first BatchNorm running statistics are consistent with an approximately zero-mean, unit-variance input distribution (documented in `spectral.py`'s module docstring), which is why this is the chosen default rather than "none" |
| Output shape | `(1, 1, 20, 400)` | PROVEN mechanically (matches the checkpoint's declared `Linear(64, num_classes)` after global average pooling, which accepts any spatial size -- so this shape check does not prove the shape is *correct*, only that it runs) |

No training notebook or exported feature-extraction script shipped with the
checkpoint. The BatchNorm-statistics argument constrains feature *scale* only
(roughly zero-mean, unit-variance); it does not identify the filterbank type,
coefficient count, or FFT parameters. **No preprocessing verification is
claimed here or anywhere in this phase.** `CNN_PREPROCESSING_VERIFIED=false`
remains correct.

## CNN Saturation Diagnostics

Run via `scripts/validate_cnn_class_order.py` (no labelled manifest available
in this environment, so synthetic-probe mode only -- see caveat below):

| Probe | P(index 0) | P(index 1) | Max probability | Entropy (bits) | Saturated (>=0.99) |
|---|---|---|---|---|---|
| synthetic_harmonic | 0.000000 | 1.000000 | 1.000000 | 0.0000 | yes |
| synthetic_broadband_noise | 0.000000 | 1.000000 | 1.000000 | 0.0000 | yes |
| synthetic_pure_tone | 1.000000 | 0.000000 | 1.000000 | 0.0000 | yes |
| synthetic_chirp | 1.000000 | 0.000000 | 1.000000 | 0.0000 | yes |

4/4 (100%) of scored probes saturated at machine-precision extremes (entropy
0 bits). **These are unlabelled synthetic signals -- sine tones, noise, a
chirp -- and this is a saturation diagnostic only, never accuracy evidence.**
It does, however, corroborate the audit's read-only diagnostic finding that
the CNN behaves as a hard binary switch rather than a graded probability
estimator across the inputs tried so far. Confirming (or ruling out) that this
also holds on real speech requires the labelled validation set; the tooling to
run that comparison exists (see below) but the audio does not yet.

## AASIST Regression Status

Verified unchanged by this phase's helper repair, all against the real V2
checkpoint:

| Property | Status |
|---|---|
| Class mapping bonafide=0/spoof=1 | Confirmed (`test_aasist_v2_checkpoint_identity_is_pinned`) |
| Target samples 64,600 | Confirmed (existing `test_real_model_preprocessing.py` AASIST-V2 tests, unaffected) |
| First crop, right zero-pad | Confirmed (existing tests, unaffected) |
| Per-waveform z-score | Confirmed (existing tests, unaffected) |
| Unnormalised waveform used | Confirmed -- this is exactly what the repaired helper now supplies |
| Real checkpoint loads (strict) | Confirmed (`test_aasist_checkpoint_loads_strictly_with_no_missing_or_unexpected_keys` in the V2 phase suite, unaffected) |
| Real inference executes | Confirmed (`test_real_aasist_v2_can_run_through_the_repaired_helper`, new) |

Fixing checkpoint resolution had one side effect worth recording plainly:
three **legacy V1-architecture** test files
(`test_real_model_adapters.py`, `test_real_model_architectures.py`,
`test_real_model_capture_targets.py`) had been silently testing nothing at
all for AASIST (checkpoint never resolved), and one of them additionally
asserted stale V1 architecture identity/parameter-count constants that
predate the V2 migration. Once checkpoint resolution was fixed:

- `test_real_model_adapters.py::test_real_aasist_branch_produces_a_valid_probability_pair`
  asserted `architecture == "aasist-light-conv1d-bigru-attnpool-recovered-v1"`
  and `parameter_count == 300_035` (V1 values) against what is now correctly
  the real V2 checkpoint. **Fixed**: updated to
  `"aasist-light-v2-finalized-baseline"` / `641_795`, matching the
  already-established, already-attested V2 identity above -- not a new claim,
  just correcting a stale test constant.
- Four tests across `test_real_model_architectures.py` and
  `test_real_model_capture_targets.py` explicitly build
  `app/models/architectures/aasist_light.py::build_aasist_light_net` (the V1
  module) and load it with the generic (non-safe-globals) loader -- a
  combination that only ever worked against V1's flat-state-dict checkpoint
  format, not V2's dict-with-training-metadata format. **Fixed** by pointing
  these four at the new `AASIST_LIGHT_V1_CHECKPOINT` constant (the V1
  artifact still present at `models/best_aasist_light_full_weighted.pth`),
  restoring their original intent (verify the V1 reconstruction against V1's
  own checkpoint) without conflating it with the deployed V2 identity. No
  production code (`app/models/architectures/aasist_light.py`,
  `app/models/capture_targets.py`) was touched.

**Discovered, explicitly out of scope, not modified:**
`app/models/capture_targets.py::aasist_capture_targets()` names V1-era module
paths (`temporal`, `pooling.attention`) and is still registered for the
`aasist` branch name in that module's dispatch table. AASIST-Light V2's real
XAI capture, however, is wired inline in
`app/models/real/inference.py::build_aasist_loader` with V2's own module
names (`frontend`, `gru`, `attention`, `classifier`) and does not call
`capture_targets.py` at all for AASIST. Whether `capture_targets.py`'s AASIST
entry is genuinely live/reachable production code or an orphaned leftover from
before the V2 migration was not determined -- XAI is explicitly out of scope
for this phase. Flagged here so it is not lost; recommend a dedicated,
narrowly-scoped XAI-capture audit before AASIST XAI capture is relied upon.

## Label-Order Validation Tool

`backend/scripts/validate_cnn_class_order.py` (new, developer-only, read-only
w.r.t. configuration).

- Loads the deployed CNN checkpoint through the exact production ingestion
  (`preprocess_audio_file`) and inference path (`_prepare_input`, the same
  private helper `app/models/real/inference.py`'s real predictor uses) --
  **one model load, one forward pass per file** -- then interprets the same
  raw 2-logit output both ways (bonafide=0/spoof=1, and spoof=0/bonafide=1),
  per the task's explicit instruction not to load two differently-configured
  models.
- Accepts a CSV manifest (`audio_path,true_label`, `bonafide`/`spoof`) and
  prints per-sample logits, both softmax values, both mappings' predictions
  and correctness, then per-mapping accuracy / balanced accuracy / bonafide
  recall / spoof recall / confusion matrix -- explicitly captioned **"SMALL
  LABELLED SANITY VALIDATION -- not a benchmark evaluation."**
- Runs a saturation diagnostic (P(index0), P(index1), max probability, entropy
  bits, `saturated = max(P0, P1) >= 0.99`) over whatever it scored -- labelled
  files when given, plus (by default, `--no-synthetic-probes` to disable) the
  same four unlabelled synthetic probes `test_real_model_discrimination.py`
  uses, captioned **"NOT accuracy evidence."**
- Ends with exactly one of three lines:
  `RECOMMEND KEEP bonafide_spoof`, `RECOMMEND INVESTIGATE spoof_bonafide`, or
  `INSUFFICIENT EVIDENCE` -- driven by a simple, stated decision rule
  (default: need >= 3 labelled samples of *each* class, and an accuracy gap
  >= 0.15 between the two mappings; otherwise insufficient evidence). **The
  script never writes `CNN_CLASS_ORDER` or any other configuration file.**
  Acting on its recommendation is an explicit, separate, future task.
- Prints the CNN checkpoint's identity (path, SHA-256, size, modified time,
  configured class order) at the top of every run, with an explicit note that
  the checkpoint carries no class-mapping metadata of its own.

Smoke-tested in this phase two ways: (1) no manifest -- correctly falls back
to synthetic-probe-only mode and reports `INSUFFICIENT EVIDENCE` with a clear
reason; (2) a throwaway two-sample manifest (placeholder tones, **not**
real/labelled speech, generated in `/tmp` and not committed) -- correctly
computed both mappings' accuracy/confusion matrices and correctly reported
`INSUFFICIENT EVIDENCE` because two samples is below the minimum-per-class
threshold. `--json`/`--output` modes verified.

## Real-World Sanity Validation Support

`backend/validation_data/real_world_sanity_set/` (new):

```
validation_data/real_world_sanity_set/
  README.md          (committed -- format + instructions)
  manifest.csv        (gitignored -- your local manifest)
  samples/.gitkeep     (committed placeholder; samples/* gitignored)
```

`backend/.gitignore` updated to ignore `manifest.csv` and everything under
`samples/` in that directory (keeping only `.gitkeep`), so nobody accidentally
commits personal or licensed audio. **No audio files were added or fabricated
in this phase** -- the directory is tooling scaffolding only, per the explicit
instruction not to fabricate metrics. The README documents the recommended
minimum (5 bonafide + 5 spoof, 10+10 better) and the exact command to run
`validate_cnn_class_order.py` against it once populated.

## Tests Repaired

| File | What changed |
|---|---|
| `tests/real_model_helpers.py` | Checkpoint resolution now settings-driven (matches production); `processed_audio()` populates `unnormalised_waveform`; added `AASIST_LIGHT_V1_CHECKPOINT` / `requires_aasist_light_v1_checkpoint` |
| `tests/test_real_model_adapters.py` | Fixed stale V1 architecture/parameter-count assertion in `test_real_aasist_branch_produces_a_valid_probability_pair` to V2's real, attested values |
| `tests/test_real_model_architectures.py` | Four V1-architecture tests repointed from `AASIST_CHECKPOINT` (now correctly V2) to the new `AASIST_LIGHT_V1_CHECKPOINT` |
| `tests/test_real_model_capture_targets.py` | Two V1-architecture capture tests repointed the same way |

## Tests Added

| File | Purpose |
|---|---|
| `tests/test_cnn_verification_guard.py` | 13 tests: checkpoint-resolution parity with production (2), independent `.env` cross-checks (2), silent-skip guard (1), pinned AASIST V2 identity (1), AASIST-not-V1 proof (1), CNN identity-recording (1), `unnormalised_waveform` presence (1), real AASIST V2 through the repaired helper (1), mechanical CNN mapping contract tests (2), AASIST spoof-index characterization test (1) |
| `scripts/validate_cnn_class_order.py` | Developer tool, not a pytest test -- see above |

## Test Results

Full backend suite (`pytest tests -q`):

**610 passed, 4 failed, 9 skipped, 3 deselected.**

The 4 failures are pre-existing and unrelated to this phase (confirmed
unchanged from before this work began; they touch `app/ingestion/audio.py`
under local FFmpeg/environment conditions, not anything modified here):

- `test_audio_preprocessing.py::test_extension_spoofing_is_rejected`
- `test_audio_preprocessing.py::test_audio_duration_limit_is_enforced_and_temp_file_is_removed`
- `test_prediction_routes.py::test_browser_recording_formats_work_with_real_ingestion[webm-...]`
- `test_prediction_routes.py::test_browser_recording_formats_work_with_real_ingestion[m4a-...]`

The 9 skips, with exact reasons (all legitimate -- none are the stale-path
failure mode this phase fixed):

```
tests/test_aasist_light_v2_api_regression.py:118  AASIST-Light V2 checkpoint or public audio fixture is unavailable.
tests/test_aasist_light_v2_api_regression.py:234  AASIST-Light V2 checkpoint or public audio fixture is unavailable.
tests/test_aasist_light_v2_fusion_validation.py:148  AASIST-Light V2 checkpoint or Phase 3 audio fixture is unavailable.
tests/test_aasist_light_v2_fusion_validation.py:320  AASIST-Light V2 checkpoint or Phase 3 audio fixture is unavailable.
tests/test_aasist_light_v2_fusion_validation.py:378  AASIST-Light V2 checkpoint or Phase 3 audio fixture is unavailable.
tests/test_aasist_light_v2_fusion_validation.py:389  AASIST-Light V2 checkpoint or Phase 3 audio fixture is unavailable.
tests/test_aasist_light_v2_fusion_validation.py:438  AASIST-Light V2 checkpoint or Phase 3 audio fixture is unavailable.
tests/test_ffmpeg_audio_formats.py:436  Installed FFmpeg cannot generate the libvorbis fixture.
tests/test_mcp_adapter.py:442  TEST_MONGODB_URI is not configured.
```

The first seven skip because `frontend/e2e/fixtures/sample-voice.wav`
(referenced by those tests) does not currently exist in this checkout --
confirmed by direct inspection; a genuinely missing optional artifact, not
touched by this phase.

Targeted run of every real-model / fusion / AASIST-phase / CNN-verification
file together:

```
tests/test_real_model_discrimination.py
tests/test_real_model_adapters.py
tests/test_real_model_architectures.py
tests/test_real_model_capture_targets.py
tests/test_real_model_preprocessing.py
tests/test_cnn_verification_guard.py
tests/test_aasist_light_v2_checkpoint_integration.py
tests/test_aasist_light_v2_standalone_inference.py
tests/test_aasist_light_v2_fusion_validation.py
tests/test_aasist_light_v2_api_regression.py
tests/test_fusion.py
tests/test_model_runtime_phase3.py
tests/test_model_base.py
tests/test_ssl_sequence_integration.py
```

**202 passed, 7 skipped (the fixture-dependent ones above), 1 deselected, 0
failed.**

`tests/test_real_model_discrimination.py` specifically (the file named in the
audit as silently skipping): **5 passed, 0 skipped** -- was "5 skipped, 0
executed" before this phase's repair.

## Remaining Unknowns

- **CNN class mapping.** Still genuinely unknown. The tooling to resolve it
  exists; the labelled audio to run it against does not, in this environment.
- **CNN feature pipeline.** Still a declared assumption (feature type,
  filterbank shape, FFT parameters) partially constrained only by BatchNorm
  running statistics. No amount of guard-test repair changes this -- it
  requires either the original training configuration or an empirical
  parity/ablation study, neither of which is in scope here.
- **AASIST spoof-index cross-check gap (B5 in the integration audit).**
  Confirmed still present and now pinned by a characterization test
  (`test_characterization_aasist_class_order_env_var_is_not_cross_checked_against_checkpoint`).
  Setting `AASIST_CLASS_ORDER=spoof_bonafide` in `.env` would currently invert
  every AASIST prediction with no load-time error, despite the checkpoint's
  own attestation disagreeing. Not fixed here per this phase's explicit scope
  (no AASIST production code changes).
- **`capture_targets.py`'s AASIST entry** -- possibly orphaned V1-era XAI
  capture wiring, possibly live and broken against V2. Not determined; XAI is
  out of scope for this phase.
- **CNN saturation on real speech.** Confirmed on synthetic probes only (100%
  saturated, unlabelled). Whether this holds on real human/synthetic speech is
  exactly what the real-world sanity validation set (once populated) would
  answer.

## Recommended Next Step

Populate `backend/validation_data/real_world_sanity_set/` with a small local,
labelled set (5+5 minimum, 10+10 better -- see that directory's README for the
manifest format), then run:

```
cd backend
python scripts/validate_cnn_class_order.py \
  --manifest validation_data/real_world_sanity_set/manifest.csv
```

Read its `RECOMMEND KEEP bonafide_spoof` / `RECOMMEND INVESTIGATE spoof_bonafide`
/ `INSUFFICIENT EVIDENCE` verdict, and only then decide, as a separate
explicit task, whether `CNN_CLASS_ORDER` should change. Do not tune fusion
weights or thresholds before that -- half the fused score's meaning is still
unestablished, exactly as the integration audit concluded.

## Verdict

**PASS**

Every item in this phase's scope was completed: checkpoint resolution now
matches production exactly (and is proven to, two independent ways); the
`ProcessedAudio` helper drives real AASIST V2; the previously-silent
discrimination guard suite now executes (5/5, not 5 skipped) against the
correct deployed checkpoints; AASIST-V2 identity is pinned; CNN identity is
recorded through the same mechanism; label-order and saturation tooling exist
and were smoke-tested; a known AASIST cross-check gap was characterized rather
than silently left undocumented; and no production model, fusion, frontend,
schema, or persistence behaviour changed. The CNN class-order question remains
open, exactly as it should given no labelled evidence exists yet -- this
phase's job was to make that question askable and answerable, not to answer
it.
