# SSL (XLS-R + Mamba) Generalization V2 Checkpoint Integration Report

**Date:** 2026-08-19
**Scope:** Integrate `model_artifacts/ssl/ssl_xlsr_mamba_generalization_v2_best.pt` into the existing `ssl_sequence` branch. Audit-first, minimal-change. No retraining, no fusion/threshold changes, no frontend changes, no API schema changes.

---

## 1. Existing SSL implementation audit

The `ssl_sequence` branch was **not a stub** — it was already a mature, fully-wired real-model branch built for a *previous* SSL checkpoint (`models/xlsr_mamba_asvspoof2019_best.pt`, ASVspoof2019-only). Audited files and findings:

| File | Finding |
|---|---|
| `app/models/architectures/xlsr_mamba.py` | Pure-PyTorch reimplementation of XLS-R + 2×Mamba + classifier. Parameter names/shapes (`in_proj`, `conv1d`, `x_proj`, `dt_proj`, `A_log`, `D`, `out_proj`) match `mamba_ssm.modules.mamba_simple.Mamba` exactly, `d_state=16`, `d_conv=4`, `expand=2`. **Already matches the brief's exact architecture — no change needed.** `mamba-ssm`/`causal-conv1d` are CUDA-only and confirmed unable to install on this machine (documented in the module's own docstring), which is why this reimplementation exists. |
| `app/models/real/ssl_sequence_inference.py` | Real loader + predictor. Loads a "metadata package" checkpoint (frozen backbone name + trained head weights), fetches the named Hugging Face backbone separately, freezes it, restores the trained head strictly. **Assumed every artifact self-describes `architecture`/`xlsr_model_name`/`xlsr_hidden_size`/`mamba_dim`/`num_classes`/`label_mapping` — the new Generalization V2 artifact does not.** This was the one real blocking gap; fixed below. |
| `app/models/preprocessing/ssl_waveform.py` | 16kHz, 96000-sample (6s) window, zero-mean/peak normalisation matching the brief's formula exactly. **Crop policy for long audio was a leading/trailing crop (`waveform[:target]`), not the brief's required deterministic center crop.** Fixed below. |
| `app/models/registry.py` / `factory.py` / `runtime.py` | Branch wiring (`ssl_sequence` canonical name, `ssl_wavlm_xlsr` public name, aliases, fusion weight, class-order/verification plumbing) is generic and complete. No changes needed. |
| `app/config/settings.py` | `ssl_model_path` existed but defaulted to `""` and only resolved the *old* checkpoint via `.env`. No `ssl_xlsr_model_name` setting existed (backbone name came only from inside the old self-describing checkpoint). Fixed below. |
| `tests/test_ssl_sequence_integration.py` | 16 hermetic tests + 1 network/integration e2e test, all previously passing against the *old* checkpoint. |
| `pyproject.toml` | `torch`/`transformers` already declared as the `models` optional extra; `mamba-ssm`/`causal-conv1d` deliberately excluded (documented reason: CUDA-only build backends). **No dependency changes were needed** — both packages are already installed in `backend/.venv` (torch 2.13.0, transformers 4.57.6), and the frozen backbone (`facebook/wav2vec2-large-xlsr-53`) was already cached under `~/.cache/huggingface` from prior work on the old checkpoint. |
| Frontend (`frontend/src/lib/copy/detector-names.ts`, `branch-state.ts`) | Already handles the `ssl_sequence`/`ssl_wavlm_xlsr` branch generically. One hardcoded UX string ("The SSL temporal and glottal branches are still in development") remains accurate only while `SSL_MODEL_MODE=disabled`, as it is left after this integration — see §14. |
| `.env` (local, deployment) | `SSL_MODEL_MODE=disabled`, `SSL_MODEL_PATH=models/xlsr_mamba_asvspoof2019_best.pt` (old checkpoint), `ssl_sequence` not in `REQUIRED_MODEL_BRANCHES`. This was a deliberate operator choice (SSL kept off pending verification), not a bug. |

**Repository-wide search** for `xlsr_mamba_asvspoof2019_best.pt` / `ssl_sequence` / `SSL_MODEL` / `xlsr` / `mamba` / `Wav2Vec2` / `facebook/wav2vec2-large-xlsr-53` confirmed no other code paths reference the old checkpoint by name outside `.env`/`.env.example` (updated) and the file itself (left in place, untouched).

---

## 2. Previous checkpoint / reference

`models/xlsr_mamba_asvspoof2019_best.pt` — self-describing package (`architecture`, `xlsr_model_name`, `xlsr_hidden_size`, `mamba_dim`, `num_classes`, `label_mapping: {"bonafide": 0, "spoof": 1}`, `trainable_model_state`, `best_epoch`, `best_dev_eer=0.042761`, `final_eval_metrics` with ASVspoof2019 LA EER 0.0568). **Not deleted, not renamed, still resolvable at its original path.**

## 3. New checkpoint

`model_artifacts/ssl/ssl_xlsr_mamba_generalization_v2_best.pt` — confirmed by direct inspection to be a **trainable-state-only** package:

```
keys: epoch, stage, source_checkpoint, train_config, train_loss, trainable_model_state, sampling_strategy
epoch: 2
stage: Generalization_V2
train_loss: 0.2633038581501354
trainable_model_state: 28 tensors, 1,173,634 total parameters
```

It does **not** contain `architecture`, `xlsr_model_name`, `xlsr_hidden_size`, `mamba_dim`, `num_classes`, or `label_mapping` — exactly as the integration brief stated. SHA-256 (independently recomputed): `336e65d36764315ed4fe036cc99d3b957daae5e07a5ec462f8b09faa35291afe`.

All 28 `trainable_model_state` keys were checked directly against `build_xlsr_mamba_net`'s reconstructed module (`projection.0/1`, `mamba1.*`, `mamba2.*`, `mamba_norm.*`, `classifier.0/3`) — every key/shape matches exactly, and the parameter count sums to **1,173,634**, matching the brief's stated figure precisely.

## 4. Exact architecture

No architecture code changed. `xlsr_mamba.py` was already the exact architecture specified (XLS-R hidden=1024, frozen; projection `LayerNorm→Linear(1024,256)→GELU→Dropout(0.1)`; 2× Mamba (`d_state=16, d_conv=4, expand=2`) with residual adds; `LayerNorm(256)`; masked mean pool (equivalent to `x.mean(dim=1)` for an unpadded window); classifier `Linear(256,128)→GELU→Dropout(0.2)→Linear(128,2)`). Verified end-to-end with the real checkpoint + real backbone (§8).

## 5. Preprocessing contract

| Aspect | Before | After | Status |
|---|---|---|---|
| Sample rate | 16000 Hz | 16000 Hz | unchanged |
| Max duration / samples | 6.0s / 96000 | 6.0s / 96000 | unchanged |
| Mono | shared decoder guarantees it | unchanged | unchanged |
| Short-clip padding | zero-pad at end | zero-pad at end | unchanged |
| Long-clip crop | **leading crop** (`waveform[:target]`, kept the start) | **deterministic center crop** (`waveform[start:start+target]`, `start=(len-target)//2`) | **fixed** — matches the brief exactly (training used random crop, "not appropriate for inference"; center crop is the declared deterministic substitute) |
| Normalisation | zero-mean, peak-normalise (`x - mean`, then `/max(abs(x))` if `>1e-7`) | unchanged | Matches the brief's formula (`audio - audio.mean()`, `/peak if peak>1e-8`) exactly; the ~1e-7 vs 1e-8 epsilon difference is cosmetic |
| Attention mask | built from real/padded sample count (never `input_values != 0`) | unchanged | Already correct per the brief's explicit warning against the `!= 0` anti-pattern |
| HF `Wav2Vec2FeatureExtractor` | not used; custom implementation | not used; custom implementation (deliberate) | See note below |

**Note on the Hugging Face feature extractor:** the brief asks for `Wav2Vec2FeatureExtractor`-style `padding=True, return_tensors="pt", return_attention_mask=True`, but also gives an explicit zero-mean/**peak**-normalisation formula. `Wav2Vec2FeatureExtractor(do_normalize=True)`'s actual default normalises to zero-mean/**unit-variance**, not peak — using the real HF class would silently contradict the given formula. The existing custom `SSLWaveformFrontEnd` already implements the exact given formula and already avoids the `!= 0` mask anti-pattern the brief warns about, so it was kept rather than replaced with a class whose default disagrees with the spec. This is a **documented judgment call**, not an oversight — flagged for the user to confirm.

Preprocessing version bumped: `ssl-xlsr-waveform-frontend-v2-training-peak` → `ssl-xlsr-waveform-frontend-v3-center-crop-peak`.

## 6. Label mapping

`bonafide=0, spoof=1` (`SSL_CLASS_ORDER=bonafide_spoof`, unchanged). The new checkpoint carries no `label_mapping` of its own; the loader now declares `{"bonafide": 0, "spoof": 1}` as a fallback (`DECLARED_LABEL_MAPPING` in `ssl_sequence_inference.py`) when the artifact doesn't supply one. Mechanically verified with controlled-logit contract tests (§10) — this proves the configured mapping is *applied* as claimed, not that it is correct; correctness rests on the training pipeline's own attestation, same trust boundary as CNN/AASIST's declared feature contracts.

## 7. Runtime dependencies

No changes. `torch` and `transformers` were already present in `backend/.venv` (2.13.0 / 4.57.6) and already declared as the `models` optional extra in `pyproject.toml`. `mamba-ssm`/`causal-conv1d` remain deliberately excluded (confirmed still cannot build on this non-CUDA machine); the existing pure-PyTorch reimplementation is used, as it already was for the old checkpoint. The `facebook/wav2vec2-large-xlsr-53` backbone (~1.2 GB) was already cached under `~/.cache/huggingface` from prior work, so no first-run download was needed during this integration — a fresh environment would still need one-time network access, exactly as already documented.

## 8. Files changed

Tracked files (git-visible):

| File | Change |
|---|---|
| `backend/app/config/settings.py` | `ssl_model_path` default now points at the new checkpoint (`ssl/ssl_xlsr_mamba_generalization_v2_best.pt`); added `ssl_xlsr_model_name` setting (default `facebook/wav2vec2-large-xlsr-53`, alias `SSL_XLSR_MODEL_NAME`); updated stale comments referencing the old checkpoint's self-description |
| `backend/app/models/real/ssl_sequence_inference.py` | Loader now supports both the old self-describing format and the new trainable-state-only format (declared fallback constants `DECLARED_XLSR_HIDDEN_SIZE/DECLARED_MAMBA_DIM/DECLARED_NUM_CLASSES/DECLARED_LABEL_MAPPING`); `LoadedSSLBranch` gained `checkpoint_epoch`/`checkpoint_stage`/`checkpoint_train_loss`; prediction metadata now includes those plus `label_mapping` and a documented `threshold: 0.5` |
| `backend/app/models/preprocessing/ssl_waveform.py` | Long-clip crop changed from leading to deterministic center crop; `as_provenance()`'s `length_policy` corrected to `center_crop_pad`; preprocessing version bumped |
| `backend/tests/test_ssl_sequence_integration.py` | One existing test updated to fall back to declared constants when reading `xlsr_hidden_size`/`mamba_dim`/`num_classes` directly from the artifact (previously assumed the self-describing format unconditionally) |
| `backend/.env.example` | Comments/paths updated to describe the new checkpoint and `SSL_XLSR_MODEL_NAME` |

Untracked (gitignored) files, also changed:

| File | Change |
|---|---|
| `backend/.env` | `SSL_MODEL_PATH` repointed at the new checkpoint; `SSL_XLSR_MODEL_NAME` added explicitly; comments updated. **`SSL_MODEL_MODE` left as `disabled`, unchanged** — see §14 |

New files:

| File | Purpose |
|---|---|
| `backend/tests/test_ssl_generalization_v2_integration.py` | 13 new tests specific to this integration (checkpoint identity pinning, loader fallback for both artifact formats, label-mapping mechanics, center-crop content verification, checkpoint-metadata-in-prediction, backbone-unavailable failure mode) |
| `backend/scripts/validate_ssl_generalization_v2_inference.py` | Standalone developer harness (loads the real branch, prints checkpoint/architecture diagnostics, optionally classifies one audio file) |
| `backend/SSL_XLSR_MAMBA_GENERALIZATION_V2_INTEGRATION_REPORT_2026-08-19.md` | This report |

**Not changed:** `xlsr_mamba.py` (architecture, already correct), CNN, AASIST, Glottal, fusion engine/weights/thresholds, API routes/schemas, database/persistence, XAI algorithms, frontend, `REQUIRED_MODEL_BRANCHES`, `SSL_MODEL_MODE`.

## 9. Tests added

13 new tests in `test_ssl_generalization_v2_integration.py`:

1. Checkpoint resolves through runtime Settings (not a guessed path)
2. Default `ssl_model_path` points at the new checkpoint, not the superseded one
3. Checkpoint SHA-256 is pinned (`336e65d3...`)
4. Resolved checkpoint is not the superseded V1 file
5. Loader fills in declared defaults for a trainable-state-only artifact (synthetic, in-memory)
6. Loader prefers embedded values over declared defaults for a self-describing artifact (proves the fallback never silently overrides an explicit value)
7. Loader still rejects a declared-architecture mismatch for a self-describing artifact
8. The real deployed artifact genuinely has no embedded self-description (guards against the fallback path silently going untested if a future export changes format)
9. Configured label mapping mechanically selects the matching logit column (contract test, controlled logits)
10. `SSL_CLASS_ORDER` default matches the declared `bonafide_spoof` convention
11. Center crop keeps the *middle* of long audio (spike-position test — proven, not just shape-checked)
12. *(integration+network)* Checkpoint epoch/stage/train_loss/threshold/label_mapping appear in real prediction metadata
13. *(integration+network)* An unavailable HF backbone name fails loudly (`ModelLoadError`, `checkpoint_missing`) rather than substituting a dummy prediction

Plus 1 existing test updated (`test_reconstructed_architecture_state_dict_matches_real_artifact_exactly`) to work against whichever artifact format is actually deployed.

## 10. Commands executed

```
./.venv/bin/python -m pytest tests/test_ssl_sequence_integration.py -q
./.venv/bin/python -m pytest tests/test_ssl_sequence_integration.py -q -m "integration and network"
./.venv/bin/python -m pytest tests/test_ssl_generalization_v2_integration.py -q
./.venv/bin/python -m pytest tests/test_ssl_generalization_v2_integration.py -q -m "integration and network"
./.venv/bin/python -m pytest tests/test_ssl_sequence_integration.py tests/test_real_model_adapters.py \
    tests/test_real_model_architectures.py tests/test_real_model_preprocessing.py \
    tests/test_model_runtime_phase3.py tests/test_fusion.py tests/test_cnn_verification_guard.py \
    tests/test_real_model_discrimination.py tests/test_real_model_capture_targets.py -q
./.venv/bin/python -m pytest -q   # full suite
./.venv/bin/python scripts/validate_ssl_generalization_v2_inference.py --audio /tmp/ssl_smoke.wav
./.venv/bin/python scripts/validate_ssl_generalization_v2_inference.py --json
```

Plus ad hoc verification scripts (not committed): direct `torch.load` inspection of both checkpoints; direct `_load_artifact_package`/`_restore_trainable_state` calls confirming strict, zero-error loading and an exact 1,173,634 trainable-parameter count; a direct `xai_attention_windows()` call against the real loaded branch.

## 11. Test results

| Suite | Result |
|---|---|
| `test_ssl_sequence_integration.py` (hermetic) | 16 passed, 1 deselected |
| `test_ssl_sequence_integration.py` (integration+network e2e) | 1 passed |
| `test_ssl_generalization_v2_integration.py` (hermetic) | 11 passed |
| `test_ssl_generalization_v2_integration.py` (integration+network) | 2 passed |
| Combined targeted regression set (9 files above) | 136 passed, 1 deselected |
| **Full backend suite** | **621 passed**, 4 failed, 9 skipped, 5 deselected |

The 4 failures (`test_extension_spoofing_is_rejected`, `test_audio_duration_limit_is_enforced_and_temp_file_is_removed`, two `test_browser_recording_formats_work_with_real_ingestion` cases) are **pre-existing, local FFmpeg/environment issues unrelated to SSL** — confirmed identical to the baseline recorded in the prior `CNN_VERIFICATION_AND_REAL_MODEL_GUARD_REPAIR_REPORT_2026-08-19.md` (same 4 failures, same 610 passed baseline before this session's +11 new hermetic tests). The 9 skips are the same pre-existing, legitimately-reasoned skips (missing `sample-voice.wav` fixture ×7, missing libvorbis codec ×1, missing `TEST_MONGODB_URI` ×1) — none newly introduced.

## 12. Standalone script verification

`scripts/validate_ssl_generalization_v2_inference.py` run against the real deployed checkpoint + real backbone:

```
Resolved checkpoint: ssl_xlsr_mamba_generalization_v2_best.pt
Checkpoint SHA-256: 336e65d36764315ed4fe036cc99d3b957daae5e07a5ec462f8b09faa35291afe
Checkpoint epoch: 2
Checkpoint stage: Generalization_V2
Checkpoint train_loss: 0.2633038581501354
XLS-R model name: facebook/wav2vec2-large-xlsr-53
Trainable parameter count: 1173634 (matches expected: True)
Prediction (synthetic 3s harmonic+noise probe): bonafide, P(spoof)=0.499496, sum=1.000000
```

`--json` mode verified structurally (valid JSON, all expected fields present).

## 13. Known limitations

* **Preprocessing not re-verified against training**: the declared zero-mean/peak/center-crop formula is taken from the integration brief, not re-derived from a training notebook (none shipped with the checkpoint). `SSL_PREPROCESSING_VERIFIED` remains `false`.
* **Label mapping not re-verified against training**: `bonafide=0, spoof=1` is a declared continuation of the same convention the superseded checkpoint documented; the new artifact carries no label map of its own. `SSL_CLASS_MAPPING_VERIFIED` remains `false`.
* **Mamba scan is a pure-PyTorch reimplementation**, not the CUDA `mamba-ssm` kernel the model was almost certainly trained with. Mathematically the same recurrence, but not yet cross-checked on this runtime against the documented Generalization V2 evaluation metrics (ASVspoof5 TEST / WaveFake unseen-generator numbers in the integration brief, §9 there).
* **HF feature extractor not used** — see §5's note; a deliberate, documented substitution, not a gap, but worth the user's explicit sign-off.
* **`app/models/capture_targets.py`'s AASIST entry references V1-era module paths** (discovered during the prior CNN verification task, unrelated to this integration, not touched here).
* **`SSL_MODEL_MODE` left `disabled`** — see §14, a deliberate scope boundary, not a limitation of the code itself (verified fully functional end-to-end).

## 14. External-validation remaining work / recommended next steps

1. **Re-run the documented Generalization V2 evaluation** (ASVspoof5 TEST: accuracy 0.7921, F1 0.8191, EER 0.1856; WaveFake unseen-generator: 90.3% spoof-detection rate) against this pure-PyTorch Mamba scan on this runtime, then flip `SSL_PREPROCESSING_VERIFIED`/`SSL_CLASS_MAPPING_VERIFIED` to `true` only if it reproduces.
2. **Decide, as a separate explicit step, whether to set `SSL_MODEL_MODE=real`** in `backend/.env`. This integration makes the branch fully load- and inference-capable (proven end-to-end with the real checkpoint and real backbone), but the mode switch itself was deliberately left untouched — flipping it changes what every production fusion result includes, which is a live-behavior decision beyond "integrate the checkpoint," and CNN/AASIST's own precedent in this repo is to run real-but-unverified models with `research_result: false` flagged rather than gate on verification. If flipped, also consider whether `ssl_sequence` should join `REQUIRED_MODEL_BRANCHES` (it currently should NOT, to preserve "SSL failing never fails the whole prediction").
3. If step 2 is taken, **update the one hardcoded frontend string** ("The SSL temporal and glottal branches are still in development", `frontend/src/lib/models/branch-state.ts:133`) — not changed here per the read-only frontend constraint.
4. The WaveFake 90.3%/1.1% figures are a **spoof-only unseen-generator detection rate**, not balanced accuracy or EER — do not conflate with the ASVspoof5 balanced metrics above when documenting this checkpoint elsewhere.
5. Note that ASVspoof5 TEST accuracy nominally *regressed* slightly from Epoch 1 (0.7944) to the selected Epoch 2 (0.7921); the epoch was chosen for cross-domain generalization (WaveFake 43.6%→90.3%), not peak in-domain accuracy — worth stating explicitly wherever these numbers are cited so the choice reads as intentional, not an unnoticed regression.

---

**Summary verdict:** the Generalization V2 checkpoint is now fully integration-capable — loads strictly (zero missing/unexpected trainable keys, exact 1,173,634-parameter match), runs real end-to-end inference and real temporal-XAI attention extraction, and is covered by 13 new regression tests plus 621 total passing tests. Its correctness (label direction, feature pipeline) remains a declared-not-proven attestation, unchanged in kind from CNN/AASIST's existing status in this repository, and the decision to make it live in production (`SSL_MODEL_MODE=real`) was deliberately left to the user.
