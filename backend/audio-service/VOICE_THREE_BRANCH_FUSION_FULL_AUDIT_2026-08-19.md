# Voice Three-Branch Fusion — Full Technical Audit

**Date:** 2026-08-19
**Scope:** CNN, AASIST, SSL/XLS-R+Mamba branches, fusion algorithm, orchestration, API, health, tests. **Audit only — no production code was modified.** All findings below are backed by reading the actual implementation and by real, executed runs (real checkpoints, real fusion function, no mocks, no invented numbers).

---

## 1. Executive summary

The three branches genuinely load their intended checkpoints (verified by independent SHA-256 recomputation), produce probabilities in the correct semantic direction (higher = more spoof), and feed a real weighted-average fusion function that correctly excludes unavailable branches (renormalizing weights, never substituting a zero score) and correctly enforces a minimum-successful-branch gate. **No critical correctness bug was found**: no wrong checkpoint, no inverted label mapping, no dummy branch masquerading as real, no fusion arithmetic error.

However, the **live deployment today only runs two of the three intended branches**. `backend/.env` currently has `SSL_MODEL_MODE=disabled` — a deliberate, previously-documented choice, not a bug — so production fusion is CNN+AASIST only (50/50 weight), not the three-branch system this audit was asked to validate. SSL was proven fully load- and inference-capable in this audit (real checkpoint, real backbone, real fusion), but is not currently live. Glottal has no checkpoint at all and is disabled.

All three branches' feature pipelines and label mappings remain **declared, not verified** against a training pipeline (none of the three checkpoints shipped with training notebooks). AASIST additionally has a latent, currently-inactive but real gap: its runtime spoof index comes only from an env setting, never cross-checked against the checkpoint's own attested class mapping, even though that attestation is already loaded and validated at checkpoint-load time.

**Readiness classification: PARTIALLY READY** (see §27 for the full justification).

## 2. Current architecture

```
Audio upload
  -> shared ingestion/preprocessing (ffprobe inspect, decode, resample to 16kHz,
     mono-convert, peak-normalise -> ProcessedAudio{waveform, unnormalised_waveform})
  -> VoiceService._predict_from_validated_upload()
       for each branch in ModelRegistry.models (SEQUENTIAL, one ThreadPoolExecutor
       per branch used only for a timeout, not for parallelism):
         CNN (lfcc_cnn_tcn)   -> BranchPrediction
         AASIST (aasist)      -> BranchPrediction
         SSL (ssl_sequence)   -> BranchPrediction   [currently: status=skipped, disabled]
         Glottal (glottal)    -> BranchPrediction   [status=skipped, disabled, no checkpoint]
  -> FusionEngine.fuse(branch_predictions)   (weighted average over successful branches only)
  -> VoicePredictionResponse{branches, fusion, provenance}
```

## 3. Real checkpoint inventory

| Branch | Impl. class | Architecture file | Adapter file | Preprocessing file | Checkpoint path (resolved) | Exists? | Loaded? | Mode (live `.env`) | Device | Label mapping | Output semantics | Used in fusion? | Fallback |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNN | `LFCCCnnTcnRealAdapter` | `app/models/architectures/cnn.py` | `app/models/real/lfcc_cnn_tcn.py` (→ `real/inference.py`) | `app/models/preprocessing/spectral.py` | `models/best_cnn_full_weighted.pth` | Yes | Yes (verified this audit) | **real** | cpu | `bonafide_spoof` (index 0=bonafide, 1=spoof), declared | softmax spoof=index 1 | Yes | Fails closed to `BranchStatus.failed`; never dummy |
| AASIST | `AASISTRealAdapter` | `app/models/architectures/aasist_light_v2.py` | `app/models/real/aasist.py` (→ `real/inference.py`) | `app/models/preprocessing/aasist_light_v2.py` | `model_artifacts/aasist/aasist_light_v2_best.pt` | Yes | Yes (verified this audit) | **real** | cpu | `bonafide_spoof`, checkpoint-attested `{"bonafide":0,"spoof":1}` | softmax spoof=index 1 | Yes | Fails closed; never dummy |
| SSL | `SSLSequenceRealAdapter` | `app/models/architectures/xlsr_mamba.py` | `app/models/real/ssl_sequence.py` (→ `real/ssl_sequence_inference.py`) | `app/models/preprocessing/ssl_waveform.py` | `model_artifacts/ssl/ssl_xlsr_mamba_generalization_v2_best.pt` | Yes | **Load-capable, verified this audit, but not started in the live `.env`** | **disabled** | n/a (would be cpu) | `bonafide_spoof`, declared (checkpoint has no self-description) | softmax spoof=index 1 | **No, currently excluded (disabled)** | `BranchStatus.skipped`, contributes nothing |
| Glottal | `GlottalRealAdapter` (unused) / `DisabledModelAdapter` | n/a | `app/models/real/glottal.py` (exists, no checkpoint wired) | n/a | none configured | No checkpoint present | No | **disabled** | n/a | n/a | n/a | **No** | `BranchStatus.skipped`, contributes nothing |

Independent SHA-256 recomputation (this audit, `checkpoint_identity_for_path` — the exact function `ModelFactory` uses in production):

```
CNN    : 4bf86dd15108a8cff43569ef907a9f3852978353af6d6e184001480a0fa2e6f8  (104,273 bytes)
AASIST : c944c69f05a0135ad9dfdaf86fb8a31815339f862d780c5f2c5d834f181e64ec  (7,753,595 bytes)
SSL    : 336e65d36764315ed4fe036cc99d3b957daae5e07a5ec462f8b09faa35291afe  (4,704,997 bytes)
```

The old/superseded checkpoints (`models/xlsr_mamba_asvspoof2019_best.pt`, `models/best_aasist_light_full_weighted.pth`) are **not referenced by any current default configuration path** — confirmed by repository-wide search for both filenames outside `.env`/`.env.example`/the files themselves; only `AASIST_LIGHT_V1_CHECKPOINT`/`requires_aasist_light_v1_checkpoint` test helpers reference the old AASIST file, deliberately, to keep legacy V1-architecture regression tests honest (see §22 of the prior `CNN_VERIFICATION_AND_REAL_MODEL_GUARD_REPAIR_REPORT_2026-08-19.md`).

## 4. CNN audit

1. **Checkpoint loaded**: `models/best_cnn_full_weighted.pth`, confirmed by direct load in this audit (SHA-256 above). Not another checkpoint.
2. **Path resolution**: via `Settings.resolved_model_root_dir` (`.env`'s `MODEL_ROOT_DIR=..`, i.e. repository root) + `CNN_MODEL_PATH=models/best_cnn_full_weighted.pth` — same mechanism `ModelFactory.branch_config` uses in production. Verified to resolve to the correct absolute path.
3. **Architecture match**: `build_cnn_acoustic_net()` (3×Conv2d/BatchNorm2d/ReLU/MaxPool2d blocks + AdaptiveAvgPool2d + Linear(64,2)) is reconstructed directly from the checkpoint's own tensor shapes (`features.0.weight [16,1,3,3]`, ..., `classifier.3.weight [2,64]`), documented layer-by-layer in `cnn.py`'s module docstring.
4. **Strict or partial loading**: **strict** (`load_state_dict_strict`, `app/models/torch_support.py`) — any missing/unexpected key raises `CheckpointCompatibilityError` → `ModelLoadError(checkpoint_incompatible)`, never a silent partial load.
5. **Missing/unexpected keys**: none — strict load succeeds (confirmed by this audit's live load).
6. **Feature extraction**: `log_mel`, 20 filters, `global_zscore` normalisation, `n_fft=512`, `hop_length=160`, `win_length=400`, `max_frames=400` (from `.env`'s `CNN_FEATURE_*` settings). **This is a declared, not verified, configuration.** No training notebook shipped with the checkpoint; the current values were chosen by sweeping 72 feature configurations and picking the one whose zero-mean/~unit-variance statistics matched the checkpoint's own first-BatchNorm running statistics (documented in `spectral.py`'s module docstring and `.env.example`). This does **not** prove the pipeline is correct — only that it isn't degenerate. **Not independently re-verified in this audit** (out of scope — this is a re-statement of the existing, still-open finding, not new evidence either way).
7. **Class mapping**: `CNN_CLASS_ORDER=bonafide_spoof` → index 0 = bonafide, index 1 = spoof. **Declared, unverified** — same status as CNN preprocessing; no training artifact confirms this. `.env`'s own comment (`CNN_CLASS_ORDER`/`AASIST_CLASS_ORDER` block) records that CNN and AASIST checkpoints agree with each other on only 1 of 8 synthetic probe signals under the same class-order assumption and disagree on the other 7 — meaning **at least one of the two is probably using the wrong order**, an unresolved finding carried forward from the prior audit, not fixed here (fix requires labelled evidence, per the existing `scripts/validate_cnn_class_order.py` tool — see §26).
8. **Fusion direction**: verified — `spoof_index = 1 if class_order == "bonafide_spoof" else 0`; `_spoof_probability_from_logits` returns `softmax(logits)[spoof_index]`, which is exactly the value passed into `real_prediction_from_spoof_probability` → `ProbabilityScores.spoof` → fusion. Higher = more spoof, confirmed mechanically (see §8 label-mapping contract tests already in the repository, `test_cnn_verification_guard.py`).
9. **Historical label-mapping bug**: none *found in code* — but see item 7's unresolved CNN/AASIST disagreement, which is exactly the shape of bug this would be if either is wrong.
10. **Do current CNN tests load the real checkpoint or mock?** Both exist, clearly separated: `test_real_model_adapters.py`/`test_real_model_architectures.py`/`test_cnn_verification_guard.py` gate on `requires_cnn_checkpoint` and run the real file (35/9/14 real-checkpoint-marked tests respectively, confirmed by this audit's grep); `test_fusion.py`/`test_voice_service.py`/`test_voice_routes.py` use synthetic `BranchPrediction`/mocks and never touch the real checkpoint.
11. **CNN saturation** (re-confirmed live in this audit, §14): on non-speech synthetic probes, CNN output is frequently **exactly** 1.0 or ~6×10⁻⁸ — a fully saturated classifier on out-of-training-distribution input. Consistent with the prior `CNN_VERIFICATION_AND_REAL_MODEL_GUARD_REPAIR_REPORT_2026-08-19.md` finding; not a new discovery, re-verified true today.
12. **Status: REAL.** Checkpoint genuinely loads and runs; correctness (feature pipeline, class order) remains unverified, which is a documented limitation, not a misconfiguration.

## 5. AASIST audit

1. **Checkpoint actually selected**: `model_artifacts/aasist/aasist_light_v2_best.pt` — verified live-loaded in this audit, SHA-256 matches. **Not** the historical `models/best_aasist_light_full_weighted.pth` (V1), which is confirmed unreachable from any current default path.
2. **Architecture compatibility**: `build_aasist_light_v2_net()`. Checkpoint load additionally validates (`load_aasist_light_v2_checkpoint_strict`, `real/inference.py`): epoch == 13, `class_mapping == {"bonafide":0,"spoof":1}`, `best_eer ≈ 0.039245587694380565` (exact float compare, `abs_tol=1e-12`), parameter count == 641,795, plus a companion `aasist_light_v2_final_summary.json` cross-check (model_name, status, epoch, label_mapping, sample_rate=16000, max_audio_samples=64600). This is the most rigorously self-validating checkpoint loader of the three branches.
3. **Preprocessing**: 16kHz, 64,600 samples, **first-crop / right-zero-pad** (keeps the clip's start, not centered — different convention from SSL's center-crop; both are declared, neither proven, and there is no requirement they match each other), per-waveform z-score (`(x - mean) / (std + 1e-6)`), silence guard `std < 1e-7` raises `model_input_invalid`. Uses the **unnormalised** decoded waveform (`requires_unnormalized_waveform = True`), not the shared peak-normalised one — confirmed correct per the checkpoint's training convention.
4. **Class mapping**: `AASIST_CLASS_ORDER=bonafide_spoof` env setting → `spoof_index=1`. The checkpoint's own attested `class_mapping` (validated during load, item 2 above) also says `{"bonafide":0,"spoof":1}` — **today these agree**. But `spoof_index` is computed purely from the env setting; it is never cross-checked against `checkpoint_metadata["class_mapping"]`, which was already loaded and validated moments earlier in the same function. See §8 and the Critical/High findings below — this is a real, currently-latent gap, already characterized by an existing test (`test_characterization_aasist_class_order_env_var_is_not_cross_checked_against_checkpoint`, `test_cnn_verification_guard.py`).
5. **Fusion direction**: confirmed — higher AASIST score = more spoof, mechanically verified.
6. **Runtime device**: cpu (`.env`: `AASIST_DEVICE` unset → falls back to `MODEL_DEVICE_POLICY=cpu`; documented reason: AASIST-Light measured ~30ms CPU vs ~192ms MPS on the reference machine).
7. **Prior phase reports** (`AASIST_LIGHT_V2_PHASE1-6`, `backend/*.md`): all confirm the same facts independently — strict load, epoch 13, 64,600 samples, z-score normalisation, bonafide=0/spoof=1, deterministic CPU inference, Phase 6 verdict "READY TO COMMIT WITH DOCUMENTED ENVIRONMENT ISSUES." Nothing in this audit contradicts those reports.
8. **Status: REAL.**

## 6. SSL audit

1. **Checkpoint selected**: `model_artifacts/ssl/ssl_xlsr_mamba_generalization_v2_best.pt` (SHA-256 above), confirmed by direct load in this audit. The old `models/xlsr_mamba_asvspoof2019_best.pt` is **not** the default and is not reachable from any current setting.
2. **Backbone**: `facebook/wav2vec2-large-xlsr-53`, loaded separately via Hugging Face (`ssl_xlsr_model_name` setting), frozen (`requires_grad_(False)` on every backbone parameter, `.eval()`). Confirmed already cached locally (`~/.cache/huggingface`), so no first-run download was needed for this audit's real end-to-end runs.
3. **Architecture**: `xlsr_mamba.py`'s `build_xlsr_mamba_net()` — `LayerNorm(1024)→Linear(1024,256)→GELU→Dropout(0.1)` projection, two `Mamba(d_state=16, d_conv=4, expand=2)` blocks each with a residual add, `LayerNorm(256)`, masked mean pool, `Linear(256,128)→GELU→Dropout(0.2)→Linear(128,2)` classifier — **matches the requested architecture exactly**, confirmed by direct inspection (not re-derived from the checkpoint, since it has no self-description; verified instead by loading the checkpoint's 28 `trainable_model_state` tensors strictly into this exact module graph, which succeeded with zero missing/unexpected keys and an exact 1,173,634-parameter match).
4. **Label mapping**: `bonafide=0, spoof=1`, declared (`SSL_CLASS_ORDER=bonafide_spoof`), because the checkpoint carries no `label_mapping` of its own (trainable-head-only package). Final spoof probability is `softmax(logits, dim=-1)[spoof_index]` = index 1 under this convention — matches the requested `softmax(logits, dim=1)[:, 1]` exactly.
5. **Preprocessing**: 16kHz, mono, mean-centre + peak-normalise (`(x-mean)/max(|x-mean|)` guarded at `1e-7`), 6s/96,000-sample window, **deterministic center crop** for long clips, zero-pad-at-end for short clips, attention mask built from real/padded sample counts (never `input_values != 0`). All of this matches the requested contract exactly; this was fixed during the prior SSL integration task (crop was previously leading, not centered — corrected then, re-verified true now).
6. **Real Hugging Face feature extractor**: **not used** — a deliberate substitution. `Wav2Vec2FeatureExtractor(do_normalize=True)`'s actual default normalises to zero-mean/**unit-variance**, which contradicts the explicitly requested zero-mean/**peak** formula; the hand-written front end implements the requested formula exactly and already avoids the `!= 0` mask anti-pattern. Documented as a judgment call in `SSL_XLSR_MAMBA_GENERALIZATION_V2_INTEGRATION_REPORT_2026-08-19.md` §5 — flagged again here for visibility, not re-litigated.
7. **Live status**: **`SSL_MODEL_MODE=disabled` in the current `.env`.** The branch is fully functional (proven in this audit — see §14) but does not run for real API requests today. This was a deliberate choice recorded in the prior integration report, not an oversight of this audit.
8. **Status: REAL (when explicitly enabled), DISABLED in the live deployment.**

## 7. SSL research result context

Reported for audit context only, not re-verified and not hard-coded anywhere in code:

- ASVspoof5 TEST: accuracy 0.7921, precision 0.724935, recall 0.941400, F1 0.819107, ROC-AUC 0.877987, EER 0.1856.
- WaveFake generator-unseen: 1000 spoof-only samples, spoof detection rate 0.9030, mean spoof probability 0.850188. **This is a spoof-only detection rate, not balanced accuracy or EER** — it says nothing about the false-positive rate on bonafide audio from unseen generators. Not conflated with the ASVspoof5 balanced metrics anywhere in this report.

## 8. Label mapping matrix

| Branch | Class 0 | Class 1 | Verified against training? |
|---|---|---|---|
| CNN | bonafide (declared) | spoof (declared) | **No** |
| AASIST | bonafide (checkpoint-attested + env setting agree) | spoof (checkpoint-attested + env setting agree) | Checkpoint attests it; **the runtime never cross-checks its own env setting against that attestation** (see §14 Critical/High findings) |
| SSL | bonafide (declared) | spoof (declared) | **No** — checkpoint carries no label map |

## 9. Probability semantic matrix (real run, "harmonic_speechlike" probe — see §14 for full data)

| Branch | Raw output | Softmax/sigmoid | Bonafide prob. | Spoof prob. | Score → fusion | Direction |
|---|---|---|---|---|---|---|
| CNN | 2 logits | softmax | 0.0 | 1.0 | 1.0 | higher = more spoof ✓ |
| AASIST | 2 logits | softmax | 0.0121 | 0.9879 | 0.9879 | higher = more spoof ✓ |
| SSL | 2 logits | softmax | 0.4194 | 0.5806 | 0.5806 | higher = more spoof ✓ |

Required semantic (0.0 = strongly bonafide, 1.0 = strongly spoof) holds for **all three branches**, mechanically confirmed by `_spoof_probability_from_logits`'s shared implementation (CNN/AASIST) and its SSL-branch equivalent, plus the live run above. **No branch was found using the opposite direction.**

## 10. Fusion algorithm

Read directly from `app/utils/fusion.py::FusionEngine`. Actual formula (weighted average, the configured default — `fusion_method` also supports `simple_average` and `majority_vote`, neither currently configured):

```
successful = [b for b in branches if b.status == success and b.probabilities is not None]
raw_weight[b] = configured_weight_for(b.model_name)   # 0.0 if branch has no configured weight
effective_weight[b] = raw_weight[b] / sum(raw_weight[b'] for b' in successful)   # renormalized over SUCCESSFUL branches only
final_spoof_score = sum(b.probabilities.spoof * effective_weight[b] for b in successful)
final_bonafide_score = sum(b.probabilities.bonafide * effective_weight[b] for b in successful)
# then re-normalised so bonafide+spoof == 1 exactly (they already do, but this guards float drift)
prediction = spoof if final_spoof_score >= spoof_threshold else bonafide
```

This **is** a confidence-agnostic linear weighted average — not logit fusion, not majority voting (in the configured mode), not threshold-based per-branch gating. Confirmed by the disagreement-scenario runs in §16 (Scenario A: CNN=0.10, AASIST=0.90, SSL=0.90 → fused = (0.10+0.90+0.90)/3 = **0.6333**, exactly the plain average — no outlier suppression, no confidence weighting).

## 11. Fusion weights

Defined in `Settings.fusion_weight_map` (`app/config/settings.py`), sourced from `.env`:

```
FUSION_WEIGHT_LFCC_CNN_TCN = 0.25   (not set in .env, Python default)
FUSION_WEIGHT_AASIST       = 0.25
FUSION_WEIGHT_SSL_SEQUENCE = 0.25
FUSION_WEIGHT_GLOTTAL      = 0.25
```

Configurable: yes, via `.env`. **Currently equal weight across all four branches**, so effectively equal weight among whichever branches actually contribute — sum to 1.0 by design of the four static values, and are **re-normalized over successful branches only** (`_normalize_weights`), confirmed live:

- CNN+AASIST live today (SSL/glottal disabled): weights renormalize to **0.5 / 0.5** (verified in §14's "current deployment" run).
- All three real branches active (this audit's forced test): weights renormalize to **1/3 / 1/3 / 1/3** (verified in §14).

**Critical questions, answered from the real implementation (not inferred):**

- *If SSL fails to load*: excluded from `successful_branches`, `excluded_branches["ssl_wavlm_xlsr"] = <error_code>`, remaining weights renormalize — confirmed (§16 Scenario D: CNN unavailable → AASIST+SSL renormalize to 0.5/0.5 automatically).
- *If AASIST fails*: same mechanism, symmetric.
- *If only CNN succeeds*: with the **production default** `fusion_min_successful_branches=2`, fusion **refuses to produce a verdict at all** — `status=failed`, `prediction=null`, warning "At least two successful branches are required for fusion." (confirmed live in §16 Scenario E, run against the actual production default). A single successful branch never reaches the API as a confident final verdict.
- *Does fusion silently substitute dummy scores?* No — `_weight_for_branch` returns `0.0` for a branch with no configured weight, but that branch is never in `successful_branches` to begin with unless its `status == success`; a failed/disabled/skipped branch is excluded from the weighted-average sum entirely, not included at a weight of 0 with a manufactured score.
- *Do unavailable branches pull the score toward bonafide?* **No** — confirmed by construction (`_average_probabilities` only iterates `successful_branches`) and by the live Scenario D/E tests: excluding a branch changes the denominator, not the numerator, so it cannot bias the result toward either class.

One real, if narrow, gap: **if a `dummy`-mode branch is active**, its deterministic placeholder score IS included in `successful_branches` (dummy adapters return `BranchStatus.success`) and DOES numerically contribute to `final_spoof_score` — flagged via `contains_dummy_branches=True` and a hard warning, and the response schema's own validator (`VoicePredictionResponse.validate_dummy_fusion_consistency`) forbids claiming research eligibility when that's true. This is correctly gated for research-result purposes but the raw returned `probabilities.spoof` number is still influenced by the dummy branch's fake score — worth knowing if any caller reads `fusion.probabilities` without checking `contains_dummy_branches` first. **Not applicable to the current live deployment** (no branch is currently in dummy mode: CNN/AASIST are real, SSL/glottal are disabled, not dummy).

## 12. Fusion threshold

`fusion_decision_threshold = 0.5` (`Settings`, `.env` does not override it). Confirmed: `prediction = spoof if probabilities.spoof >= threshold else bonafide`. Same threshold is applied uniformly to the **fused** score; no per-branch threshold affects the fusion decision itself (each branch's own `prediction`/`confidence` field, computed via the same 0.5 midpoint independently in `real_prediction_from_spoof_probability`, is informational only and does not gate fusion). No evidence anywhere in settings, `.env`, `.env.example`, or code comments of this threshold ever having been calibrated (no EER-optimal threshold, no ROC-based selection). Per the standing instruction:

**"Current fusion threshold is 0.5 and appears uncalibrated."**

## 13. Unavailable branch behavior

Covered fully in §11/§16. Summary: exclusion + weight renormalization, never zero-substitution; a hard minimum-branch gate (currently 2) blocks single-branch verdicts; every exclusion is recorded with a machine-readable reason in `FusionResult.excluded_branches`.

## 14. Real-model loading verification

All three branches were loaded from their **actual configured checkpoint files** via `ModelFactory` (the same class production uses) in this audit, with no mocking:

```
CNN    load time:     346.7 ms   (checkpoint deserialise + strict load)
AASIST load time:      17.6 ms
SSL    load time:    2565.3 ms   (backbone construction + strict head restore; backbone
                                   weights were already cached locally, so this excludes
                                   any network download time)
```

Strict-load results: **zero missing/unexpected keys for all three**, matching each branch's own audit section above.

## 15. Three-branch inference results (real checkpoints, real fusion, no audio corpus available)

No labelled or unlabelled audio exists anywhere in this repository (`validation_data/real_world_sanity_set/samples/` is empty except its `.gitkeep`; no `.wav`/`.mp3`/`.flac` files exist anywhere outside `.venv`). Per the working rule against inventing labels, three **synthetic, unlabelled probe signals** (not speech, not known bonafide/spoof — explicitly not claimed as such) were used purely to prove the real pipeline executes end-to-end and to obtain genuine (not fabricated) numbers. All three real branches (CNN, AASIST, SSL forced to `real` for this test — see §6 for its live-disabled status) ran on each probe; fusion ran on the real outputs.

| Probe | CNN spoof | AASIST spoof | SSL spoof | CNN ms | AASIST ms | SSL ms | Fused spoof | Verdict |
|---|---|---|---|---|---|---|---|---|
| harmonic_speechlike (140Hz+280Hz tones + light noise, 3s) | 1.000000 | 0.987906 | 0.580555 | 157.4 | 56.5 | 7179.5 | 0.856154 | spoof |
| broadband_noise (Gaussian noise, 3s) | 1.000000 | 0.000143 | 0.606610 | 173.4 | 58.4 | 8235.7 | 0.535584 | spoof |
| pure_tone_440hz (single sine, 3s) | 0.0000001 | 0.999967 | 0.641682 | 166.0 | 85.3 | 8066.6 | 0.547217 | spoof |

Fusion weights for every probe (SSL forced on, glottal disabled): `cnn_acoustic=1/3, aasist=1/3, ssl_wavlm_xlsr=1/3`; `eligible_for_research_evaluation=false`; `research_blockers=["incomplete_branch_set", "unverified_branch_contributed"]` in every case (glottal never contributes, and none of the three branches carry a `research_result: true` attestation).

**Observation, not a claimed bug**: CNN saturates to machine-precision extremes on all three non-speech probes (exactly 1.0 or ~10⁻⁷); AASIST also swings to near-extremes but varies meaningfully between probes; SSL stays in a comparatively narrow 0.58–0.64 band across three very different signal types. None of this proves correctness or incorrectness for real speech — these are synthetic probes, not a labelled evaluation set (see §26 for the recommended follow-up: populate `validation_data/real_world_sanity_set/`).

**Current live deployment (unmodified `.env`, SSL genuinely disabled as configured)**, same harmonic probe:

```
cnn_acoustic : status=success, mode=real,     spoof=1.000000
aasist       : status=success, mode=real,     spoof=0.987906
ssl_wavlm_xlsr: status=skipped, mode=disabled, spoof=null   (excluded: branch_disabled)
glottal_features: status=skipped, mode=disabled, spoof=null (excluded: branch_disabled)

fusion: status=success, prediction=spoof, spoof=0.993953
        weights={cnn_acoustic: 0.5, aasist: 0.5}
        eligible_for_research_evaluation=false
        research_blockers=["incomplete_branch_set", "unverified_branch_contributed"]
```

This is what a real API request produces **today**, exactly as configured — not a hypothetical.

## 16. Fusion calculation examples (disagreement scenarios, run through the actual `FusionEngine.fuse()`)

All five scenarios below used `fusion_min_successful_branches=1` **except** where explicitly marked "(production default)", to isolate the weighted-average mechanics from the minimum-branch gate; the production-default behavior is shown separately since it materially changes the outcome for Scenario E.

| Scenario | CNN | AASIST | SSL | Fused spoof | Verdict | Weights used | Notes |
|---|---|---|---|---|---|---|---|
| A: CNN low, others high | 0.10 | 0.90 | 0.90 | **0.6333** | spoof | 1/3 each | Plain average: (0.10+0.90+0.90)/3 |
| B: CNN high, others low | 0.90 | 0.10 | 0.10 | **0.3667** | bonafide | 1/3 each | Mirror of A, confirms symmetry |
| C: all ≈0.50 | 0.500001 | 0.500001 | 0.500001 | **0.500001** | spoof (boundary) | 1/3 each | `>=` threshold, ties resolve to spoof |
| D: CNN unavailable | (failed) | 0.90 | 0.90 | **0.9000** | spoof | 0.5 / 0.5 | Weight cleanly renormalized, no bonafide pull |
| E: only CNN available | 0.90 | (failed) | (failed) | **0.9000** | spoof | 1.0 | Only reachable with `min_successful_branches=1` |
| **E (production default, min=2)** | 0.90 | (failed) | (failed) | **n/a — fusion FAILS** | **no verdict** | none | `status=failed`, warning "At least two successful branches are required for fusion." This is what actually happens in production if only CNN succeeds. |

**Cross-branch disagreement resolution**: confirmed to be a **pure fixed-weight linear average** — no confidence-weighted dominance, no majority-vote override (that logic exists in `FusionEngine` but is not the configured method), and no threshold-driven surprise behavior beyond the final `>=0.5` cutoff applied once to the already-averaged score.

## 17. Runtime dependency audit

```
torch          2.13.0     (installed, backend/.venv)
transformers   4.57.6     (installed)
torchaudio     NOT INSTALLED  — confirmed not required (spectral.py is deliberately
                                implemented directly on torch to avoid this dependency)
mamba_ssm      NOT INSTALLED  — confirmed not required; xlsr_mamba.py ships a pure-
                                PyTorch reimplementation of the same published selective-
                                scan algorithm specifically because mamba-ssm's CUDA-only
                                build backend cannot install on this (non-CUDA) machine
numpy          2.4.6
```

The environment SSL was originally validated with (`torch 2.9.0`, `torchaudio 2.9.0`, `mamba-ssm 2.3.1`, CUDA) is **not** what this backend runs — this backend runs a newer CPU-only torch build with no `mamba-ssm` at all, substituting the pure-PyTorch scan. This is a known, documented, deliberate divergence (see `xlsr_mamba.py`'s module docstring) — not an unnoticed incompatibility. **No conflicts found** between CNN/AASIST/SSL's actual runtime requirements on this machine; all three coexist in the same process (confirmed in §14/§15's combined runs).

## 18. Latency results

From §14/§15's real runs (CPU, sequential execution — confirmed by reading `voice_service.py`'s branch loop, which is a plain `for` loop, not `asyncio.gather`/`ThreadPoolExecutor(max_workers=N>1)` across branches):

```
Model load (once, at first use):
  CNN     347 ms
  AASIST   18 ms
  SSL    2565 ms   (backbone construction, weights already cached)

Per-audio inference (3s clip):
  CNN      157–173 ms
  AASIST    56–85 ms
  SSL     7179–8236 ms   <- dominant cost by ~40-140x
```

Because branches run **sequentially, not in parallel**, a request with SSL enabled would add roughly its full ~7–8s inference time to the total request latency (CNN+AASIST alone total well under 300ms). **This is the single biggest latency-relevant finding of this audit**: enabling SSL live, as currently coded, would take typical request latency from sub-second to ~7-8+ seconds, entirely because of sequential execution, not because SSL itself is unreasonably slow for a frozen 316M-parameter transformer on CPU. Parallelizing the branch loop (not attempted in this audit — read-only) would let CNN/AASIST/SSL overlap and bring wall-clock latency close to `max(branch latencies)` instead of their sum.

## 19. Memory / runtime findings

Measured via `ps -o rss` on the actual running process (not estimated from parameter counts alone, though those are also reported):

```
Parameter counts (real, from loaded modules):
  CNN                : 23,650 params        (~0.09 MB float32)
  AASIST              : 641,795 params       (~2.4 MB float32)
  SSL (trainable head): 1,173,634 params      (~4.5 MB float32)
  SSL (head + frozen XLS-R backbone, total): 316,612,354 params  (~1.2 GB float32, theoretical)

Measured process RSS (macOS, backend/.venv, cold process):
  after imports only                    : 79 MB
  after CNN + AASIST + SSL .load()      : 270 MB
  after CNN predict_safe()              : 267 MB
  after AASIST predict_safe()           : 274 MB
  after SSL predict_safe() (first call) : 535 MB
```

RSS after `.load()` alone (270 MB) is far below the SSL backbone's theoretical ~1.2 GB float32 footprint — consistent with `transformers`/`safetensors` lazily paging in weight pages on first actual use rather than fully materializing them at construction time; RSS roughly doubles (270→535 MB) only after the first real SSL forward pass. **All three branches coexist without any observed memory error on this machine** (16GB+ class development laptop, exact spec not probed by this audit). No GPU/VRAM measurement was possible or relevant — `MODEL_DEVICE_POLICY=cpu` throughout; no CUDA/MPS device is in use for any branch in the current configuration.

## 20. Model health findings

`ModelRegistry.health()`/`readiness()`, read directly and exercised live in this audit:

- Accurately distinguishes `mode: real` (CNN, AASIST) from `mode: disabled` (SSL, glottal) — no branch can report `ready: true` while silently running a dummy model instead of the configured real one; `is_loaded`/`lifecycle_state`/`adapter_type` are all independently checkable and consistent with the real load results in this audit.
- `research_ready` is correctly `false` for every branch today (verification flags all default `false`), so "healthy" never implies "research-validated."
- The disabled SSL/glottal branches still report `checkpoint_configured`/`checkpoint_valid`/`checkpoint_hash_short` for SSL (informational — the checkpoint IS valid, it's just not loaded because the branch is disabled) — this is accurate, not misleading, since `mode: disabled` and `is_loaded: false` are unambiguous alongside it.
- Overall `readiness.ready == true` today because both `required_model_branches` (`lfcc_cnn_tcn`, `aasist`) are `real` and loaded; SSL/glottal being disabled does not block readiness because neither is in the required list — this is working exactly as designed, confirmed live.

**No instance found of "healthy" being returned while a required branch silently used a dummy model** — the dummy-vs-real distinction is structural (different adapter classes entirely; see §22), not a runtime toggle that could silently misreport.

## 21. API findings

`VoicePredictionResponse`/`BranchPrediction`/`FusionResult` (`app/schemas/prediction.py`) correctly represent: final fused verdict + confidence (`fusion.prediction`/`fusion.confidence`), per-branch results including `status`/`mode`/`probabilities`/rich `metadata` (architecture, checkpoint hash, verification flags), branch availability (`status: skipped` + `excluded_branches` reason codes), and branch model identity (`model_name`, `metadata["architecture"]`, `metadata["checkpoint_metadata"]` where applicable) — without ever leaking a raw filesystem path. A model-level Pydantic validator (`validate_dummy_fusion_consistency`) enforces that a response containing any dummy branch cannot also claim `eligible_for_research_evaluation`. This is a well-designed, self-guarding contract; **no API redesign is warranted by this audit.**

One item carried forward from the prior `VOICE_BACKEND_FRONTEND_INTEGRATION_AND_LABEL_MAPPING_AUDIT_2026-08-19.md` (finding B1: the frontend reads a `preprocessing.decision_threshold` field that does not exist in this schema) was **not re-verified in this session** — it is a frontend-side consumption bug, out of this backend-focused audit's primary scope, and is called out here only so it isn't lost between audits.

## 22. Existing test-quality audit

| File | Tests | Real-checkpoint-gated | Uses mocks/monkeypatch | What it actually proves |
|---|---|---|---|---|
| `test_fusion.py` | 11 | 0 | 0 | Fusion **mechanics only** (synthetic `BranchPrediction` fixtures) — weighting, exclusion, thresholds, majority-vote/simple-average paths. Never touches a real checkpoint. |
| `test_real_model_adapters.py` | 26 | 35 markers | 0 | Real CNN/AASIST checkpoint loading, class-order mechanics, path-escape guards. |
| `test_real_model_architectures.py` | 14 | 9 markers | 0 | Real checkpoint ↔ reconstructed architecture strict-load parity (CNN, AASIST V1/V2). |
| `test_real_model_discrimination.py` | 5 | 10 markers | 0 | Real CNN/AASIST/SSL saturation and discrimination probes (see prior CNN report). |
| `test_ssl_sequence_integration.py` | 17 | 5 markers | 2 | Real SSL checkpoint load, preprocessing parity, one true e2e (real checkpoint + real backbone) inference test. |
| `test_ssl_generalization_v2_integration.py` (added this session's prior task) | 13 | multiple | 0 | Real Generalization V2 checkpoint identity, loader-format fallback, real e2e metadata check. |
| `test_aasist_light_v2_fusion_validation.py` | 24 | 2 markers | 0 | **Mostly fusion-mechanics tests using AASIST's public branch key/weight in synthetic scenarios** — not real three-branch fusion. |
| `test_aasist_light_v2_api_regression.py` | 8 | 3 markers | 6 | API-contract regression with a mix of real-checkpoint and mocked-dependency tests. |
| `test_cnn_verification_guard.py` | 13 | 14 markers | 2 | Real checkpoint resolution/identity pinning, label-mapping mechanical contract tests, the AASIST B5 characterization test. |
| `test_voice_service.py` | 13 | 0 | 2 | Orchestration contract (mocked registry/models) — not real inference. |
| `test_voice_routes.py` | 7 | 0 | 12 | API-layer contract (mocked service) — not real inference. |

**Key finding**: every fusion-specific test file (`test_fusion.py`, most of `test_aasist_light_v2_fusion_validation.py`) tests fusion **arithmetic and contract behavior with synthetic scores**, never with three genuinely-loaded real checkpoints running together. **No pre-existing automated test exercised CNN+AASIST+SSL together with real checkpoints through the real `FusionEngine`** before this audit — §15/§16 above are, as far as this audit found, the first time that specific combination has been run and recorded. This is a **test-coverage gap**, not evidence the system is broken (the manual run in this audit found no bug), but it does mean **passing tests alone were never sufficient evidence that real three-branch fusion works** — consistent with this audit's explicit instruction not to treat passing tests as proof.

## 23. Critical issues

**None found.** No wrong checkpoint is loaded, no label mapping is proven inverted, no dummy branch is used as if it were real, and the fusion arithmetic is correct and behaves safely when branches are unavailable.

## 24. High-priority issues

1. **CNN and AASIST feature pipelines and label mappings are declared, not verified**, against any training artifact (neither checkpoint shipped with a training notebook). This is long-standing and already tracked (`CNN_PREPROCESSING_VERIFIED=false`, etc.), re-confirmed still true.
2. **CNN vs AASIST class-order disagreement across probe signals** (documented in `.env`'s own comments: agree on 1/8 probes under the same assumed order, disagree on 7/8) remains unresolved — at least one of the two branches may be using the wrong class order. `scripts/validate_cnn_class_order.py` exists to help resolve this with real labelled data but has not been run against real labelled data (none exists in the repo).
3. **AASIST's runtime `spoof_index` is never cross-checked against the checkpoint's own attested `class_mapping`**, even though that attestation is loaded and validated moments earlier in the same load path. Currently both agree (`bonafide_spoof`), so there is no active incorrect behavior today — but nothing in the code would catch a future accidental `AASIST_CLASS_ORDER=spoof_bonafide` misconfiguration before it silently inverted every AASIST prediction. Already characterized by an existing test.
4. **Sequential (not parallel) branch execution** means enabling SSL adds its full ~7-8s inference latency on top of CNN+AASIST's sub-300ms total, rather than overlapping with it. Not a correctness bug, but a material latency risk if/when SSL is enabled live without also parallelizing branch execution.

## 25. Medium/low issues

**Medium:**
- No probability calibration exists anywhere in the pipeline (no temperature/Platt/isotonic scaling, no branch-specific calibrated threshold) — three branches' "0.90" scores are not proven comparable to each other. Explicitly not implemented in this audit, per instructions.
- `eligible_for_research_evaluation` can structurally never be `true` today, even with all three real branches succeeding, because Glottal (part of `FULL_SYSTEM_BRANCHES`) has no checkpoint and can never contribute — correctly reported, but worth knowing this is a ceiling, not a bug to fix by adjusting fusion.
- No automated test exercises real three-branch fusion together (§22) — a coverage gap this audit had to work around manually.
- Frontend finding B1 (reads a nonexistent `preprocessing.decision_threshold` field) carried forward from the prior audit, not re-verified this session.

**Low:**
- CNN checkpoint remains at `models/best_cnn_full_weighted.pth` rather than a `model_artifacts/cnn/` location consistent with AASIST's `model_artifacts/aasist/` and SSL's `model_artifacts/ssl/`. **Recommended, not performed** (explicitly out of scope for this audit).
- Old superseded checkpoints (`models/xlsr_mamba_asvspoof2019_best.pt`, `models/best_aasist_light_full_weighted.pth`) remain in place, correctly untouched, but add minor confusion for anyone browsing `models/` without reading this audit.

## 26. Recommended fixes (proposed only — not implemented in this audit)

1. Populate `validation_data/real_world_sanity_set/` with real labelled bonafide/spoof audio (minimum 5+5, ideally 10+10) and run `scripts/validate_cnn_class_order.py` against it to resolve the CNN/AASIST class-order disagreement with actual evidence, rather than the current probe-signal-only heuristic.
2. Add a runtime cross-check between AASIST's (and, if ever made self-describing, CNN's) configured class order and its checkpoint's own attested class mapping, failing loudly on mismatch instead of silently trusting the env setting.
3. Add an automated integration test that loads all three real checkpoints and asserts on the actual `FusionEngine.fuse()` output — closing the coverage gap identified in §22, ideally using the same probe methodology as §15/§16 once real audio exists.
4. If/when `SSL_MODEL_MODE` is flipped to `real`, parallelize branch execution (or at minimum measure and document the added end-to-end latency) so a ~7-8s SSL inference doesn't silently become the dominant cost of every prediction.
5. Consider (not urgent) moving the CNN checkpoint into `model_artifacts/cnn/` for artifact-location consistency with AASIST/SSL — purely organizational.
6. Decide, as a separate explicit task (already flagged in the prior SSL integration report), whether/when to set `SSL_MODEL_MODE=real` in production — this audit found no technical blocker, only the standing verification/latency considerations above.

## 27. Current readiness classification

**PARTIALLY READY**

Justification against the instruction ("Do not return READY unless all three intended real models genuinely load and the real three-branch fusion output has been verified"):

- All three intended real models **do** genuinely load and **were** verified to run together through the real fusion function in this audit (§14–§16) — so this is not `NOT READY`.
- But the **live, currently-deployed configuration only runs two of the three** (SSL is deliberately disabled today), feature pipelines/label mappings for all three remain unverified against any training artifact, one latent class-order cross-check gap exists (AASIST), and no pre-existing automated test covered the three-branch-together case before this audit — collectively too much outstanding verification/completeness work to call this `READY` or even `READY WITH FIXES` (there is no single fix; it's a verification and enablement-decision backlog, most of it already known and tracked).

---

## A. CNN checkpoint actually used
`models/best_cnn_full_weighted.pth` (resolved absolute path confirmed; SHA-256 `4bf86dd15108a8cff43569ef907a9f3852978353af6d6e184001480a0fa2e6f8`)

## B. AASIST checkpoint actually used
`model_artifacts/aasist/aasist_light_v2_best.pt` (SHA-256 `c944c69f05a0135ad9dfdaf86fb8a31815339f862d780c5f2c5d834f181e64ec`)

## C. SSL checkpoint actually used
`model_artifacts/ssl/ssl_xlsr_mamba_generalization_v2_best.pt` (SHA-256 `336e65d36764315ed4fe036cc99d3b957daae5e07a5ec462f8b09faa35291afe`) — **correctly configured as the default, but the branch itself is disabled in the live `.env`.**

## D. Branch status
```
CNN    = REAL (live, active in production fusion today)
AASIST = REAL (live, active in production fusion today)
SSL    = REAL-CAPABLE, verified fully functional in this audit, but DISABLED in the live .env (not currently contributing)
```

## E. Label semantics
```
CNN spoof index    = 1 (declared, unverified against training)
AASIST spoof index = 1 (declared AND checkpoint-attested to agree; runtime never cross-checks the two against each other)
SSL spoof index    = 1 (declared, unverified against training)
```

## F. Scores passed to fusion (harmonic probe, real run)
```
CNN    = 1.000000
AASIST = 0.987906
SSL    = 0.580555   (not live today — forced on for this audit's verification run)
```

## G. Fusion formula
```
effective_weight[b] = configured_weight[b] / sum(configured_weight[b'] for b' in successful_branches)
final_spoof = sum(branch.spoof_probability * effective_weight[branch] for branch in successful_branches)
prediction  = spoof if final_spoof >= fusion_decision_threshold else bonafide
```
Plain weighted average over successful branches only; not logit fusion, not confidence-weighted, not majority vote (in the configured mode).

## H. Fusion weights
```
CNN     = 0.25 (configured) -> renormalizes to 0.5 today (SSL/glottal excluded) or 1/3 with all three active
AASIST  = 0.25 (configured) -> same renormalization
SSL     = 0.25 (configured) -> currently excluded entirely (branch disabled)
Glottal = 0.25 (configured) -> currently excluded entirely (no checkpoint)
```

## I. Fusion threshold
`0.5`, uncalibrated (no evidence of EER/ROC-based calibration anywhere in the codebase).

## J. Real three-branch fusion verified
**YES** — real checkpoints, real `FusionEngine.fuse()`, no mocks, results recorded in §15/§16. (Not the same as saying it's live in production today — see D/§7.)

## K. Dummy fallback involved
**NO** — no dummy branch was used anywhere in this audit's real-model runs, and the codebase has no mechanism to silently substitute a dummy model for a real one that fails to load (real-mode failures fail closed to `BranchStatus.failed`, never to a dummy adapter).

## L. Top critical problems
**None at CRITICAL severity.** Top HIGH-severity items: (1) CNN/AASIST feature pipeline and label mapping remain unverified against training; (2) CNN and AASIST class-order assumptions disagree on 7/8 synthetic probes, meaning one of them is probably wrong; (3) AASIST's env-configured spoof index is never cross-checked against its own checkpoint's attested mapping; (4) sequential branch execution means enabling SSL live will add ~7-8s to every request unless parallelized.

## M. Current readiness
**PARTIALLY READY**

## N. Exact next actions
1. Populate `validation_data/real_world_sanity_set/` with real labelled bonafide/spoof audio and run `scripts/validate_cnn_class_order.py` to resolve the CNN/AASIST class-order disagreement with evidence.
2. Add a runtime assertion cross-checking AASIST's configured class order against its checkpoint's own attested `class_mapping`, failing loudly on mismatch.
3. Decide explicitly whether/when to enable `SSL_MODEL_MODE=real` in production, and if so, parallelize branch execution first so SSL's ~7-8s inference doesn't dominate every request's latency.
