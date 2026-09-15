# MULTI-SCOPE Voice Pipeline Architecture Audit

**Date:** 2026-08-20
**Scope:** Audit the current backend voice-classification pipeline against the final research architecture (CNN + AASIST + SSL/XLS-R+Mamba, score-level simple-average fusion, threshold `0.5519237850482265`, Glottal as auxiliary/XAI-only evidence). **Audit only — no files were modified, no code was written.** Branch: `feat/aasist-light-v2-integration`.

Findings are based on direct inspection of the backend source tree and the live `backend/.env` at the time of this report, cross-checked against three internal audit reports already present in the repository (`VOICE_THREE_BRANCH_FUSION_FULL_AUDIT_2026-08-19.md`, `SSL_XLSR_MAMBA_GENERALIZATION_V2_INTEGRATION_REPORT_2026-08-19.md`, `docs/SSL_CLASS_MAPPING_AND_INFERENCE_AUDIT_2026-08-19.md`). No implementation is inferred beyond what the cited files contain.

---

## 1. Executive summary

The backend runs a real, working three-branch pipeline today — CNN, AASIST, and SSL/XLS-R+Mamba all load genuine checkpoints and genuinely execute (confirmed directly: `backend/.env` currently sets `CNN_MODEL_MODE=real`, `AASIST_MODEL_MODE=real`, `SSL_MODEL_MODE=real`). Fusion is a score-level average over whichever branches succeed, computed in `app/utils/fusion.py:236-251`. Glottal has no checkpoint anywhere in the repository and is disabled (`GLOTTAL_MODEL_MODE=disabled`, `GLOTTAL_MODEL_PATH=` empty) — it already does not affect the primary score, which happens to match the target architecture's intent, though not for a designed reason.

Three concrete points block a clean match to the final research architecture:

1. **Decision threshold is wrong.** `Settings.fusion_decision_threshold` defaults to `0.5` (`app/config/settings.py:291`) and nothing in `backend/.env` overrides it. The required `0.5519237850482265` is not present anywhere in the codebase.
2. **CNN checkpoint and feature pipeline don't match the target contract.** The live checkpoint is `models/best_cnn_full_weighted.pth`, not `cnn_v2_lfcc_delta_aug_asvspoof2019_inference_best.pt` (which does not exist anywhere in this repository). The live feature pipeline is **log-mel, 20 filters, no deltas** (`CNN_FEATURE_TYPE=log_mel`, `backend/.env`), not the target's LFCC-40 + delta + delta2.
3. **SSL checkpoint is a different, later artifact than the one named in the target.** The live checkpoint is `model_artifacts/ssl/ssl_xlsr_mamba_generalization_v2_best.pt`, a "Generalization V2" fine-tune whose own metadata records `ssl_asvspoof5_best.pt` as its Colab-side *parent* checkpoint (`source_checkpoint` key), not itself. Whether the target intends the parent or this descendant is a research decision this audit cannot make.

Fusion mechanics, once the threshold is corrected, are close to the target: with all three branches equally weighted at `0.25` each (`app/config/settings.py:293-296`) and Glottal permanently excluded (disabled, no checkpoint), the weighted-average renormalizes to exactly `1/3, 1/3, 1/3` across CNN/AASIST/SSL — numerically identical to the requested simple average, even though the code path is `weighted_average`, not the separate `simple_average` method that already exists unused in the same file (`app/utils/fusion.py:253-260`).

XAI's semantic (acoustic/SHAP) explanation layer is running in **mock mode** today (`XAI_MODE=mock`, `backend/.env`) — it does not consume real branch scores. Temporal (SSL attention) XAI is real and independent of that setting.

**Overall readiness: Partially Ready.** The three-branch real pipeline exists and runs end-to-end; what's missing is a threshold correction, a decision on the CNN/SSL checkpoint identity question, and either code changes or an explicit accept of the current numeric fusion result — no architectural rebuild is required.

---

## 2. Current voice API flow

Two prediction routes exist. Only one is live traffic.

```
POST /api/v1/predictions  (primary, authenticated)
  -> app/api/v1/prediction_routes.py
  -> PredictionSubmissionService.submit()             app/services/prediction_submission_service.py:77
  -> PredictionSubmissionService.execute_prediction()  app/services/prediction_submission_service.py:130
       -> save_validated_audio_upload_blocking()        app/ingestion/audio.py:295   (validate + save)
       -> storage.upload_audio()                        app/storage/cloudinary_storage.py
       -> VoiceService._predict_from_validated_upload() app/services/voice_service.py:110
            -> preprocess_audio_file()                  app/ingestion/audio.py:450  (decode/resample/mono/normalise)
            -> for model in ModelRegistry.models:        app/services/voice_service.py:157  (SEQUENTIAL loop)
                 -> model.predict_safe(processed_audio)  app/models/real/base.py
            -> FusionEngine.fuse(branch_predictions)     app/utils/fusion.py:104
       -> VoicePredictionResponse                        app/schemas/prediction.py:130
       -> persistence.save_result() (MongoDB)             app/services/prediction_persistence_service.py
       -> xai_orchestrator.enqueue() (best-effort, async) app/voice_xai/orchestrator.py
  -> PredictionSubmissionResponse                        app/schemas/prediction_submission.py:10

POST /api/v1/voice/predict  (deprecated compat route)
  -> app/api/v1/voice_routes.py:89     disabled unless ENABLE_LEGACY_ANONYMOUS_PREDICTION=true; never in prod

GET  /api/v1/voice/models/health
  -> app/api/v1/voice_routes.py:127  -> VoiceService.model_health()
```

**Dead code found:** `backend/app/routes/voice_routes.py` is not imported anywhere in `app/main.py` (confirmed by repository-wide search) — it is an orphaned duplicate of `app/api/v1/voice_routes.py` that no request ever reaches.

Model dispatch is entirely settings-driven: `app/models/factory.py:55-68` builds one adapter per canonical branch (`lfcc_cnn_tcn`, `aasist`, `ssl_sequence`, `glottal`) from `app/config/settings.py`'s `*_model_mode` fields, choosing a real / dummy / disabled adapter class per branch independently. `ModelRegistry` (`app/models/registry.py:16`) holds the ordered list the request loop above iterates.

---

## 3. CNN branch audit

| Contract item | Target | Current (live) | Status |
|---|---|---|---|
| Checkpoint file | `cnn_v2_lfcc_delta_aug_asvspoof2019_inference_best.pt` | `models/best_cnn_full_weighted.pth` | **Mismatch** |
| Checkpoint exists in repo? | — | Target filename: **not found anywhere** in the repository | **Missing** |
| Sample rate | 16 kHz | 16 kHz (`TARGET_SAMPLE_RATE=16000`) | Match |
| Duration / samples | 4 s / 64,000 samples | Not an explicit raw-sample crop; feature budget is `max_frames=400 × hop_length=160 = 64,000 samples` (≈4.0 s) in the frame domain | Different mechanism, same nominal length |
| Feature type | LFCC, 40 coefficients | `log_mel`, 20 filters (`CNN_FEATURE_TYPE=log_mel`, `CNN_FEATURE_FILTERS=20`, `backend/.env`) | **Mismatch** |
| Delta / delta2 | LFCC + Δ + ΔΔ | `CNN_FEATURE_INCLUDE_DELTAS` not set → defaults `False` (`app/config/settings.py:247`) | **Mismatch** |
| Normalization | — | `global_zscore` (`CNN_FEATURE_NORMALIZATION`) | declared, unverified against training |
| Class mapping | bonafide=0, spoof=1 | `CNN_CLASS_ORDER=bonafide_spoof` → spoof index 1 | Declared match |
| Spoof probability | `softmax(logits)[:,1]` | `_spoof_probability_from_logits`: `softmax(logits, dim=-1)[spoof_index]`, `spoof_index=1` | Match |
| Error handling | — | Strict state-dict load (`app/models/torch_support.py`); mismatch → `CheckpointCompatibilityError` → fails closed, never falls back to dummy | Sound |

Architecture: `app/models/architectures/cnn.py`'s `build_cnn_acoustic_net()`, three Conv2d/BatchNorm2d/ReLU/MaxPool2d blocks + AdaptiveAvgPool2d + `Linear(64, 2)`, reconstructed directly from the checkpoint's own tensor shapes (no training notebook shipped). Feature config is built by `app/config/settings.py:570-586` (`cnn_feature_config`) and consumed by `app/models/preprocessing/spectral.py`.

The current `log_mel`/20-filter/no-delta choice is explicitly **not** a training-verified value — it's documented in `spectral.py:1-30` and `settings.py:225-238` as the best of 72 swept configurations against the checkpoint's own first-BatchNorm statistics, chosen because the actual training feature pipeline was never shipped with this checkpoint. `CNN_PREPROCESSING_VERIFIED=false` and `CNN_CLASS_MAPPING_VERIFIED=false` in the live `.env` reflect this directly. Whether this checkpoint even accepts LFCC-40+Δ+ΔΔ input is untested — the network's global-average-pool head is shape-agnostic and will return a confident, meaningless answer on the wrong front end rather than an error.

**Net effect:** the live CNN branch matches neither the target checkpoint file nor the target feature contract. It is a real, working, but architecturally different CNN branch from the one specified.

---

## 4. AASIST branch audit

| Contract item | Target | Current (live) | Status |
|---|---|---|---|
| Checkpoint file | `aasist_light_v2_best.pt` | `model_artifacts/aasist/aasist_light_v2_best.pt` | Match |
| Sample rate | 16 kHz | 16 kHz, enforced in `AasistLightV2WaveformConfig.__post_init__` (`app/models/preprocessing/aasist_light_v2.py:27-29`) | Match |
| Samples | 64,600 | 64,600, enforced identically (`aasist_light_v2.py:15,30-31`) | Match |
| Mono | required | guaranteed upstream by shared decoder | Match |
| Crop/pad policy | crop/pad to 64,600 (unspecified direction) | **Leading crop**, right-zero-pad (keeps clip start), hardcoded in `aasist_light_v2.py`, provenance key `"first_crop_right_zero_pad"` | Declared, direction unspecified in target |
| Normalization | per-waveform z-score | per-waveform z-score, `(x - mean) / (std + 1e-6)`, silence guard at `std < 1e-7` | Match |
| Class mapping | bonafide=0, spoof=1 | `AASIST_CLASS_ORDER=bonafide_spoof` AND checkpoint self-attests `class_mapping={"bonafide":0,"spoof":1}` at load time | Match (best-evidenced of the three branches) |
| Spoof probability | `softmax(logits)[:,1]` | identical shared formula to CNN/SSL | Match |

The checkpoint loader (`app/models/real/inference.py`, `load_aasist_light_v2_checkpoint_strict`) is the most rigorously self-validating of the three: it asserts epoch `13`, the exact `class_mapping`, `best_eer ≈ 0.03925` to 1e-12 tolerance, and a `641,795` parameter count against a companion `aasist_light_v2_final_summary.json`.

**Dead configuration found:** `backend/.env` sets `AASIST_TARGET_SAMPLES=64000` (not 64,600), `AASIST_WAVEFORM_NORMALIZATION=peak`, and `AASIST_LENGTH_POLICY=center_crop_pad` — but `app/models/real/inference.py:124` (`build_aasist_loader`) constructs `AasistLightV2WaveformConfig()` with no arguments, so **none of these three env vars reach the live AASIST branch at all**. The branch's real behavior (64,600 samples, z-score, leading-crop) is hardcoded in `aasist_light_v2.py` and happens to be correct; the env vars are orphaned and would mislead an operator who edits them expecting an effect. (These settings still drive the separate, unused legacy `aasist_waveform_config` / generic `preprocessing/waveform.py` path.)

Known latent gap, not active today: `AASIST_CLASS_ORDER` is read from the env setting alone and is never cross-checked against the checkpoint's own attested `class_mapping`, even though that attestation is loaded and validated moments earlier in the same load function. Both currently agree, but nothing would catch a future misconfiguration.

---

## 5. SSL branch audit

| Contract item | Target | Current (live) | Status |
|---|---|---|---|
| Checkpoint file | `ssl_asvspoof5_best.pt` | `model_artifacts/ssl/ssl_xlsr_mamba_generalization_v2_best.pt` | **Different artifact** |
| Relationship | — | Live checkpoint's own `source_checkpoint` metadata field names a Colab path ending `.../checkpoints/ssl_asvspoof5_best.pt` as its training parent (epoch 2, stage `Generalization_V2`, fine-tuned further from that file) | Live file is a **descendant**, not the named file itself |
| Backbone | `facebook/wav2vec2-large-xlsr-53` | same, via `SSL_XLSR_MODEL_NAME` (`backend/.env`) | Match |
| Backbone frozen | required | `requires_grad_(False)` on every backbone param, `.eval()` | Match |
| Classifier head | Mamba | 2× pure-PyTorch Mamba reimplementation (`d_state=16, d_conv=4, expand=2`) — `mamba-ssm` is CUDA-only and cannot install on this machine | Mathematically equivalent, not the original kernel |
| Sample rate | 16 kHz | 16 kHz | Match |
| Max duration / samples | 6 s / 96,000 | 6 s / 96,000 (`SSL_TARGET_SAMPLES=96000`) | Match |
| Crop policy | deterministic center crop | deterministic center crop, `start=(len-target)//2` — fixed in the prior integration task (was leading-crop before) | Match |
| Short-clip padding | zero-padding | zero-pad at end | Match |
| Normalization | valid-audio mean subtraction + peak normalization | `(x - mean) / max(|x - mean|)`, guarded at `1e-7` | Match |
| Padded region stays zero | required | confirmed — normalization computed pre-pad, pad applied after | Match |
| Attention mask | required, real/padded aware | built from real/padded sample counts, never the `input_values != 0` anti-pattern | Match |
| HF feature-extractor renormalization | must NOT be used | **Not used.** `Wav2Vec2FeatureExtractor(do_normalize=True)` defaults to zero-mean/*unit-variance*, which would silently contradict the required zero-mean/*peak* formula — a hand-written front end (`app/models/preprocessing/ssl_waveform.py`) implements the exact required formula instead | Correctly avoided |
| Class mapping | bonafide=0, spoof=1 | `SSL_CLASS_ORDER=bonafide_spoof`, declared — **the checkpoint carries zero label-map metadata of its own** | Declared, not attested |

Live mode: `SSL_MODEL_MODE=real` in `backend/.env` right now. **This drifted since the prior integration task**, which deliberately left it `disabled`; a subsequent, separately-documented audit (`backend/docs/SSL_CLASS_MAPPING_AND_INFERENCE_AUDIT_2026-08-19.md`) found it already flipped to `real` and treated that as current state. SSL is genuinely contributing to every production fusion result today, weighted equally with CNN/AASIST.

An investigation triggered by a known ElevenLabs sample producing CNN≈99.5%/AASIST≈100% spoof but SSL≈2.2% spoof traced the entire logit→softmax→API→frontend path and found it mechanically clean (no swap, no complement, no double-softmax). The checkpoint's own reported ASVspoof5 TEST EER is `0.1856` and its WaveFake unseen-generator spoof-detection rate is `90.3%` (not 100%) — the disagreement is most consistent with a genuine generalization gap on a commercial TTS system outside the training corpora, not a mapping bug, at moderate-high (not certain) confidence. No labelled audio exists anywhere in the repository to raise this to certainty.

---

## 6. Glottal branch audit

**Expected artifacts:** `glottal_logreg_selected20_v1.joblib`, `glottal_selected_features_v1.json` — **neither exists**. A repository-wide search for `*glottal*` outside `.venv`/build returns only source code, no artifact files.

**Live configuration:** `GLOTTAL_MODEL_MODE=disabled`, `GLOTTAL_MODEL_PATH=` empty. Not a required branch — `REQUIRED_MODEL_BRANCHES=lfcc_cnn_tcn,aasist` excludes it.

`app/models/real/glottal.py` defines `GlottalRealAdapter` as a bare pass-through subclass of `RealModelAdapter` with no glottal-specific logic at all (no joblib load, no logreg inference, no feature-selection code). `app/models/factory.py:244-245,257-259` (`_default_real_loader`/`_default_real_predictor`) return `None` for the `glottal` branch by design — if `GLOTTAL_MODEL_MODE` were ever set to `real` today, the branch would immediately fail with `error_code="model_not_implemented"` (`app/models/real/base.py:71,87`), not run a scikit-learn logistic-regression model.

| Question | Answer |
|---|---|
| Required for the final classification decision today? | No — disabled, never enters `successful_branches` |
| Used in the primary fusion arithmetic? | No — `FusionEngine._weighted_average` only sums over branches with `status == success`; a disabled branch is excluded entirely from both numerator and denominator |
| Used in XAI? | Structurally referenced, not functionally used — no acoustic/glottal evidence is produced anywhere since no glottal model runs |
| Can it be made auxiliary without breaking the current API? | Partially already true, one structural blocker remains — see below |

Glottal's absence already does *not* affect the primary spoof-probability number. But it is still hard-wired into `FULL_SYSTEM_BRANCHES` in `app/utils/fusion.py:49-54`, which every fusion result checks in `_research_blockers()` (`fusion.py:226-227`) to decide `eligible_for_research_evaluation`. Because Glottal can never contribute, `eligible_for_research_evaluation` is structurally pinned to `false` forever, regardless of how good CNN+AASIST+SSL's real, verified output eventually becomes. Reclassifying Glottal as auxiliary-only (per the target architecture) would require removing it from `FULL_SYSTEM_BRANCHES` — a small, well-isolated code change, not attempted in this audit-only pass.

---

## 7. Fusion logic audit

Single implementation: `app/utils/fusion.py`, class `FusionEngine`. No other fusion path exists (no logistic regression, no XGBoost, no ensemble stacking anywhere in the codebase).

| Parameter | Current (live) | Source |
|---|---|---|
| Method | `weighted_average` (`simple_average` and `majority_vote` exist in the same class but are not selected) | `settings.py:290`, not overridden in `.env` |
| Weights | CNN `0.25`, AASIST `0.25`, SSL `0.25`, Glottal `0.25` — renormalized over *successful* branches only | `settings.py:293-296` |
| Effective weights today | CNN/AASIST/SSL all real and succeeding, Glottal disabled → renormalizes to **1/3, 1/3, 1/3** (numerically = target's simple average) | `_normalize_weights`, `fusion.py:334-341` |
| Decision threshold | `0.5` | `settings.py:291`, not overridden |
| Required threshold | `0.5519237850482265` | **Not present anywhere in the codebase** |
| Minimum successful branches | `2` of the 4 canonical branches | `settings.py:292` |
| Fallback if a branch fails | Excluded from both sum and weight denominator; remaining branches renormalize automatically. Never zero-substituted, never biases toward either class by construction. | `fusion.py:236-251, 285-302` |
| Fallback if fewer than 2 branches succeed | Fusion returns `status=failed`, `prediction=null`, explicit warning — no single-branch verdict ever reaches the API as a confident result | `fusion.py:138-151` |

Glottal in the primary fusion formula: **not present** — confirmed by construction (excluded whenever disabled/failed, per branch above). Glottal *is* present in the separate research-eligibility gate (`fusion.py:49-54, 226-227`), which is metadata, not the spoof-probability arithmetic.

**Threshold gap is the single largest deviation from the target spec.** `0.5` vs. `0.5519237850482265` is not a rounding difference — it will flip the verdict for every sample whose fused spoof probability falls in `[0.5, 0.5519237850482265)`, moving them from "spoof" to "bonafide" under the target's calibrated threshold.

One structural difference from the literal target formula: the target's `(cnn + aasist + ssl) / 3` assumes exactly three branches are always present with no defined fallback. The live system instead tolerates a 2-of-3 failure (renormalizing to a 2-way average) rather than failing outright. This is arguably safer, but it is a behavior the target spec doesn't define either way and is worth an explicit decision.

---

## 8. XAI integration audit

XAI schema: `app/schemas/xai.py`, classes `ClassifierSnapshot` / `ClassifierBranchSnapshot`. The handoff from classifier to XAI is a frozen contract, `ClassifierInferenceBundle` (`app/voice_xai/contracts.py:29`), carrying the full `VoicePredictionResponse` plus the exact `ProcessedAudio` already decoded for classification (no re-decode).

**Requested shape:** `prediction`, `spoof_probability`, `threshold` (`0.5519...`), `branch_scores: {cnn, aasist, ssl}`, `auxiliary_evidence: {glottal_spoof_probability, used_for_primary_decision}`.

**Actual `ClassifierSnapshot`:** `verdict` (matches "prediction"), `spoof_probability` (present, matches), `decision_threshold` (present, currently `0.5`), `branches[]` (flat list incl. Glottal (skipped) — not split into an "auxiliary_evidence" group), `contains_dummy_branches` (present, not requested but useful).

**Mismatches only** (per the audit instruction not to report matches):

- No `auxiliary_evidence` grouping exists. Glottal's `ClassifierBranchSnapshot` (status `skipped`, `spoof_probability=None`) sits in the same flat `branches` list as CNN/AASIST/SSL — nothing marks it as auxiliary or as excluded-from-primary-decision.
- No `used_for_primary_decision` field exists anywhere in the XAI or prediction schemas.
- `decision_threshold` will carry whatever `fusion_decision_threshold` resolves to at request time — currently `0.5`, propagating the same threshold gap from §7 into every XAI report.

**Semantic (acoustic/SHAP) XAI is running in mock mode.** Live `.env`: `XAI_ENABLED=true`, `XAI_MODE=mock`. `app/main.py:173-189` only constructs the real `ProductionSemanticExplanationService` (loading `esvas-acoustic-28-v1.json` + a SHAP background + a semantic model) when `XAI_MODE=real` — and `Settings.xai_semantic_model_path`/`xai_semantic_shap_background_path` default to empty strings (`settings.py:130,134`), unset in `.env`. Under `mock`, acoustic/glottal-adjacent explanation evidence returned to callers today is a deterministic placeholder, not derived from real branch scores.

Temporal XAI (SSL attention rollout) is the one component confirmed real regardless of `XAI_MODE` (per commit `263ee54`, "Temporal XAI now always uses the real XLS-R attention-rollout service") — it calls back into the already-loaded real SSL branch via `VoiceService.xai_ssl_attention_windows` (`voice_service.py:270-284`) whenever the SSL branch is genuinely in `real` mode.

---

## 9. Current response contract

Returned by `POST /api/v1/predictions`: `PredictionSubmissionResponse` (`app/schemas/prediction_submission.py:10-28`), wrapping the internal `VoicePredictionResponse` (`app/schemas/prediction.py:130`).

| Field | Type / source | Status |
|---|---|---|
| `prediction_id`, `request_id`, `status`, `source_type`, `created_at` | submission bookkeeping | present |
| `audio` | `PredictionHistoryAudioMetadata` | present |
| `branches[]` | `BranchPrediction` — `model_name`, `status`, `mode`, `prediction`, `confidence`, `probabilities.{bonafide,spoof}`, `metadata` | present, matches "branch_scores" intent |
| `fusion` | `FusionResult` — `prediction`, `confidence`, `probabilities`, `method`, `branch_weights`, `contributing_branches`, `excluded_branches`, `eligible_for_research_evaluation` | present, matches "spoof_probability"/"prediction" intent |
| `research_eligible` | bool, mirrors `fusion.eligible_for_research_evaluation` | present |
| Decision threshold | — | **Missing from this response entirely** — only exists inside `VoicePredictionResponse.provenance.fusion.threshold`, and `PredictionSubmissionResponse` has no `provenance` field at all |
| Glottal / auxiliary evidence marker | — | **Missing** — Glottal appears only as one more `branches[]` entry with `status=skipped` |

A prior audit's finding, carried forward unverified in that report and re-confirmed structurally true here: the frontend is documented to read `preprocessing.decision_threshold`, a field that does not exist anywhere in `PredictionSubmissionResponse`. Whatever UI depends on the numeric threshold today cannot be getting it from this response.

---

## 10. Model artifact inventory

| Branch | File | Loaded by | Exists? | Env key | Status |
|---|---|---|---|---|---|
| CNN | `models/best_cnn_full_weighted.pth` | `app/models/real/inference.py` (`build_cnn_loader`) | Yes | `CNN_MODEL_PATH` | Current, wrong file for target |
| AASIST | `model_artifacts/aasist/aasist_light_v2_best.pt` + `aasist_light_v2_final_summary.json` | `app/models/real/inference.py` (`build_aasist_loader`) | Yes | `AASIST_MODEL_PATH` | Current, matches target |
| SSL | `model_artifacts/ssl/ssl_xlsr_mamba_generalization_v2_best.pt` | `app/models/real/ssl_sequence_inference.py` (`build_ssl_loader`) | Yes | `SSL_MODEL_PATH` | Current, different artifact than target name |
| SSL backbone | `facebook/wav2vec2-large-xlsr-53` | Hugging Face (`AutoModel.from_pretrained`), cached under `~/.cache/huggingface` | Yes (cached locally) | `SSL_XLSR_MODEL_NAME` | Matches target |
| Glottal | `glottal_logreg_selected20_v1.joblib`, `glottal_selected_features_v1.json` | n/a — no loader wired (`_default_real_loader` returns `None`) | **No** | `GLOTTAL_MODEL_PATH` (empty) | Missing entirely |
| CNN/AASIST (superseded) | `models/xlsr_mamba_asvspoof2019_best.pt`, `models/best_aasist_light_full_weighted.pth` | Not referenced by any live default path | Yes, unused | — | Obsolete, kept for lineage/legacy tests |
| XAI semantic | `app/voice_xai/semantic/manifests/esvas-acoustic-28-v1.json` | `ProductionSemanticExplanationService` (only if `XAI_MODE=real`) | Yes | `xai_semantic_manifest_path` | present but unused (`XAI_MODE=mock` live) |
| XAI semantic model / SHAP background | — | same | **No** — paths default to `""`, unset in `.env` | `xai_semantic_model_path`, `xai_semantic_shap_background_path` | not configured |
| XAI temporal calibration | `app/voice_xai/temporal/calibrations/partialspoof_v1_2_dev_attention_threshold.json` | `XLSRTemporalAttentionService` | Yes | `xai_temporal_threshold_config_path` | in use |

Two model directories exist at the repository root: `models/` (legacy, flat) and `model_artifacts/<branch>/` (current, structured). CNN is the one branch still resolving from the legacy `models/` directory rather than a `model_artifacts/cnn/` location consistent with AASIST/SSL.

---

## 11. Latency architecture

| Aspect | Finding |
|---|---|
| Branch execution | **Sequential**, not parallel — a plain `for model in self._model_registry.models` loop (`voice_service.py:157`). Each branch does get its own single-worker `ThreadPoolExecutor`, but only to enforce a per-branch timeout, not to overlap branches. |
| CPU/GPU | `MODEL_DEVICE_POLICY=cpu`, `MODEL_DEFAULT_DEVICE=cpu` throughout. No CUDA/MPS in the live path. |
| Model init strategy | `MODEL_LOAD_STRATEGY=startup` (live `.env`) — real branches load once at process start via `ModelRegistry.load_startup_models()` (`registry.py:38-44`), not per request. |
| Reload per request? | No — loaded models are cached on the adapter instance for the process lifetime. |
| Caching | No prediction-result cache; every request re-runs inference. HF backbone weights are cached on disk (`~/.cache/huggingface`) across process restarts. |
| Queue / background jobs | `InlinePredictionJobRunner` (`app/services/prediction_job_runner.py`) bounds concurrency via `MAX_CONCURRENT_PREDICTIONS=2`; XAI runs on its own separate async queue (`AsynchronousExplanationQueue`), decoupled from the prediction response. |
| ffmpeg/audio conversion | One ffprobe inspect + one ffmpeg decode per upload (`app/ingestion/audio.py`), each with its own timeout (`FFMPEG_TIMEOUT_SECONDS=30`, `FFPROBE_TIMEOUT_SECONDS=15`). |
| XAI latency | Fully decoupled — enqueued best-effort after the prediction response is already persisted (`prediction_submission_service.py:267-275`); never adds to request latency. |

**Dominant bottleneck: SSL inference on CPU, run sequentially.** Independently measured in the prior full-pipeline audit: CNN ≈160–175ms, AASIST ≈55–85ms, SSL **≈7,200–8,200ms** per 3-second clip — SSL alone is 40–140x either other branch. Because branches run sequentially and SSL is now live (`SSL_MODEL_MODE=real`), every production request today likely takes on the order of several seconds, dominated almost entirely by SSL. Parallelizing the branch loop (not attempted here) would bring wall-clock latency down toward `max(branch latencies)` instead of their sum.

---

## 12. Failure handling

| Scenario | Behavior | Degrades gracefully? |
|---|---|---|
| CNN fails | Branch marked `failed`, excluded from fusion; AASIST+SSL renormalize to 0.5/0.5 and still produce a verdict (2 ≥ min_successful_branches) | Yes |
| AASIST fails | Symmetric to CNN — CNN+SSL renormalize | Yes |
| SSL fails | Symmetric — CNN+AASIST renormalize to 0.5/0.5 (this was the standing live behavior before SSL was flipped to real) | Yes |
| Two branches fail | Fewer than `fusion_min_successful_branches=2` succeed → `FusionResult.status=failed`, no verdict at all → `NoUsableModelBranchesError` raised in `prediction_submission_service.py:253-259` | Fails closed, no silent wrong answer |
| Glottal fails / absent | Already excluded (disabled) — no effect on primary decision, only pins `eligible_for_research_evaluation=false` | Yes, by design |
| XAI fails | Caught and logged in `_enqueue_xai_without_affecting_prediction` (`prediction_submission_service.py:357-395`) — explicitly documented as "post-persistence best effort," never affects the already-saved prediction | Yes, fully isolated |
| Invalid / corrupted audio | `CorruptedAudioError` raised during `inspect_audio_file`/`decode_audio_file` (`app/ingestion/audio.py`) before any model runs | Fails fast with a typed error |
| Unsupported format | `UnsupportedAudioFormatError` (`audio.py:465`), mapped to HTTP 415 | Fails fast, typed |
| Very short audio | `AudioTooShortError` against `min_audio_duration_seconds` (default 1.0s); AASIST additionally has its own silence guard (`std < 1e-7`) independent of duration | Fails fast, typed |
| GPU unavailable | Not applicable in the live configuration — policy is CPU-only everywhere; `resolve_device()` (`app/models/runtime.py`) would fall back to CPU if a GPU device were requested and unavailable (`model_allow_cpu_fallback=true`) | N/A today, but has a defined fallback |

Overall: no scenario in this pipeline silently substitutes a dummy model for a real one that fails — real-mode failures fail closed to `BranchStatus.failed`, and the response schema's own Pydantic validators (`app/schemas/prediction.py:110-127,139-150`) structurally forbid claiming a dummy or blocked result is research-eligible.

---

## 13. Tests

| Coverage area | File(s) | Real checkpoint? |
|---|---|---|
| CNN branch | `test_real_model_adapters.py`, `test_real_model_architectures.py`, `test_cnn_verification_guard.py` | Yes, real-checkpoint-gated markers |
| AASIST branch | `test_aasist_light_v2_*.py` (4 files), `test_real_model_adapters.py` | Yes |
| SSL branch | `test_ssl_sequence_integration.py`, `test_ssl_generalization_v2_integration.py` | Yes, incl. real e2e (network) tests |
| Fusion mechanics | `test_fusion.py` (11 tests), part of `test_aasist_light_v2_fusion_validation.py` | No — synthetic `BranchPrediction` fixtures only |
| Real 3-branch fusion together | — | **No automated test exists** — only manually run in the prior audit reports, not committed as a test |
| API endpoint (prediction routes) | `test_voice_routes.py`, `test_prediction_routes.py` | No — mocked service layer |
| XAI contract | `test_voice_xai_contract.py`, `test_voice_xai_schemas.py`, `test_voice_xai_orchestrator.py`, `test_voice_xai_real_temporal.py` | Temporal: partially real; semantic: mocked |
| Invalid audio | `test_audio_preprocessing.py`, `test_ffmpeg_audio_formats.py` | n/a |
| Branch failure paths | `test_voice_service.py`, `test_model_runtime_phase3.py` | No — mocked |
| Glottal | Referenced in 15 test files as a fusion-branch *name* in synthetic scenarios | No real-checkpoint test exists — there is no checkpoint to test |

**Missing tests, explicitly:**

- An integration test that loads CNN+AASIST+SSL together with real checkpoints and asserts on the actual `FusionEngine.fuse()` output.
- Any test pinning the required threshold `0.5519237850482265` — the value does not appear anywhere in the test suite (nor the source) today.
- A labelled-audio sanity test for any branch — no `.wav`/`.mp3`/`.flac` fixture with a known bonafide/spoof label exists anywhere in the repository outside `.venv`.
- A regression test for the XAI response shape requested in §8 (`auxiliary_evidence`, `used_for_primary_decision`) — it doesn't exist in the schema yet, so no test could assert on it.

---

## 14. Final gap analysis

| Area | Current state | Required state | Gap | Severity | Files affected |
|---|---|---|---|---|---|
| Fusion threshold | `0.5` | `0.5519237850482265` | Not configured anywhere | **Critical** | `app/config/settings.py`, `.env` |
| CNN checkpoint | `models/best_cnn_full_weighted.pth` | `cnn_v2_lfcc_delta_aug_asvspoof2019_inference_best.pt` | Target file absent from repo | **Critical** | `app/config/settings.py`, `.env`, checkpoint files |
| CNN feature pipeline | log-mel, 20 filters, no deltas | LFCC-40 + Δ + ΔΔ | Different feature family entirely | **Critical** | `app/models/preprocessing/spectral.py`, `.env` |
| SSL checkpoint identity | `ssl_xlsr_mamba_generalization_v2_best.pt` | `ssl_asvspoof5_best.pt` | Live file is a later fine-tune of the named one, not itself | **High** | `app/config/settings.py`, `.env`, checkpoint files |
| Glottal artifacts | None present, no loader wired | Auxiliary-only branch producing XAI evidence | Both the model files and the loader/predictor implementation are missing | **High** | `app/models/real/glottal.py`, `app/models/factory.py`, model artifact files |
| Fusion / research-eligibility coupling | Glottal listed in `FULL_SYSTEM_BRANCHES`, permanently blocking `eligible_for_research_evaluation` | Glottal auxiliary, not gating primary result eligibility | One constant needs to change | **Medium** | `app/utils/fusion.py` |
| XAI response shape | Flat `branches[]` list, no auxiliary grouping | `auxiliary_evidence.{glottal_spoof_probability, used_for_primary_decision}` | Field/grouping doesn't exist | **Medium** | `app/schemas/xai.py` |
| Threshold visibility in the API response | Not present in `PredictionSubmissionResponse` at all | Client-visible threshold value | Field missing from the externally returned schema | **Medium** | `app/schemas/prediction_submission.py` |
| Semantic XAI | `XAI_MODE=mock`, no model/SHAP background configured | Real evidence consuming real branch scores | Not configured, not just disabled | **Medium** | `.env`, `app/main.py` |
| Branch execution model | Sequential, SSL ≈7–8s dominates every request now that it's live | Reasonable production latency | Not parallelized | **High** | `app/services/voice_service.py` |
| AASIST dead config | `AASIST_TARGET_SAMPLES`/`_NORMALIZATION`/`_LENGTH_POLICY` env vars silently ignored by the live V2 branch | Config that's read should be the config that's used | Orphaned settings, misleading to operators | **Low** | `app/models/real/inference.py`, `.env` |
| Dead route file | `app/routes/voice_routes.py` never imported | — | Orphaned duplicate | **Low** | `app/routes/voice_routes.py` |
| 3-branch fusion test coverage | No automated test with 3 real checkpoints | Regression coverage for the exact target formula | Test doesn't exist | **Medium** | `tests/` |

---

## 15. Integration plan

Recommended order only — nothing below was implemented in this audit.

1. **Resolve the CNN checkpoint question first.** Confirm whether `cnn_v2_lfcc_delta_aug_asvspoof2019_inference_best.pt` exists in the research/Colab environment; if so, copy it into `model_artifacts/cnn/` and point `CNN_MODEL_PATH` at it. This blocks the feature-pipeline fix below, since the correct LFCC-40+Δ+ΔΔ front end must match whatever this specific checkpoint was trained on.
2. **Update the CNN feature pipeline** (`app/config/settings.py` / `.env`: `CNN_FEATURE_TYPE=lfcc`, `CNN_FEATURE_FILTERS=40`, `CNN_FEATURE_COEFFICIENTS=40`, `CNN_FEATURE_INCLUDE_DELTAS=true`) once the matching checkpoint is in place, then re-run the checkpoint's own BatchNorm-statistics sanity check the current code already does (see §3) against the new configuration.
3. **Resolve the SSL checkpoint question.** Decide explicitly whether the target means the literal `ssl_asvspoof5_best.pt` parent or the currently-deployed `ssl_xlsr_mamba_generalization_v2_best.pt` descendant; if the parent is required, copy it in and repoint `SSL_MODEL_PATH`.
4. **Set the fusion threshold** to `0.5519237850482265` via `FUSION_DECISION_THRESHOLD` in `.env` (no code change needed — `Settings.fusion_decision_threshold` already reads this directly). This is the lowest-effort, highest-impact single change on this list.
5. **Confirm the fusion method choice.** Either leave `fusion_method=weighted_average` with equal `0.25` weights (already numerically equivalent to a simple average whenever exactly CNN+AASIST+SSL succeed) or switch explicitly to the already-implemented `simple_average` method for architectural clarity. Functionally equivalent for the 3-branch case; a documentation/intent decision, not a behavior fix.
6. **Reclassify Glottal as auxiliary.** Remove `glottal` from `FULL_SYSTEM_BRANCHES` in `app/utils/fusion.py` so real CNN+AASIST+SSL output can become research-eligible without a Glottal contribution. Do this only after step 4, since eligibility should reflect a correctly-thresholded result.
7. **Implement the Glottal branch** (once `glottal_logreg_selected20_v1.joblib` and `glottal_selected_features_v1.json` are available) as a real loader/predictor in `app/models/real/glottal.py`, explicitly wired to feed XAI only — never added to `FUSION_WEIGHT_GLOTTAL`'s consumption in the primary score path.
8. **Extend the XAI schema** (`app/schemas/xai.py`) with an `auxiliary_evidence` group carrying `glottal_spoof_probability` and `used_for_primary_decision: false`, sourced from the newly-real Glottal branch's output.
9. **Expose the threshold in the external response** (`app/schemas/prediction_submission.py`) so frontend/XAI consumers stop needing to read the internal-only `provenance` field that isn't returned today.
10. **Parallelize branch execution** in `app/services/voice_service.py` before or immediately after enabling all of the above in a live deployment, since SSL alone already dominates request latency sequentially.
11. **Add the missing 3-branch real-fusion regression test** and a test pinning the exact threshold value, so the corrected configuration cannot silently regress.

---

## 16. Required files from research/Colab

| File | Category | Why |
|---|---|---|
| `cnn_v2_lfcc_delta_aug_asvspoof2019_inference_best.pt` | **Required — primary inference** | Named target CNN checkpoint; absent from the repo entirely, current CNN checkpoint and feature pipeline both deviate from spec |
| `ssl_asvspoof5_best.pt` (if the parent, not the deployed descendant, is intended) | **Needs a decision, then required — primary inference** | Named target SSL checkpoint; only its lineage is present today via the deployed Generalization V2 descendant |
| `aasist_light_v2_best.pt` | Already present | No action needed — matches the target exactly |
| `glottal_logreg_selected20_v1.joblib` | **Required — XAI only** | Per the target architecture, Glottal must not gate the primary decision; needed to produce real auxiliary evidence, not fusion input |
| `glottal_selected_features_v1.json` | **Required — XAI only** | Feature-selection manifest paired with the joblib model above |
| SSL semantic/SHAP explainer model + background set | Required — XAI only | `xai_semantic_model_path`/`xai_semantic_shap_background_path` are both unset; needed to move `XAI_MODE` from `mock` to `real` |
| `models/xlsr_mamba_asvspoof2019_best.pt` | Research/archive only | Superseded SSL checkpoint, kept for lineage; not reachable from any live default path |
| `models/best_aasist_light_full_weighted.pth` | Research/archive only | Superseded AASIST V1 checkpoint, kept only for legacy regression tests |
| CNN/SSL training notebooks or feature configs | Not required, but recommended | Neither shipped checkpoint has one; this is why CNN's feature pipeline and both branches' label maps remain "declared, not verified" (`*_PREPROCESSING_VERIFIED=false`, `*_CLASS_MAPPING_VERIFIED=false` throughout `.env`) |
| Labelled bonafide/spoof audio (5–10 each, multiple generators) | Not required for inference, but blocking for verification | No labelled or even unlabelled real audio exists anywhere in the repository; nothing can move any `*_VERIFIED` flag to `true` without it |

---

## 17. Final verdict

| Question | Answer |
|---|---|
| What is already integrated? | A real, running three-branch pipeline (CNN, AASIST, SSL/XLS-R+Mamba) with genuine checkpoints, sequential inference, and a working score-level fusion engine that already excludes Glottal from the primary calculation. |
| What is outdated? | The fusion decision threshold (`0.5` instead of the calibrated `0.5519237850482265`); the CNN checkpoint and feature pipeline (log-mel vs. required LFCC+Δ+ΔΔ); several orphaned AASIST env-var overrides that no longer reach the live branch. |
| What is missing? | The target CNN checkpoint file itself; a decision on the SSL checkpoint's exact identity; the entire Glottal branch implementation and its two artifact files; the XAI schema's auxiliary-evidence grouping; a real (non-mock) semantic XAI model. |
| What should be changed first? | The fusion threshold — a one-line `.env` change (`FUSION_DECISION_THRESHOLD=0.5519237850482265`) with no code change required, and the single highest-leverage fix available. |
| Can the backend currently reproduce the final research detector exactly? | **No.** It reproduces the architecture's shape (three real branches, score-level average fusion, Glottal excluded from the primary decision) but not its exact numeric behavior: wrong threshold, a different CNN checkpoint/feature pipeline, and an SSL checkpoint one fine-tuning stage removed from the one named in the target. |
