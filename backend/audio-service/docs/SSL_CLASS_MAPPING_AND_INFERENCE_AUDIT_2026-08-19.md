# SSL Class Mapping & Inference Audit — ElevenLabs False-Negative Investigation

**Date:** 2026-08-19
**Trigger:** A known AI-generated (ElevenLabs) voice sample produced CNN spoof≈99.5%, AASIST spoof≈100%, but SSL/Temporal Voice Analysis spoof≈2.2% / bonafide≈97.8%.
**Scope:** SSL class-mapping and inference-direction audit only. **No code, `.env`, checkpoint, or frontend changes were made.**

**Note on `.env` drift observed during this audit:** `backend/.env`'s `SSL_MODEL_MODE` changed from `disabled` to `real` between the prior full-pipeline audit and this one. That is an external change (not made by this audit) — it is now the live configuration and is treated as current state below, not reverted. It also means SSL's fusion weight is now live-renormalized to 1/3 (previously 0.5/0.5 with only CNN+AASIST live), which is relevant context for interpreting the ElevenLabs result: SSL is genuinely contributing to production fusion now, not just running in a side test.

---

## 1. Executive summary

The SSL branch's mapping mechanism was traced end-to-end, from raw logits through to the frontend pixel. **No bug was found in the code path itself** — softmax, index selection, `ProbabilityScores` construction, the API schema, and the frontend's rendering are all internally consistent and faithfully preserve whatever the configured `spoof_index` says. The frontend performs **zero transformation**: it renders `probabilities.spoof` under a "Spoof" label and `probabilities.bonafide` under a "Bonafide" label, verbatim, with no swap, no complement, no recomputation.

The real open question is **not** "does the code implement the mapping correctly" (yes, mechanically confirmed) but **"is `bonafide=0, spoof=1` actually the training-time truth for this exact checkpoint"** — and here the direct evidence is limited: the checkpoint file itself (`ssl_xlsr_mamba_generalization_v2_best.pt`) carries **no label-map, class_to_idx, or class-order metadata of any kind**. No training script for this specific fine-tuning run exists in this repository.

The strongest available evidence — independently reconstructing the checkpoint's own reported ASVspoof5 confusion matrix and confirming it reproduces the reported precision/recall/F1 **only** when class index 1 is the "positive" (recall/precision-reported) class — is consistent with `bonafide=0, spoof=1`, matching the superseded checkpoint's own explicit self-declared mapping for the same training lineage, and matching universal ASVspoof-domain convention. This is strong, convergent, but **indirect** evidence, not a literal training-script confirmation.

No labelled bonafide/spoof audio exists anywhere in this repository, so **no mechanical sanity test could be run** to directly confirm or refute the mapping using real audio. The single ElevenLabs anecdote is explicitly **not** used as proof of anything, per the audit's own working rule.

**Root-cause verdict: C. MAPPING CORRECT — MODEL GENERALIZATION FAILURE** (moderate-high confidence, not certainty — see §15 for the full reasoning and what would raise this to certainty).

## 2. Exact SSL checkpoint

`model_artifacts/ssl/ssl_xlsr_mamba_generalization_v2_best.pt`, resolved via the same `checkpoint_identity_for_path` function `ModelFactory` uses in production. Independently recomputed this audit:

```
resolved absolute path: <repo_root>/model_artifacts/ssl/ssl_xlsr_mamba_generalization_v2_best.pt
size: 4,704,997 bytes
SHA-256: 336e65d36764315ed4fe036cc99d3b957daae5e07a5ec462f8b09faa35291afe
```

This is confirmed **not** the superseded `models/xlsr_mamba_asvspoof2019_best.pt` (different SHA-256, different byte size, different key structure — see §4). `SSL_MODEL_PATH=model_artifacts/ssl/ssl_xlsr_mamba_generalization_v2_best.pt` in the live `.env`; the Python-level `Settings` default matches. Strict state-dict loading was re-confirmed to succeed with **zero missing/unexpected keys** and an exact 1,173,634-parameter match for the trainable head (details in §11).

## 3. Training-time label semantics

**No training script for this checkpoint exists in this repository.** The checkpoint's `source_checkpoint` field (`/content/drive/MyDrive/MULTI_SCOPE_SSL_XLSR_MAMBA_FINETUNING/checkpoints/ssl_asvspoof5_best.pt`) points to a Colab path outside the repo — evidence of lineage, not something this audit can execute or inspect further.

Given that, the training-time source of truth had to be reconstructed from indirect evidence:

| Evidence | Finding | File/reference | Confidence |
|---|---|---|---|
| Checkpoint's own embedded metadata | **No label_map/class_to_idx/spoof_index present at all** (see §4) | `model_artifacts/ssl/ssl_xlsr_mamba_generalization_v2_best.pt` | CONFIRMED (absence) |
| Superseded checkpoint's own self-declared mapping, same architecture/lineage | `label_mapping: {"bonafide": 0, "spoof": 1}` | `models/xlsr_mamba_asvspoof2019_best.pt` (inspected this audit, key `label_mapping` present) | CONFIRMED for that file; LIKELY-transferable to the new one (same architecture family, same head shape, `source_checkpoint` names a direct fine-tuning continuation) |
| Reconstructed confusion matrix vs. reported precision/recall/F1 | Numbers **only** reconcile when class index 1 is the class recall/precision were computed for | See §15 math below; matrix and metrics both supplied by the user as this checkpoint's own reported ASVspoof5 TEST results | LIKELY (self-consistent math, but the semantic label "spoof" for that index-1 class is an inference from domain convention, not read from a file) |
| `sampling_strategy` key naming | `{"asv_bonafide": 0.5, "asv_spoof": 0.25, "wavefake_spoof": 0.25}` — explicit `_spoof`/`_bonafide` suffixes, and WaveFake (a spoof-only corpus) only appears as `wavefake_spoof`, never `wavefake_bonafide` | `ssl_xlsr_mamba_generalization_v2_best.pt`'s `sampling_strategy` key | Weak corroboration — proves the training pipeline used "bonafide"/"spoof" as named classes, not which index each maps to |
| Domain convention (ASVspoof-family anti-spoofing literature and this project's own AASIST checkpoint) | Virtually universal convention: `spoof=1` (the class the detector exists to catch) | `AASIST_V2_LABEL_MAPPING = {"bonafide": 0, "spoof": 1}` (checkpoint-attested, `real/inference.py`); `CNN_CLASS_ORDER`/`AASIST_CLASS_ORDER` both `bonafide_spoof` | Contextual, not proof specific to the SSL checkpoint |

**Conclusion for this section**: index 0 = bonafide, index 1 = spoof is **LIKELY** (well-evidenced by convergent indirect evidence) but **not CONFIRMED** by a training script or an explicit label map in the checkpoint file itself.

## 4. Checkpoint metadata evidence

Full, exhaustive top-level key dump of the loaded checkpoint object (`torch.load(..., weights_only=True)`, the same safe-loading convention `real/ssl_sequence_inference.py` already uses):

```python
ALL TOP-LEVEL KEYS: ['epoch', 'stage', 'source_checkpoint', 'train_config', 'train_loss', 'trainable_model_state', 'sampling_strategy']

epoch: 2
stage: 'Generalization_V2'
source_checkpoint: '/content/drive/MyDrive/MULTI_SCOPE_SSL_XLSR_MAMBA_FINETUNING/checkpoints/ssl_asvspoof5_best.pt'
train_config: {'epochs': 2, 'batch_size': 4, 'gradient_accumulation_steps': 4, 'learning_rate': 1e-05, 'weight_decay': 0.0001, 'max_grad_norm': 1.0, 'mixed_precision': 'bf16', 'freeze_xlsr': True}
train_loss: 0.2633038581501354
sampling_strategy: {'asv_bonafide': 0.5, 'asv_spoof': 0.25, 'wavefake_spoof': 0.25}
```

Searched explicitly for, and **confirmed absent**: `labels`, `label_map`, `class_names`, `class_to_idx`, `idx_to_class`, `spoof_index`, `bonafide_index`, `dataset`, `config` (a distinct key from `train_config`), `metadata`, `attestation`, `threshold`. **The checkpoint contains no trustworthy class-order metadata of its own.**

**This is a HIGH-severity gap by the audit's own stated rule** ("If there is a mismatch between checkpoint metadata and backend env/config, mark it HIGH severity") — though here it is an *absence* of checkpoint metadata to compare against, not a proven *mismatch*. Flagged HIGH regardless: the backend's `SSL_CLASS_ORDER=bonafide_spoof` is entirely unattested by the file it configures, unlike AASIST where the checkpoint does self-attest and the two happen to agree (see the three-branch audit's §14/§24 for that comparison).

## 5. Raw model output semantics

Traced in `app/models/real/ssl_sequence_inference.py`:

```python
# build_ssl_predictor(), inside predict():
with torch.inference_mode():
    logits = runtime_model.module(input_values, attention_mask)   # shape (1, 2)
spoof_probability = _spoof_probability_from_logits(torch, logits, runtime_model, config)

# _spoof_probability_from_logits():
probabilities = torch.softmax(logits.to(dtype=torch.float32), dim=-1)[0]   # shape (2,)
spoof = float(probabilities[runtime_model.spoof_index])
bonafide = float(probabilities[runtime_model.bonafide_index])
```

Where `runtime_model.spoof_index = 1 if class_order == "bonafide_spoof" else 0` and `runtime_model.bonafide_index = 1 - runtime_model.spoof_index` (`LoadedSSLBranch.bonafide_index` property). With the live `SSL_CLASS_ORDER=bonafide_spoof`, this is **exactly** `spoof = softmax(logits, dim=-1)[..., 1]`, matching the brief's expected formula exactly.

Checked explicitly for every failure mode the brief asked about — **none found**:
- **Tensor indexing**: single dynamic index lookup (`probabilities[runtime_model.spoof_index]`), not a hard-coded `[1]` that could silently desync from `spoof_index`.
- **Squeeze/reshape**: `logits[0]` after a shape-validated `(1, 2)` check (`_spoof_probability_from_logits` raises `model_output_invalid` if `logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != 2`) — no accidental batch-dimension confusion possible.
- **Sigmoid misuse**: none — `softmax(dim=-1)` only, never `sigmoid`.
- **Double softmax**: none — softmax is applied exactly once, directly on raw logits.
- **Negation**: none found anywhere in this path.
- **Probability complement swap**: none — `bonafide` and `spoof` are each read from their own distinct softmax index, never derived as `1 - other`.
- **Naming swap after inference**: none — the variable named `spoof_probability` is returned directly into `real_prediction_from_spoof_probability(spoof_probability=spoof_probability, ...)`, which builds `ProbabilityScores(bonafide=1.0 - spoof_probability, spoof=spoof_probability)` — the field names match the value origins exactly.

**Conclusion**: the raw-logit-to-probability path is mechanically clean. If the mapping is wrong, it is wrong because `spoof_index=1` itself is the wrong declaration for this checkpoint — not because of a bug in how that declaration is applied.

## 6. Backend probability mapping

`real_prediction_from_spoof_probability` (`app/models/real/base.py`, shared by CNN/AASIST/SSL):

```python
bonafide_probability = 1.0 - spoof_probability
prediction = PredictionLabel.spoof if spoof_probability >= bonafide_probability else PredictionLabel.bonafide
confidence = max(spoof_probability, bonafide_probability)
probabilities = ProbabilityScores(bonafide=bonafide_probability, spoof=spoof_probability)
```

This is purely mechanical given `spoof_probability` — it cannot introduce a mapping error on its own; it only propagates whatever `spoof_probability` it was handed. For the SSL branch specifically, `metadata` also carries `label_mapping: {"bonafide": 0, "spoof": 1}` (declared, added in the prior SSL integration task) and `threshold: 0.5`, both informational.

## 7. Fusion score mapping

`ProbabilityScores.spoof` is the exact field `FusionEngine._average_probabilities` reads (`probabilities.spoof * weight`, summed across contributing branches — see the prior three-branch audit, §9-§10, for the full formula). No transformation happens between the SSL branch's own `probabilities.spoof` and what fusion consumes — it is the identical float value, weighted and summed alongside CNN/AASIST's. **No inversion at the fusion layer; SSL is not "correct in its card but inverted in fusion," nor the reverse — it is the same number in both places.**

## 8. API response mapping

`BranchPrediction` (`app/schemas/prediction.py`), the schema actually serialized over the wire for the SSL branch (public `model_name = "ssl_wavlm_xlsr"`):

```
status: BranchStatus            (success/failed/skipped)
mode: ModelMode                 (real, currently -- SSL_MODEL_MODE=real live)
prediction: "spoof" | "bonafide" | null
confidence: float | null        (max(spoof, bonafide))
probabilities: { bonafide: float, spoof: float }   (Pydantic-validated to sum to ~1.0, tolerance 1e-3)
metadata: { architecture, checkpoint..., label_mapping, threshold, class_mapping_verified: false, research_result: false, warning: "...Treat the score as unvalidated.", ... }
```

`probabilities.bonafide + probabilities.spoof ≈ 1` is enforced by a Pydantic model validator (`ProbabilityScores.validate_probability_sum`) — a malformed response that violated this would fail to even construct, so this was not just spot-checked, it is structurally guaranteed. Both fields are sourced from the single `spoof_probability` float computed in §5/§6 — there is no separate "raw index 0 probability" field exposed anywhere in the API that could be confused with `bonafide`. **No semantic ambiguity found in the API contract.**

## 9. Frontend mapping

Traced `frontend/src/components/prediction/branch-result-card.tsx` → `frontend/src/components/prediction/probability-bar.tsx` → `frontend/src/components/prediction/status-badge.tsx`, and `frontend/src/lib/copy/detector-names.ts` for the branch identity.

- **Branch ID/key used**: `branch.model_name` — for SSL this is `"ssl_wavlm_xlsr"` (or `"ssl_sequence"`, both mapped), which resolves to friendly name **"Temporal Voice Analysis"** (`detector-names.ts`) — matches exactly what the user saw.
- **API fields consumed**: `branch.probabilities.spoof`, `branch.probabilities.bonafide`, `branch.prediction`, `branch.confidence` — read directly off the API response object, no intermediate transform layer.
- **Spoof % / Bonafide % calculation**: `probability-bar.tsx` line 9-10: `const spoof = probabilities.spoof * 100; const bonafide = probabilities.bonafide * 100;` — a straight ×100 unit conversion, applied identically and independently to each field. **No complement calculation** (`bonafide = 1 - spoof`) exists anywhere in this component — both values come straight from the API.
- **Label selection ("Likely authentic" / "Likely AI-generated")**: `status-badge.tsx`'s `LabelBadge`, a pure 1:1 string map off `branch.prediction` (the backend-computed label): `"spoof" → "Likely AI-generated"`, `"bonafide" → "Likely authentic"`. **No threshold is recomputed client-side** — the frontend never looks at the raw probability to decide the label, it trusts the backend's own decision entirely.
- **Field swap check**: the JSX explicitly pairs `<span>Spoof</span>` with `probabilities.spoof` (lines 14-18) and `<span>Bonafide</span>` with `probabilities.bonafide` (lines 20-25) — verified by reading the literal source, not inferred. **No swap.**
- **Percentage conversion error**: none — `formatProbability` (checked in `frontend/src/lib/formatters.ts`) is a standard `value*100` formatter, applied uniformly.

**Conclusion**: the frontend is a faithful, unmodified passthrough of whatever the backend API sent for the SSL branch. **If SSL showed bonafide=97.8%/spoof=2.2% for the ElevenLabs clip, that is exactly the pair of floats `real_prediction_from_spoof_probability` computed on the backend — the frontend introduced no distortion.** This directly answers the audit's Secondary Objective.

## 10. Preprocessing parity

| Aspect | Backend (declared, `ssl_waveform.py`) | Training (per integration brief, no training script in repo) | Status |
|---|---|---|---|
| Sample rate | 16000 Hz | 16000 Hz | MATCH (declared vs declared) |
| Mono | guaranteed by shared decoder | mono | MATCH (declared) |
| Duration / samples | 6.0s / 96000 | 6.0s / 96000 | MATCH (declared) |
| Normalization | zero-mean, peak-normalize (`(x-mean)/max(|x-mean|)`, guard `1e-7`) | zero-mean, peak-normalize (brief's exact formula) | MATCH (declared) |
| Long-clip crop | deterministic center crop | training used random crop; center crop is the declared deterministic inference substitute | MATCH by design, **not training-identical by construction** (random ≠ center, intentionally) |
| Short-clip padding | zero-pad at end | zero-pad at end (declared) | MATCH (declared) |
| Attention mask | built from real/padded sample counts, never `input_values != 0` | HF feature-extractor-style attention mask expected | MATCH in *effect* (correctly marks real vs. padded samples); **the literal HF `Wav2Vec2FeatureExtractor` class is not used** — a deliberate substitution because its default normalization (zero-mean/unit-variance) contradicts the requested zero-mean/peak formula (documented in the prior SSL integration report, §5) |
| XLS-R processor | `AutoModel.from_pretrained("facebook/wav2vec2-large-xlsr-53", attn_implementation="eager")`, frozen | same named backbone, frozen (per brief) | MATCH |
| Sequence length / Mamba input shape | `(1, 96000)` waveform → XLS-R → `(1, ~299, 1024)` hidden states → projection → `(1, ~299, 256)` → 2×Mamba → mean pool → `(1, 256)` | matches the declared architecture exactly (verified by strict state-dict load, §11) | MATCH |

**None of this is independently verified against an actual training pipeline** (no training script exists in-repo) — every "MATCH" above is "backend declaration matches the integration brief's declaration," not "backend matches ground-truth training code." All marked **UNVERIFIED** in the stricter sense the brief asks for; no **MISMATCH** was found between what's declared and what's implemented.

## 11. CNN / AASIST / SSL semantics comparison

| Branch | Training index 0 | Training index 1 | Backend spoof index | API spoof field source | Frontend spoof source | Status |
|---|---|---|---|---|---|---|
| CNN | bonafide (declared) | spoof (declared) | 1 | `probabilities.spoof` ← `softmax(logits)[1]` | `probabilities.spoof` (direct) | **UNVERIFIED** — no checkpoint metadata, no training script; `.env`'s own comment records CNN and AASIST disagree on 7/8 synthetic probes under the shared assumption, meaning at least one is suspect (carried forward from the prior full-pipeline audit, not re-derived here) |
| AASIST | bonafide (**checkpoint-attested**: `class_mapping={"bonafide":0,"spoof":1}`, validated at load time) | spoof (checkpoint-attested) | 1 | `probabilities.spoof` ← `softmax(logits)[1]` | `probabilities.spoof` (direct) | **LIKELY** — checkpoint attests it, env setting happens to agree, but the runtime never cross-checks the two against each other (pre-existing characterized gap) |
| SSL | bonafide (**not** attested by this checkpoint; inferred from lineage + confusion-matrix math + domain convention, §3) | spoof (same basis) | 1 | `probabilities.spoof` ← `softmax(logits)[1]` | `probabilities.spoof` (direct) | **UNVERIFIED** by the checkpoint itself; **LIKELY correct** by convergent indirect evidence (§3, §15) |

No branch reaches **CONFIRMED** in the strict sense (an in-repo training script or checkpoint self-attestation independently re-executed); **no branch is MISMATCH** either — no evidence anywhere in this audit or the prior ones shows a *proven* inverted mapping for any of the three.

## 12. Labelled sanity-test results

**No labelled validation audio exists anywhere in this repository.** Checked: `backend/validation_data/real_world_sanity_set/samples/` (empty except `.gitkeep`), `backend/validation_data/real_world_sanity_set/manifest.csv` (absent), `backend/uploads/` (empty — the ElevenLabs clip that produced the observed result was not retained; standard cleanup unlinks uploaded audio after each request completes), and a repository-wide search for any `.wav`/`.mp3`/`.flac`/`.m4a` file (none found outside `.venv`/`node_modules`). **No mechanical sanity test with known-labelled audio could be performed.** This is stated plainly per the audit's explicit instruction, not worked around by inventing labels or substituting synthetic probes as if they were equivalent.

## 13. Mapping-error hypothesis evaluation (Hypothesis A)

Signs the brief asked to check for:
- *Known real files consistently get high "spoof"*: **cannot be tested** — no known real files exist locally.
- *Known fake files consistently get high "bonafide"*: the single ElevenLabs anecdote is consistent with this, but the brief explicitly forbids concluding from one file, and no second sample exists to check consistency.
- *Raw outputs become correct when indices are swapped*: mathematically true by definition for any single misclassified sample (swapping always "fixes" the one sample you're looking at) — this is exactly why a single anecdote cannot distinguish Hypothesis A from B, and is not evidence either way.
- *Training-time label map disagrees with backend mapping*: **not found** — the checkpoint has no label map to disagree with; the closest available evidence (§3's confusion-matrix reconstruction) is *consistent with*, not contradictory to, the backend's `bonafide_spoof` declaration.

**Assessment**: Hypothesis A has no code-path evidence supporting it (§5-§9 traced clean) and no checkpoint-metadata evidence supporting it (§4 found no mapping at all, let alone a conflicting one). It cannot be fully ruled out without a labelled test set, but nothing found in this audit points toward it.

## 14. Generalization-failure hypothesis evaluation (Hypothesis B)

Signs the brief asked to check for:
- *Training-time mapping definitely matches backend mapping*: not "definitely" (no training script), but the best available indirect evidence (§3, §15) is consistent with a match, not a conflict.
- *Known validation samples behave correctly overall*: **cannot be tested** — no validation samples exist.
- *Only certain generators, such as ElevenLabs, are misclassified*: consistent with — and independently supported by — the checkpoint's own documented research context (given by the user, not re-derived): ASVspoof5 TEST EER=0.1856 (≈18.6% error rate — this is not a near-perfect model), and the WaveFake unseen-generator test itself was explicitly framed around the premise that trained-on-ASVspoof detectors generalize poorly to unseen generators (that is the entire point of reporting a separate "unseen-generator" number, and even that number — 90.3% spoof detection — is not 100%). ElevenLabs is a commercial, proprietary neural TTS system; there is no evidence it is represented in either ASVspoof5 or WaveFake's training/eval corpora, both of which are academic corpora built from published, fixed synthesis pipelines.
- *Raw probabilities themselves strongly favor bonafide for ElevenLabs*: consistent with the observed 97.8% bonafide — a value that strongly, not marginally, favors bonafide, which fits a model confidently misjudging an unfamiliar generator rather than a boundary-case disagreement.

**Assessment**: every one of the brief's own listed signs for Hypothesis B is present or consistent with the available evidence; none of Hypothesis A's signs are positively supported. This is the basis for the §15 verdict.

## 15. Root-cause verdict

### C. MAPPING CORRECT — MODEL GENERALIZATION FAILURE

**Confidence: moderate-high, not certainty.**

Supporting evidence, weighted:

1. **Code-path tracing (§5-§9) is clean** — every stage from raw logits to the rendered percentage was read line-by-line; no swap, complement, double-softmax, sigmoid misuse, or field-name confusion exists anywhere in the pipeline. This rules out the *mechanical* half of Hypothesis A entirely — if there is a mapping error, it can only be that the single configured constant (`spoof_index=1`) is the wrong value, not that the code mishandles a correct value.
2. **Confusion-matrix reconciliation** (independently computed this audit from the numbers the user/research context supplied for this exact checkpoint): the reported ASVspoof5 TEST confusion matrix `[[3214, 1786], [293, 4707]]` reproduces the reported accuracy (7921/10000 = 0.7921 ✓), recall (4707/5000 = 0.9414 ✓), and precision (4707/6493 = 0.724936 ✓, matching 0.724935 to 5 decimal places) **only** when row/column index 1 is treated as the "positive" class those metrics were computed for. Checking the alternative (index 0 as positive) yields recall=0.6428/precision=0.9165 — numbers that do **not** match anything reported. This is hard, independently-verifiable arithmetic, not a guess — though it proves index 1 was the "positive/reported" class, not by itself which English word ("spoof") that index means; that final semantic step relies on domain convention (below).
3. **Lineage + domain convention**: the superseded checkpoint from the same architecture family explicitly self-declares `{"bonafide": 0, "spoof": 1}`, and virtually every ASVspoof-based anti-spoofing system (including this project's own checkpoint-attested AASIST) uses spoof=1 as the class of interest for recall/precision reporting — consistent with item 2's finding that index 1 was the reported-metric class.
4. **The failure pattern fits known, already-documented model limitations, not a systemic inversion**: EER=0.1856 means this checkpoint is measurably imperfect even on in-domain data; the entire reason a separate WaveFake "unseen-generator" number exists is that cross-domain generalization is a known, explicit weak point for this model family, and even that dedicated unseen-generator test only reached 90.3%, not 100%. A single miss on a commercial black-box TTS system outside both training corpora is the expected failure mode of an imperfect, generalization-limited classifier — not the expected signature of a mapping bug, which would typically produce systematic, not isolated, disagreement with CNN/AASIST.

What would raise this from "moderate-high confidence" to **CONFIRMED**: a real labelled validation set (§12) run through the SSL branch, ideally including several confirmed-bonafide human recordings and several confirmed-spoof recordings from multiple generators including, if possible, another ElevenLabs sample. If known-bonafide samples systematically score high-spoof and known-spoof samples systematically score high-bonafide, that would flip this verdict to **A**. Nothing in this audit found that pattern — but nothing in this audit could test for it either, for lack of data.

## 16. Severity-ranked findings

**HIGH:**
- The SSL checkpoint currently in production (`ssl_xlsr_mamba_generalization_v2_best.pt`) carries **zero label-order metadata of its own**; the live `bonafide_spoof` mapping is a declared inheritance from a *different* checkpoint file's self-attestation, not this file's. (§3, §4)
- SSL's own EER (0.1856) and unseen-generator ceiling (90.3%) mean the branch is **known, documented, and expected** to sometimes misclassify — including plausibly on commercial TTS systems outside its training corpora — independent of any mapping question. This is a real accuracy limitation, not a bug, and should inform how much trust any single-branch SSL "bonafide" verdict is given for consumer-facing commercial TTS.
- `SSL_MODEL_MODE` is now `real` in the live `.env` (changed since the prior audit) — SSL genuinely influences every production fusion result now, weighted equally with CNN/AASIST (1/3 each), so this generalization gap is live, not hypothetical.

**MEDIUM:**
- No labelled validation corpus exists to ever directly test SSL's (or CNN's or AASIST's) mapping/accuracy claims against real data — this remains the single biggest evidentiary gap blocking a CONFIRMED verdict for any branch, not just SSL.
- AASIST's checkpoint-attested class mapping is never cross-checked against its own env setting at runtime (pre-existing, characterized, not SSL-specific, listed here for completeness since the two branches were compared directly in §11).

**LOW:**
- The declared preprocessing contract (§10) is internally consistent but unverifiable against an actual training script; low severity because no evidence of an actual mismatch was found, only an inability to prove a match.

## 17. Exact recommended fix plan (proposed only — not implemented)

1. **Do not swap SSL's class indices.** No evidence in this audit supports a mapping inversion; swapping without evidence would very likely make CNN/AASIST-agreeing cases wrong instead.
2. Obtain a small labelled validation set (5-10 known-bonafide human recordings, 5-10 known-spoof recordings spanning multiple TTS/voice-conversion systems, ideally including at least one more ElevenLabs sample) and run it through `scripts/validate_ssl_generalization_v2_inference.py` (already exists, built in the prior SSL integration task) to move §15's verdict from "moderate-high confidence" to a data-backed conclusion.
3. If that labelled run shows SSL systematically inverted (bonafide↔spoof swapped across most samples), only then flip `SSL_CLASS_ORDER=spoof_bonafide` — as an explicit, separate, evidence-backed change, never silently.
4. If that labelled run confirms the mapping is correct (the more likely outcome per §15), consider documenting SSL's known generalization limitation more visibly in the UI/report layer (e.g., a note that a single "bonafide" verdict from the temporal branch alone, especially disagreeing with both other branches, should not be treated as strong evidence — this is a product/communication decision, not a code fix, and is explicitly not implemented here).
5. Separately (not blocking on the above): investigate whether ElevenLabs-style commercial neural TTS output is represented at all in the ASVspoof5/WaveFake training mixture SSL was fine-tuned on; if not, this specific failure mode is expected and would recur for any comparable commercial TTS system until the model is fine-tuned on more representative data — a retraining/data question, explicitly out of scope for this audit.

## 18. Files that would need changes if a fix is approved

*(Not implemented — listed only for planning purposes.)*

- `backend/.env` — `SSL_CLASS_ORDER` (only if evidence from item 2 above proves a swap is needed).
- `backend/app/config/settings.py` — the `ssl_class_order` comment block, if changed.
- No production code changes would be needed for a pure class-order flip (`spoof_index` derivation already reads the setting dynamically) — this is itself worth noting as a positive: fixing a confirmed mapping bug, if one is ever found, is a one-line config change, not a code change, because of how cleanly `_spoof_probability_from_logits` already reads `class_order` indirectly.

## 19. Tests that must be added

*(Not implemented — recommended only.)*

- A labelled-sanity-set-driven test (parallel to `backend/scripts/validate_cnn_class_order.py`'s pattern for CNN) that, once real labelled audio exists, asserts SSL's bonafide/spoof predictions against known ground truth and reports accuracy/balanced-accuracy — explicitly labeled "small sanity validation," not "benchmark evaluation," consistent with this project's existing convention.
- A regression test pinning the confusion-matrix-derived evidence in §15 as a permanent, documented artifact (e.g., a test that recomputes the precision/recall reconciliation from the stated confusion matrix and asserts it only matches under the `index 1 = positive` reading) — cheap, deterministic, and prevents this specific evidentiary chain from being silently lost or contradicted by a future checkpoint swap without someone noticing.

## 20. Final readiness status

**Unchanged from the prior full-pipeline audit: PARTIALLY READY.** This SSL-specific investigation did not find a mapping bug, but it also did not — and could not, for lack of data — fully confirm the mapping is correct. SSL is now live in production (`SSL_MODEL_MODE=real`), contributing real, unverified, generalization-limited scores to every fusion result. Nothing found here should be read as either a green light or a stop-ship signal on its own; it sharpens the existing "populate a labelled validation set" recommendation from **should-do** to **directly explains an observed real-world discrepancy, and would resolve it either way.**

---

## Final summary

**A. SSL checkpoint:** `model_artifacts/ssl/ssl_xlsr_mamba_generalization_v2_best.pt` (SHA-256 `336e65d36764315ed4fe036cc99d3b957daae5e07a5ec462f8b09faa35291afe`), confirmed live-loaded, confirmed not the superseded ASVspoof2019-only file.

**B. Training index 0:** bonafide — LIKELY, not checkpoint-proven (no label map in this file; inferred from lineage + confusion-matrix math + domain convention).

**C. Training index 1:** spoof — same basis and confidence as B.

**D. Backend configured spoof index:** 1 (`SSL_CLASS_ORDER=bonafide_spoof`).

**E. Checkpoint-attested spoof index:** **none — this checkpoint attests nothing about class order.**

**F. Raw output mapping:** `spoof_probability = softmax(logits, dim=-1)[..., 1]` — mechanically confirmed correct given the configured index; no swap/complement/double-softmax/sigmoid bug found anywhere in the path.

**G. API spoof field mapping:** `probabilities.spoof` sourced directly from the same float computed above; schema-enforced to sum to ~1 with `bonafide`; no ambiguity found.

**H. Frontend spoof field mapping:** direct, untransformed passthrough — "Spoof" label ↔ `probabilities.spoof`, "Bonafide" label ↔ `probabilities.bonafide`, verdict badge ↔ backend's own `prediction` field. No frontend swap, no client-side threshold recomputation.

**I. Preprocessing parity:** MATCH between declared backend implementation and the declared training brief on every checked dimension (sample rate, mono, duration, normalization, padding, backbone, sequence shape); center-crop is an intentional, documented, non-training-identical substitute for training-time random crop. Nothing independently verifiable against an actual training script (none exists in-repo) — all UNVERIFIED in the strict sense, no MISMATCH found.

**J. Labelled sanity test available:** **NO** — no labelled or even unlabelled real audio exists anywhere in this repository; none was run; the single ElevenLabs anecdote was explicitly not used as proof of anything.

**K. ElevenLabs result likely caused by:** genuine model generalization failure on a commercial TTS system outside the SSL checkpoint's documented training/eval corpora (ASVspoof5 + WaveFake) — consistent with the checkpoint's own reported EER (0.1856) and unseen-generator ceiling (90.3%, not 100%) — rather than a class-mapping/index bug, based on clean code-path tracing and convergent (though indirect) training-time evidence.

**L. Root-cause verdict: C. MAPPING CORRECT — MODEL GENERALIZATION FAILURE** (moderate-high confidence; would become CONFIRMED with a real labelled validation run, which does not currently exist).

**M. Severity:** HIGH (checkpoint carries no self-attested label metadata; the live mapping rests on inherited/inferred evidence, not an in-file guarantee) — but not CRITICAL, since no evidence of an actual inversion was found anywhere the code path was traced.

**N. Exact next action:** Obtain a small labelled validation set (bonafide + spoof, multiple generators, ideally including another ElevenLabs sample) and run it through `scripts/validate_ssl_generalization_v2_inference.py` before making any mapping change — do not swap `SSL_CLASS_ORDER` without that evidence.
