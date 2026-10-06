# Audio Service Initialization & ESVAS Technical Specification

# Explainable Spoof Voice Analysis System (ESVAS)

## 1. System Overview

ESVAS is a non-invasive analytical explainability layer for deepfake voice detection.

### Research Gaps Addressed
- **G3 (Temporal Localization):** Pinpoints manipulated segments via Attention Rollout.
- **G6 (Faithfulness Validation):** Proves causal accuracy using AOPC metric.
- **G10 (Acoustic Interpretability):** Applies SHAP on physical features rather than raw pixels.


### Core Principles
1. **Non-invasive:** Uses PyTorch hooks without altering classifier weights.
2. **Unidirectional:** Explanations cannot alter prediction outcomes.
3. **Modular:** Independent execution phases.


### Dual-View Explanation Concept
- **View A (Temporal):** Identifies *where* spoofing occurs.
- **View B (Semantic):** Identifies *what* acoustic features indicate synthetic origin.


```
[Audio Input] -> [Multi-Branch Classifier] -> [Prediction Output]
                         |
            (PyTorch Hooks / Immutable Read)
                         v
                    [ESVAS Layer] -> [Dual-View Report]
```


### Decoupling Rules
- Phase 1 (Extraction) runs inline during inference pass.
- Phase 2 (Rollout) & Phase 3 (SHAP) execute asynchronously in background task.
- Phase 4 (Synthesis) merges outputs into unified schema.


### Execution Requirements
- **Python:** 3.11+
- **PyTorch:** 2.x
- **Device:** CPU / CUDA / MPS (Default: CPU for small model efficiency)


### Component Interaction
- `HookManager`: Intercepts branch activations.
- `AttentionRolloutEngine`: Calculates temporal heatmaps.
- `SemanticXGBExplainer`: Calculates SHAP attributions.
- `SynthesisEvaluator`: Evaluates IoU & AOPC consistency.


### Target Audience & Persona
- **Forensic Auditors:** Detailed mathematical attributions and raw tensors.
- **End Users:** Clear natural language text summary with visual heatmaps.


### Ethical Boundaries
- Operates strictly in detection/explanation mode.
- Zero synthesis or voice modification capabilities.
- Privacy preserved: No PII stored; transient audio lifecycle enforced.


## 2. Phase 1 — Feature & Attention Extraction

### PyTorch Forward Hooks
- Registered via `module.register_forward_hook(hook_fn)`.
- Captures intermediate branch hidden states and attention matrices.

### LFCC Feature Extraction
- 19 linear-frequency cepstral coefficients.
- Extracted per 20ms frame with 10ms overlap.

### Pitch (F0) Analysis
- Fundamental frequency F0 mean and variance tracked per frame.
- Captures pitch monotonicity characteristic of neural vocoders.

### Micro-prosodic Jitter
- Period-to-period variability of fundamental frequency.
- Quantifies subtle vocal cord perturbation anomalies.

### Shimmer Analysis
- Frame-to-frame amplitude variation coefficient.
- Measures glottal flow dynamics stability.

### Harmonics-to-Noise Ratio (HNR)
- Evaluates acoustic signal purity in dB.
- Highlights high-frequency spectral phase distortion.

### REAPER GCI Irregularity
- Epoch-based Glottal Closure Instant detection.
- Measures interval jitter between consecutive vocal fold closures.

### Memory Buffer Management
- Tensors stored in process-isolated thread-local storage.
- Automatically purged after Phase 2 & 3 completion.

### Attention Weight Interception
- Intercepts self-attention weights from Transformer/Conformer encoder blocks.
- Preserves multi-head tensor shapes: `[batch, heads, seq_len, seq_len]`.

### Windowing Parameters
- Window Size: 20 ms (320 samples at 16 kHz)
- Hop Length: 10 ms (160 samples at 16 kHz)
- Window Function: Hann window

### Buffer Cleanup Protocol
- `HookManager.clear_buffers()` called in `finally` block.
- Explicit `torch.cuda.empty_cache()` or Python `gc.collect()` invocation.

### Feature Normalization
- Zero-mean unit-variance per utterance over non-padded audio frames.
- Prevents silence padding from distorting feature variance.

### Thread Safety
- Context-local dictionary keyed by `prediction_id`.
- Ensures multi-threaded concurrent predictions do not leak hook data.

### REAPER Fallback
- If REAPER binary fails or returns empty GCI, fall back to Parselmouth/Praat pitch bounds.
- Log warning and set `gci_reliability_flag = false`.

### Extraction Latency Budget
- Maximum allowed overhead: < 80ms for 6.0s audio clip.
- Efficient C-extensions used for LFCC & Praat calls.

## 3. Phase 2 — Temporal Attention Rollout (ARS)

### Mathematical Formulation
```
R_l = A_hat_l * R_{l-1}
```
Recursive multiplication over sequence of layers $l=1 \dots L$.

### Identity Matrix Integration
```
A_hat_l = 0.5 * A_l + 0.5 * I
```
Accounts for residual connections by adding identity matrix $I$.

### Layer Normalization
- Row-normalization ensures each row of $\hat{A}_l$ sums to 1.0.
- Prevents exploding attention rollout weights across deep layers.

### XLSR-53 Integration
- First documented application of Attention Rollout to Wav2Vec2-XLSR backbone.
- Tracks attention flow from raw audio frames to latent speech representations.

### Attention Rollout Spectrogram (ARS)
- 1D temporal density map aligned with audio time axis.
- Maps model attention weight focus per 20ms frame.

### Anomaly Candidate Thresholding
- Top 20% highest attention values (80th percentile) marked as candidate fake regions.
- Output format: list of time interval tuples `[(t_start, t_end)]`.

### Frame-to-Time Mapping
```
timestamp_seconds = frame_index * hop_length_samples / sample_rate
```
Ensures millisecond-level precision in frontend player alignment.

### Edge-Smoothing Filter
- 1D Gaussian filter ($\\sigma=1.5$) applied to ARS density vector.
- Removes single-frame noise spikes and smooths region transitions.

### Multi-Head Aggregation
- Options: Mean over heads, Max over heads, or Attention-weighted head selection.
- Default: Mean over heads for stable representation.

### Overlapping Window Fusion
- 6.0s windows with 1.0s overlap.
- Cosine taper applied at window boundaries prior to stitching ARS segments.

### PartialSpoof Calibration
- Calibrated using PartialSpoof v1.2 DEV set.
- Fixed threshold JSON: `partialspoof_v1_2_dev_attention_threshold.json`.

### Short Utterance Handling
- Minimum duration: 1.0s.
- Right-zero padding used; rollout density zeroed out for padded positions.

### Export Formats
- JSON: `{ "timestamps": [...], "densities": [...], "anomalies": [...] }`
- Binary: Compact Float32 array for frontend WebGL canvas rendering.

### Bounding Box Generation
- Merges contiguous frame intervals above threshold separated by < 50ms.
- Assigns candidate severity score based on mean rollout density.

### Noise Floor Filter
- Subtracts baseline ambient noise attention profile.
- Prevents silent audio gaps from triggering false positive attention regions.

### Single-Layer Fallback
- If full depth rollout fails, extract raw last-layer cross-attention.
- Flag report with `rollout_mode = 'single_layer_fallback'`.

### Phase 2 Performance
- Computation time: ~45ms for 6s audio clip on CPU.
- Peak RAM overhead: < 12 MB per prediction.

## 4. Phase 3 — Semantic Explanation (XGBoost-SHAP)

### XGBoost Surrogate Model
- Trained to mimic primary classifier decisions on physical acoustic features.
- Model version: `v4_semantic_xgboost`.

### SHAP TreeExplainer Integration
- Fast exact SHAP calculation for tree ensembles.
- Computes Shapley values for each extracted acoustic feature.

### Attribution Direction
- **Positive SHAP value (> 0):** Pushes prediction towards **Spoof**.
- **Negative SHAP value (< 0):** Pushes prediction towards **Bonafide**.

### 148 Acoustic Feature Vector
- 19 LFCCs x (mean, std, delta, delta-delta) = 76
- Spectral Centroid, Bandwidth, Contrast, Rolloff = 32
- F0, Jitter, Shimmer, HNR, GCI stats = 40

### Feature Importance Ranking
- Top-N features ranked by absolute SHAP value $|\\phi_i|$.
- Identifies dominant physical contributors to spoofing decision.

### Beeswarm Plot Data Structure
- Formats global feature impact & feature value intensity (high/low).
- Serves summary view across evaluation datasets.

### Waterfall Plot Schema
- Base value (expected model output) $E[f(x)]$.
- Stepwise contribution of top 10 acoustic features to final logit.

### Artifact Integrity Verification
- SHA-256 checksum verification for model weights (`.json`), column mapping (`.json`), and baseline (`.pkl`).
- Prevents silently running mismatched surrogate models.

### Feature Column Alignment
- Strict validation of input dataframe column names and order against `v4_column_order.json`.
- Raises `ValueError` on column mismatch.

### Acoustic Property Categories
- **Spectral:** LFCC coefficients, spectral slope, tilt.
- **Prosodic:** F0 trajectory, duration, energy contours.
- **Glottal:** GCI jitter, shimmer, HNR, glottal pulse shape.

### Background Dataset Baseline
- 500 representative bonafide audio samples from ASVspoof 2019 LA training set.
- Pre-computed and cached in `background_baseline.pkl`.

### Top-K Thresholding
- Extracts top 5 positive (spoof-inducing) and top 5 negative (bonafide-inducing) features.
- Ignores features with $|\\phi_i| < 0.01$.

### Collinearity Management
- TreeExplainer naturally distributes attribution among correlated features.
- Grouped category attribution aggregates SHAP values per feature family.

### SHAP JSON Schema
```json
{
  "base_value": 0.12,
  "prediction": 0.94,
  "top_features": [
    {"name": "lfcc_coef_2_std", "shap_value": +0.31, "value": 4.12, "category": "spectral"}
  ]
}
```

### Natural Language Narrative
- Template engine converts top SHAP attributions into human-readable sentences:
  *"High LFCC spectral variation and unnatural F0 stability strongly indicate synthetic vocoder origin."*

### Surrogate Model Fidelity
- R^2 score between surrogate output and primary ensemble score.
- Fidelity target: R^2 > 0.91 on evaluation benchmark.

### Explainer Caching
- `shap.TreeExplainer` instance initialized once at startup.
- Retained in memory to avoid ~300ms explainer creation latency per request.

### Phase 3 Summary
- Total execution time: ~65ms.
- Returns structured semantic attribution dictionary.

## 5. Phase 4 — Unified Synthesis & Faithfulness Evaluation

### Cross-Modal Consistency (IoU)
- Computes Intersection over Union (IoU) between ARS high-attention regions and temporal windows of top SHAP acoustic features.

### Consistency Rule
```
IoU(ARS_anomalies, SHAP_feature_windows) > 0.75 => Consistent (Status: Verified)
```
Flagged as highly trustworthy dual-view explanation.

### AOPC Metric Formula
```
AOPC = (1 / (L + 1)) * sum_{k=0}^L [ f(x) - f(x \ \Omega_l^k) ]
```
Measures prediction degradation when top explanation segments $\\Omega_l^k$ are masked.

### Perturbation Strategy
- Replaces top-ranked 20ms audio frames with zero-padded silence or shaped background noise.
- Evaluates classifier output drop $f(x) - f(x')$.

### Faithfulness Validation
- High AOPC score confirms explanation features are causally responsible for classification.
- Low AOPC triggers warning: `Unverified Explanation (Potential Artifact)`.

### Unified Report Structure
1. **Executive Summary:** Verdict & Faithfulness score.
2. **Temporal Map:** ARS timeline with flagged timestamps.
3. **Acoustic Breakdown:** Top SHAP features with physical descriptions.
4. **Audit Metrics:** IoU and AOPC quantitative scores.

### Explanation Confidence Index (ECI)
```
ECI = 0.5 * IoU_score + 0.5 * Normalized_AOPC
```
Ranges from 0.0 (unreliable) to 1.0 (highly faithful).

### Conflict Resolution Rules
- If IoU < 0.40 (Temporal & Semantic disagree):
  * Rely on AOPC score to determine which view is causally superior.
  * Annotate report with explicit conflict advisory.

### Summary Generator Rules
- Combines ECI score, top time window, and primary acoustic category into a 3-sentence summary for non-technical users.

### Export Specs
- JSON: Full telemetry and raw arrays.
- HTML/PDF: Formatted report with embedded vector charts.

### AOPC Step Sizes
- Steps $L = 5, 10, 15, 20$ frames perturbed iteratively.
- Step size: 20ms per iteration.

### Ground-Truth Comparison
- Compares predicted ARS anomaly regions against FakeSound2 segment ground truth annotations.
- Measures precision, recall, and frame-level F1 score.

### Faithfulness Levels
- **High:** ECI >= 0.75
- **Medium:** 0.50 <= ECI < 0.75
- **Low:** ECI < 0.50

### Rollout Unavailable Fallback
- Generates semantic-only explanation report.
- Explicitly notes missing temporal view due to branch config.

### Phase 4 Summary
- Completes ESVAS pipeline execution.
- Outputs audit-compliant explanation document.

## 6. Evaluation Datasets & Setup

### ASVspoof 2019 LA
- ~121,000 utterances across logical access attacks (TTS & VC).
- Used for training and validating the XGBoost surrogate model.

### ASVspoof 2021 LA
- ~180,000 utterances including telephony codec and transmission channel variations.
- Tests ESVAS robustness under bandpass filtering and lossy compression.

### FakeSound2 Benchmark
- ~30,000 audio clips with precise segment-level temporal manipulation timestamps.
- Used as primary ground truth for IoU and ARS temporal localization accuracy.

### PartialSpoof v1.2 DEV
- Partially spoofed audio files with mixed bonafide/spoof segments.
- Calibrates attention rollout detection thresholds.

### Data Splitting Strategy
- Train: 70% of ASVspoof 2019 LA train set.
- Validation: 15% for hyperparameter tuning.
- Test: 15% evaluation split (unseen speakers).

### Dataset Preprocessing Protocol
- Audio downsampled to 16 kHz mono.
- Peak normalization to -1 dBFS.
- Silence trimming: leading/trailing silence < -40 dB removed.

### Augmentation Protocols
- Additive Gaussian noise (SNR 10dB to 30dB).
- RIR (Room Impulse Response) reverberation convolution.
- Evaluates explanation stability under noisy conditions.

### Class Imbalance Handling
- Scale_pos_weight parameter tuned in XGBoost.
- Stratified K-Fold cross-validation (K=5).

### License & Compliance
- Academic research license compliance for ASVspoof & FakeSound2.
- No commercial redistribution of raw audio samples.

### Cross-Dataset Generalization
- Surrogate trained on ASVspoof 2019 evaluated directly on In-the-Wild dataset.
- Measures feature attribution shift across out-of-domain samples.

### Audio Standardization
- Uniform 16,000 Hz, 16-bit PCM mono format required before feature extraction.

### Metadata Schema
- Tracks dataset source, speaker ID, attack algorithm, and channel codec for each test sample.

### Benchmark Summary Matrix
| Dataset | Role | Metric |
|---|---|---|
| ASVspoof 2019 | Surrogate Train | R^2 > 0.91 |
| FakeSound2 | ARS Eval | IoU > 0.75 |
| PartialSpoof | Calibration | EER / Threshold |


## 7. API Integration & Service Architecture

### Endpoint Definition
`GET /api/v1/me/predictions/{prediction_id}/explanation`
- Returns dual-view explanation object for prediction owner.

### Async Background Execution
- ESVAS processing runs in background worker task after prediction persistence.
- Audio retained temporarily during XAI execution pass.

### Payload Validation
- Validates `prediction_id` format (MongoDB ObjectId).
- Ensures prediction status is `completed` before returning explanation.

### Access Control Scopes
- Enforces Clerk authentication user ID matching prediction owner ID.
- 403 Forbidden returned on unauthorized attempts.

### Transient Audio Lifecycle
- Raw PCM audio buffer retained in memory during XAI run.
- Automatically deleted upon completion of Phase 4 synthesis.

