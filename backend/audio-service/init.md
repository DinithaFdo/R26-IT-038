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

