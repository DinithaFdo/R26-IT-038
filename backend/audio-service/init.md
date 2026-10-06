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


