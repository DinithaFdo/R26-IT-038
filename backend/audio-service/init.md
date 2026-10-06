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


