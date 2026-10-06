text + xai

# Multi-Modal Text Authenticity Detector - Classification Backend

## 1. System Overview
The text classification backend service provides high-throughput, adversarially robust classification of digital text into AI-Generated, Human-Written, or Mixed categories.

### 1.1 Module Ownership
- **Classification Backend**: Ishara (IT22251046) - Core semantic inference, stylometric feature extraction, CSS-v2 calibration, and input sanitization.
- **Explainability (XAI)**: Dinitha - HNIF feature attribution and interpretability reporting.

### 1.2 Core Runtime Stack
- **API Framework**: FastAPI with ASGI Uvicorn server
- **Deep Learning Engine**: PyTorch (CUDA L4 GPU / CPU fallback)
- **Transformer Library**: HuggingFace Transformers (DeBERTa-v3-large architecture)
- **Stylometric Engine**: XGBoost Classifier with Scikit-learn preprocessing
- **Explainability**: SHAP (TreeExplainer) for stylometric feature contribution
- **NLP Pipeline**: spaCy (`en_core_web_sm`) for syntactic and lexical dependency parsing

### 1.3 Application Lifespan and Routing
The service is instantiated via `app.py`, which configures FastAPI lifespan handlers to load heavy neural network models into memory during startup and mount the `/classify` and `/xai` sub-routers.

## 2. Request Processing Pipeline
```
POST /classify {text}
  │
  ├── [1] Sanitization Layer: Cleans text across 10 adversarial evasion categories
