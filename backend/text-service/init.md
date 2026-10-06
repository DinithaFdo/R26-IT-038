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
  ├── [2] Semantic Inference: DeBERTa-v3-large predicts probabilities (prob_ai, prob_human)
  ├── [3] Token Highlighting: Computes attribution scores for UI visual emphasis (optional)
  ├── [4] Stylometric Branch: Extracts 28 linguistic features and computes calibrated probability
  ├── [5] CSS-v2 Engine: Evaluates branch disagreement score (C) and conflict category
  └── [6] Final Label Resolver: Assigns final category (AI, Human, or Mixed) with directional lean
```

## 3. API Endpoints
### 3.1 Service Discovery
- `GET /`: Returns service identity and Swagger documentation link (`{"message": "AI Detection API running", "docs": "/docs"}`).

### 3.2 Classification Endpoint
- `POST /classify`: Primary entrypoint executing full multi-modal text classification and conflict analysis.

### 3.3 Classification Health Probe
- `GET /classify/health`: Health and readiness probe verifying model availability in memory.

### 3.4 Explainability Audit Endpoint
- `POST /xai/audit`: Dual-pipeline endpoint running classification combined with HNIF token-level heatmap and LLM interpretability synthesis.

### 3.5 Interactive Documentation
- `GET /docs`: Auto-generated OpenAPI (Swagger UI) providing schema validation and interactive request debugging.

### 3.6 Error Handling Strategy
Exceptions encountered during downstream prediction are caught and re-raised as `HTTPException(status_code=500, detail=str(exc))` to preserve stack-trace diagnostics in logging.

## 4. Data Contracts & Schemas
### 4.1 Request Contract
`ClassifyRequest` requires:
- `text`: non-empty string (`min_length=1`).

### 4.2 ClassifyResponse Fields
- `label`: Raw semantic prediction (`"AI-Generated"` | `"Human-Written"`).
- `prob_ai` & `prob_human`: Softmax confidence probabilities computed by the semantic transformer branch.
- `sanitization`: Metadata object containing `was_attacked` (boolean), `attack_report` (list of strings), and `clean_text`.
- `token_highlights`: Sequence of highlighted tokens with relevance score (0.0 to 1.0) and discretized intensity (`high`, `mid`, `low`).
- `model_version`: Identifier string designating semantic backbone (`deberta-v3-large-model2-domainfix`).
- `processing_time_ms`: End-to-end execution latency in milliseconds for performance telemetry.
- `signal_analysis`: Detailed diagnostics object holding stylometric signals, CSS score, and SHAP feature contributions.
- `final_label`: Resolved multi-class categorization (`"AI-Generated"`, `"Human-Written"`, or `"Mixed"`).
- `leans_toward`: Directional indication (`"AI-Generated"` | `"Human-Written"`) populated when `final_label` evaluates to `Mixed`.

## 5. Adversarial Input Sanitization Layer
The sanitization layer cleans input strings prior to classification across ten sequential defensive stages.

- **Stage 1 (Null & Control Chars)**: `_remove_null_bytes` strips null bytes (`\x00`) and unprintable ASCII control characters.
- **Stage 2 (Zero-Width Characters)**: `_remove_zero_width_chars` purges zero-width spaces, joiners, non-joiners, and invisible delimiters (`\u200b`, `\u200c`, `\u200d`, `\ufeff`).
- **Stage 3 (Unicode Normalization)**: `_normalize_unicode` applies NFKC decomposition to standardize mathematical monospace, bold, and italic unicode alphabets.
- **Stage 4 (Homoglyph Normalization)**: `_normalize_homoglyphs` resolves Cyrillic, Greek, and fullwidth look-alike characters to standard Latin equivalents via `HOMOGLYPH_MAP`.
- **Stage 5 (HTML Entity Decoding)**: `_remove_html_entities` translates entity sequences (`&amp;`, `&lt;`, `&#x20;`) back to raw characters.
- **Stage 6 (URL & Email Redaction)**: `_remove_urls` redacts web links and email addresses to avoid out-of-vocabulary token distortion.
