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
- **Stage 7 (Structural Redaction)**: `_remove_structural_patterns` cleans markdown artifacts, Reddit quote syntax, and conversational headers.
- **Stage 8 (Punctuation Normalization)**: `_normalize_quotes_and_dashes` standardizes curly smart quotes, em-dashes, and en-dashes to standard ASCII delimiters.
- **Stage 9 (Repeated Punctuation)**: `_normalize_repeated_punctuation` condenses multi-exclamations (`!!!!`) and multi-periods (`....`) to bounded representations.
- **Stage 10 (Whitespace Normalization)**: `_normalize_whitespace` compresses horizontal tabs and spaces, limits consecutive newlines to two, and strips outer whitespace.

### 5.1 Sanitization Audit Output
The sanitizer returns `{clean_text, attacks_detected, attack_report, original_length, clean_length}` which is passed down to logging and response schemas.

> **Security Boundary Notice**: The sanitization layer mitigates known heuristic and visual spoofing attacks; it does not offer cryptographic robustness against arbitrary adversarial perturbations.

## 6. Semantic Branch (DeBERTa-v3-large)
### 6.1 Model Architecture
The semantic classification branch utilizes `DeBERTa-v3-large` fine-tuned under the Model 2 'domainfix' setup for cross-domain text authenticity detection.

### 6.2 Model Loading Strategy
`model_loader.load_models()` handles eager loading of `AutoModelForSequenceClassification` and `AutoTokenizer` into memory during application startup.
- Model weights are resolved from the `MODEL2_PATH` environment variable (default: `./models/branch1_deberta_large_model2_domainfix_final`).
- `_ensure_model_directory` performs fail-fast validation, verifying that the target model path exists and contains required weights before server binding.

### 6.3 Tokenizer Configuration
- `MAX_LENGTH = 512` with standard truncation and batch padding for tensor alignment.
- PyTorch 2.6+ compatibility patch: `torch.load` is wrapped with `weights_only=False` to safely load custom model checkpoints.
- Inferences execute inside `torch.no_grad()` contexts on device resolved via `get_device()` (CUDA L4 GPU in production, CPU fallback locally).

### 6.4 Classification Decision Boundary
- If `config.id2label` is not present, index defaults are pinned to `0 = Human-Written` and `1 = AI-Generated`.
- Raw binary decision logic: `label = "AI-Generated" if prob_ai >= 0.5 else "Human-Written"`.
- Temperature scaling constant `T = 1.7084` (loaded from `DEBERTA_TEMPERATURE_PATH`) is applied strictly to calibrate probabilities for CSS calculation without altering the raw label.

### 6.5 Token Highlighting Extraction
- Subword token attribution scores are extracted and mapped into word-level highlights with `high`, `mid`, and `low` saliency tiers.

### 6.6 Semantic Branch Empirical Benchmarks
- **In-Distribution Performance**: Achieves 99.75% accuracy with an AUROC of 0.99988 on test splits.
- **Out-of-Distribution Generalization**: 87.9% average across external datasets (M4GT: 78.7%, Ghostbuster: 87.6%, CHEAT: 97.3%).

## 7. Stylometric Branch (XGBoost)
### 7.1 Architecture & Role
The stylometric branch evaluates structural writing patterns, lexical variety, and syntactic distributions using an XGBoost gradient-boosted decision tree.

### 7.2 Artifact Dependencies
Requires four coordinated serialized artifacts:
1. XGBoost model binary (`XGB_MODEL_PATH`)
2. StandardScaler feature scaler (`XGB_SCALER_PATH`)
3. Feature names list (`XGB_FEATURES_PATH`)
4. Isotonic probability calibrator (`XGB_CALIBRATOR_PATH`)
