text + xai# Explainable AI (XAI) Service Documentation & Implementation Tracker
## 1. Overview
Provides deep-learning interpretability by extracting causal mathematical weights.
- Architecture: Follows strict Domain-Driven Design (DDD) decoupling validation, extraction, and filtering.
## 2. Core Schemas & Validation (schemas.py)
Defines strict Pydantic schemas for request validation and serialization.
- XAIRequest: Input contract wrapping incoming text string for explainability processing.
- TokenScore: Schema capturing token string, raw score, normalized score, and noise flag.
- XAIResponse: Output contract containing predicted_class, confidence, heatmap_data, and interpretability_report.
- Swagger UI: Auto-generated OpenAPI v3 schemas ensure seamless API contract verification.
## 3. Mathematical Extraction Engine (core_extractor.py)
Extracts transformer attention and gradient attributions.
- Attention Extraction: Captures multi-head attention weights from the final transformer encoder layer.
- Attention Aggregation: Averages attention heads across tokens to produce a unified 1D vector.
- Integrated Gradients: Utilizes Captum LayerIntegratedGradients targeting the word embedding layer.
- Baseline Strategy: Uses zero/pad token embedding tensor baselines for path-integral calculation.
- Riemann Approximation: Computes m-step path integrals (default n_steps=50) for numerical stability.
- Vector Norm: Applies Euclidean L2-norm across hidden dimensions to convert embedding gradients to scalar token scores.
- Hybrid Fusion Equation: Score = Final Layer Base Attention * Captum Integrated Gradients.
- Target Class Attribution: Selects the predicted class logit index as the target for gradient backpropagation.
- Attribution Clamping: Evaluates positive causal attribution to prevent negative cancellation.
- Min-Max Normalization: Rescales attribution scores into [0.0, 1.0] interval for heatmap rendering.
- Subword Reconstruction: Handles subword tokens by matching offsets to full lexical tokens.
## 4. NLP & Linguistic Filtering (linguistic_filter.py)
Applies linguistic and part-of-speech filtering to eliminate non-causal grammatical noise.
- spaCy Model: Uses en_core_web_sm pipeline with parser and tagger enabled.
- POS Filter (PUNCT): Marks punctuation tokens (commas, periods, quotation marks) as noise.
- POS Filter (CCONJ): Filters coordinating conjunctions (and, but, or) to prevent spurious importance.
- POS Filter (DET): Masks definite and indefinite articles (the, a, an) as non-causal tokens.
- POS Filter (ADP): Filters prepositions and adpositions (in, to, for, with).
- POS Filter (SPACE): Identifies and suppresses newline and space tokens from heatmap consideration.
- POS Filter (PART): Eliminates grammatical particles (e.g., possessive markers, negative particles).
- Score Suppression: Sets normalized_score = 0.0 and is_noise = True for all filtered tokens.
- Subword Merging: Strips subword artifacts (e.g. ## or byte-pair markers) before linguistic analysis.
- Semantic Saliency: Preserves content words (NOUN, VERB, ADJ, ADV) to isolate genuine causal triggers.
## 5. Agentic LLM Translation Layer (llm_translator.py)
Translates raw attribution tensors into human-readable forensic reports.
- LLM Provider: Integrates Google Gemini API via official SDK client.
- Configuration: Reads GEMINI_API_KEY from environment variables with graceful fallback validation.
- Top-K Token Selection: Extracts the top-5 highest-scoring non-noise tokens to construct the prompt.
- System Prompt: Instructs the model to act as a forensic AI interpretability auditor.
- Anti-Hallucination: Strictly constrains generated explanations to provided salient tokens.
- Report Length Constraint: Enforces a concise 3-sentence summary targeting non-technical stakeholders.
- Fault Tolerance: Implements template-based deterministic fallback when API limits or errors occur.
- Generation Hyperparameters: Configures low temperature (0.2) and top_p for deterministic auditing.
## 6. Service Orchestration Pipeline (service.py)
Coordinates end-to-end execution of classification, extraction, and explanation.
- State Extraction: Accesses app.state.model and app.state.tokenizer without redundant reloads.
