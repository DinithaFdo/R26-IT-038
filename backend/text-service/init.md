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
