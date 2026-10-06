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
