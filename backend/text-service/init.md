text + xai# Explainable AI (XAI) Service Documentation & Implementation Tracker
## 1. Overview
Provides deep-learning interpretability by extracting causal mathematical weights.
- Architecture: Follows strict Domain-Driven Design (DDD) decoupling validation, extraction, and filtering.
## 2. Core Schemas & Validation (schemas.py)
Defines strict Pydantic schemas for request validation and serialization.
- XAIRequest: Input contract wrapping incoming text string for explainability processing.
