# Explainable AI (XAI) Module Architecture

This document provides a comprehensive overview of the **Hybrid Neural Interpretability Framework (HNIF)** implemented within the `xai` module.

The module provides deep-learning interpretability by extracting causal mathematical weights from Transformer models and translating them into human-readable forensic audits.

---

# 🏗️ Architectural Overview

The XAI module follows strict **Domain-Driven Design (DDD)** principles, ensuring that validation, mathematical extraction, NLP filtering, and external API integrations are fully decoupled.

## Module Structure

### `schemas.py`

Defines strict **Pydantic data contracts** for incoming requests and outgoing API responses.

Responsibilities:

- Request validation
- Response serialization
- Automatic Swagger UI documentation generation

---

### `core_extractor.py`

The intrinsic extraction engine of the XAI framework.

Implements **Hybrid Fusion**, which calculates:

```
Final Layer Base Attention × Captum Integrated Gradients
```

Targeting the model's word embedding layer to identify causal token importance.

Responsibilities:

- Extract transformer attention weights
- Generate Integrated Gradients using Captum
- Perform mathematical fusion of interpretability signals
- Produce raw token importance scores

---

### `linguistic_filter.py`

Applies Natural Language Processing (NLP) using **spaCy**.

Responsibilities:

- Remove transformer subword artifacts
- Identify grammatical noise
- Mask non-semantic tokens

The following POS categories are filtered and assigned a score of `0.0`:

```
PUNCT
CCONJ
DET
ADP
SPACE
PART
```

This isolates the true semantic causal triggers behind model predictions.

---

### `llm_translator.py`

The Agentic LLM integration layer.

Uses the **Google Gemini API** to translate filtered interpretability tensors into a human-readable forensic explanation.

Responsibilities:

- Convert mathematical outputs into natural language
- Prevent hallucination through strict prompting
- Generate a concise 3-sentence audit report

Output target:

- Non-technical users
- Human-readable explanations
- Transparent AI decision auditing

---

### `service.py`

The orchestration layer of the XAI pipeline.

Responsibilities:

1. Receive the shared model from FastAPI application state
2. Execute classification
3. Extract interpretability weights
4. Apply linguistic filtering
5. Generate LLM explanation

Pipeline:

```
Classification
      ↓
Feature Extraction
      ↓
Linguistic Filtering
      ↓
LLM Translation
      ↓
Final XAI Audit Response
```

---

### `router.py`

FastAPI endpoint definition.

Endpoint:

```
POST /xai/audit
```

Responsibilities:

- Dependency injection
- Request handling
- Prevent redundant model loading
- Avoid memory duplication

---

# ⚙️ Integration Details

The XAI module is designed to run alongside the core classification module without duplicating model memory overhead.

It uses **FastAPI Dependency Injection** through `app.state`.

The main application (`app.py`) loads the model and tokenizer once and attaches them globally.

Example:

```python
# app.py

from classification.model_loader import load_models
import classification.model_loader as model_loader

# Attach to global state during startup lifespan

app.state.model = getattr(
    model_loader,
    'model',
    None
)

app.state.tokenizer = getattr(
    model_loader,
    'tokenizer',
    None
)
```

Benefits:

- Single model instance in memory
- Faster inference
- Reduced GPU/CPU memory consumption
- Cleaner module separation

---

# 🐳 Docker Deployment & Model Management

To keep the Docker image lightweight and deployment scalable, machine learning model weights are **not baked into the Docker image**.

Instead, the system uses a **Docker Volume Mount strategy** through `docker-compose`.

---

## Folder Setup

The project structure should contain a separate `models` directory:

```
ai-detection-app/
│
├── models/                     <-- Excluded from Git and Docker
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
│
├── docker-compose.yml
│
├── Dockerfile
│
└── app.py
```

The `models` directory should be excluded using:

- `.gitignore`
- `.dockerignore`

---

# Running the System

Start the fully containerized application using Docker Compose:

```bash
docker-compose up --build -d
```

The compose configuration automatically mounts:

```
Local:
./models

Container:
 /app/models
```

with read-only permission:

```
:ro
```

This allows the API to load model weights while preventing accidental modification or corruption.

---

# 🚀 API Usage

## Endpoint

```
POST /xai/audit
```

---

## Request Payload

```json
{
  "text": "The algorithmic progression of artificial intelligence systems demonstrates a remarkable capacity for generating syntactically uniform prose."
}
```

---

## Response Payload

The endpoint returns an `XAIResponse` containing:

- Original classification result
- Confidence score
- Normalized token heatmap
- LLM-generated interpretability report

Example:

```json
{
  "status": "success",
  "predicted_class": "AI-Generated",
  "confidence": 98.5,
  "heatmap_data": [
    {
      "original_token": "algorithmic",
      "display_token": "algorithmic",
      "is_noise": false,
      "raw_score": 0.082,
      "normalized_score": 1.0
    }
  ],
  "interpretability_report": "The model has been classified as AI-generated with a confidence of 98.50%. The highest impact tokens driving this decision were 'algorithmic', 'progression', and 'syntactically'. The network's decision was primarily driven by these specific causal triggers indicating uniform prose structure."
}
```

---

# Summary

The XAI module provides a complete interpretability pipeline by combining:

- Transformer attention analysis
- Captum Integrated Gradients
- NLP-based semantic filtering
- LLM-powered explanation generation
- Docker-based scalable deployment

The Hybrid Neural Interpretability Framework enables transparent, explainable, and human-readable AI decision auditing.