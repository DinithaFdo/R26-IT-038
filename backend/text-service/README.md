# AI Text Detection API

## Overview

FastAPI backend for AI-generated text detection using DeBERTa-v3-large
Model 2, with an input sanitization layer and token-level explainability
(attention-based highlighting), plus a placeholder XAI/HNIF endpoint for
deeper interpretability.

## Project Structure

```
ai-detection-app/
│
├── app.py                     # FastAPI app entry point — lifespan, CORS, routers, uvicorn runner
├── requirements.txt           # Pinned Python dependencies
├── .env.example                # Template for required environment variables
├── .gitignore                 # Git exclusions (env files, models, caches, etc.)
├── README.md                  # This file
│
├── classification/            # Core AI-text classification service
│   ├── __init__.py            # Package marker
│   ├── router.py               # FastAPI router: POST /classify, GET /classify/health
│   ├── predictor.py            # Prediction + token-highlighting logic
│   ├── model_loader.py         # Singleton tokenizer/model/device loader
│   ├── sanitization.py         # Input sanitization / adversarial-attack detection
│   └── schemas.py              # (reserved for classification-specific schemas)
│
├── xai/                       # Explainability (XAI) service
│   ├── __init__.py            # Package marker
│   ├── router.py                # FastAPI router: POST /xai/explain (placeholder)
│   ├── schemas.py               # (reserved for xai-specific schemas)
│   └── hnif/                    # Hybrid Neural Interpretability Framework (teammate-owned, WIP)
│
└── shared/                    # Schemas/utilities shared across services
    ├── __init__.py            # Package marker
    └── schemas.py              # Pydantic models: ClassifyRequest/Response, TokenHighlight, SanitizationResult
```

## Setup

### Prerequisites

- Python 3.10+
- Google Drive access to the Model 2 folder

### Installation

1. Clone the repository
2. Create virtual environment: `python -m venv venv`
3. Activate:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and set `MODEL2_PATH` to your local Model 2 folder

### Running the API

```
uvicorn app:app --reload
```

### API Endpoints

| Method | Endpoint            | Description                        |
|--------|----------------------|-------------------------------------|
| GET    | `/`                  | API status                          |
| POST   | `/classify`          | Classify text as AI or Human        |
| GET    | `/classify/health`   | Health check                        |
| POST   | `/xai/explain`       | XAI explanation (coming soon)       |

### API Docs

After running the server: http://localhost:8000/docs

## Team

- Component 3 (Classification): IT22251046
- Component 2 (XAI/HNIF): IT22090744

## Model

DeBERTa-v3-large, domain-fix dataset.
Accuracy: 99.75%, AUROC: 0.99988
