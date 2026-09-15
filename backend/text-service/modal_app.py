import os
import sys
import logging
import modal

# Configure root logger to output INFO to Modal console
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = modal.App("ai-detection-backend")

# Define secrets and volume at top level
dockerhub_secret = modal.Secret.from_name("dockerhub-secret")
frontend_secret = modal.Secret.from_name("frontend")
models_volume = modal.Volume.from_name("ai-models")

image = (
    modal.Image.from_registry(
        "dinithafdo/ai-detection-backend:latest",
        secret=dockerhub_secret,
        force_build=True,
    )
    .pip_install("accelerate")  # <-- INSTALLS ACCELERATE ON THE FLY
    .env(
        {
            "MODEL2_PATH": "/app/models/branch1_deberta_large_model2_domainfix_final",
            "XGB_MODEL_PATH": "/app/models/branch2_lexical_model_v4_domainfix/model.pkl",
            "XGB_SCALER_PATH": "/app/models/branch2_lexical_model_v4_domainfix/scaler.pkl",
            "XGB_FEATURES_PATH": "/app/models/branch2_lexical_model_v4_domainfix/feature_names.pkl",
            "XGB_CALIBRATOR_PATH": "/app/models/branch2_lexical_model_v4_domainfix/isotonic_calibrator.pkl",
            "DEBERTA_TEMPERATURE_PATH": "/app/models/deberta_temperature.json",
            "CSS_MIN_WORD_COUNT": "150",
            "PYTHONPATH": "/app",
        }
    )
)

models_volume = modal.Volume.from_name("ai-models")


@app.function(
    image=image,
    gpu="L4",
    volumes={"/app/models": models_volume.with_mount_options(sub_path="/models")},
    secrets=[
        modal.Secret.from_name(
            "frontend"
        ),  # <-- INJECTS YOUR FRONTEND URL / CORS SECRET
    ],
    scaledown_window=180,
    timeout=600,
)
@modal.asgi_app()
def serve():
    sys.path.insert(0, "/app")

    from classification.model_loader import (
        load_models,
        is_css_available,
        is_qwen_available,
    )

    load_models()
    print(
        f"--> [MODAL STATUS] CSS-v2 / XGBoost available: {is_css_available()}",
        flush=True,
    )
    print(
        f"--> [MODAL STATUS] Qwen model available: {is_qwen_available()}",
        flush=True,
    )

    from app import app as fastapi_app

    return fastapi_app
