import os
import logging
import json
import joblib

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger("uvicorn.error")

load_dotenv()

# Patch torch.load for PyTorch 2.6+ compatibility
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

MODEL2_PATH = os.getenv(
    "MODEL2_PATH", "./models/branch1_deberta_large_model2_domainfix_final"
)

QWEN_PATH = os.getenv("QWEN_MODEL_PATH", "./models/qwen-0.5b")

MAX_LENGTH = 512


def _ensure_model_directory(model_path: str):
    """Fail fast with a clear error if the mounted model directory is missing."""
    if not model_path:
        raise RuntimeError(
            "MODEL2_PATH is not configured. Set MODEL2_PATH to the mounted model directory."
        )

    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model directory not found: '{model_path}'. "
            "Mount your models folder to /app/models or set MODEL2_PATH to the correct path."
        )


# Validate the configured model path early so the startup error is explicit.
_ensure_model_directory(MODEL2_PATH)

_tokenizer = None
_model = None
_device = None

# Qwen model globals (for interpretability generation)
_qwen_tokenizer = None
_qwen_model = None

# CSS-v2 globals
_xgb_model = None
_xgb_scaler = None
_xgb_feature_names = None
_xgb_calibrator = None
_deberta_temperature = None
_css_min_word_count = 150


def load_models():
    """Load the tokenizer, model, device, and Qwen once. Call this at app startup."""
    global _tokenizer, _model, _device, _qwen_tokenizer, _qwen_model

    if _model is not None and _tokenizer is not None:
        return

    _ensure_model_directory(MODEL2_PATH)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "CUDA available: %s | selected device: %s", torch.cuda.is_available(), _device
    )
    logger.info(f"Loading Model 2 from: {MODEL2_PATH}")

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL2_PATH,
        local_files_only=True,
    )

    _model = AutoModelForSequenceClassification.from_pretrained(
        MODEL2_PATH,
        local_files_only=True,
        output_attentions=True,
    )
    _model.to(_device)
    _model.eval()

    # ── Load Qwen model for interpretability generation ──────────────────────────
    logger.info(f"Loading Qwen model from: {QWEN_PATH}")
    if os.path.isdir(QWEN_PATH):
        try:
            _qwen_tokenizer = AutoTokenizer.from_pretrained(
                QWEN_PATH,
                local_files_only=True,
            )
            _qwen_model = AutoModelForSequenceClassification.from_pretrained(
                QWEN_PATH,
                local_files_only=True,
                device_map="auto",
                torch_dtype=torch.float32,
            )
            _qwen_model.eval()
            logger.info(
                "Qwen model loaded successfully and ready for interpretability."
            )
        except Exception as e:
            logger.warning(
                f"Qwen model failed to load: {e}. Interpretability will be limited."
            )
    else:
        logger.warning(
            f"Qwen model path not found: {QWEN_PATH}. Interpretability will be limited."
        )

    # ── CSS-v2: load XGBoost artifacts ──────────────────────────
    global _xgb_model, _xgb_scaler, _xgb_feature_names
    global _xgb_calibrator, _deberta_temperature, _css_min_word_count

    xgb_model_path = os.getenv("XGB_MODEL_PATH")
    xgb_scaler_path = os.getenv("XGB_SCALER_PATH")
    xgb_features_path = os.getenv("XGB_FEATURES_PATH")
    xgb_calibrator_path = os.getenv("XGB_CALIBRATOR_PATH")
    temperature_path = os.getenv("DEBERTA_TEMPERATURE_PATH")

    if all(
        [
            xgb_model_path,
            xgb_scaler_path,
            xgb_features_path,
            xgb_calibrator_path,
            temperature_path,
        ]
    ):
        try:
            _xgb_model = joblib.load(xgb_model_path)
            logger.info("XGBoost model loaded successfully from: %s", xgb_model_path)
            _xgb_scaler = joblib.load(xgb_scaler_path)
            logger.info("XGBoost scaler loaded successfully from: %s", xgb_scaler_path)
            _xgb_feature_names = joblib.load(xgb_features_path)
            logger.info(
                "XGBoost feature names loaded successfully from: %s",
                xgb_features_path,
            )
            _xgb_calibrator = joblib.load(xgb_calibrator_path)
            logger.info(
                "XGBoost calibrator loaded successfully from: %s",
                xgb_calibrator_path,
            )
            with open(temperature_path, "r") as f:
                _deberta_temperature = json.load(f)["temperature"]
            logger.info(
                "DeBERTa temperature loaded successfully from: %s",
                temperature_path,
            )
            _css_min_word_count = int(os.getenv("CSS_MIN_WORD_COUNT", "150"))
            logger.info("CSS-v2 XGBoost artifacts loaded successfully.")
        except Exception as e:
            logger.warning(
                f"CSS-v2 artifacts failed to load: {e}. "
                f"signal_analysis will be null on all responses."
            )
    else:
        logger.warning("CSS-v2 env vars not set. signal_analysis will be null.")


def get_tokenizer():
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not loaded.")
    return _tokenizer


def get_model():
    if _model is None:
        raise RuntimeError("Model not loaded.")
    return _model


def get_device():
    if _device is None:
        raise RuntimeError("Device not set.")
    return _device


def get_xgb_model():
    return _xgb_model


def get_xgb_scaler():
    return _xgb_scaler


def get_xgb_feature_names():
    return _xgb_feature_names


def get_xgb_calibrator():
    return _xgb_calibrator


def get_deberta_temperature():
    return _deberta_temperature


def get_css_min_word_count():
    return _css_min_word_count


def get_qwen_tokenizer():
    return _qwen_tokenizer


def get_qwen_model():
    return _qwen_model


def is_css_available():
    """Returns True only if all CSS-v2 artifacts loaded successfully."""
    return all(
        [
            _xgb_model is not None,
            _xgb_scaler is not None,
            _xgb_feature_names is not None,
            _xgb_calibrator is not None,
            _deberta_temperature is not None,
        ]
    )


def is_qwen_available():
    """Returns True only if Qwen model and tokenizer loaded successfully."""
    return _qwen_model is not None and _qwen_tokenizer is not None
