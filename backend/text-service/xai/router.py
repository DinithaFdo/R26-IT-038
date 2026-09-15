import logging
import time
import hashlib

from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends, Request

from .schemas import XAIRequest, XAIResponse
from .service import XAIService

# Global cache dictionary to store previously analyzed text results
response_cache = {}


def get_text_hash(text: str) -> str:
    """Generates a unique MD5 hash for the input text to use as a cache key."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


# Import Ishara's classify function directly to eliminate HTTP latency
try:
    from classification.predictor import classify
except ImportError:
    logging.error(
        "Component 3 'classify' function not found. "
        "Ensure the classification module exists."
    )

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/xai", tags=["Explainable AI (XAI)"])


def get_xai_service(request: Request) -> XAIService:
    """
    Get the XAI service using the models loaded into FastAPI application state.
    """
    model = getattr(request.app.state, "model", None)
    tokenizer = getattr(request.app.state, "tokenizer", None)
    qwen_model = getattr(request.app.state, "qwen_model", None)
    qwen_tokenizer = getattr(request.app.state, "qwen_tokenizer", None)

    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=500,
            detail="Machine learning models are not initialized on the server.",
        )

    return XAIService(
        model=model,
        tokenizer=tokenizer,
        qwen_model=qwen_model,
        qwen_tokenizer=qwen_tokenizer,
    )


@router.post("/audit", response_model=XAIResponse)
async def audit_text(
    payload: XAIRequest,
    request: Request,
    xai_service: XAIService = Depends(get_xai_service),
):
    """
    Executes a high-speed forensic Interpretability Audit.
    Component 3's classification pipeline is called directly in memory.
    """
    logger.info(f"Received XAI audit request for sequence length: {len(payload.text)}")
    start_time = time.time()
    telemetry_data = {}  # Initialize telemetry dictionary

    # ============================================================
    # CACHE CHECK
    # ============================================================
    text_hash = get_text_hash(payload.text)

    if text_hash in response_cache:
        logger.info("⚡ Cache hit! Returning instantaneous XAI response.")
        cached_result = response_cache[text_hash].copy()
        cache_processing_time_ms = (time.time() - start_time) * 1000
        cached_result["processing_time_ms"] = cache_processing_time_ms

        # Add cache hit telemetry
        cached_result["telemetry"] = {
            "total_time_ms": cache_processing_time_ms,
            "cache_hit": True,
        }
        return XAIResponse(**cached_result)

    try:
        # ============================================================
        # STEP 1: Execute Component 3 (Classification)
        # ============================================================
        t0 = time.time()
        # include_highlights=False saves a massive DeBERTa forward pass!
        classify_response = classify(payload.text, include_highlights=False)
        telemetry_data["classification_time_ms"] = (time.time() - t0) * 1000

        # Extract predicted class and confidence
        predicted_class = classify_response.label
        if predicted_class == "AI-Generated":
            confidence = round(classify_response.prob_ai * 100, 2)
        else:
            confidence = round(classify_response.prob_human * 100, 2)

        # Extract sanitization information
        clean_text = classify_response.sanitization.clean_text
        sanitization_data = {
            "was_attacked": classify_response.sanitization.was_attacked,
            "attack_report": classify_response.sanitization.attack_report,
            "clean_text": clean_text,
        }

        # Convert signal analysis Pydantic model to dict
        signal_analysis_dict = (
            classify_response.signal_analysis.model_dump()
            if classify_response.signal_analysis
            else None
        )

        # ============================================================
        # STEP 2: Run XAI Audit (Math + LLM)
        # ============================================================
        # We pass the telemetry dict down to be populated by the service
        audit_result = xai_service.generate_audit(
            text=clean_text,
            predicted_class=predicted_class,
            confidence=confidence,
            signal_analysis=signal_analysis_dict,
            telemetry=telemetry_data,  # Pass telemetry tracker
            final_label=classify_response.final_label,
            leans_toward=classify_response.leans_toward,
        )

        # ============================================================
        # STEP 3: Finalize Telemetry and Metadata
        # ============================================================
        total_processing_time_ms = (time.time() - start_time) * 1000
        telemetry_data["total_time_ms"] = total_processing_time_ms
        telemetry_data["cache_hit"] = False

        # Log the final telemetry breakdown clearly in the console
        logger.info("⏱️ [XAI-TELEMETRY] Pipeline Execution Times:")
        logger.info(
            f"   - Component 3 (Classification/Stylometrics): {telemetry_data.get('classification_time_ms', 0):.2f} ms"
        )
        logger.info(
            f"   - HNIF Extraction (Attention + IG): {telemetry_data.get('math_extraction_time_ms', 0):.2f} ms"
        )
        logger.info(
            f"   - Qwen LLM Generation: {telemetry_data.get('llm_generation_time_ms', 0):.2f} ms"
        )
        logger.info(f"   - TOTAL TIME: {total_processing_time_ms:.2f} ms")

        # Inject classification metadata
        audit_result["model_version"] = classify_response.model_version
        audit_result["processing_time_ms"] = total_processing_time_ms
        audit_result["sanitization"] = sanitization_data
        audit_result["signal_analysis"] = signal_analysis_dict
        audit_result["final_label"] = classify_response.final_label
        audit_result["leans_toward"] = classify_response.leans_toward

        # Add telemetry to the payload so the frontend can display it
        audit_result["telemetry"] = telemetry_data

        # ============================================================
        # STEP 4: Save to Cache and Return
        # ============================================================
        response_cache[text_hash] = audit_result
        return XAIResponse(**audit_result)

    except Exception as e:
        logger.exception(f"Failed to process XAI audit: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}",
        )
