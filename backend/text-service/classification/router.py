from fastapi import APIRouter, HTTPException

from classification.predictor import classify
from shared.schemas import ClassifyRequest, ClassifyResponse

# All routes on this router are mounted under /classify and grouped under
# the "classification" tag in the OpenAPI docs.
router = APIRouter(prefix="/classify", tags=["classification"])


@router.post("", response_model=ClassifyResponse)
def classify_text(request: ClassifyRequest):
    """
    Classify input text as AI-Generated or Human-Written.

    Runs the full pipeline (sanitization -> prediction -> token
    highlighting) and returns the classification result along with
    explainability data.
    """
    try:
        return classify(request.text)
    except Exception as exc:
        # Surface a generic 500 with the error detail rather than letting an
        # unhandled exception crash the request.
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/health")
def health_check():
    """Simple liveness check for the classification service. No auth required."""
    return {"status": "ok", "service": "classification"}
