from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.api.dependencies import (
    get_voice_service,
    get_prediction_submission_service,
)
from app.auth.clerk_auth import require_clerk_user
from app.auth.schemas import AuthPrincipal
from app.config.settings import settings
from app.schemas.common import ErrorResponse, SourceType
from app.schemas.model_health import ModelHealthResponse
from app.schemas.prediction_submission import PredictionSubmissionResponse
from app.services.prediction_submission_service import PredictionSubmissionService
from app.services.voice_service import VoiceService

router = APIRouter(prefix="/voice", tags=["Voice Classification"])

PREDICT_DESCRIPTION = f"""
Run the MULTI-SCOPE multi-branch voice deepfake classification pipeline.

    Accepted formats: WAV, FLAC, MP3, M4A, AAC, Opus, OGG, and audio-only WebM.
    Files are inspected by ffprobe. Uploads with video streams are rejected.
Maximum upload size: {settings.max_upload_size_mb} MB.
Maximum audio duration: {settings.max_audio_duration_seconds} seconds.

Development note: branches may run in deterministic dummy mode. Dummy outputs are
for system development and integration testing only; they are not research
results and must not be used for evaluation.
"""


def require_legacy_prediction_enabled() -> None:
    if (
        settings.app_env.lower() == "production"
        or not settings.enable_legacy_anonymous_prediction
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Not found.",
        )


@router.post(
    "/predict",
    response_model=PredictionSubmissionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict whether an uploaded voice sample is bonafide or spoofed",
    description=(
        PREDICT_DESCRIPTION
        + "\nDeprecated: use POST /api/v1/predictions. This compatibility "
        "route is disabled by default, never available in production, and "
        "requires Clerk authentication when explicitly enabled for local "
        "development."
    ),
    deprecated=True,
    dependencies=[Depends(require_legacy_prediction_enabled)],
    responses={
        400: {
            "model": ErrorResponse,
            "description": (
                "Invalid, empty, corrupted, over-duration, or unusable audio."
            ),
        },
        413: {
            "model": ErrorResponse,
            "description": "Uploaded audio exceeds the configured size limit.",
        },
        415: {
            "model": ErrorResponse,
            "description": (
                "Unsupported format, mismatched container/codec, or video content."
            ),
        },
        422: {"model": ErrorResponse, "description": "Request validation problem."},
        500: {
            "model": ErrorResponse,
            "description": "Unexpected internal server error.",
        },
        503: {
            "model": ErrorResponse,
            "description": (
                "Audio processing tools or usable model branches are unavailable."
            ),
        },
    },
)
async def predict_voice(
    file: Annotated[
        UploadFile,
        File(
            description=(
                "Audio file to classify. Supported formats: WAV, FLAC, MP3, "
                "M4A, AAC, Opus, OGG, and audio-only WebM. "
                "Dummy branch outputs are development placeholders only."
            )
        ),
    ],
    request: Request,
    principal: AuthPrincipal = Depends(require_clerk_user),
    submission_service: PredictionSubmissionService = Depends(
        get_prediction_submission_service
    ),
) -> PredictionSubmissionResponse:
    return await submission_service.submit(
        file=file,
        principal=principal,
        source_type=SourceType.dashboard_upload,
        request_id=request.state.request_id,
        client_correlation_id=request.state.client_correlation_id,
    )


@router.get(
    "/models/health",
    response_model=list[ModelHealthResponse],
    summary="Get voice model branch health",
    description=(
        "Reports each branch model name, display name, configured mode, and loaded "
        "status. Branches in dummy mode are development placeholders and are not "
        "research-result producers. `ready` means the branch can run as configured; "
        "`research_ready` additionally requires explicit preprocessing and class "
        "mapping verification."
    ),
)
def model_health(
    voice_service: VoiceService = Depends(get_voice_service),
) -> list[dict[str, Any]]:
    if not settings.model_health_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not found.")
    return [
        {
            **branch_health,
            "uses_dummy_mode": branch_health.get("mode") == "dummy",
            "warning": (
                "Dummy mode is for system development only; not a research result."
                if branch_health.get("mode") == "dummy"
                else None
            ),
        }
        for branch_health in voice_service.model_health()
    ]
