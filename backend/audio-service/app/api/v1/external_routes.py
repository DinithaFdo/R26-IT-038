from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile, status

from app.api.dependencies import get_prediction_submission_service
from app.auth.api_keys import require_api_key_scope
from app.auth.schemas import AuthPrincipal
from app.config.settings import settings
from app.schemas.api_keys import ApiKeyScope
from app.schemas.common import ErrorResponse, SourceType
from app.schemas.prediction_submission import PredictionSubmissionResponse
from app.services.prediction_submission_service import PredictionSubmissionService

router = APIRouter(prefix="/external", tags=["API Keys / External"])


@router.post(
    "/predictions",
    response_model=PredictionSubmissionResponse,
    status_code=status.HTTP_200_OK,
    summary="Create a prediction using an API key",
    description=(
        "Third-party integrations can submit an audio file with "
        "Authorization: Bearer msk_live_... . The API key must include the "
        "prediction:create scope. Upload size and audio duration are limited "
        f"by the configured {settings.max_upload_size_mb} MB and "
        f"{settings.max_audio_duration_seconds} second limits."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Invalid API key."},
        403: {"model": ErrorResponse, "description": "Missing API-key scope."},
        413: {"model": ErrorResponse, "description": "Upload is too large."},
        415: {"model": ErrorResponse, "description": "Unsupported audio format."},
        422: {"model": ErrorResponse, "description": "Request validation problem."},
    },
)
async def create_external_prediction(
    request: Request,
    file: Annotated[
        UploadFile,
        File(
            description=(
                "Required audio file for prediction. Example filenames: "
                "human_voice.wav, synthetic_voice.wav, sample-voice.wav."
            )
        ),
    ],
    client_filename: Annotated[str | None, Form()] = None,
    idempotency_key: Annotated[str | None, Form()] = None,
    principal: AuthPrincipal = Depends(
        require_api_key_scope(ApiKeyScope.prediction_create)
    ),
    submission_service: PredictionSubmissionService = Depends(
        get_prediction_submission_service
    ),
) -> PredictionSubmissionResponse:
    return await submission_service.submit(
        file=file,
        principal=principal,
        source_type=SourceType.public_api,
        request_id=request.state.request_id,
        client_correlation_id=request.state.client_correlation_id,
        client_filename=client_filename,
        idempotency_key=idempotency_key,
    )
