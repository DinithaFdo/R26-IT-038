from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from app.api.dependencies import get_prediction_submission_service
from app.auth.clerk_auth import require_clerk_user
from app.auth.schemas import AuthPrincipal
from app.config.settings import settings
from app.schemas.common import ErrorResponse, SourceType
from app.schemas.prediction_submission import (
    PredictionJobStatusResponse,
    PredictionSubmissionResponse,
)
from app.services.prediction_submission_service import PredictionSubmissionService

router = APIRouter(prefix="/predictions", tags=["Predictions"])

PREDICTION_DESCRIPTION = f"""
Submit audio for authenticated MULTI-SCOPE deepfake voice classification from
a dashboard upload or browser live-recording submission.

This endpoint is record-then-analyse only. It does not implement WebSockets or
streaming inference. Dashboard uploads and live recordings use the same secure
ingestion, FFmpeg/ffprobe validation, audio preprocessing, model-branch
prediction, score-level fusion, Cloudinary/no-op storage, MongoDB persistence,
and optional post-prediction Voice XAI handoff.

Accepted browser recording examples include audio/webm with Opus, audio/ogg
with Opus, and audio/mp4 or M4A with AAC. The backend does not trust extension
or Content-Type alone; actual container and codec are inspected. Browser
recordings are limited to {settings.max_audio_duration_seconds} seconds.
Maximum upload size is {settings.max_upload_size_mb} MB.

Authentication: use Swagger's Authorize button with a Clerk bearer JWT.
"""


@router.post(
    "",
    response_model=PredictionSubmissionResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit audio for deepfake voice classification",
    description=PREDICTION_DESCRIPTION,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid audio submission."},
        401: {"model": ErrorResponse, "description": "Authentication failed."},
        409: {
            "model": ErrorResponse,
            "description": (
                "idempotency_conflict: the supplied idempotency_key is already "
                "bound to a different logical request."
            ),
        },
        413: {"model": ErrorResponse, "description": "Upload is too large."},
        415: {"model": ErrorResponse, "description": "Unsupported audio format."},
        422: {"model": ErrorResponse, "description": "Request validation problem."},
        500: {"model": ErrorResponse, "description": "Internal server error."},
        503: {"model": ErrorResponse, "description": "Required service unavailable."},
    },
)
async def create_prediction(
    request: Request,
    file: Annotated[
        UploadFile,
        File(
            description=(
                "Required audio file. Browser recordings such as audio/webm "
                "Opus, audio/ogg Opus, and audio/mp4 or M4A AAC are accepted "
                "when ffprobe confirms a supported audio-only container. "
                "Example filenames: human_voice.wav, synthetic_voice.wav, "
                "sample-voice.wav."
            )
        ),
    ],
    source_type: Annotated[
        SourceType,
        Form(description="dashboard_upload or live_recording.", examples=["dashboard_upload"]),
    ],
    client_filename: Annotated[
        str | None,
        Form(
            description=(
                "Optional user-facing filename. Live recordings without this "
                "field receive a generated Recording timestamp display name."
            )
        ),
    ] = None,
    idempotency_key: Annotated[
        str | None,
        Form(
            description=(
                "Optional owner-scoped idempotency key. Leave it empty for a "
                "normal new submission; every request without a key creates a "
                "new prediction. Send a NEW unique value (for example a UUID4 "
                "such as 550e8400-e29b-41d4-a716-446655440000) for every new "
                "audio submission, and reuse the same value only when retrying "
                "the exact same logical request -- that replays the existing "
                "prediction instead of running inference twice. Reusing a key "
                "for a different logical request (different source_type, "
                "client_filename, or uploaded filename) is rejected with HTTP "
                "409 idempotency_conflict. This example is documentation only: "
                "it is deliberately not a default and Swagger does not submit "
                "it for you."
            )
            # No `examples=[...]` on purpose: Swagger UI pre-fills form inputs
            # from schema examples, which would make every "Try it out" send
            # the same static key and 409 on the second differing upload.
        ),
    ] = None,
    principal: AuthPrincipal = Depends(require_clerk_user),
    submission_service: PredictionSubmissionService = Depends(
        get_prediction_submission_service
    ),
) -> PredictionSubmissionResponse:
    if source_type not in {SourceType.dashboard_upload, SourceType.live_recording}:
        raise HTTPException(
            status_code=422,
            detail="source_type must be dashboard_upload or live_recording.",
        )
    return await submission_service.submit(
        file=file,
        principal=principal,
        source_type=source_type,
        request_id=request.state.request_id,
        client_correlation_id=request.state.client_correlation_id,
        client_filename=client_filename,
        idempotency_key=idempotency_key,
    )


@router.get(
    "/{prediction_id}/status",
    response_model=PredictionJobStatusResponse,
    summary="Get authenticated prediction status",
    description=(
        "Return the owner-scoped job status for a prediction. The current MVP "
        "executes jobs inline, but the response shape is compatible with a "
        "future HTTP 202 plus durable-worker flow. Use the prediction_id returned "
        "by POST /api/v1/predictions."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Authentication failed."},
        404: {"model": ErrorResponse, "description": "Prediction not found."},
        500: {"model": ErrorResponse, "description": "Internal server error."},
    },
)
async def get_prediction_status(
    prediction_id: str,
    principal: AuthPrincipal = Depends(require_clerk_user),
    submission_service: PredictionSubmissionService = Depends(
        get_prediction_submission_service
    ),
) -> PredictionJobStatusResponse:
    return await submission_service.get_status(
        principal=principal,
        prediction_id=prediction_id,
    )
