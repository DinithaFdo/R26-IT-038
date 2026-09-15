from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, Form, Query, Request

from app.api.dependencies import (
    get_api_key_service,
    get_prediction_history_service,
    get_prediction_rerun_service,
)
from app.auth.clerk_auth import require_clerk_user
from app.auth.schemas import AuthPrincipal
from app.schemas.common import PredictionLabel, PredictionStatus, SourceType
from app.schemas.api_keys import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyDeleteResponse,
    ApiKeyListResponse,
)
from app.schemas.prediction_history import (
    PredictionAudioPlaybackResponse,
    PredictionDeleteResponse,
    PredictionDetailResponse,
    PredictionHistoryListResponse,
)
from app.schemas.prediction_submission import PredictionSubmissionResponse
from app.services.prediction_history_service import PredictionHistoryService
from app.services.prediction_rerun_service import PredictionRerunService
from app.services.api_key_service import ApiKeyService

router = APIRouter(prefix="/me", tags=["Users"])


@router.post(
    "/api-keys",
    response_model=ApiKeyCreateResponse,
    tags=["API Keys / External"],
    summary="Create an API key for third-party integrations",
    description=(
        "Creates an API key for the verified Clerk user. The full key is "
        "returned only in this response; only a keyed hash and safe metadata "
        "are stored."
    ),
)
async def create_my_api_key(
    payload: ApiKeyCreateRequest,
    principal: AuthPrincipal = Depends(require_clerk_user),
    api_key_service: ApiKeyService = Depends(get_api_key_service),
) -> ApiKeyCreateResponse:
    return await api_key_service.create_api_key(
        principal=principal,
        request=payload,
    )


@router.get(
    "/api-keys",
    response_model=ApiKeyListResponse,
    tags=["API Keys / External"],
    summary="List the authenticated user's API keys",
)
async def list_my_api_keys(
    principal: AuthPrincipal = Depends(require_clerk_user),
    api_key_service: ApiKeyService = Depends(get_api_key_service),
) -> ApiKeyListResponse:
    return await api_key_service.list_api_keys(principal=principal)


@router.delete(
    "/api-keys/{api_key_id}",
    response_model=ApiKeyDeleteResponse,
    tags=["API Keys / External"],
    summary="Revoke one authenticated-user API key",
)
async def delete_my_api_key(
    api_key_id: str,
    principal: AuthPrincipal = Depends(require_clerk_user),
    api_key_service: ApiKeyService = Depends(get_api_key_service),
) -> ApiKeyDeleteResponse:
    return await api_key_service.delete_api_key(
        principal=principal,
        api_key_id=api_key_id,
    )


@router.get(
    "/predictions",
    response_model=PredictionHistoryListResponse,
    tags=["Predictions"],
    summary="List the authenticated user's prediction history",
    description=(
        "Returns lightweight prediction history for the verified Clerk user. "
        "The owner is always derived from the bearer token; owner_user_id is not "
        "accepted from clients. Soft-deleted records are hidden."
    ),
)
async def list_my_predictions(
    principal: AuthPrincipal = Depends(require_clerk_user),
    history_service: PredictionHistoryService = Depends(
        get_prediction_history_service
    ),
    page: Annotated[int, Query(ge=1)] = 1,
    limit: Annotated[int, Query(ge=1, le=50)] = 20,
    status_filter: Annotated[
        PredictionStatus | None,
        Query(alias="status"),
    ] = None,
    source_type: SourceType | None = None,
    prediction_label: PredictionLabel | None = None,
    created_from: datetime | None = None,
    created_to: datetime | None = None,
) -> PredictionHistoryListResponse:
    return await history_service.list_predictions(
        principal=principal,
        page=page,
        limit=limit,
        status_filter=status_filter,
        source_type=source_type,
        prediction_label=prediction_label,
        created_from=created_from,
        created_to=created_to,
    )


@router.get(
    "/predictions/{prediction_id}",
    response_model=PredictionDetailResponse,
    tags=["Predictions"],
    summary="Get one authenticated-user prediction",
    description=(
        "Returns full prediction details only when the record belongs to the "
        "verified Clerk user. Cross-user and deleted records return 404."
    ),
)
async def get_my_prediction(
    prediction_id: str,
    principal: AuthPrincipal = Depends(require_clerk_user),
    history_service: PredictionHistoryService = Depends(
        get_prediction_history_service
    ),
) -> PredictionDetailResponse:
    return await history_service.get_prediction_detail(
        principal=principal,
        prediction_id=prediction_id,
    )


@router.get(
    "/predictions/{prediction_id}/audio",
    response_model=PredictionAudioPlaybackResponse,
    tags=["Predictions"],
    summary="Create a signed playback URL for one authenticated-user prediction",
    description=(
        "Returns a short-lived signed Cloudinary playback URL for the owner. "
        "Cloudinary public IDs are not returned."
    ),
)
async def get_my_prediction_audio(
    prediction_id: str,
    principal: AuthPrincipal = Depends(require_clerk_user),
    history_service: PredictionHistoryService = Depends(
        get_prediction_history_service
    ),
) -> PredictionAudioPlaybackResponse:
    return await history_service.get_audio_playback_url(
        principal=principal,
        prediction_id=prediction_id,
    )


@router.post(
    "/predictions/{prediction_id}/rerun",
    response_model=PredictionSubmissionResponse,
    tags=["Predictions"],
    summary="Re-run one authenticated-user prediction",
    description=(
        "Creates a new prediction record from the owner-scoped private "
        "Cloudinary audio asset attached to a historical prediction. The "
        "original prediction is never overwritten."
    ),
)
async def rerun_my_prediction(
    prediction_id: str,
    request: Request,
    rerun_reason: Annotated[
        str | None,
        Form(description="Optional user-facing reason for the rerun."),
    ] = None,
    principal: AuthPrincipal = Depends(require_clerk_user),
    rerun_service: PredictionRerunService = Depends(
        get_prediction_rerun_service
    ),
) -> PredictionSubmissionResponse:
    return await rerun_service.rerun_prediction(
        principal=principal,
        prediction_id=prediction_id,
        request_id=request.state.request_id,
        rerun_reason=rerun_reason,
    )


@router.delete(
    "/predictions/{prediction_id}",
    response_model=PredictionDeleteResponse,
    tags=["Predictions"],
    summary="Delete one authenticated-user prediction",
    description=(
        "Deletes the owner-scoped Cloudinary audio asset when present, then "
        "removes every owner-scoped Voice XAI explanation and its private artifacts, "
        "then soft-deletes the MongoDB prediction record. The operation is idempotent "
        "for records already deleted by the same owner."
    ),
)
async def delete_my_prediction(
    prediction_id: str,
    principal: AuthPrincipal = Depends(require_clerk_user),
    history_service: PredictionHistoryService = Depends(
        get_prediction_history_service
    ),
) -> PredictionDeleteResponse:
    return await history_service.delete_prediction(
        principal=principal,
        prediction_id=prediction_id,
    )
