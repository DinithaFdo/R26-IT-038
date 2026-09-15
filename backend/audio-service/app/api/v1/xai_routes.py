"""Owner-scoped endpoints for post-prediction Voice XAI runs."""

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.api.dependencies import get_app_settings, get_rate_limiter, get_xai_orchestrator
from app.auth.clerk_auth import require_clerk_user
from app.auth.schemas import AuthPrincipal
from app.config.settings import Settings
from app.core.rate_limit import InMemoryRateLimiter
from app.schemas.common import ErrorResponse
from app.schemas.xai import (
    CombinedExplanationReport,
    NarrativeExplanation,
    SemanticExplanation,
    TemporalExplanation,
    XaiExplanationResponse,
)
from app.voice_xai.orchestrator import (
    VoiceXaiOrchestrator,
    XaiArtifactUnavailableError,
    XaiNarrativeRetryError,
    XaiOrchestrationError,
    XaiReplayUnavailableError,
    XaiRetryRequiredError,
)
from app.voice_xai.status import XaiExplanationNotFoundError

router = APIRouter(prefix="/me/predictions", tags=["Explainability / XAI"])


def _owner_user_id(principal: AuthPrincipal) -> str:
    return principal.user_id or principal.subject


def _not_found() -> HTTPException:
    return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Explanation not found.")


def _unavailable(error: XaiOrchestrationError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error))


def _conflict(error: XaiOrchestrationError) -> HTTPException:
    return HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error))


def _enforce_xai_action_rate_limit(
    *,
    action: str,
    principal: AuthPrincipal,
    limiter: InMemoryRateLimiter,
    app_settings: Settings,
) -> None:
    """Bound expensive explanation trigger/retry calls per authenticated user.

    The generic ``RateLimitMiddleware`` keys on the literal request path,
    which includes ``prediction_id`` -- so it does not actually bound how
    many distinct explanation jobs one user can create across their own
    predictions (see SEC-3 in the 2026-08-08 reliability fixes). This uses
    the same shared limiter, keyed by the server-derived owner ID only, never
    a client-supplied identifier.
    """

    if action == "trigger":
        limit = app_settings.xai_trigger_rate_limit_requests
        window_seconds = app_settings.xai_trigger_rate_limit_window_seconds
    elif action == "narrative_retry":
        limit = app_settings.xai_narrative_retry_rate_limit_requests
        window_seconds = app_settings.xai_narrative_retry_rate_limit_window_seconds
    else:
        limit = app_settings.xai_retry_rate_limit_requests
        window_seconds = app_settings.xai_retry_rate_limit_window_seconds
    decision = limiter.check(
        f"xai:{action}:{_owner_user_id(principal)}",
        limit=limit,
        window_seconds=window_seconds,
    )
    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Voice XAI request rate limit exceeded.",
            headers={"Retry-After": str(decision.retry_after_seconds)},
        )


@router.post(
    "/{prediction_id}/explanation",
    response_model=XaiExplanationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger a Voice XAI explanation for a completed prediction",
    description=(
        "Creates or returns the latest owner-scoped Voice XAI run for a completed "
        "prediction. The path `prediction_id` must be an existing prediction owned "
        "by the authenticated Clerk user. The classifier prediction has already "
        "been persisted; explanation work is queued asynchronously when workers "
        "are available. XAI-disabled or unavailable states return controlled "
        "errors without changing the classifier prediction."
    ),
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def trigger_explanation(
    prediction_id: str,
    principal: AuthPrincipal = Depends(require_clerk_user),
    orchestrator: VoiceXaiOrchestrator = Depends(get_xai_orchestrator),
    rate_limiter: InMemoryRateLimiter = Depends(get_rate_limiter),
    app_settings: Settings = Depends(get_app_settings),
) -> XaiExplanationResponse:
    _enforce_xai_action_rate_limit(
        action="trigger",
        principal=principal,
        limiter=rate_limiter,
        app_settings=app_settings,
    )
    try:
        return await orchestrator.trigger_for_prediction(
            prediction_id=prediction_id, owner_user_id=_owner_user_id(principal)
        )
    except XaiExplanationNotFoundError as error:
        raise _not_found() from error
    except (XaiRetryRequiredError, XaiReplayUnavailableError) as error:
        raise _conflict(error) from error
    except XaiOrchestrationError as error:
        if "completed prediction" in str(error):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        raise _unavailable(error) from error


@router.get(
    "/{prediction_id}/explanation",
    response_model=XaiExplanationResponse,
    summary="Get the latest owner-scoped Voice XAI explanation",
    description=(
        "Returns the latest explanation state for an authenticated user's completed "
        "prediction. This does not trigger a new run."
    ),
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def get_explanation(
    prediction_id: str,
    principal: AuthPrincipal = Depends(require_clerk_user),
    orchestrator: VoiceXaiOrchestrator = Depends(get_xai_orchestrator),
) -> XaiExplanationResponse:
    try:
        return await orchestrator.get_for_prediction(
            prediction_id=prediction_id, owner_user_id=_owner_user_id(principal)
        )
    except XaiExplanationNotFoundError as error:
        raise _not_found() from error


@router.get(
    "/{prediction_id}/explanation/temporal",
    response_model=TemporalExplanation,
    summary="Get owner-scoped temporal evidence",
    description=(
        "Returns the persisted XLS-R attention-rollout evidence for an existing "
        "Voice XAI run. First call POST /me/predictions/{prediction_id}/explanation "
        "(or create a completed prediction while XAI is enabled), then poll "
        "GET /me/predictions/{prediction_id}/explanation until temporal is completed."
    ),
    responses={
        401: {"model": ErrorResponse},
        404: {
            "model": ErrorResponse,
            "description": "Prediction/XAI run not found, or temporal evidence was not produced.",
        },
        409: {
            "model": ErrorResponse,
            "description": "The XAI run exists but temporal evidence is still being generated.",
        },
    },
)
async def get_temporal_explanation(
    prediction_id: str,
    principal: AuthPrincipal = Depends(require_clerk_user),
    orchestrator: VoiceXaiOrchestrator = Depends(get_xai_orchestrator),
) -> TemporalExplanation:
    try:
        explanation = await orchestrator.get_for_prediction(
            prediction_id=prediction_id, owner_user_id=_owner_user_id(principal)
        )
    except XaiExplanationNotFoundError as error:
        raise _not_found() from error
    if explanation.temporal is None and explanation.status.value in {"queued", "running"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Temporal evidence is still being generated. Poll the explanation endpoint.",
        )
    if explanation.temporal is None:
        temporal_error = explanation.component_errors.temporal
        run_error = explanation.error
        if temporal_error is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=temporal_error.message)
        if run_error is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=run_error.message)
        raise _not_found()
    return explanation.temporal


@router.get(
    "/{prediction_id}/explanation/semantic",
    response_model=SemanticExplanation,
    summary="Get owner-scoped semantic evidence",
    description=(
        "Returns semantic evidence for the latest explanation when available. "
        "The endpoint is owner-scoped by the Clerk bearer token."
    ),
)
async def get_semantic_explanation(
    prediction_id: str,
    principal: AuthPrincipal = Depends(require_clerk_user),
    orchestrator: VoiceXaiOrchestrator = Depends(get_xai_orchestrator),
) -> SemanticExplanation:
    try:
        explanation = await orchestrator.get_for_prediction(
            prediction_id=prediction_id, owner_user_id=_owner_user_id(principal)
        )
    except XaiExplanationNotFoundError as error:
        raise _not_found() from error
    if explanation.semantic is None:
        if explanation.status.value in {"queued", "running"}:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Semantic evidence is still being generated. Poll the explanation endpoint.",
            )
        semantic_error = explanation.component_errors.semantic
        run_error = explanation.error
        if semantic_error is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=semantic_error.message)
        if run_error is not None:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=run_error.message)
        raise _not_found()
    return explanation.semantic


@router.get(
    "/{prediction_id}/explanation/report",
    response_model=CombinedExplanationReport,
    summary="Get the owner-scoped deterministic XAI report",
    description=(
        "Returns the combined deterministic XAI report for the latest explanation "
        "when the report has been produced."
    ),
)
async def get_explanation_report(
    prediction_id: str,
    principal: AuthPrincipal = Depends(require_clerk_user),
    orchestrator: VoiceXaiOrchestrator = Depends(get_xai_orchestrator),
) -> CombinedExplanationReport:
    try:
        explanation = await orchestrator.get_for_prediction(
            prediction_id=prediction_id, owner_user_id=_owner_user_id(principal)
        )
    except XaiExplanationNotFoundError as error:
        raise _not_found() from error
    if explanation.combined_report is None:
        raise _not_found()
    return explanation.combined_report


@router.get(
    "/{prediction_id}/explanation/narrative",
    response_model=NarrativeExplanation,
    summary="Get the optional owner-scoped AI XAI narrative",
    description=(
        "Returns the validated Alibaba Model Studio Qwen narrative for the latest "
        "explanation. The deterministic report endpoint remains authoritative and "
        "is not changed by this optional rendering layer."
    ),
    responses={
        401: {"model": ErrorResponse},
        404: {
            "model": ErrorResponse,
            "description": "Prediction/XAI run not found, or no narrative is available.",
        },
        409: {
            "model": ErrorResponse,
            "description": "The narrative is pending or could not be generated.",
        },
    },
)
async def get_explanation_narrative(
    prediction_id: str,
    principal: AuthPrincipal = Depends(require_clerk_user),
    orchestrator: VoiceXaiOrchestrator = Depends(get_xai_orchestrator),
) -> NarrativeExplanation:
    try:
        explanation = await orchestrator.get_for_prediction(
            prediction_id=prediction_id, owner_user_id=_owner_user_id(principal)
        )
    except XaiExplanationNotFoundError as error:
        raise _not_found() from error
    if explanation.narrative is not None:
        return explanation.narrative
    narrative_status = explanation.component_statuses.narrative
    if narrative_status.value in {"queued", "running"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="AI narrative is still being generated. Poll the explanation endpoint.",
        )
    narrative_error = explanation.component_errors.narrative
    if narrative_error is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=narrative_error.message)
    raise _not_found()


@router.post(
    "/{prediction_id}/explanation/narrative/retry",
    response_model=XaiExplanationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Retry only the optional AI narrative",
    description=(
        "Queues a bounded retry of the optional AI narrative using the persisted "
        "validated XAI evidence. It never re-runs the classifier, temporal or "
        "semantic evidence, or deterministic report."
    ),
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
        429: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def retry_explanation_narrative(
    prediction_id: str,
    principal: AuthPrincipal = Depends(require_clerk_user),
    orchestrator: VoiceXaiOrchestrator = Depends(get_xai_orchestrator),
    rate_limiter: InMemoryRateLimiter = Depends(get_rate_limiter),
    app_settings: Settings = Depends(get_app_settings),
) -> XaiExplanationResponse:
    _enforce_xai_action_rate_limit(
        action="narrative_retry",
        principal=principal,
        limiter=rate_limiter,
        app_settings=app_settings,
    )
    try:
        return await orchestrator.retry_narrative_for_prediction(
            prediction_id=prediction_id,
            owner_user_id=_owner_user_id(principal),
        )
    except XaiExplanationNotFoundError as error:
        raise _not_found() from error
    except XaiNarrativeRetryError as error:
        raise _conflict(error) from error
    except XaiOrchestrationError as error:
        raise _unavailable(error) from error


@router.get(
    "/{prediction_id}/explanation/artifacts/{artifact_id}",
    summary="Download a private owner-scoped Voice XAI artifact",
    description=(
        "Downloads a private artifact belonging to the latest owner-scoped XAI "
        "run. The route requires both the prediction id and artifact id."
    ),
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)
async def get_explanation_artifact(
    prediction_id: str,
    artifact_id: str,
    principal: AuthPrincipal = Depends(require_clerk_user),
    orchestrator: VoiceXaiOrchestrator = Depends(get_xai_orchestrator),
) -> Response:
    try:
        artifact = await orchestrator.get_artifact_for_prediction(
            prediction_id=prediction_id,
            artifact_id=artifact_id,
            owner_user_id=_owner_user_id(principal),
        )
    except XaiExplanationNotFoundError as error:
        raise _not_found() from error
    except XaiArtifactUnavailableError as error:
        raise _conflict(error) from error
    return Response(content=artifact.content, media_type=artifact.reference.content_type)


@router.post(
    "/{prediction_id}/explanation/retry",
    response_model=XaiExplanationResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create a new owner-scoped Voice XAI explanation run",
    description=(
        "Queues a new versioned explanation run for an authenticated user's "
        "completed prediction. It does not rewrite terminal evidence from a "
        "previous run and does not change prediction status."
    ),
)
async def retry_explanation(
    prediction_id: str,
    principal: AuthPrincipal = Depends(require_clerk_user),
    orchestrator: VoiceXaiOrchestrator = Depends(get_xai_orchestrator),
    rate_limiter: InMemoryRateLimiter = Depends(get_rate_limiter),
    app_settings: Settings = Depends(get_app_settings),
) -> XaiExplanationResponse:
    _enforce_xai_action_rate_limit(
        action="retry",
        principal=principal,
        limiter=rate_limiter,
        app_settings=app_settings,
    )
    try:
        return await orchestrator.trigger_for_prediction(
            prediction_id=prediction_id,
            owner_user_id=_owner_user_id(principal),
            retry=True,
        )
    except XaiExplanationNotFoundError as error:
        raise _not_found() from error
    except (XaiRetryRequiredError, XaiReplayUnavailableError) as error:
        raise _conflict(error) from error
    except XaiOrchestrationError as error:
        if "completed prediction" in str(error):
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from error
        raise _unavailable(error) from error
