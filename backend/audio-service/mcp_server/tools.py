from __future__ import annotations

import base64
from dataclasses import dataclass
import logging
from pathlib import Path
from tempfile import SpooledTemporaryFile
from time import perf_counter
from typing import Any, Awaitable, Callable, TypeVar
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from app.config.settings import Settings, settings
from app.core.request_context import normalize_request_id, reset_request_id, set_request_id
from app.repositories.mongodb import MongoApiKeyRepository, MongoPredictionRepository
from app.repositories.protocols import ApiKeyRepository, PredictionRepository
from app.schemas.common import PredictionLabel, PredictionStatus, SourceType
from app.schemas.prediction_history import PredictionDeleteResponse
from app.schemas.prediction_submission import PredictionSubmissionResponse
from app.services.prediction_history_service import PredictionHistoryService
from app.services.prediction_job_runner import InlinePredictionJobRunner, PredictionJobRunner
from app.services.prediction_persistence_service import PredictionPersistenceService
from app.services.prediction_submission_service import PredictionSubmissionService
from app.services.voice_service import ModelRegistry, VoiceService
from app.storage.factory import get_audio_storage
from app.storage.protocols import AudioStorage
from app.voice_xai.artifacts.service import LocalExplanationArtifactStore
from app.voice_xai.persistence.mongodb import MongoXaiExplanationRepository

from mcp_server.auth import MCPAuthError, authenticate_mcp_token
from mcp_server.schemas import (
    MCPCreatePredictionInput,
    MCPDeletePredictionInput,
    MCPGetPredictionInput,
    MCPListPredictionsInput,
    MCPModelStatusInput,
    MCPToolResponse,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class MCPToolContext:
    api_key_repository: ApiKeyRepository
    prediction_repository: PredictionRepository
    storage: AudioStorage
    model_registry: ModelRegistry
    job_runner: PredictionJobRunner
    submission_service: PredictionSubmissionService
    history_service: PredictionHistoryService
    voice_service: VoiceService


def create_default_context(app_settings: Settings = settings) -> MCPToolContext:
    prediction_repository = MongoPredictionRepository()
    api_key_repository = MongoApiKeyRepository()
    storage = get_audio_storage(app_settings)
    model_registry = ModelRegistry(app_settings=app_settings)
    voice_service = VoiceService(
        model_registry=model_registry,
        app_settings=app_settings,
    )
    persistence = PredictionPersistenceService(prediction_repository)
    job_runner: PredictionJobRunner = InlinePredictionJobRunner(app_settings)
    xai_repository = MongoXaiExplanationRepository()
    xai_artifact_store = LocalExplanationArtifactStore(
        app_settings.resolve_xai_artifact_path(app_settings.xai_artifact_root),
        retention_seconds=app_settings.xai_artifact_retention_seconds,
    )
    return MCPToolContext(
        api_key_repository=api_key_repository,
        prediction_repository=prediction_repository,
        storage=storage,
        model_registry=model_registry,
        job_runner=job_runner,
        submission_service=PredictionSubmissionService(
            repository=prediction_repository,
            persistence=persistence,
            storage=storage,
            voice_service=voice_service,
            job_runner=job_runner,
            app_settings=app_settings,
        ),
        history_service=PredictionHistoryService(
            repository=prediction_repository,
            storage=storage,
            xai_repository=xai_repository,
            xai_artifact_store=xai_artifact_store,
        ),
        voice_service=voice_service,
    )


_shared_context: MCPToolContext | None = None


def set_shared_context(context: MCPToolContext) -> None:
    global _shared_context
    _shared_context = context


def clear_shared_context() -> None:
    global _shared_context
    _shared_context = None


def get_shared_context() -> MCPToolContext:
    if _shared_context is None:
        raise MCPToolError(
            code="mcp_server_not_ready",
            message="MCP server is not ready.",
        )
    return _shared_context


def get_optional_shared_context() -> MCPToolContext | None:
    return _shared_context


async def multiscope_create_prediction(
    *,
    api_token: str,
    filename: str | None = None,
    content_type: str | None = None,
    client_filename: str | None = None,
    idempotency_key: str | None = None,
    secure_audio_asset_id: str | None = None,
    signed_upload_reference: str | None = None,
    small_audio_payload_base64: str | None = None,
    request_id: str | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    payload = MCPCreatePredictionInput(
        api_token=api_token,
        filename=filename,
        content_type=content_type,
        client_filename=client_filename,
        idempotency_key=idempotency_key,
        secure_audio_asset_id=secure_audio_asset_id,
        signed_upload_reference=signed_upload_reference,
        small_audio_payload_base64=small_audio_payload_base64,
    )
    return await _run_tool(
        tool_name="multiscope_create_prediction",
        api_token=payload.api_token,
        request_id=request_id,
        context=context,
        action=lambda principal, resolved_request_id, tool_context: _create_prediction(
            payload=payload,
            principal=principal,
            request_id=resolved_request_id,
            context=tool_context,
        ),
    )


async def multiscope_get_prediction(
    *,
    api_token: str,
    prediction_id: str,
    request_id: str | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    payload = MCPGetPredictionInput(api_token=api_token, prediction_id=prediction_id)
    return await _run_tool(
        tool_name="multiscope_get_prediction",
        api_token=payload.api_token,
        request_id=request_id,
        context=context,
        action=lambda principal, _request_id, tool_context: _get_prediction(
            payload=payload,
            principal=principal,
            context=tool_context,
        ),
    )


async def multiscope_list_predictions(
    *,
    api_token: str,
    page: int = 1,
    limit: int = 20,
    status: PredictionStatus | None = None,
    source_type: SourceType | None = None,
    prediction_label: PredictionLabel | None = None,
    created_from=None,
    created_to=None,
    request_id: str | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    payload = MCPListPredictionsInput(
        api_token=api_token,
        page=page,
        limit=limit,
        status=status,
        source_type=source_type,
        prediction_label=prediction_label,
        created_from=created_from,
        created_to=created_to,
    )
    return await _run_tool(
        tool_name="multiscope_list_predictions",
        api_token=payload.api_token,
        request_id=request_id,
        context=context,
        action=lambda principal, _request_id, tool_context: _list_predictions(
            payload=payload,
            principal=principal,
            context=tool_context,
        ),
    )


async def multiscope_get_model_status(
    *,
    api_token: str,
    request_id: str | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    payload = MCPModelStatusInput(api_token=api_token)
    return await _run_tool(
        tool_name="multiscope_get_model_status",
        api_token=payload.api_token,
        request_id=request_id,
        context=context,
        action=lambda _principal, _request_id, tool_context: _get_model_status(
            context=tool_context,
        ),
    )


async def multiscope_delete_prediction(
    *,
    api_token: str,
    prediction_id: str,
    request_id: str | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    payload = MCPDeletePredictionInput(api_token=api_token, prediction_id=prediction_id)
    return await _run_tool(
        tool_name="multiscope_delete_prediction",
        api_token=payload.api_token,
        request_id=request_id,
        context=context,
        action=lambda principal, _request_id, tool_context: _delete_prediction(
            payload=payload,
            principal=principal,
            context=tool_context,
        ),
    )


async def _create_prediction(
    *,
    payload: MCPCreatePredictionInput,
    principal,
    request_id: str,
    context: MCPToolContext,
) -> dict[str, Any]:
    if payload.secure_audio_asset_id or payload.signed_upload_reference:
        raise MCPToolError(
            code="unsupported_audio_reference",
            message=(
                "This controlled MCP implementation currently accepts only "
                "bounded small_audio_payload_base64 submissions."
            ),
        )

    upload_file = _upload_file_from_small_payload(payload)
    try:
        response = await context.submission_service.submit(
            file=upload_file,
            principal=principal,
            source_type=SourceType.mcp,
            request_id=request_id,
            client_filename=payload.client_filename or payload.filename,
            idempotency_key=payload.idempotency_key,
        )
    finally:
        await upload_file.close()
    return _compact_submission(response)


async def _get_prediction(
    *,
    payload: MCPGetPredictionInput,
    principal,
    context: MCPToolContext,
) -> dict[str, Any]:
    detail = await context.history_service.get_prediction_detail(
        principal=principal,
        prediction_id=payload.prediction_id,
    )
    return {
        "prediction_id": detail.prediction_id,
        "request_id": detail.request_id,
        "status": detail.status.value,
        "source_type": detail.source_type.value,
        "audio": detail.audio.model_dump(mode="json"),
        "branches": [
            _compact_branch(branch.model_dump(mode="python"))
            for branch in detail.branches
        ],
        "fusion": detail.fusion.model_dump(mode="json") if detail.fusion else None,
        "research_eligible": detail.research_eligible,
        "warnings": detail.warnings,
        "created_at": detail.created_at.isoformat(),
        "completed_at": detail.completed_at.isoformat()
        if detail.completed_at
        else None,
    }


async def _list_predictions(
    *,
    payload: MCPListPredictionsInput,
    principal,
    context: MCPToolContext,
) -> dict[str, Any]:
    response = await context.history_service.list_predictions(
        principal=principal,
        page=payload.page,
        limit=payload.limit,
        status_filter=payload.status,
        source_type=payload.source_type,
        prediction_label=payload.prediction_label,
        created_from=payload.created_from,
        created_to=payload.created_to,
    )
    return {
        "items": [item.model_dump(mode="json") for item in response.items],
        "page": response.page,
        "limit": response.limit,
        "has_next": response.has_next,
    }


async def _get_model_status(*, context: MCPToolContext) -> dict[str, Any]:
    """Report each branch's real state to an MCP client.

    An agent consuming this needs to distinguish three things a plain
    ``is_loaded`` flag conflates: a branch with no trained model in this
    deployment, a placeholder, and a real checkpoint whose feature pipeline or
    label mapping has not been verified. All three mean "do not treat this as a
    research result", but for different reasons.
    """

    return {
        "models": [
            {
                "model_name": item.get("model_name"),
                "display_name": item.get("display_name"),
                "mode": item.get("mode"),
                "is_loaded": bool(item.get("is_loaded", item.get("loaded", False))),
                "development_placeholder": item.get("mode") == "dummy",
                "available": item.get("mode") != "disabled",
                "research_ready": bool(item.get("research_ready", False)),
                "preprocessing_verified": bool(item.get("preprocessing_verified", False)),
                "class_mapping_verified": bool(item.get("class_mapping_verified", False)),
            }
            for item in context.voice_service.model_health()
        ]
    }


async def _delete_prediction(
    *,
    payload: MCPDeletePredictionInput,
    principal,
    context: MCPToolContext,
) -> dict[str, Any]:
    response: PredictionDeleteResponse = await context.history_service.delete_prediction(
        principal=principal,
        prediction_id=payload.prediction_id,
    )
    return response.model_dump(mode="json")


def _upload_file_from_small_payload(payload: MCPCreatePredictionInput) -> UploadFile:
    if not payload.small_audio_payload_base64:
        raise MCPToolError(
            code="missing_audio_payload",
            message="A bounded small_audio_payload_base64 value is required.",
        )
    try:
        decoded = base64.b64decode(payload.small_audio_payload_base64, validate=True)
    except Exception as error:
        raise MCPToolError(
            code="invalid_audio_payload",
            message="small_audio_payload_base64 is not valid base64.",
        ) from error
    if not decoded:
        raise MCPToolError(
            code="empty_audio_payload",
            message="Audio payload is empty.",
        )
    if len(decoded) > settings.mcp_small_payload_max_bytes:
        raise MCPToolError(
            code="audio_payload_too_large",
            message=(
                "MCP small audio payload exceeds the configured byte limit. "
                "Use a secure upload handoff for normal-length audio."
            ),
        )

    filename = _safe_filename(payload.filename or payload.client_filename)
    file_object = SpooledTemporaryFile(max_size=settings.mcp_small_payload_max_bytes)
    file_object.write(decoded)
    file_object.seek(0)
    return UploadFile(
        file=file_object,
        filename=filename,
        headers=Headers({"content-type": payload.content_type or "application/octet-stream"}),
    )


async def _run_tool(
    *,
    tool_name: str,
    api_token: str,
    request_id: str | None,
    context: MCPToolContext | None,
    action: Callable[[Any, str, MCPToolContext], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    resolved_request_id = normalize_request_id(request_id)
    token = set_request_id(resolved_request_id)
    start_time = perf_counter()
    result_status = "failed"
    principal_id = "-"
    try:
        tool_context = context or get_shared_context()
        principal = await authenticate_mcp_token(
            api_token=api_token,
            tool_name=tool_name,
            repository=tool_context.api_key_repository,
        )
        principal_id = principal.user_id or principal.subject
        data = await action(principal, resolved_request_id, tool_context)
        result_status = str(data.get("status", "success"))
        return MCPToolResponse(
            ok=True,
            request_id=resolved_request_id,
            data=data,
        ).model_dump(mode="json")
    except MCPAuthError as error:
        return _error_response(
            request_id=resolved_request_id,
            code=error.code,
            message=error.message,
        )
    except MCPToolError as error:
        return _error_response(
            request_id=resolved_request_id,
            code=error.code,
            message=error.message,
        )
    except HTTPException as error:
        return _error_response(
            request_id=resolved_request_id,
            code=_http_error_code(error),
            message=_safe_http_message(error),
        )
    except Exception:
        logger.exception(
            "MCP tool failed unexpectedly.",
            extra={
                "tool_name": tool_name,
                "principal_id": principal_id,
                "request_id": resolved_request_id,
            },
        )
        return _error_response(
            request_id=resolved_request_id,
            code="internal_error",
            message="MCP tool execution failed.",
        )
    finally:
        logger.info(
            "MCP tool finished.",
            extra={
                "tool_name": tool_name,
                "principal_id": principal_id,
                "request_id": resolved_request_id,
                "status": result_status,
                "duration_ms": _elapsed_ms(start_time),
            },
        )
        reset_request_id(token)


def _compact_submission(response: PredictionSubmissionResponse) -> dict[str, Any]:
    fusion = response.fusion.model_dump(mode="json") if response.fusion else None
    return {
        "prediction_id": response.prediction_id,
        "request_id": response.request_id,
        "status": response.status.value,
        "source_type": response.source_type.value,
        "audio": response.audio.model_dump(mode="json"),
        "final_prediction": fusion.get("prediction") if fusion else None,
        "confidence": fusion.get("confidence") if fusion else None,
        "mode_summary": _mode_summary(response.branches),
        "fusion": fusion,
        "research_eligible": response.research_eligible,
        "created_at": response.created_at.isoformat(),
    }


def _compact_branch(branch: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": branch.get("model_name"),
        "display_name": branch.get("display_name"),
        "status": _enum_value(branch.get("status")),
        "mode": _enum_value(branch.get("mode")),
        "prediction": _enum_value(branch.get("prediction")),
        "confidence": branch.get("confidence"),
        "processing_time_ms": branch.get("processing_time_ms"),
        "development_placeholder": (
            _enum_value(branch.get("mode")) == "dummy"
            or bool((branch.get("metadata") or {}).get("development_placeholder"))
        ),
        "research_result": bool((branch.get("metadata") or {}).get("research_result")),
    }


def _mode_summary(branches) -> dict[str, Any]:
    dummy = 0
    real = 0
    for branch in branches:
        mode = getattr(branch, "mode", None)
        mode_value = mode.value if hasattr(mode, "value") else mode
        if mode_value == "dummy":
            dummy += 1
        elif mode_value == "real":
            real += 1
    return {
        "dummy": dummy,
        "real": real,
        "contains_dummy": dummy > 0,
    }


def _safe_filename(value: str | None) -> str:
    if value is None or not value.strip():
        return f"mcp-upload-{uuid4().hex}.wav"
    filename = Path(value.strip()).name
    if filename != value.strip() or not filename:
        raise MCPToolError(
            code="unsafe_filename",
            message="MCP audio filename must not include path components.",
        )
    return filename[:180]


def _error_response(*, request_id: str, code: str, message: str) -> dict[str, Any]:
    return MCPToolResponse(
        ok=False,
        request_id=request_id,
        error={
            "code": code,
            "message": message,
            "details": None,
        },
    ).model_dump(mode="json")


def _http_error_code(error: HTTPException) -> str:
    if error.status_code == 401:
        return "authentication_failed"
    if error.status_code == 403:
        return "permission_denied"
    if error.status_code == 404:
        return "not_found"
    if error.status_code == 409:
        return "conflict"
    if error.status_code == 413:
        return "payload_too_large"
    if error.status_code == 415:
        return "unsupported_media_type"
    if error.status_code == 422:
        return "validation_error"
    if error.status_code == 503:
        return "service_unavailable"
    return "http_error"


def _safe_http_message(error: HTTPException) -> str:
    detail = error.detail
    if isinstance(detail, str) and detail.strip():
        return detail.strip()[:240]
    return "MCP tool request failed."


def _enum_value(value: Any) -> Any:
    return value.value if hasattr(value, "value") else value


def _elapsed_ms(start_time: float) -> float:
    return max((perf_counter() - start_time) * 1000, 0.0)


@dataclass
class MCPToolError(Exception):
    code: str
    message: str
