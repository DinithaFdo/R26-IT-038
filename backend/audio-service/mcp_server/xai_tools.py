"""API-key authenticated adapters for the existing Voice XAI orchestrator."""

from __future__ import annotations

import base64
from typing import Any, Literal

from app.schemas.xai import XaiExplanationResponse
from app.voice_xai.workflow.orchestrator import (
    XaiArtifactUnavailableError,
    XaiNarrativeRetryError,
    XaiOrchestrationError,
    XaiReplayUnavailableError,
    XaiRetryRequiredError,
)
from app.voice_xai.workflow.status import XaiExplanationNotFoundError
from mcp_server.schemas import MCPExplanationArtifactInput, MCPExplanationInput
from mcp_server.tools import MCPToolContext, MCPToolError, _run_tool

MAX_MCP_ARTIFACT_BYTES = 1_048_576
Operation = Literal["create", "get", "retry", "narrative_retry", "artifact"]


async def multiscope_create_explanation(
    *, api_token: str, prediction_id: str, request_id: str | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Queue or reuse XAI for an owned completed prediction. Requires xai:create.

    Poll multiscope_get_explanation while queued/running. Manual runs need
    retained source audio; temporal analysis needs original-pass XLS-R evidence.
    """
    return await _call_xai_tool(
        "multiscope_create_explanation", api_token, prediction_id,
        request_id=request_id, context=context, operation="create",
    )


async def multiscope_get_explanation(
    *, api_token: str, prediction_id: str, request_id: str | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Get the latest owned XAI run, statuses and available evidence. Requires xai:read."""
    return await _call_xai_tool(
        "multiscope_get_explanation", api_token, prediction_id,
        request_id=request_id, context=context,
    )


async def multiscope_get_temporal_explanation(
    *, api_token: str, prediction_id: str, request_id: str | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Get an owned run's persisted temporal evidence. Requires xai:read."""
    return await _call_xai_tool(
        "multiscope_get_temporal_explanation", api_token, prediction_id,
        request_id=request_id, context=context, component="temporal",
    )


async def multiscope_get_semantic_explanation(
    *, api_token: str, prediction_id: str, request_id: str | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Get an owned run's persisted semantic/SHAP evidence. Requires xai:read."""
    return await _call_xai_tool(
        "multiscope_get_semantic_explanation", api_token, prediction_id,
        request_id=request_id, context=context, component="semantic",
    )


async def multiscope_get_explanation_report(
    *, api_token: str, prediction_id: str, request_id: str | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Get an owned run's deterministic combined XAI report. Requires xai:read."""
    return await _call_xai_tool(
        "multiscope_get_explanation_report", api_token, prediction_id,
        request_id=request_id, context=context, component="report",
    )


async def multiscope_get_explanation_narrative(
    *, api_token: str, prediction_id: str, request_id: str | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Get an owned run's optional AI narrative when available. Requires xai:read."""
    return await _call_xai_tool(
        "multiscope_get_explanation_narrative", api_token, prediction_id,
        request_id=request_id, context=context, component="narrative",
    )


async def multiscope_get_explanation_artifact(
    *, api_token: str, prediction_id: str, artifact_id: str,
    request_id: str | None = None, context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Read an owned public XAI artifact as base64 (up to 1 MiB). Requires xai:read.

    Use an artifact_id returned by multiscope_get_explanation. Internal temporal
    retry evidence and local filesystem paths are never exposed.
    """
    payload = MCPExplanationArtifactInput(
        api_token=api_token, prediction_id=prediction_id, artifact_id=artifact_id,
    )
    return await _call_xai_tool(
        "multiscope_get_explanation_artifact", payload.api_token, payload.prediction_id,
        request_id=request_id, context=context, operation="artifact",
        artifact_id=payload.artifact_id,
    )


async def multiscope_retry_explanation(
    *, api_token: str, prediction_id: str, request_id: str | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Queue a new versioned owned XAI run without reclassifying audio. Requires xai:retry."""
    return await _call_xai_tool(
        "multiscope_retry_explanation", api_token, prediction_id,
        request_id=request_id, context=context, operation="retry",
    )


async def multiscope_retry_explanation_narrative(
    *, api_token: str, prediction_id: str, request_id: str | None = None,
    context: MCPToolContext | None = None,
) -> dict[str, Any]:
    """Retry only the optional AI narrative using saved evidence. Requires xai:retry."""
    return await _call_xai_tool(
        "multiscope_retry_explanation_narrative", api_token, prediction_id,
        request_id=request_id, context=context, operation="narrative_retry",
    )


async def _call_xai_tool(
    tool_name: str, api_token: str, prediction_id: str, *,
    request_id: str | None, context: MCPToolContext | None,
    operation: Operation = "get", component: str | None = None,
    artifact_id: str | None = None,
) -> dict[str, Any]:
    payload = MCPExplanationInput(api_token=api_token, prediction_id=prediction_id)

    async def action(principal, _request_id, tool_context):
        orchestrator = tool_context.xai_orchestrator
        if orchestrator is None:
            raise MCPToolError("service_unavailable", "Voice XAI is not configured.")
        owner_user_id = principal.user_id or principal.subject
        arguments = {"prediction_id": payload.prediction_id, "owner_user_id": owner_user_id}
        try:
            if operation in {"create", "retry", "narrative_retry"}:
                _enforce_action_rate_limit(operation, owner_user_id, tool_context)
                if operation == "narrative_retry":
                    response = await orchestrator.retry_narrative_for_prediction(**arguments)
                else:
                    response = await orchestrator.trigger_for_prediction(
                        **arguments, retry=operation == "retry",
                    )
                return response.model_dump(mode="json")
            if operation == "artifact":
                artifact = await orchestrator.get_artifact_for_prediction(
                    **arguments, artifact_id=artifact_id,
                )
                if len(artifact.content) > MAX_MCP_ARTIFACT_BYTES:
                    raise MCPToolError(
                        "payload_too_large", "The artifact exceeds the MCP 1 MiB limit.",
                    )
                return {
                    "artifact": artifact.reference.model_dump(mode="json"),
                    "encoding": "base64",
                    "content_base64": base64.b64encode(artifact.content).decode("ascii"),
                }
            response = await orchestrator.get_for_prediction(**arguments)
            if component is not None:
                return _component_data(response, component)
            return response.model_dump(mode="json")
        except XaiExplanationNotFoundError as error:
            raise MCPToolError("not_found", "Explanation or artifact not found.") from error
        except (
            XaiRetryRequiredError, XaiReplayUnavailableError,
            XaiNarrativeRetryError, XaiArtifactUnavailableError,
        ) as error:
            raise MCPToolError("conflict", str(error)) from error
        except XaiOrchestrationError as error:
            code = "conflict" if "completed prediction" in str(error) else "service_unavailable"
            raise MCPToolError(code, str(error)) from error

    return await _run_tool(
        tool_name=tool_name, api_token=payload.api_token, request_id=request_id,
        context=context, action=action,
    )


def _component_data(response: XaiExplanationResponse, component: str) -> dict[str, Any]:
    value = getattr(response, "combined_report" if component == "report" else component)
    if value is not None:
        return value.model_dump(mode="json")
    component_status = getattr(response.component_statuses, component).value
    if response.status.value in {"queued", "running"} or component_status in {"queued", "running"}:
        raise MCPToolError("conflict", "Evidence is still being generated. Poll the explanation tool.")
    component_error = getattr(response.component_errors, component)
    error = component_error or response.error
    if error is not None:
        raise MCPToolError("conflict", error.message)
    raise MCPToolError("not_found", "The requested explanation component is unavailable.")


def _enforce_action_rate_limit(
    operation: str, owner_user_id: str, context: MCPToolContext,
) -> None:
    action = "trigger" if operation == "create" else operation
    prefix = {"trigger": "xai_trigger", "retry": "xai_retry", "narrative_retry": "xai_narrative_retry"}[action]
    decision = context.rate_limiter.check(
        f"xai:{action}:{owner_user_id}",
        limit=getattr(context.app_settings, f"{prefix}_rate_limit_requests"),
        window_seconds=getattr(context.app_settings, f"{prefix}_rate_limit_window_seconds"),
    )
    if not decision.allowed:
        raise MCPToolError(
            "rate_limited",
            f"Voice XAI request rate limit exceeded. Retry after {decision.retry_after_seconds} seconds.",
        )
