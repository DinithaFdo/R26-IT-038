from dataclasses import dataclass

from fastapi import HTTPException

from app.auth.api_keys import API_KEY_PREFIX, verify_api_key_token
from app.auth.schemas import AuthPrincipal
from app.repositories.protocols import ApiKeyRepository
from app.schemas.api_keys import ApiKeyScope


MCP_SCOPE_MAP = {
    "multiscope_create_prediction": ApiKeyScope.prediction_create,
    "multiscope_get_prediction": ApiKeyScope.prediction_read,
    "multiscope_list_predictions": ApiKeyScope.prediction_list,
    "multiscope_get_model_status": ApiKeyScope.prediction_read,
    "multiscope_delete_prediction": ApiKeyScope.prediction_delete,
}


@dataclass
class MCPAuthError(Exception):
    code: str
    message: str


async def authenticate_mcp_token(
    *,
    api_token: str,
    tool_name: str,
    repository: ApiKeyRepository,
) -> AuthPrincipal:
    token = _normalize_api_token(api_token)
    try:
        principal = await verify_api_key_token(
            token,
            repository=repository,
            request_path=f"mcp:{tool_name}",
        )
    except HTTPException as error:
        raise MCPAuthError(
            code="authentication_failed",
            message="Invalid MCP API token.",
        ) from error

    required_scope = MCP_SCOPE_MAP[tool_name]
    if required_scope.value not in principal.scopes:
        try:
            await repository.record_audit_event(
                {
                    "event_type": "mcp_api_key_missing_scope",
                    "owner_user_id": principal.user_id,
                    "api_key_id": principal.api_key_id,
                    "scope": required_scope.value,
                    "path": f"mcp:{tool_name}",
                }
            )
        except Exception:
            pass
        raise MCPAuthError(
            code="permission_denied",
            message="MCP API token does not have the required scope.",
        )

    return principal


def _normalize_api_token(value: str) -> str:
    token = value.strip()
    if token.lower().startswith("bearer "):
        token = token.split(None, 1)[1].strip()
    if not token.startswith(API_KEY_PREFIX):
        raise MCPAuthError(
            code="authentication_failed",
            message="Invalid MCP API token.",
        )
    return token
