from datetime import UTC, datetime
import hashlib
import hmac
import logging
import secrets
from typing import Any
from uuid import uuid4

from fastapi import Depends, Header, HTTPException, Request, status

from app.auth.schemas import AuthPrincipal
from app.config.settings import Settings, settings
from app.repositories.mongodb import MongoApiKeyRepository
from app.repositories.protocols import ApiKeyRepository
from app.schemas.api_keys import ApiKeyScope

logger = logging.getLogger(__name__)

API_KEY_PREFIX = "msk_live_"
API_KEY_SECRET_BYTES = 32
API_KEY_LOOKUP_PREFIX_LENGTH = 24


def generate_api_key() -> str:
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(API_KEY_SECRET_BYTES)}"


def key_prefix(api_key: str) -> str:
    return api_key[:API_KEY_LOOKUP_PREFIX_LENGTH]


def hash_api_key(api_key: str, app_settings: Settings = settings) -> str:
    return hmac.new(
        app_settings.api_key_hash_secret.encode("utf-8"),
        api_key.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


async def get_api_key_repository() -> ApiKeyRepository:
    return MongoApiKeyRepository()


async def require_api_key_principal(
    request: Request,
    authorization: str | None = Header(default=None),
    repository: ApiKeyRepository = Depends(get_api_key_repository),
) -> AuthPrincipal:
    token = _extract_api_key(authorization)
    return await verify_api_key_token(
        token,
        repository=repository,
        request_path=request.url.path if request is not None else None,
    )


async def verify_api_key_token(
    token: str,
    *,
    repository: ApiKeyRepository,
    request_path: str | None = None,
) -> AuthPrincipal:
    prefix = key_prefix(token)
    document = await repository.get_api_key_by_prefix(prefix)
    if document is None:
        await _audit(repository, None, "api_key_auth_failed", request_path)
        _raise_invalid_api_key()

    if not _hash_matches(token, document):
        await _audit(repository, document, "api_key_auth_failed", request_path)
        _raise_invalid_api_key()
    if document.get("revoked_at") is not None:
        await _audit(repository, document, "api_key_auth_revoked", request_path)
        _raise_invalid_api_key()
    if _is_expired(document.get("expires_at")):
        await _audit(repository, document, "api_key_auth_expired", request_path)
        _raise_invalid_api_key()

    await repository.mark_api_key_used(str(document["id"]))
    await _audit(repository, document, "api_key_auth_success", request_path)
    return AuthPrincipal(
        subject=f"api_key:{document['id']}",
        principal_type="api_key",
        user_id=document["owner_user_id"],
        api_key_id=str(document["id"]),
        scopes=list(document.get("scopes") or []),
    )


def require_api_key_scope(required_scope: ApiKeyScope):
    async def dependency(
        request: Request,
        principal: AuthPrincipal = Depends(require_api_key_principal),
        repository: ApiKeyRepository = Depends(get_api_key_repository),
    ) -> AuthPrincipal:
        if required_scope.value not in principal.scopes:
            await repository.record_audit_event(
                {
                    "event_type": "api_key_missing_scope",
                    "owner_user_id": principal.user_id,
                    "api_key_id": principal.api_key_id,
                    "scope": required_scope.value,
                    "path": request.url.path if request is not None else None,
                    "created_at": datetime.now(UTC),
                }
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key does not have the required scope.",
            )
        return principal

    return dependency


def _extract_api_key(authorization: str | None) -> str:
    if authorization is None:
        _raise_invalid_api_key()
    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        _raise_invalid_api_key()
    token = parts[1].strip()
    if not token.startswith(API_KEY_PREFIX):
        _raise_invalid_api_key()
    return token


def _hash_matches(api_key: str, document: dict[str, Any]) -> bool:
    stored_hash = str(document.get("key_hash") or "")
    computed_hash = hash_api_key(api_key)
    return hmac.compare_digest(stored_hash, computed_hash)


def _is_expired(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, datetime):
        expires_at = value
    else:
        try:
            expires_at = datetime.fromisoformat(str(value))
        except ValueError:
            return True
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at.astimezone(UTC) <= datetime.now(UTC)


async def _audit(
    repository: ApiKeyRepository,
    document: dict[str, Any] | None,
    event_type: str,
    request_path: str | None,
) -> None:
    try:
        await repository.record_audit_event(
            {
                "id": str(uuid4()),
                "event_type": event_type,
                "owner_user_id": document.get("owner_user_id") if document else None,
                "api_key_id": str(document.get("id")) if document else None,
                "path": request_path,
                "created_at": datetime.now(UTC),
            }
        )
    except Exception:
        logger.exception("API key audit event failed.")


def _raise_invalid_api_key() -> None:
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API key.",
        headers={"WWW-Authenticate": "Bearer"},
    )
