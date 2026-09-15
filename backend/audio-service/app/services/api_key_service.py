from datetime import UTC, datetime
from uuid import uuid4

from app.auth.api_keys import generate_api_key, hash_api_key, key_prefix
from app.auth.schemas import AuthPrincipal
from app.repositories.protocols import ApiKeyRepository
from app.schemas.api_keys import (
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyDeleteResponse,
    ApiKeyListResponse,
    ApiKeyMetadata,
)


class ApiKeyService:
    def __init__(self, repository: ApiKeyRepository) -> None:
        self._repository = repository

    async def create_api_key(
        self,
        *,
        principal: AuthPrincipal,
        request: ApiKeyCreateRequest,
    ) -> ApiKeyCreateResponse:
        full_key = generate_api_key()
        now = datetime.now(UTC)
        document = {
            "id": str(uuid4()),
            "key_hash": hash_api_key(full_key),
            "key_prefix": key_prefix(full_key),
            "owner_user_id": _owner_user_id(principal),
            "name": request.name.strip(),
            "scopes": [scope.value for scope in request.scopes],
            "created_at": now,
            "last_used_at": None,
            "expires_at": request.expires_at,
            "revoked_at": None,
            "usage_count": 0,
        }
        await self._repository.create_api_key(document)
        await self._repository.record_audit_event(
            {
                "event_type": "api_key_created",
                "owner_user_id": document["owner_user_id"],
                "api_key_id": document["id"],
                "created_at": now,
            }
        )
        return ApiKeyCreateResponse(
            **_metadata_payload(document),
            api_key=full_key,
        )

    async def list_api_keys(
        self,
        *,
        principal: AuthPrincipal,
    ) -> ApiKeyListResponse:
        documents = await self._repository.list_api_keys_for_owner(
            _owner_user_id(principal)
        )
        return ApiKeyListResponse(
            items=[ApiKeyMetadata(**_metadata_payload(document)) for document in documents]
        )

    async def delete_api_key(
        self,
        *,
        principal: AuthPrincipal,
        api_key_id: str,
    ) -> ApiKeyDeleteResponse:
        revoked = await self._repository.revoke_api_key_for_owner(
            api_key_id=api_key_id,
            owner_user_id=_owner_user_id(principal),
        )
        await self._repository.record_audit_event(
            {
                "event_type": "api_key_revoked",
                "owner_user_id": _owner_user_id(principal),
                "api_key_id": api_key_id,
                "created_at": datetime.now(UTC),
            }
        )
        return ApiKeyDeleteResponse(api_key_id=api_key_id, revoked=revoked)


def _metadata_payload(document: dict) -> dict:
    return {
        "api_key_id": str(document["id"]),
        "key_prefix": document["key_prefix"],
        "name": document["name"],
        "scopes": list(document.get("scopes") or []),
        "created_at": document["created_at"],
        "last_used_at": document.get("last_used_at"),
        "expires_at": document.get("expires_at"),
        "revoked_at": document.get("revoked_at"),
    }


def _owner_user_id(principal: AuthPrincipal) -> str:
    return principal.user_id or principal.subject
