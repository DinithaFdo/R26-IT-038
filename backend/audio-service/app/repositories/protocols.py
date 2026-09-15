from typing import Any, Protocol

from app.auth.schemas import AuthPrincipal
from app.schemas.common import PredictionLabel, PredictionStatus, SourceType
from app.schemas.prediction import VoicePredictionResponse


class PredictionRepository(Protocol):
    """Persistence contract for future prediction records."""

    async def create_prediction(
        self,
        *,
        request_id: str,
        owner_user_id: str,
        source_type: SourceType,
        client_correlation_id: str | None = None,
        idempotency_key: str | None = None,
        logical_request: dict[str, Any] | None = None,
        parent_prediction_id: str | None = None,
        rerun_reason: str | None = None,
        preprocessing_version: str | None = None,
        model_versions: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError

    async def reserve_prediction(
        self,
        *,
        request_id: str,
        owner_user_id: str,
        source_type: SourceType,
        client_correlation_id: str | None,
        idempotency_key: str,
        logical_request: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError

    async def update_prediction_status(
        self,
        request_id: str,
        status: PredictionStatus,
        *,
        error_code: str | None = None,
        error_stage: str | None = None,
    ) -> None:
        raise NotImplementedError

    async def attach_upload_metadata(
        self,
        request_id: str,
        upload_metadata,
    ) -> None:
        raise NotImplementedError

    async def attach_cloudinary_asset(
        self,
        request_id: str,
        storage_metadata,
    ) -> None:
        raise NotImplementedError

    async def save_prediction_result(
        self,
        request_id: str,
        prediction: VoicePredictionResponse,
        *,
        status: PredictionStatus,
        error_codes: list[dict],
    ) -> None:
        raise NotImplementedError

    async def save_prediction(
        self,
        prediction: VoicePredictionResponse,
        *,
        principal: AuthPrincipal | None = None,
        source_type: SourceType = SourceType.public_api,
    ) -> str:
        raise NotImplementedError

    async def get_prediction_by_request_id(
        self,
        request_id: str,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    async def list_predictions_for_owner(
        self,
        *,
        owner_user_id: str,
        page: int,
        limit: int,
        status: PredictionStatus | None = None,
        source_type: SourceType | None = None,
        prediction_label: PredictionLabel | None = None,
        created_from: Any | None = None,
        created_to: Any | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def get_prediction_for_owner(
        self,
        *,
        prediction_id: str,
        owner_user_id: str,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    async def soft_delete_prediction_for_owner(
        self,
        *,
        prediction_id: str,
        owner_user_id: str,
    ) -> bool:
        raise NotImplementedError

    async def record_storage_reconciliation_event(
        self,
        *,
        request_id: str,
        event: Any,
    ) -> None:
        raise NotImplementedError


class UserRepository(Protocol):
    """Persistence contract for future user and organisation lookups."""

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    async def get_user_by_clerk_user_id(
        self,
        clerk_user_id: str,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    async def upsert_user(self, user: dict[str, Any]) -> dict[str, Any] | None:
        raise NotImplementedError


class ApiKeyRepository(Protocol):
    """Persistence contract for future API-key authentication."""

    async def create_api_key(self, document: dict[str, Any]) -> str:
        raise NotImplementedError

    async def list_api_keys_for_owner(
        self,
        owner_user_id: str,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def get_api_key_by_prefix(
        self,
        key_prefix: str,
    ) -> dict[str, Any] | None:
        raise NotImplementedError

    async def revoke_api_key_for_owner(
        self,
        *,
        api_key_id: str,
        owner_user_id: str,
    ) -> bool:
        raise NotImplementedError

    async def mark_revoked(self, key_prefix: str) -> bool:
        raise NotImplementedError

    async def mark_api_key_used(self, api_key_id: str) -> None:
        raise NotImplementedError

    async def record_audit_event(self, event: dict[str, Any]) -> None:
        raise NotImplementedError
