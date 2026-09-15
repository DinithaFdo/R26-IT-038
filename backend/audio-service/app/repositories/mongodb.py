from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from app.auth.schemas import AuthPrincipal
from app.database.collections import (
    API_KEYS_COLLECTION,
    AUDIT_EVENTS_COLLECTION,
    PREDICTIONS_COLLECTION,
    USERS_COLLECTION,
)
from app.database.mongodb import get_mongodb_database

from app.schemas.common import (
    BranchStatus,
    ModelMode,
    PredictionLabel,
    PredictionStatus,
    SourceType,
    is_valid_prediction_status_transition,
)
from app.schemas.prediction import AudioStorageMetadata, VoicePredictionResponse

try:
    from bson import ObjectId
except ImportError:
    ObjectId = None

try:
    from pymongo import ReturnDocument
except ImportError:
    ReturnDocument = None


class MongoUserRepository:
    def __init__(self, database: Any | None = None) -> None:
        self._database = database

    @property
    def collection(self):
        database = self._database or get_mongodb_database()
        return database[USERS_COLLECTION]

    async def get_user(self, user_id: str) -> dict[str, Any] | None:
        document = await self.collection.find_one({"_id": _to_object_id(user_id)})
        return _serialize_document(document)

    async def get_user_by_clerk_user_id(
        self,
        clerk_user_id: str,
    ) -> dict[str, Any] | None:
        document = await self.collection.find_one({"clerk_user_id": clerk_user_id})
        return _serialize_document(document)

    async def upsert_user(self, user: dict[str, Any]) -> dict[str, Any] | None:
        now = _utc_now()
        # created_at is server-owned and must only ever be written by
        # $setOnInsert. Excluding it here (rather than trusting callers never
        # to pass it) keeps $set and $setOnInsert from ever touching the same
        # path, which MongoDB rejects outright (WriteError code 40) even on
        # the very first insert.
        mutable_fields = {
            key: value for key, value in user.items() if key != "created_at"
        }
        mutable_fields["updated_at"] = now
        clerk_user_id = mutable_fields["clerk_user_id"]
        await self.collection.update_one(
            {"clerk_user_id": clerk_user_id},
            {"$set": mutable_fields, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        return await self.get_user_by_clerk_user_id(clerk_user_id)


class MongoPredictionRepository:
    def __init__(self, database: Any | None = None) -> None:
        self._database = database

    @property
    def collection(self):
        database = self._database or get_mongodb_database()
        return database[PREDICTIONS_COLLECTION]

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
        prediction_id = str(uuid4())
        document = _new_prediction_document(
            prediction_id=prediction_id,
            request_id=request_id,
            owner_user_id=owner_user_id,
            source_type=source_type,
            client_correlation_id=client_correlation_id,
            idempotency_key=idempotency_key,
            logical_request=logical_request,
            parent_prediction_id=parent_prediction_id,
            rerun_reason=rerun_reason,
            preprocessing_version=preprocessing_version,
            model_versions=model_versions,
        )
        await self.collection.insert_one(document)
        return prediction_id

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
        prediction_id = str(uuid4())
        document = _new_prediction_document(
            prediction_id=prediction_id,
            request_id=request_id,
            owner_user_id=owner_user_id,
            source_type=source_type,
            client_correlation_id=client_correlation_id,
            idempotency_key=idempotency_key,
            logical_request=logical_request,
        )
        reserved = await self.collection.find_one_and_update(
            {
                "owner_user_id": owner_user_id,
                "idempotency_key": idempotency_key,
            },
            {"$setOnInsert": document},
            upsert=True,
            return_document=(
                ReturnDocument.AFTER if ReturnDocument is not None else True
            ),
        )
        serialized = _serialize_document(reserved)
        return {
            "created": serialized["request_id"] == request_id,
            "document": serialized,
        }

    async def update_prediction_status(
        self,
        request_id: str,
        status: PredictionStatus,
        *,
        error_code: str | None = None,
        error_stage: str | None = None,
    ) -> None:
        now = _utc_now()
        current_status = await self._current_status_for_request(request_id)
        _validate_status_transition(current_status, status)
        update = {
            "$set": {
                "status": status.value,
                "updated_at": now,
            },
            "$push": {
                "status_history": _status_history_entry(status),
            },
        }
        if status in {PredictionStatus.completed, PredictionStatus.failed}:
            update["$set"]["completed_at"] = now
        if status == PredictionStatus.deleted:
            update["$set"]["deleted_at"] = now
        if error_code is not None:
            update["$push"]["error_summary"] = _error_summary_entry(
                stage=error_stage or "unknown",
                code=error_code,
            )
        result = await self.collection.update_one(
            {"request_id": request_id, "status": current_status.value},
            update,
        )
        _require_matched_one(result, "update_prediction_status")

    async def attach_upload_metadata(
        self,
        request_id: str,
        upload_metadata,
    ) -> None:
        result = await self.collection.update_one(
            {"request_id": request_id},
            {
                "$set": {
                    "original_filename": upload_metadata.sanitized_filename,
                    "original_extension": upload_metadata.original_extension,
                    "detected_container": upload_metadata.detected_container,
                    "detected_codec": upload_metadata.detected_codec,
                    "duration_seconds": upload_metadata.duration_seconds,
                    "sample_rate": upload_metadata.sample_rate,
                    "channels": upload_metadata.channels,
                    "size_bytes": upload_metadata.size_bytes,
                    "updated_at": _utc_now(),
                }
            },
        )
        _require_matched_one(result, "attach_upload_metadata")

    async def attach_cloudinary_asset(
        self,
        request_id: str,
        storage_metadata: AudioStorageMetadata,
    ) -> None:
        result = await self.collection.update_one(
            {"request_id": request_id},
            {
                "$set": {
                    "cloudinary_asset": _cloudinary_asset_document(storage_metadata),
                    "storage_status": storage_metadata.status.value,
                    "storage_error": storage_metadata.error,
                    "updated_at": _utc_now(),
                }
            },
        )
        _require_matched_one(result, "attach_cloudinary_asset")

    async def save_prediction_result(
        self,
        request_id: str,
        prediction: VoicePredictionResponse,
        *,
        status: PredictionStatus,
        error_codes: list[dict],
    ) -> None:
        now = _utc_now()
        current_status = await self._current_status_for_request(request_id)
        _validate_status_transition(current_status, status)
        research_eligible = _research_eligible(prediction)
        update = {
            "$set": {
                "status": status.value,
                "preprocessing": _preprocessing_document(prediction),
                "provenance": _provenance_document(prediction),
                "branches": [
                    branch.model_dump(mode="python") for branch in prediction.branches
                ],
                "fusion": prediction.fusion.model_dump(mode="python"),
                "research_eligible": research_eligible,
                "total_processing_time_ms": prediction.total_processing_time_ms,
                "updated_at": now,
                "completed_at": now,
            },
            "$push": {
                "status_history": _status_history_entry(status),
            },
        }
        if error_codes:
            update["$push"]["error_summary"] = {"$each": error_codes}
        result = await self.collection.update_one(
            {"request_id": request_id, "status": current_status.value},
            update,
        )
        _require_matched_one(result, "save_prediction_result")

    async def save_prediction(
        self,
        prediction: VoicePredictionResponse,
        *,
        principal: AuthPrincipal | None = None,
        source_type: SourceType = SourceType.public_api,
    ) -> str:
        now = _utc_now()
        document = prediction.model_dump(mode="python")
        document.update(
            {
                "owner_user_id": principal.user_id if principal else None,
                "organisation_id": principal.organisation_id if principal else None,
                "source_type": source_type.value,
                "status": _prediction_status_from_response(prediction).value,
                "created_at": _ensure_utc(prediction.created_at),
                "persisted_at": now,
            }
        )
        result = await self.collection.insert_one(document)
        return str(result.inserted_id)

    async def get_prediction_by_request_id(
        self,
        request_id: str,
    ) -> dict[str, Any] | None:
        document = await self.collection.find_one({"request_id": request_id})
        return _serialize_document(document)

    async def list_predictions_for_owner(
        self,
        *,
        owner_user_id: str,
        page: int,
        limit: int,
        status: PredictionStatus | None = None,
        source_type: SourceType | None = None,
        prediction_label: PredictionLabel | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        filter_document = _owner_history_filter(
            owner_user_id=owner_user_id,
            status=status,
            source_type=source_type,
            prediction_label=prediction_label,
            created_from=created_from,
            created_to=created_to,
        )
        cursor = (
            self.collection.find(filter_document)
            .sort("created_at", -1)
            .skip((page - 1) * limit)
            .limit(limit + 1)
        )
        documents = await cursor.to_list(length=limit + 1)
        return [_serialize_document(document) for document in documents]

    async def get_prediction_for_owner(
        self,
        *,
        prediction_id: str,
        owner_user_id: str,
        include_deleted: bool = False,
    ) -> dict[str, Any] | None:
        filter_document: dict[str, Any] = {
            "id": prediction_id,
            "owner_user_id": owner_user_id,
        }
        if not include_deleted:
            filter_document["status"] = {"$ne": PredictionStatus.deleted.value}
        document = await self.collection.find_one(filter_document)
        return _serialize_document(document)

    async def soft_delete_prediction_for_owner(
        self,
        *,
        prediction_id: str,
        owner_user_id: str,
    ) -> bool:
        now = _utc_now()
        document = await self.get_prediction_for_owner(
            prediction_id=prediction_id,
            owner_user_id=owner_user_id,
            include_deleted=True,
        )
        if document is None:
            return False
        current_status = PredictionStatus(document["status"])
        if current_status == PredictionStatus.deleted:
            return True
        _validate_status_transition(current_status, PredictionStatus.deleted)
        result = await self.collection.update_one(
            {
                "id": prediction_id,
                "owner_user_id": owner_user_id,
                "status": current_status.value,
            },
            {
                "$set": {
                    "status": PredictionStatus.deleted.value,
                    "cloudinary_asset": None,
                    "updated_at": now,
                    "deleted_at": now,
                    "audio_deleted_at": now,
                    "playback_revoked_at": now,
                },
                "$push": {
                    "status_history": _status_history_entry(
                        PredictionStatus.deleted
                    ),
                },
            },
        )
        _require_matched_one(result, "soft_delete_prediction_for_owner")
        return True

    async def record_storage_reconciliation_event(
        self,
        *,
        request_id: str,
        event: Any,
    ) -> None:
        now = _utc_now()
        event_document = _storage_reconciliation_event_document(event, now)
        update = {
            "$set": {
                "storage_reconciliation": {
                    "required": bool(
                        getattr(event, "retry_required", False)
                        or getattr(event, "deletion_failed", False)
                    ),
                    "last_event_type": getattr(event, "event_type", None),
                    "last_error": (
                        getattr(event, "event_type", None)
                        if getattr(event, "deletion_failed", False)
                        else None
                    ),
                    "updated_at": now,
                },
                "updated_at": now,
            },
            "$push": {
                "storage_reconciliation_events": event_document,
            },
        }
        if getattr(event, "retry_required", False) or getattr(
            event, "deletion_failed", False
        ):
            update["$inc"] = {"storage_reconciliation.attempts": 1}
        result = await self.collection.update_one(
            {"request_id": request_id},
            update,
        )
        _require_matched_one(result, "record_storage_reconciliation_event")

    async def _current_status_for_request(self, request_id: str) -> PredictionStatus:
        document = await self.collection.find_one(
            {"request_id": request_id},
        )
        if document is None:
            raise PersistenceConsistencyError("Prediction record was not found.")
        return PredictionStatus(document["status"])


class MongoApiKeyRepository:
    def __init__(self, database: Any | None = None) -> None:
        self._database = database

    @property
    def collection(self):
        database = self._database or get_mongodb_database()
        return database[API_KEYS_COLLECTION]

    @property
    def audit_collection(self):
        database = self._database or get_mongodb_database()
        return database[AUDIT_EVENTS_COLLECTION]

    async def create_api_key(self, document: dict[str, Any]) -> str:
        payload = {
            **document,
            "created_at": _ensure_utc(document["created_at"]),
        }
        await self.collection.insert_one(payload)
        return str(payload["id"])

    async def list_api_keys_for_owner(
        self,
        owner_user_id: str,
    ) -> list[dict[str, Any]]:
        cursor = self.collection.find({"owner_user_id": owner_user_id}).sort(
            "created_at",
            -1,
        )
        documents = await cursor.to_list(length=500)
        return [_serialize_document(document) for document in documents]

    async def get_api_key_by_prefix(
        self,
        key_prefix: str,
    ) -> dict[str, Any] | None:
        document = await self.collection.find_one({"key_prefix": key_prefix})
        return _serialize_document(document)

    async def revoke_api_key_for_owner(
        self,
        *,
        api_key_id: str,
        owner_user_id: str,
    ) -> bool:
        result = await self.collection.update_one(
            {"id": api_key_id, "owner_user_id": owner_user_id, "revoked_at": None},
            {"$set": {"revoked_at": _utc_now()}},
        )
        return result.modified_count == 1

    async def mark_revoked(self, key_prefix: str) -> bool:
        result = await self.collection.update_one(
            {"key_prefix": key_prefix, "revoked_at": None},
            {"$set": {"revoked_at": _utc_now()}},
        )
        return result.modified_count == 1

    async def mark_api_key_used(self, api_key_id: str) -> None:
        await self.collection.update_one(
            {"id": api_key_id},
            {
                "$set": {"last_used_at": _utc_now()},
                "$inc": {"usage_count": 1},
            },
        )

    async def record_audit_event(self, event: dict[str, Any]) -> None:
        payload = {
            "id": str(uuid4()),
            **event,
            "created_at": _ensure_utc(event.get("created_at", _utc_now())),
        }
        await self.audit_collection.insert_one(payload)


def _new_prediction_document(
    *,
    prediction_id: str,
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
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "id": prediction_id,
        "request_id": request_id,
        "client_correlation_id": client_correlation_id,
        "owner_user_id": owner_user_id,
        "source_type": source_type.value,
        "status": PredictionStatus.queued.value,
        "idempotency_key": idempotency_key,
        "idempotency_logical_request": logical_request,
        "parent_prediction_id": parent_prediction_id,
        "rerun_reason": rerun_reason,
        "preprocessing_version": preprocessing_version,
        "model_versions": model_versions or {},
        "original_filename": None,
        "original_extension": None,
        "detected_container": None,
        "detected_codec": None,
        "duration_seconds": None,
        "sample_rate": None,
        "channels": None,
        "size_bytes": None,
        "cloudinary_asset": None,
        "storage_status": None,
        "storage_error": None,
        "storage_reconciliation": {
            "required": False,
            "attempts": 0,
            "last_event_type": None,
            "last_error": None,
            "updated_at": None,
        },
        "storage_reconciliation_events": [],
        "preprocessing": {},
        "provenance": None,
        "branches": [],
        "fusion": None,
        "research_eligible": False,
        "error_summary": [],
        "status_history": [
            _status_history_entry(PredictionStatus.queued),
        ],
        "total_processing_time_ms": None,
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
        "deleted_at": None,
    }


class PersistenceConsistencyError(RuntimeError):
    """Raised when MongoDB prediction updates do not affect exactly one record."""


def _owner_history_filter(
    *,
    owner_user_id: str,
    status: PredictionStatus | None,
    source_type: SourceType | None,
    prediction_label: PredictionLabel | None,
    created_from: datetime | None,
    created_to: datetime | None,
) -> dict[str, Any]:
    filter_document: dict[str, Any] = {
        "owner_user_id": owner_user_id,
        "status": {"$ne": PredictionStatus.deleted.value},
    }
    if status is not None:
        if status == PredictionStatus.deleted:
            filter_document["status"] = PredictionStatus.deleted.value
        else:
            filter_document["status"] = status.value
    if source_type is not None:
        filter_document["source_type"] = source_type.value
    if prediction_label is not None:
        filter_document["fusion.prediction"] = prediction_label.value
    created_range = {}
    if created_from is not None:
        created_range["$gte"] = _ensure_utc(created_from)
    if created_to is not None:
        created_range["$lte"] = _ensure_utc(created_to)
    if created_range:
        filter_document["created_at"] = created_range
    return filter_document


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _to_object_id(value: str):
    if ObjectId is None:
        return value
    try:
        return ObjectId(value)
    except Exception:
        return value


def _serialize_document(document):
    if document is None:
        return None
    if ObjectId is not None and isinstance(document, ObjectId):
        return str(document)
    if isinstance(document, datetime):
        return _ensure_utc(document)
    if isinstance(document, list):
        return [_serialize_document(item) for item in document]
    if isinstance(document, dict):
        return {key: _serialize_document(value) for key, value in document.items()}
    return document


def _status_history_entry(status: PredictionStatus) -> dict[str, Any]:
    return {
        "status": status.value,
        "created_at": _utc_now(),
    }


def _validate_status_transition(
    current_status: PredictionStatus,
    next_status: PredictionStatus,
) -> None:
    if not is_valid_prediction_status_transition(current_status, next_status):
        raise PersistenceConsistencyError(
            f"Invalid prediction status transition: "
            f"{current_status.value} -> {next_status.value}."
        )


def _prediction_status_from_response(
    prediction: VoicePredictionResponse,
) -> PredictionStatus:
    if (
        prediction.fusion.status == BranchStatus.failed
        or not any(branch.status == BranchStatus.success for branch in prediction.branches)
    ):
        return PredictionStatus.failed
    return PredictionStatus.completed


def _require_matched_one(result: Any, operation: str) -> None:
    if getattr(result, "matched_count", None) != 1:
        raise PersistenceConsistencyError(
            f"MongoDB prediction update did not match exactly one record: {operation}."
        )


def _error_summary_entry(*, stage: str, code: str) -> dict[str, Any]:
    return {
        "stage": _safe_identifier(stage),
        "code": _safe_identifier(code),
        "created_at": _utc_now(),
    }


def _storage_reconciliation_event_document(event: Any, created_at: datetime) -> dict[str, Any]:
    return {
        "event_type": _safe_identifier(str(getattr(event, "event_type", "unknown"))),
        "public_id": getattr(event, "public_id", None),
        "owner_user_id": getattr(event, "owner_user_id", None),
        "asset_exists": bool(getattr(event, "asset_exists", False)),
        "deleted": bool(getattr(event, "deleted", False)),
        "deletion_failed": bool(getattr(event, "deletion_failed", False)),
        "retry_required": bool(getattr(event, "retry_required", False)),
        "created_at": created_at,
    }


def _cloudinary_asset_document(
    storage_metadata: AudioStorageMetadata,
) -> dict[str, Any] | None:
    if storage_metadata.status != BranchStatus.success:
        return None
    return {
        "asset_id": storage_metadata.asset_id,
        "public_id": storage_metadata.public_id,
        "resource_type": storage_metadata.resource_type,
        "version": storage_metadata.version,
        "format": storage_metadata.format,
        "bytes": storage_metadata.bytes,
        "duration": storage_metadata.duration,
        "created_at": storage_metadata.created_at,
    }


def _preprocessing_document(prediction: VoicePredictionResponse) -> dict[str, Any]:
    if prediction.provenance is not None:
        return prediction.provenance.preprocessing.model_dump(mode="python")
    return {
        "shared_representation": "mono_float32_waveform",
        "input_sample_rate": prediction.audio.sample_rate,
        "input_channels": prediction.audio.channels,
        "input_duration_seconds": prediction.audio.duration_seconds,
        "target_sample_rate": prediction.audio.sample_rate,
        "target_channels": prediction.audio.channels,
        "resampled": False,
        "mono_conversion_applied": False,
        "normalisation_applied": False,
        "preprocessing_version": "unknown",
        "ffmpeg_version": None,
        "ffprobe_version": None,
    }


def _provenance_document(prediction: VoicePredictionResponse) -> dict[str, Any] | None:
    if prediction.provenance is None:
        return None
    return prediction.provenance.model_dump(mode="python")


def _research_eligible(prediction: VoicePredictionResponse) -> bool:
    contains_dummy = any(branch.mode == ModelMode.dummy for branch in prediction.branches)
    return prediction.fusion.eligible_for_research_evaluation and not contains_dummy


def _safe_identifier(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in value.strip().lower()
    )[:80]
