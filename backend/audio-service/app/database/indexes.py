from typing import Any

from app.database.collections import (
    API_KEYS_COLLECTION,
    AUDIT_EVENTS_COLLECTION,
    PREDICTIONS_COLLECTION,
    USERS_COLLECTION,
    XAI_EXPLANATIONS_COLLECTION,
)

ASCENDING = 1
DESCENDING = -1


async def ensure_indexes(database: Any) -> None:
    users = database[USERS_COLLECTION]
    predictions = database[PREDICTIONS_COLLECTION]
    api_keys = database[API_KEYS_COLLECTION]
    audit_events = database[AUDIT_EVENTS_COLLECTION]
    xai_explanations = database[XAI_EXPLANATIONS_COLLECTION]

    await users.create_index(
        [("clerk_user_id", ASCENDING)],
        unique=True,
        name="uniq_users_clerk_user_id",
    )

    await predictions.create_index(
        [("owner_user_id", ASCENDING), ("created_at", DESCENDING)],
        name="idx_predictions_owner_created_at_desc",
    )
    await predictions.create_index(
        [("request_id", ASCENDING)],
        unique=True,
        name="uniq_predictions_request_id",
    )
    await predictions.create_index(
        [("status", ASCENDING)],
        name="idx_predictions_status",
    )
    await predictions.create_index(
        [("source_type", ASCENDING)],
        name="idx_predictions_source_type",
    )
    await predictions.create_index(
        [("owner_user_id", ASCENDING), ("idempotency_key", ASCENDING)],
        unique=True,
        partialFilterExpression={"idempotency_key": {"$type": "string"}},
        name="uniq_predictions_owner_idempotency_key",
    )

    await api_keys.create_index(
        [("key_prefix", ASCENDING)],
        unique=True,
        name="uniq_api_keys_key_prefix",
    )
    await api_keys.create_index(
        [("owner_user_id", ASCENDING)],
        name="idx_api_keys_owner_user_id",
    )
    await api_keys.create_index(
        [("revoked_at", ASCENDING)],
        name="idx_api_keys_revoked_at",
    )
    await api_keys.create_index(
        [("expires_at", ASCENDING)],
        name="idx_api_keys_expires_at",
    )

    await audit_events.create_index(
        [("owner_user_id", ASCENDING), ("created_at", DESCENDING)],
        name="idx_audit_events_owner_created_at_desc",
    )
    await audit_events.create_index(
        [("api_key_id", ASCENDING)],
        name="idx_audit_events_api_key_id",
    )

    await xai_explanations.create_index(
        [("id", ASCENDING)],
        unique=True,
        name="uniq_xai_explanations_id",
    )
    await xai_explanations.create_index(
        [
            ("owner_user_id", ASCENDING),
            ("prediction_id", ASCENDING),
            ("created_at", DESCENDING),
        ],
        name="idx_xai_explanations_owner_prediction_created_desc",
    )
    await xai_explanations.create_index(
        [("prediction_id", ASCENDING), ("created_at", DESCENDING)],
        name="idx_xai_explanations_prediction_created_desc",
    )
    await xai_explanations.create_index(
        [("status", ASCENDING), ("updated_at", ASCENDING)],
        name="idx_xai_explanations_status_updated",
    )
    await xai_explanations.create_index(
        [("owner_user_id", ASCENDING), ("prediction_id", ASCENDING)],
        unique=True,
        partialFilterExpression={
            "status": {"$in": ["queued", "running", "partial"]},
        },
        name="uniq_xai_active_owner_prediction",
    )
