from typing import Any

USERS_COLLECTION = "users"
PREDICTIONS_COLLECTION = "predictions"
API_KEYS_COLLECTION = "api_keys"
AUDIT_EVENTS_COLLECTION = "audit_events"
XAI_EXPLANATIONS_COLLECTION = "xai_explanations"

REQUIRED_COLLECTIONS = (
    USERS_COLLECTION,
    PREDICTIONS_COLLECTION,
    API_KEYS_COLLECTION,
    AUDIT_EVENTS_COLLECTION,
    XAI_EXPLANATIONS_COLLECTION,
)


async def ensure_collections(database: Any) -> None:
    existing_collections = set(await database.list_collection_names())
    for collection_name in REQUIRED_COLLECTIONS:
        if collection_name not in existing_collections:
            await database.create_collection(collection_name)
