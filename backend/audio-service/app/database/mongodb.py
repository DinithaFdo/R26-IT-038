from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import inspect
import logging
from typing import Any
from urllib.parse import parse_qsl, urlparse

from app.config.settings import Settings, settings
from app.database.collections import ensure_collections
from app.database.indexes import ensure_indexes

logger = logging.getLogger(__name__)

_client: Any | None = None
_database: Any | None = None
_lifecycle_lock = asyncio.Lock()


class MongoDBStartupError(RuntimeError):
    """MongoDB startup failed with a sanitized, operator-actionable reason."""

    def __init__(self, reason: str, *, category: str, cause: Exception | None = None):
        super().__init__(reason)
        self.reason = reason
        self.category = category
        self.original_error = cause


@asynccontextmanager
async def mongodb_lifespan(
    app_settings: Settings = settings,
) -> AsyncIterator[None]:
    await connect_to_mongodb(app_settings)
    try:
        yield
    finally:
        await close_mongodb()


async def connect_to_mongodb(app_settings: Settings = settings) -> None:
    global _client, _database

    async with _lifecycle_lock:
        _validate_mongodb_settings(app_settings)
        if not app_settings.mongodb_uri.strip():
            logger.info("MongoDB is not configured; startup will continue without it.")
            _client = None
            _database = None
            return

        summary = mongodb_safe_summary(app_settings)
        logger.info(
            "MongoDB connection attempt started.",
            extra={
                "mongodb_host": summary["host"],
                "mongodb_database": summary["database"],
                "mongodb_mode": summary["mode"],
            },
        )
        client = None
        try:
            client_class = _load_async_mongo_client_class()
            client = client_class(
                app_settings.mongodb_uri.strip(),
                connectTimeoutMS=app_settings.mongodb_connect_timeout_ms,
                serverSelectionTimeoutMS=(
                    app_settings.mongodb_server_selection_timeout_ms
                ),
                socketTimeoutMS=app_settings.mongodb_socket_timeout_ms,
                retryReads=app_settings.mongodb_retry_reads,
                retryWrites=app_settings.mongodb_retry_writes,
                tz_aware=True,
            )
            logger.info("MongoDB DNS/SRV discovery succeeded.")
            database = client[app_settings.mongodb_database.strip()]

            ping_ok = await _ping_mongodb_client(client)
            if not ping_ok:
                raise MongoDBStartupError(
                    "MongoDB ping failed.",
                    category="ping_failed",
                )
            logger.info("MongoDB ping succeeded.")

            await ensure_collections(database)
            await ensure_indexes(database)
            logger.info("MongoDB configured database validated.")

            if _client is not None and _client is not client:
                await _close_client(_client)
            _client = client
            _database = database
            logger.info("MongoDB connection ready.")
        except MongoDBStartupError:
            if client is not None:
                await _close_client(client)
            _client = None
            _database = None
            raise
        except Exception as error:
            startup_error = classify_mongodb_startup_error(error)
            logger.error(
                "MongoDB startup connection failed. Reason: %s",
                startup_error.reason,
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            if client is not None:
                await _close_client(client)
            _client = None
            _database = None
            raise startup_error from None


async def close_mongodb() -> None:
    global _client, _database

    async with _lifecycle_lock:
        if _client is None:
            _database = None
            return

        client = _client
        _client = None
        _database = None
        await _close_client(client)
        logger.info("MongoDB connection closed.")


def get_mongodb_client() -> Any:
    if _client is None:
        raise RuntimeError("MongoDB client is not connected.")
    return _client


def get_mongodb_database() -> Any:
    if _database is None:
        raise RuntimeError("MongoDB database is not connected.")
    return _database


def is_mongodb_configured(app_settings: Settings = settings) -> bool:
    return bool(app_settings.mongodb_uri.strip())


def build_dedicated_async_mongo_client(
    app_settings: Settings = settings,
) -> tuple[Any | None, Any | None]:
    """Create a new, independent ``(client, database)`` pair.

    PyMongo's async client is documented as unsafe to share across event
    loops. The main application client returned by :func:`get_mongodb_client`
    is bound to the primary asyncio event loop created at process startup.
    Callers that run coroutines on a *different* persistent event loop (for
    example, the Voice XAI background worker loop) must call this instead of
    reusing the global client, and should call it from inside that other
    loop's thread so the client is constructed for the loop that will use it.

    Returns ``(None, None)`` when MongoDB is not configured, matching
    :func:`connect_to_mongodb`'s behaviour for the primary client.
    """

    if not app_settings.mongodb_uri.strip():
        return None, None
    client_class = _load_async_mongo_client_class()
    client = client_class(
        app_settings.mongodb_uri.strip(),
        connectTimeoutMS=app_settings.mongodb_connect_timeout_ms,
        serverSelectionTimeoutMS=app_settings.mongodb_server_selection_timeout_ms,
        socketTimeoutMS=app_settings.mongodb_socket_timeout_ms,
        retryReads=app_settings.mongodb_retry_reads,
        retryWrites=app_settings.mongodb_retry_writes,
        tz_aware=True,
    )
    database = client[app_settings.mongodb_database.strip()]
    return client, database


async def ping_mongodb() -> bool:
    if _client is None:
        return False
    return await _ping_mongodb_client(_client)


async def mongodb_readiness(app_settings: Settings = settings) -> dict[str, bool]:
    configured = is_mongodb_configured(app_settings)
    available = False
    if configured and _client is not None:
        try:
            available = await ping_mongodb()
        except Exception:
            logger.exception("MongoDB readiness ping failed.")
            available = False
    return {
        "mongodb_configured": configured,
        "mongodb_available": available,
    }


def _load_async_mongo_client_class():
    try:
        from pymongo import AsyncMongoClient
    except ImportError as top_level_error:
        try:
            from pymongo.asynchronous.mongo_client import AsyncMongoClient
        except ImportError as nested_error:
            raise RuntimeError(
                "PyMongo with AsyncMongoClient support is required for MongoDB."
            ) from nested_error
        if AsyncMongoClient is None:
            raise top_level_error
    return AsyncMongoClient


async def _ping_mongodb_client(client: Any) -> bool:
    result = await client.admin.command("ping")
    return result.get("ok") == 1


async def _close_client(client: Any) -> None:
    close_result = client.close()
    if inspect.isawaitable(close_result):
        await close_result


def _validate_mongodb_settings(app_settings: Settings) -> None:
    if not app_settings.mongodb_uri.strip():
        if app_settings.mongodb_required:
            raise MongoDBStartupError(
                "MONGODB_URI is required when MongoDB is required.",
                category="missing_uri",
            )
        return
    parsed = urlparse(app_settings.mongodb_uri.strip())
    if parsed.scheme not in {"mongodb", "mongodb+srv"}:
        raise MongoDBStartupError(
            "MONGODB_URI must start with mongodb:// or mongodb+srv://.",
            category="invalid_uri",
        )
    if not parsed.hostname:
        raise MongoDBStartupError(
            "MONGODB_URI must include a MongoDB host.",
            category="invalid_uri",
        )
    if not app_settings.mongodb_database.strip():
        raise MongoDBStartupError(
            "MONGODB_DATABASE is required.",
            category="missing_database",
        )


def classify_mongodb_startup_error(error: Exception) -> MongoDBStartupError:
    message = str(error)
    lowered = message.lower()
    category = "server_selection_failed"
    reason = "MongoDB server selection failed."

    try:
        from pymongo.errors import (
            AutoReconnect,
            ConfigurationError,
            InvalidURI,
            NetworkTimeout,
            OperationFailure,
            ServerSelectionTimeoutError,
        )
    except ImportError:
        AutoReconnect = ConfigurationError = InvalidURI = NetworkTimeout = (  # type: ignore[assignment]
            OperationFailure
        ) = ServerSelectionTimeoutError = ()  # type: ignore[assignment]

    if isinstance(error, InvalidURI) or "invalid uri" in lowered:
        category = "invalid_uri"
        reason = "MongoDB URI is invalid."
    elif isinstance(error, ConfigurationError):
        if "dns" in lowered or "srv" in lowered:
            category = "dns_failure"
            reason = "MongoDB DNS/SRV discovery failed."
        else:
            category = "invalid_configuration"
            reason = "MongoDB client configuration is invalid."
    elif isinstance(error, OperationFailure):
        category = "authentication_failure"
        reason = "MongoDB authentication or authorization failed."
    elif "replicasetnoprimary" in lowered or "no primary" in lowered:
        category = "replica_set_no_primary"
        reason = "Replica set discovered but no writable PRIMARY is available."
    elif isinstance(error, NetworkTimeout) or "timed out" in lowered:
        category = "network_timeout"
        reason = "MongoDB network connection timed out."
    elif isinstance(error, ServerSelectionTimeoutError):
        category = "server_selection_failed"
        reason = "MongoDB server selection failed."
    elif isinstance(error, AutoReconnect):
        category = "network_failure"
        reason = "MongoDB network connection failed."
    if "tls" in lowered or "ssl" in lowered or "certificate" in lowered:
        category = "tls_failure"
        reason = "MongoDB TLS/certificate validation failed."

    return MongoDBStartupError(reason, category=category, cause=error)


def mongodb_safe_summary(app_settings: Settings) -> dict[str, str]:
    uri = app_settings.mongodb_uri.strip()
    parsed = urlparse(uri)
    mode = "Atlas SRV" if parsed.scheme == "mongodb+srv" else "Standard URI"
    host = _redact_mongodb_host(parsed.hostname or "")
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    return {
        "host": host,
        "database": app_settings.mongodb_database.strip(),
        "mode": mode,
        "replica_set_configured": str("replicaSet" in query),
        "read_preference": query.get("readPreference", "driver-default"),
        "tls_configured": str(any(key.lower() in {"tls", "ssl"} for key in query)),
        "server_selection_timeout_ms": str(
            app_settings.mongodb_server_selection_timeout_ms
        ),
        "connect_timeout_ms": str(app_settings.mongodb_connect_timeout_ms),
        "socket_timeout_ms": str(app_settings.mongodb_socket_timeout_ms),
        "retry_reads": str(app_settings.mongodb_retry_reads),
        "retry_writes": str(app_settings.mongodb_retry_writes),
    }


def _redact_mongodb_host(hostname: str) -> str:
    if not hostname:
        return "<missing>"
    if hostname.endswith(".mongodb.net"):
        return "*.mongodb.net"
    parts = hostname.split(".")
    if len(parts) >= 2:
        return f"*.{'.'.join(parts[-2:])}"
    return hostname
