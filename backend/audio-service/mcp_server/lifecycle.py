from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import asyncio
import logging
from typing import Any

from app.config.settings import Settings, settings
from app.database.mongodb import (
    close_mongodb,
    connect_to_mongodb,
    mongodb_readiness,
)
from app.storage.factory import storage_readiness

from mcp_server import tools
from mcp_server.tools import MCPToolContext

logger = logging.getLogger(__name__)

_startup_lock = asyncio.Lock()
_started = False


class MCPServerLifecycleError(RuntimeError):
    """Expected MCP startup failure with a client-safe message."""


async def startup_mcp_server(app_settings: Settings = settings) -> MCPToolContext:
    """Initialise one reusable MCP runtime context for the process."""

    global _started

    async with _startup_lock:
        existing_context = tools.get_optional_shared_context()
        if _started and existing_context is not None:
            return existing_context

        if not app_settings.mongodb_uri.strip():
            raise MCPServerLifecycleError(
                "MCP server requires MongoDB configuration. Set MONGODB_URI before "
                "startup."
            )

        logger.info("MCP server startup started.")
        try:
            await connect_to_mongodb(app_settings)
            mongodb_status = await mongodb_readiness(app_settings)
            if not (
                mongodb_status["mongodb_configured"]
                and mongodb_status["mongodb_available"]
            ):
                raise MCPServerLifecycleError(
                    "MCP server startup failed because MongoDB is unavailable."
                )

            context = tools.create_default_context(app_settings)
            load_startup_models = getattr(
                context.voice_service,
                "load_startup_models",
                None,
            )
            if callable(load_startup_models):
                load_startup_models()
            tools.set_shared_context(context)
            _started = True

            storage_status = await storage_readiness(app_settings)
            logger.info(
                "MCP service readiness checked.",
                extra={
                    "mongodb_available": mongodb_status["mongodb_available"],
                    "storage_enabled": storage_status["storage_enabled"],
                    "storage_available": storage_status["storage_available"],
                    "model_count": len(context.model_registry.models),
                },
            )
            logger.info("MCP server startup complete.")
            return context
        except MCPServerLifecycleError:
            await _safe_shutdown_after_failed_startup()
            raise
        except Exception as error:
            logger.exception("MCP server startup failed.")
            await _safe_shutdown_after_failed_startup()
            raise MCPServerLifecycleError("MCP server startup failed.") from error


async def shutdown_mcp_server() -> None:
    """Release process-wide MCP resources."""

    global _started

    async with _startup_lock:
        context = tools.get_optional_shared_context()
        if not _started and context is None:
            return
        if context is not None:
            unload_models = getattr(context.voice_service, "unload_models", None)
            if callable(unload_models):
                unload_models()
            _shutdown_job_runner(context.job_runner)
        tools.clear_shared_context()
        await close_mongodb()
        _started = False
        logger.info("MCP server shutdown complete.")


@asynccontextmanager
async def mcp_lifespan(_server: Any) -> AsyncIterator[MCPToolContext]:
    context = await startup_mcp_server()
    try:
        yield context
    finally:
        await shutdown_mcp_server()


def is_mcp_started() -> bool:
    return _started and tools.get_optional_shared_context() is not None


async def _safe_shutdown_after_failed_startup() -> None:
    try:
        context = tools.get_optional_shared_context()
        if context is not None:
            _shutdown_job_runner(context.job_runner)
        tools.clear_shared_context()
        await close_mongodb()
    finally:
        global _started
        _started = False


def _shutdown_job_runner(job_runner: Any) -> None:
    shutdown = getattr(job_runner, "shutdown", None)
    if callable(shutdown):
        shutdown()
