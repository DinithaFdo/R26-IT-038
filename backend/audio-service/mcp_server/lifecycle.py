from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import asyncio
import contextlib
import inspect
import logging
from typing import Any

from app.config.settings import Settings, settings
from app.database.mongodb import (
    close_mongodb,
    connect_to_mongodb,
    mongodb_readiness,
)
from app.storage.factory import storage_readiness
from app.voice_xai.semantic.contracts import SemanticWindowOptions
from app.voice_xai.semantic.service import ProductionSemanticExplanationService
from app.voice_xai.workflow.recovery import recover_stale_explanations

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
            # Publish before initialization so partial startup is also cleaned up.
            tools.set_shared_context(context)
            load_startup_models = getattr(
                context.voice_service,
                "load_startup_models",
                None,
            )
            if callable(load_startup_models):
                load_startup_models()
            if app_settings.xai_enabled and context.xai_orchestrator is not None:
                if app_settings.xai_mode == "real":
                    semantic_service = await asyncio.to_thread(
                        ProductionSemanticExplanationService.load_once,
                        window_options=SemanticWindowOptions.from_settings(app_settings),
                        model_dir=app_settings.resolve_xai_artifact_path(
                            app_settings.xai_semantic_model_dir
                        ),
                    )
                    context.xai_orchestrator.set_semantic_service(semantic_service)
                await _recover_stale_runs(context, app_settings)
                context.artifact_cleanup_task = asyncio.create_task(
                    _artifact_cleanup_loop(context, app_settings),
                    name="mcp-voice-xai-artifact-cleanup",
                )
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
            await _close_context(context)
        try:
            await close_mongodb()
        finally:
            tools.clear_shared_context()
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
            await _close_context(context)
        tools.clear_shared_context()
        await close_mongodb()
    finally:
        global _started
        _started = False


async def _close_context(context: MCPToolContext) -> None:
    if context.artifact_cleanup_task is not None:
        context.artifact_cleanup_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await context.artifact_cleanup_task
        context.artifact_cleanup_task = None
    # Prediction workers can enqueue XAI, so drain them before closing XAI.
    for resource, method in (
        (context.job_runner, "shutdown"),
        (context.xai_queue, "close"),
        (context.voice_service, "unload_models"),
    ):
        callback = getattr(resource, method, None)
        if not callable(callback):
            continue
        try:
            result = await asyncio.to_thread(callback) if method == "close" else callback()
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("MCP resource cleanup failed: %s", method)


async def _recover_stale_runs(context: MCPToolContext, app_settings: Settings) -> None:
    if context.xai_repository is None:
        return
    try:
        await recover_stale_explanations(
            context.xai_repository,
            stale_after_seconds=app_settings.xai_stale_run_recovery_threshold_seconds,
        )
    except Exception:
        logger.exception("MCP Voice XAI stale-run recovery failed.")


async def _artifact_cleanup_loop(context: MCPToolContext, app_settings: Settings) -> None:
    cleanup = getattr(context.xai_artifact_store, "cleanup_expired", None)
    if not callable(cleanup):
        return
    while True:
        await asyncio.sleep(app_settings.xai_artifact_cleanup_interval_seconds)
        try:
            await asyncio.to_thread(cleanup)
        except Exception:
            logger.exception("MCP Voice XAI artifact cleanup failed.")
