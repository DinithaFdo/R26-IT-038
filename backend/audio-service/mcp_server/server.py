from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import wraps
import inspect
import sys
from typing import Any, get_type_hints

from app.core.logging import configure_logging

from mcp_server.lifecycle import (
    MCPServerLifecycleError,
    shutdown_mcp_server,
    startup_mcp_server,
)
from mcp_server import tools, xai_tools

MCP_SERVER_NAME = "MULTI-SCOPE MCP Adapter"
MCP_TOOL_NAMES = (
    "multiscope_create_prediction",
    "multiscope_get_prediction",
    "multiscope_list_predictions",
    "multiscope_get_model_status",
    "multiscope_delete_prediction",
    "multiscope_create_explanation",
    "multiscope_get_explanation",
    "multiscope_get_temporal_explanation",
    "multiscope_get_semantic_explanation",
    "multiscope_get_explanation_report",
    "multiscope_get_explanation_narrative",
    "multiscope_get_explanation_artifact",
    "multiscope_retry_explanation",
    "multiscope_retry_explanation_narrative",
)


def create_mcp_server():
    """Create an inspector-compatible MCP server when the SDK is installed.

    The adapter is intentionally independent from the FastAPI app object. Tool
    functions call the existing application service layer directly.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as error:
        raise RuntimeError(
            "The MCP Python SDK is required to run this adapter. Install the "
            "mcp package in the MCP server environment."
        ) from error

    server = FastMCP(MCP_SERVER_NAME)
    for tool_name in MCP_TOOL_NAMES:
        module = tools if hasattr(tools, tool_name) else xai_tools
        _register_tool(server, tool_name, getattr(module, tool_name))
    return server


def _register_tool(server: Any, name: str, function: Callable[..., Any]) -> None:
    # Context contains repositories/models and is an internal test seam, never
    # client input. Hide it from FastMCP's schema and always use shared context.
    hints = get_type_hints(function)
    signature = inspect.signature(function)

    @wraps(function)
    async def public_tool(**kwargs):
        kwargs.pop("context", None)
        return await function(**kwargs)

    public_tool.__signature__ = signature.replace(
        parameters=[
            parameter.replace(annotation=hints.get(parameter.name, parameter.annotation))
            for parameter in signature.parameters.values()
            if parameter.name != "context"
        ],
        return_annotation=hints.get("return", signature.return_annotation),
    )
    server.tool(name=name)(public_tool)


async def run_mcp_server(server: Any) -> None:
    """Keep MongoDB, tools and shutdown on the same event loop (SDK 1.2)."""
    await startup_mcp_server()
    try:
        await server.run_stdio_async()
    finally:
        await shutdown_mcp_server()


def main() -> None:
    server = create_mcp_server()
    configure_logging(stream=sys.stderr)
    try:
        asyncio.run(run_mcp_server(server))
    except MCPServerLifecycleError as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
