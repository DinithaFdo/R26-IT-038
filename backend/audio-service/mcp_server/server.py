from __future__ import annotations

import asyncio
from collections.abc import Callable
import inspect
from typing import Any

from app.core.logging import configure_logging

from mcp_server.lifecycle import (
    MCPServerLifecycleError,
    mcp_lifespan,
    shutdown_mcp_server,
    startup_mcp_server,
)
from mcp_server import tools

MCP_SERVER_NAME = "MULTI-SCOPE MCP Adapter"
MCP_TOOL_NAMES = (
    "multiscope_create_prediction",
    "multiscope_get_prediction",
    "multiscope_list_predictions",
    "multiscope_get_model_status",
    "multiscope_delete_prediction",
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

    try:
        server = FastMCP(MCP_SERVER_NAME, lifespan=mcp_lifespan)
        server._multiscope_lifecycle_managed = True
    except TypeError:
        server = FastMCP(MCP_SERVER_NAME)
        server._multiscope_lifecycle_managed = False
    for tool_name in MCP_TOOL_NAMES:
        _register_tool(server, tool_name, getattr(tools, tool_name))
    return server


def _register_tool(server: Any, name: str, function: Callable[..., Any]) -> None:
    server.tool(name=name)(function)


def main() -> None:
    configure_logging()
    server = create_mcp_server()
    if getattr(server, "_multiscope_lifecycle_managed", False):
        server.run()
        return

    try:
        asyncio.run(startup_mcp_server())
    except MCPServerLifecycleError as error:
        raise SystemExit(str(error)) from None

    try:
        result = server.run()
        if inspect.isawaitable(result):
            asyncio.run(result)
    finally:
        asyncio.run(shutdown_mcp_server())


if __name__ == "__main__":
    main()
