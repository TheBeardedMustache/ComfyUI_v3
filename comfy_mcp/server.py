"""Stdio MCP server for ComfyUI. Requires a running ComfyUI instance at COMFYUI_HOST."""

from __future__ import annotations

import json
import os

from mcp.server.fastmcp import FastMCP

from comfy_mcp.tools import execute_tool

# MCP subprocess talks to ComfyUI over HTTP (ComfyUI must already be running).
os.environ.setdefault("COMFY_MCP_USE_HTTP", "1")

mcp = FastMCP(
    "comfyui",
    instructions=(
        "Tools for a local ComfyUI server. Set COMFYUI_HOST (default http://127.0.0.1:8188). "
        "Use search_nodes and get_node_info instead of guessing node wiring."
    ),
)


@mcp.tool()
async def search_nodes(query: str, limit: int = 15) -> str:
    """Search installed ComfyUI nodes by keyword."""
    result = await execute_tool("search_nodes", {"query": query, "limit": limit})
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def get_node_info(class_type: str) -> str:
    """Get input/output schema for a node class_type."""
    result = await execute_tool("get_node_info", {"class_type": class_type})
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def list_model_files(folder: str = "checkpoints", limit: int = 25) -> str:
    """List model files in a ComfyUI models folder."""
    result = await execute_tool("list_model_files", {"folder": folder, "limit": limit})
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def get_hardware() -> str:
    """Get GPU/VRAM/CPU stats and workflow recommendations."""
    result = await execute_tool("get_hardware", {})
    return json.dumps(result, ensure_ascii=False)


@mcp.tool()
async def validate_workflow(workflow_api: dict) -> str:
    """Validate a ComfyUI API workflow."""
    result = await execute_tool("validate_workflow", {"workflow_api": workflow_api})
    return json.dumps(result, ensure_ascii=False)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
