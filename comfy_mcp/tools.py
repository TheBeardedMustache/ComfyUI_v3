from __future__ import annotations

import json
import os
from typing import Any

import aiohttp

COMFYUI_HOST = os.environ.get("COMFYUI_HOST", "http://127.0.0.1:8188").rstrip("/")

COMFYUI_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_nodes",
            "description": (
                "Search installed ComfyUI nodes (core + custom) by name, category, or keyword. "
                "Returns class_type names only — call get_node_info for wiring details."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search text, e.g. 'KSampler' or 'load image'"},
                    "limit": {"type": "integer", "description": "Max results (default 12)", "default": 12},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_node_info",
            "description": "Get input/output schema for one node class_type (single node only).",
            "parameters": {
                "type": "object",
                "properties": {
                    "class_type": {"type": "string", "description": "Exact node class_type, e.g. CheckpointLoaderSimple"},
                },
                "required": ["class_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_model_files",
            "description": "List installed model filenames in a ComfyUI models folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "Model folder: checkpoints, loras, vae, controlnet, etc.",
                        "default": "checkpoints",
                    },
                    "limit": {"type": "integer", "default": 20},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hardware",
            "description": "Get GPU/VRAM/CPU info and workflow sizing recommendations for this machine.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_workflow",
            "description": "Validate a ComfyUI API-format workflow (connections, required inputs, topology).",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_api": {
                        "type": "object",
                        "description": "Workflow keyed by node id strings",
                        "additionalProperties": True,
                    }
                },
                "required": ["workflow_api"],
            },
        },
    },
]


def _use_http() -> bool:
    return os.environ.get("COMFY_MCP_USE_HTTP", "0").lower() in {"1", "true", "yes"}


async def execute_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    arguments = arguments or {}
    if _use_http():
        return await _execute_tool_http(name, arguments)
    return await _execute_tool_local(name, arguments)


async def execute_copilot_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """In-process MCP tools for ComfyUI Copilot (never loads the full node catalog)."""
    return await _execute_tool_local(name, arguments or {})


async def _execute_tool_local(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    from app import copilot_manager

    if name == "search_nodes":
        query = str(arguments.get("query") or "")
        limit = int(arguments.get("limit") or 12)
        nodes_list = copilot_manager.search_node_class_types(
            query=query,
            limit=max(1, min(limit, 25)),
        )
        return {"nodes": nodes_list, "total": len(nodes_list)}

    if name == "get_node_info":
        class_type = str(arguments.get("class_type") or "").strip()
        if not class_type:
            return {"error": "class_type is required"}
        entry = copilot_manager.get_node_info_entry(class_type)
        if entry:
            return {"node": entry}
        return {"error": f"Unknown class_type: {class_type}"}

    if name == "list_model_files":
        folder = str(arguments.get("folder") or "checkpoints")
        limit = int(arguments.get("limit") or 20)
        models = copilot_manager.build_model_context(limit_per_folder=max(1, min(limit, 30)))
        return {"folder": folder, "files": models.get(folder, [])}

    if name == "get_hardware":
        hardware = copilot_manager.build_hardware_context()
        return {
            "cpu_only": hardware.get("cpu_only"),
            "vram_total_gb": hardware.get("vram_total_gb"),
            "vram_free_gb": hardware.get("vram_free_gb"),
            "recommendations": hardware.get("recommendations"),
        }

    if name == "validate_workflow":
        workflow_api = arguments.get("workflow_api") or {}
        if not isinstance(workflow_api, dict):
            return {"error": "workflow_api must be an object"}
        manager = copilot_manager.CopilotManager(prompt_server=None)
        validation = await manager._validate_workflow(workflow_api, strict_topology=True)
        validation["missing_models"] = copilot_manager.collect_missing_models(workflow_api, None)
        return copilot_manager.summarize_validation_for_llm(validation)

    return {"error": f"Unknown tool: {name}"}


async def _execute_tool_http(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if name == "search_nodes":
            params = {
                "q": arguments.get("query", ""),
                "limit": max(1, min(int(arguments.get("limit") or 12), 25)),
            }
            data = await _http_json(session, "GET", f"{COMFYUI_HOST}/api/copilot/node_catalog", params=params)
            nodes_list = data.get("nodes") or []
            slim = [
                {
                    "class_type": node.get("class_type"),
                    "display_name": node.get("display_name"),
                    "category": node.get("category"),
                }
                for node in nodes_list
                if isinstance(node, dict) and node.get("class_type")
            ]
            return {"nodes": slim, "total": len(slim)}

        if name == "get_node_info":
            class_type = str(arguments.get("class_type") or "").strip()
            if not class_type:
                return {"error": "class_type is required"}
            params = {"class_type": class_type}
            data = await _http_json(session, "GET", f"{COMFYUI_HOST}/api/copilot/node_catalog", params=params)
            nodes_list = data.get("nodes") or []
            if nodes_list:
                return {"node": nodes_list[0]}
            data = await _http_json(session, "GET", f"{COMFYUI_HOST}/object_info/{class_type}")
            return {"node": data}

        if name == "list_model_files":
            folder = str(arguments.get("folder") or "checkpoints")
            data = await _http_json(session, "GET", f"{COMFYUI_HOST}/models/{folder}")
            limit = int(arguments.get("limit") or 20)
            files = data if isinstance(data, list) else data.get(folder, [])
            return {"folder": folder, "files": (files or [])[:limit]}

        if name == "get_hardware":
            return await _http_json(session, "GET", f"{COMFYUI_HOST}/api/copilot/hardware")

        if name == "validate_workflow":
            payload = {"workflow_api": arguments.get("workflow_api") or {}}
            return await _http_json(session, "POST", f"{COMFYUI_HOST}/api/copilot/validate", json_body=payload)

    return {"error": f"Unknown tool: {name}"}


async def _http_json(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> Any:
    async with session.request(method, url, params=params, json=json_body) as response:
        text = await response.text()
        try:
            data = json.loads(text) if text else {}
        except json.JSONDecodeError:
            data = {"error": text[:2000]}
        if response.status >= 400 and isinstance(data, dict) and "error" not in data:
            data = {"error": data, "status": response.status}
        return data
