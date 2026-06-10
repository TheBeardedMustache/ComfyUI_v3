import json
import sys
import types

nodes = types.ModuleType("nodes")
nodes.NODE_CLASS_MAPPINGS = {}
nodes.NODE_DISPLAY_NAME_MAPPINGS = {}
nodes.EXTENSION_WEB_DIRS = {}
sys.modules["nodes"] = nodes

execution = types.ModuleType("execution")


async def validate_prompt(_prompt_id, _workflow_api, _partial_execution_targets):
    return True, None, [], {}


execution.validate_prompt = validate_prompt
sys.modules["execution"] = execution

folder_paths = types.ModuleType("folder_paths")
folder_paths.folder_names_and_paths = {}
folder_paths.get_folder_paths = lambda folder: folder_paths.folder_names_and_paths[folder][0][:]
folder_paths.get_filename_list = lambda _folder: []
sys.modules["folder_paths"] = folder_paths

comfy_api = types.ModuleType("comfy_api")
comfy_api_internal = types.ModuleType("comfy_api.internal")
comfy_api_internal._ComfyNodeInternal = type("_ComfyNodeInternal", (), {})
sys.modules["comfy_api"] = comfy_api
sys.modules["comfy_api.internal"] = comfy_api_internal

from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from app.copilot_manager import COPILOT_VERSION, CopilotManager


@pytest.mark.asyncio
async def test_copilot_chat_llm_payload_stays_small():
    captured: dict = {}

    async def fake_http_json(self, method, url, **kwargs):
        if method == "POST" and "chat/completions" in url:
            if "payload" not in captured:
                captured["payload"] = kwargs.get("json") or {}
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "assistant_message": "ok",
                                    "workflow": {},
                                    "workflow_complete": False,
                                }
                            )
                        }
                    }
                ]
            }
        raise AssertionError(f"unexpected HTTP {method} {url}")

    app = web.Application()
    routes = web.RouteTableDef()
    manager = CopilotManager(prompt_server=None)
    manager.add_routes(routes)
    app.add_routes(routes)

    async with TestClient(TestServer(app)) as client:
        with patch.object(CopilotManager, "_http_json", fake_http_json):
            resp = await client.post(
                "/copilot/chat",
                json={
                    "prompt": "build a simple txt2img workflow",
                    "workflow_api": {},
                    "workflow_ui": {},
                    "messages": [],
                    "api_key": "test-key",
                    "model": "gpt-4o-mini",
                },
            )
            assert resp.status == 200
            payload = captured.get("payload") or {}
            size = len(json.dumps(payload, ensure_ascii=False))
            assert size < 50_000, f"LLM payload too large: {size} chars"
            user_blob = next(m["content"] for m in payload["messages"] if m["role"] == "user")
            assert "installed_nodes" not in user_blob


@pytest.mark.asyncio
async def test_context_preview_reports_mcp_version():
    app = web.Application()
    routes = web.RouteTableDef()
    manager = CopilotManager(prompt_server=None)
    manager.add_routes(routes)
    app.add_routes(routes)

    async with TestClient(TestServer(app)) as client:
        resp = await client.post(
            "/copilot/context_preview",
            json={"prompt": "hello", "workflow_api": {}, "workflow_ui": {}},
        )
        assert resp.status == 200
        data = await resp.json()
        assert data["copilot_version"] == COPILOT_VERSION
        assert data["bulk_node_catalog_in_context"] is False
        assert data["estimated_tokens"] < 20_000
