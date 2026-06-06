from aiohttp.test_utils import make_mocked_request

import nodes
from app import copilot_manager


class DummyLoader:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    CATEGORY = "test"
    DESCRIPTION = "Loads a test image."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": (["example.png"], {"default": "example.png"})}}


class DummyPreview:
    RETURN_TYPES = ()
    RETURN_NAMES = ()
    CATEGORY = "test"
    DESCRIPTION = "Displays a test image."
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE", {})}}


def test_extract_json_object_from_fenced_llm_response():
    result = copilot_manager._extract_json_object(
        'Sure.\n```json\n{"assistant_message":"ok","workflow":{}}\n```'
    )

    assert result == {"assistant_message": "ok", "workflow": {}}


def test_llm_settings_support_copilot_headers():
    request = make_mocked_request(
        "POST",
        "/api/copilot/chat",
        headers={
            "X-Copilot-Provider": "anthropic",
            "X-Copilot-Base-Url": "https://api.anthropic.com/v1",
            "X-Copilot-Model": "claude-test",
            "X-Copilot-Api-Key": "secret",
        },
    )

    settings = copilot_manager._llm_settings_from_request(request, {})

    assert settings == {
        "provider": "anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "api_key": "secret",
        "model": "claude-test",
    }


def test_lint_workflow_topology_reports_orphan_nodes(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "NODE_CLASS_MAPPINGS",
        {
            "DummyLoader": DummyLoader,
            "DummyPreview": DummyPreview,
        },
    )
    monkeypatch.setattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", {})

    warnings = copilot_manager.lint_workflow_topology(
        {
            "1": {"class_type": "DummyLoader", "inputs": {"image": "example.png"}},
            "2": {"class_type": "DummyPreview", "inputs": {"images": ["1", 0]}},
            "3": {"class_type": "DummyLoader", "inputs": {"image": "example.png"}},
        }
    )

    assert warnings == ["orphan nodes not connected to an output: 3"]
