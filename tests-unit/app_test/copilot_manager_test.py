import sys
import types

from aiohttp.test_utils import make_mocked_request


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

from app import copilot_manager

copilot_manager.nodes = nodes
copilot_manager.folder_paths = folder_paths
copilot_manager.execution = execution


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


class DummyCheckpointLoader:
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("MODEL", "CLIP", "VAE")
    CATEGORY = "test"
    DESCRIPTION = "Loads a test checkpoint."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"ckpt_name": (["installed.safetensors"], {})}}


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


def test_collect_missing_models_merges_declared_download(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "NODE_CLASS_MAPPINGS",
        {
            "DummyCheckpointLoader": DummyCheckpointLoader,
        },
    )
    monkeypatch.setattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", {})
    monkeypatch.setattr(
        folder_paths,
        "folder_names_and_paths",
        {"checkpoints": (["/tmp/models/checkpoints"], {".safetensors"})},
    )
    monkeypatch.setattr(
        folder_paths,
        "get_filename_list",
        lambda folder: ["installed.safetensors"] if folder == "checkpoints" else [],
    )
    copilot_manager.folder_paths = folder_paths
    copilot_manager.nodes = nodes

    missing = copilot_manager.collect_missing_models(
        {
            "1": {
                "class_type": "DummyCheckpointLoader",
                "inputs": {"ckpt_name": "missing.safetensors"},
            }
        },
        {
            "model_downloads": [
                {
                    "folder": "checkpoints",
                    "filename": "missing.safetensors",
                    "url": "https://example.com/missing.safetensors",
                    "reason": "needed for this generated workflow",
                }
            ]
        },
    )

    assert missing == [
        {
            "folder": "checkpoints",
            "filename": "missing.safetensors",
            "url": "https://example.com/missing.safetensors",
            "node_id": "1",
            "class_type": "DummyCheckpointLoader",
            "input_name": "ckpt_name",
            "reason": "needed for this generated workflow",
        }
    ]


def test_safe_model_filename_rejects_path_traversal():
    assert copilot_manager._safe_model_filename("../secret.safetensors") == ""
    assert copilot_manager._safe_model_filename("subdir/model.safetensors") == "subdir/model.safetensors"


def test_merge_workflow_into_current_preserves_unmentioned_nodes():
    current = {
        "1": {"class_type": "DummyLoader", "inputs": {"image": "example.png"}},
        "2": {"class_type": "DummyPreview", "inputs": {"images": ["1", 0]}},
    }
    candidate = {
        "workflow": {
            "2": {"class_type": "DummyPreview", "inputs": {"images": ["1", 0]}},
            "3": {"class_type": "DummyLoader", "inputs": {"image": "other.png"}},
        },
        "removed_node_ids": [],
    }

    merged = copilot_manager.merge_workflow_into_current(current, candidate)

    assert "1" in merged
    assert "2" in merged
    assert "3" in merged
    assert merged["3"]["inputs"]["image"] == "other.png"


def test_merge_workflow_into_current_honors_removed_node_ids():
    current = {
        "1": {"class_type": "DummyLoader", "inputs": {"image": "example.png"}},
        "2": {"class_type": "DummyPreview", "inputs": {"images": ["1", 0]}},
    }
    candidate = {"workflow": {"2": current["2"]}, "removed_node_ids": ["1"]}

    merged = copilot_manager.merge_workflow_into_current(current, candidate)

    assert "1" not in merged
    assert "2" in merged


def test_build_copilot_messages_omits_bulk_node_catalog(monkeypatch):
    monkeypatch.setattr(copilot_manager, "build_hardware_context", lambda: {"cpu_only": True, "recommendations": []})

    messages = copilot_manager.build_copilot_messages(
        prompt="build workflow",
        history=[],
        current_workflow={},
        current_ui_workflow={},
    )
    user_message = next(message for message in messages if message["role"] == "user")
    context = __import__("json").loads(user_message["content"])
    assert "installed_nodes" not in context
    assert "available_models" not in context


def test_build_copilot_messages_stays_within_context_budget(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "NODE_CLASS_MAPPINGS",
        {
            "DummyLoader": DummyLoader,
            "DummyPreview": DummyPreview,
            "DummyCheckpointLoader": DummyCheckpointLoader,
        },
    )
    monkeypatch.setattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", {})
    monkeypatch.setattr(copilot_manager, "build_hardware_context", lambda: {"cpu_only": True})
    monkeypatch.setattr(copilot_manager, "build_model_context", lambda **_: {"checkpoints": ["installed.safetensors"]})

    messages = copilot_manager.build_copilot_messages(
        prompt="build a simple image preview workflow",
        history=[{"role": "user", "content": "hello"}],
        current_workflow={},
        current_ui_workflow={},
    )

    user_message = next(message for message in messages if message["role"] == "user")
    assert len(user_message["content"]) < copilot_manager.COPILOT_CONTEXT_CHAR_BUDGET


def test_build_relevant_node_catalog_prioritizes_workflow_nodes(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "NODE_CLASS_MAPPINGS",
        {
            "DummyLoader": DummyLoader,
            "DummyPreview": DummyPreview,
            "DummyCheckpointLoader": DummyCheckpointLoader,
        },
    )
    monkeypatch.setattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", {})

    catalog = copilot_manager.build_relevant_node_catalog(
        query="preview",
        workflow_api={
            "1": {"class_type": "DummyCheckpointLoader", "inputs": {"ckpt_name": "installed.safetensors"}},
        },
        limit=10,
    )

    class_types = {entry["class_type"] for entry in catalog}
    assert "DummyCheckpointLoader" in class_types


def test_search_node_class_types_matches_camel_case_names(monkeypatch):
    class WanImageToVideoNode:
        CATEGORY = "conditioning/video_models"

    monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "WanImageToVideo", WanImageToVideoNode)
    monkeypatch.setitem(nodes.NODE_DISPLAY_NAME_MAPPINGS, "WanImageToVideo", "Wan Image To Video")

    hits = copilot_manager.search_node_class_types(query="wan image to video", limit=5)
    assert any(hit["class_type"] == "WanImageToVideo" for hit in hits)


def test_get_node_info_entry_returns_single_node(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "NODE_CLASS_MAPPINGS",
        {
            "DummyLoader": DummyLoader,
            "DummyPreview": DummyPreview,
        },
    )
    monkeypatch.setattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", {})

    entry = copilot_manager.get_node_info_entry("DummyPreview")
    assert entry is not None
    assert entry["class_type"] == "DummyPreview"
    assert "inputs" in entry


def test_trim_messages_for_llm_budget_drops_old_messages():
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "x" * 5000},
        {"role": "assistant", "content": "y" * 5000},
        {"role": "user", "content": "latest"},
    ]
    trimmed = copilot_manager.trim_messages_for_llm_budget(messages, char_budget=8000)
    assert trimmed[-1]["content"] == "latest"
    assert len(trimmed) < len(messages)


def test_lint_workflow_connections_reports_missing_required_input(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "NODE_CLASS_MAPPINGS",
        {
            "DummyLoader": DummyLoader,
            "DummyPreview": DummyPreview,
        },
    )
    monkeypatch.setattr(nodes, "NODE_DISPLAY_NAME_MAPPINGS", {})

    issues = copilot_manager.lint_workflow_connections(
        {
            "1": {"class_type": "DummyLoader", "inputs": {"image": "example.png"}},
            "2": {"class_type": "DummyPreview", "inputs": {}},
        }
    )

    assert any("missing required input" in issue for issue in issues)
