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
