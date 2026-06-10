import json
import sys
import types

nodes = types.ModuleType("nodes")
nodes.NODE_CLASS_MAPPINGS = {}
nodes.NODE_DISPLAY_NAME_MAPPINGS = {}
nodes.EXTENSION_WEB_DIRS = {}
sys.modules["nodes"] = nodes

folder_paths = types.ModuleType("folder_paths")
folder_paths.folder_names_and_paths = {}
folder_paths.get_filename_list = lambda _folder: []
sys.modules["folder_paths"] = folder_paths

from app import copilot_recipes


class SaveImageNode:
    OUTPUT_NODE = True


class SaveVideoNode:
    OUTPUT_NODE = True


def test_detect_wan_i2v_recipe():
    assert copilot_recipes.detect_workflow_recipe("Create a WAN 2.1 image to video workflow") == "wan_i2v"


def test_wan_i2v_recipe_has_full_pipeline():
    workflow = copilot_recipes.build_recipe_workflow("wan_i2v")
    class_types = {node["class_type"] for node in workflow.values()}
    assert len(workflow) >= 10
    assert "WanImageToVideo" in class_types
    assert "SaveVideo" in class_types
    assert "LoadImage" in class_types
    assert "KSampler" in class_types


def test_enrich_expands_two_node_stub():
    nodes.NODE_CLASS_MAPPINGS.update(
        {
            "LoadImage": object(),
            "WanImageToVideo": object(),
            "SaveVideo": SaveVideoNode,
            "SaveImage": SaveImageNode,
        }
    )
    partial = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "test.png"}},
        "2": {
            "class_type": "WanImageToVideo",
            "inputs": {"width": 832, "height": 480, "length": 81, "batch_size": 1},
        },
    }
    enriched = copilot_recipes.enrich_candidate_with_recipe(
        {"assistant_message": "partial", "workflow": partial},
        prompt="Create a WAN image to video workflow",
        current_workflow={},
    )
    workflow = enriched["workflow"]
    assert len(workflow) >= 10
    class_types = {node["class_type"] for node in workflow.values()}
    assert "SaveVideo" in class_types
    assert "WanImageToVideo" in class_types


def test_build_copilot_messages_includes_recipe(monkeypatch):
    import execution

    execution = types.ModuleType("execution")

    async def validate_prompt(_prompt_id, _workflow_api, _partial_execution_targets):
        return True, None, [], {}

    execution.validate_prompt = validate_prompt
    sys.modules["execution"] = execution

    comfy_api = types.ModuleType("comfy_api")
    comfy_api_internal = types.ModuleType("comfy_api.internal")
    comfy_api_internal._ComfyNodeInternal = type("_ComfyNodeInternal", (), {})
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.internal"] = comfy_api_internal

    from app import copilot_manager

    copilot_manager.nodes = nodes
    copilot_manager.folder_paths = folder_paths
    copilot_manager.execution = execution
    monkeypatch.setattr(copilot_manager, "build_hardware_context", lambda: {"cpu_only": True, "recommendations": []})

    messages = copilot_manager.build_copilot_messages(
        prompt="Create a WAN 2.1 image to video workflow",
        history=[],
        current_workflow={},
        current_ui_workflow={},
    )
    user_message = next(message for message in messages if message["role"] == "user")
    context = json.loads(user_message["content"])
    assert context["workflow_recipe"]["id"] == "wan_i2v"
    assert len(context["workflow_recipe"]["scaffold"]) >= 10
