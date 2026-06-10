import pytest

from comfy_mcp import tools


@pytest.mark.asyncio
async def test_execute_tool_search_nodes_local(monkeypatch):
  import nodes
  from app import copilot_manager

  class DummyNode:
      RETURN_TYPES = ("IMAGE",)
      RETURN_NAMES = ("IMAGE",)
      CATEGORY = "test"
      OUTPUT_NODE = True

      @classmethod
      def INPUT_TYPES(cls):
          return {"required": {"image": ("IMAGE", {})}}

  monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "DummyPreview", DummyNode)
  monkeypatch.setitem(nodes.NODE_DISPLAY_NAME_MAPPINGS, "DummyPreview", "Dummy Preview")
  monkeypatch.setattr(copilot_manager, "nodes", nodes)

  result = await tools.execute_copilot_tool("search_nodes", {"query": "DummyPreview", "limit": 5})
  assert "nodes" in result
  assert any(node.get("class_type") == "DummyPreview" for node in result["nodes"])
  assert "inputs" not in (result["nodes"][0] or {})


@pytest.mark.asyncio
async def test_execute_copilot_tool_get_node_info_single(monkeypatch):
  import nodes
  from app import copilot_manager

  class DummyNode:
      RETURN_TYPES = ("IMAGE",)
      RETURN_NAMES = ("IMAGE",)
      CATEGORY = "test"
      OUTPUT_NODE = True

      @classmethod
      def INPUT_TYPES(cls):
          return {"required": {"image": ("IMAGE", {})}}

  monkeypatch.setitem(nodes.NODE_CLASS_MAPPINGS, "DummyPreview", DummyNode)
  monkeypatch.setitem(nodes.NODE_DISPLAY_NAME_MAPPINGS, "DummyPreview", "Dummy Preview")
  monkeypatch.setattr(copilot_manager, "nodes", nodes)

  result = await tools.execute_copilot_tool("get_node_info", {"class_type": "DummyPreview"})
  assert result["node"]["class_type"] == "DummyPreview"
  assert "inputs" in result["node"]
