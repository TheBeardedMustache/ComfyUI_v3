import pytest

from comfy_mcp import tools


@pytest.mark.asyncio
async def test_execute_tool_search_nodes_local(monkeypatch):
  import nodes

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

  result = await tools.execute_tool("search_nodes", {"query": "DummyPreview", "limit": 5})
  assert "nodes" in result
  assert any(node.get("class_type") == "DummyPreview" for node in result["nodes"])
