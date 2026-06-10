#!/usr/bin/env python3
"""Standalone WAN i2v Copilot prompt check (loads full ComfyUI node registry)."""

import json
import sys
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import comfy.options

comfy.options.enable_args_parsing()
sys.argv = ["wan-i2v-check", "--cpu"]

import nodes  # noqa: E402
from app import copilot_manager  # noqa: E402

WAN_I2V_PROMPT = (
    "Create a WAN 2.1 image to video workflow: load an input image, "
    "use WanImageToVideo conditioning, sample, decode, and save a video."
)


async def _load_nodes() -> None:
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)


def main() -> int:
    asyncio.run(_load_nodes())
    hits = copilot_manager.search_node_class_types(query="wan image to video", limit=15)
    class_types = {hit["class_type"] for hit in hits}
    if "WanImageToVideo" not in class_types:
        print("FAIL: WanImageToVideo not in search results:", class_types)
        return 1

    hints = copilot_manager.build_inline_node_hints({}, prompt=WAN_I2V_PROMPT)
    hint_types = {entry["class_type"] for entry in hints}
    if "WanImageToVideo" not in hint_types:
        print("FAIL: WanImageToVideo not in node_hints:", hint_types)
        return 1

    entry = copilot_manager.get_node_info_entry("WanImageToVideo")
    if entry is None:
        print("FAIL: get_node_info_entry(WanImageToVideo) returned None")
        return 1

    messages = copilot_manager.build_copilot_messages(
        prompt=WAN_I2V_PROMPT,
        history=[],
        current_workflow={},
        current_ui_workflow={},
    )
    user_message = next(message for message in messages if message["role"] == "user")
    context = json.loads(user_message["content"])
    if "installed_nodes" in context:
        print("FAIL: bulk installed_nodes still in context")
        return 1
    if len(user_message["content"]) >= copilot_manager.COPILOT_CONTEXT_CHAR_BUDGET:
        print("FAIL: context too large:", len(user_message["content"]))
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "node_class_count": len(nodes.NODE_CLASS_MAPPINGS),
                "wan_search_hits": list(class_types)[:8],
                "node_hint_count": len(hints),
                "context_chars": len(user_message["content"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
