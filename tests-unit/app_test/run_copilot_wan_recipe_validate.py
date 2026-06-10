#!/usr/bin/env python3
"""Validate WAN i2v recipe wiring against the real ComfyUI node registry."""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import comfy.options

comfy.options.enable_args_parsing()
sys.argv = ["wan-recipe-validate", "--cpu"]

import nodes  # noqa: E402
from app import copilot_manager  # noqa: E402
from app.copilot_recipes import (  # noqa: E402
    build_recipe_workflow,
    complete_workflow_from_recipe,
    enrich_candidate_with_recipe,
)


async def main() -> int:
    await nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False)
    manager = copilot_manager.CopilotManager(prompt_server=None)

    scaffold = build_recipe_workflow("wan_i2v")
    validation = await manager._validate_workflow(scaffold, strict_topology=True)
    partial = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "test.png"}},
        "2": {"class_type": "WanImageToVideo", "inputs": {"width": 832, "height": 480, "length": 81, "batch_size": 1}},
    }
    enriched = enrich_candidate_with_recipe(
        {"workflow": partial},
        prompt="Create a WAN 2.1 image to video workflow",
        current_workflow={},
    )
    enriched_validation = await manager._validate_workflow(enriched["workflow"], strict_topology=True)

    report = {
        "node_class_count": len(nodes.NODE_CLASS_MAPPINGS),
        "scaffold_nodes": len(scaffold),
        "scaffold_validation_success": validation["success"],
        "scaffold_connection_issues": validation.get("connection_issues") or [],
        "scaffold_topology_warnings": validation.get("topology_warnings") or [],
        "enriched_nodes": len(enriched["workflow"]),
        "enriched_validation_success": enriched_validation["success"],
        "enriched_connection_issues": enriched_validation.get("connection_issues") or [],
        "enriched_topology_warnings": enriched_validation.get("topology_warnings") or [],
    }
    print(json.dumps(report, indent=2))

    if len(enriched["workflow"]) < 10:
        return 1
    if report["scaffold_connection_issues"] or report["enriched_connection_issues"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
