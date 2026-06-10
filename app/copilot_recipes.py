from __future__ import annotations

import re
from contextlib import suppress
from typing import Any

import folder_paths


RECIPE_TXT2IMG = "txt2img"
RECIPE_WAN_I2V = "wan_i2v"


def detect_workflow_recipe(prompt: str) -> str | None:
    text = (prompt or "").lower()
    if re.search(r"\bwan\b", text) and re.search(
        r"(image\s*(to|2)\s*video|img2vid|i2v|image.?to.?video|start\s*image.*video)",
        text,
    ):
        return RECIPE_WAN_I2V
    if re.search(r"(txt2img|text\s*(to|2)\s*image|text.?to.?image|sd\s*image)", text):
        return RECIPE_TXT2IMG
    if re.search(r"\bwan\b", text) and re.search(r"(text\s*(to|2)\s*video|t2v|txt2vid)", text):
        return RECIPE_WAN_I2V
    return None


def is_create_workflow_intent(prompt: str) -> bool:
    return bool(
        re.search(
            r"\b(create|build|make|generate|setup|design|add|wire|connect)\b",
            prompt or "",
            re.I,
        )
    )


def _first_model(folder: str, fallback: str = "") -> str:
    try:
        files = folder_paths.get_filename_list(folder)
    except Exception:
        files = []
    return files[0] if files else fallback


def build_recipe_workflow(recipe_id: str) -> dict[str, dict[str, Any]]:
    if recipe_id == RECIPE_WAN_I2V:
        return _wan_i2v_workflow()
    if recipe_id == RECIPE_TXT2IMG:
        return _txt2img_workflow()
    raise ValueError(f"Unknown workflow recipe: {recipe_id}")


def recipe_required_class_types(recipe_id: str) -> set[str]:
    return {node["class_type"] for node in build_recipe_workflow(recipe_id).values()}


def recipe_node_count(recipe_id: str) -> int:
    return len(build_recipe_workflow(recipe_id))


def _txt2img_workflow() -> dict[str, dict[str, Any]]:
    ckpt = _first_model("checkpoints", "model.safetensors")
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": ckpt},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "beautiful scenery", "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "blurry, low quality", "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 0,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "ComfyUI", "images": ["6", 0]},
        },
    }


def _wan_i2v_workflow() -> dict[str, dict[str, Any]]:
    unet = _first_model("diffusion_models", _first_model("unet", "wan2.1_i2v.safetensors"))
    clip = _first_model("text_encoders", _first_model("clip", "umt5_xxl.safetensors"))
    vae = _first_model("vae", "wan_vae.safetensors")
    return {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unet, "weight_dtype": "default"},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": clip, "type": "wan"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae},
        },
        "4": {
            "class_type": "LoadImage",
            "inputs": {"image": "example.png"},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "cinematic motion, smooth animation", "clip": ["2", 0]},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "static, blurry, low quality", "clip": ["2", 0]},
        },
        "7": {
            "class_type": "WanImageToVideo",
            "inputs": {
                "positive": ["5", 0],
                "negative": ["6", 0],
                "vae": ["3", 0],
                "width": 832,
                "height": 480,
                "length": 81,
                "batch_size": 1,
                "start_image": ["4", 0],
            },
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": 0,
                "steps": 20,
                "cfg": 5.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["1", 0],
                "positive": ["7", 0],
                "negative": ["7", 1],
                "latent_image": ["7", 2],
            },
        },
        "9": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
        },
        "10": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["9", 0], "fps": 16.0},
        },
        "11": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["10", 0], "filename_prefix": "video/ComfyUI", "format": "auto", "codec": "auto"},
        },
    }


def workflow_has_output_node(workflow_api: dict[str, Any]) -> bool:
    import nodes
    from comfy_api.internal import _ComfyNodeInternal

    for node in workflow_api.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if class_type not in nodes.NODE_CLASS_MAPPINGS:
            continue
        obj = nodes.NODE_CLASS_MAPPINGS[class_type]
        if getattr(obj, "OUTPUT_NODE", False):
            return True
        if isinstance(obj, type) and issubclass(obj, _ComfyNodeInternal):
            with suppress(Exception):
                if obj.GET_NODE_INFO_V1().get("output_node"):
                    return True
    return False


def workflow_needs_recipe_completion(
    workflow_api: dict[str, Any],
    recipe_id: str,
) -> bool:
    if not isinstance(workflow_api, dict) or not workflow_api:
        return True
    scaffold = build_recipe_workflow(recipe_id)
    required = recipe_required_class_types(recipe_id)
    present = {node.get("class_type") for node in workflow_api.values() if isinstance(node, dict)}
    if len(required - present) >= 2:
        return True
    if len(workflow_api) < max(6, len(scaffold) // 2):
        return True
    if not workflow_has_output_node(workflow_api):
        return True
    return False


def complete_workflow_from_recipe(
    workflow_api: dict[str, Any],
    recipe_id: str,
) -> dict[str, dict[str, Any]]:
    """Fill missing nodes/links from a known-good recipe scaffold."""
    scaffold = build_recipe_workflow(recipe_id)
    merged = {node_id: dict(node) for node_id, node in scaffold.items()}
    scaffold_ids_by_type: dict[str, list[str]] = {}
    for node_id, node in merged.items():
        class_type = node.get("class_type")
        if class_type:
            scaffold_ids_by_type.setdefault(str(class_type), []).append(node_id)

    used_ids: set[str] = set()
    next_id = max((int(key) for key in merged if str(key).isdigit()), default=0) + 1

    for cand_id, cand_node in (workflow_api or {}).items():
        if not isinstance(cand_node, dict) or not cand_node.get("class_type"):
            continue
        class_type = str(cand_node["class_type"])
        target_id = None
        for scaffold_id in scaffold_ids_by_type.get(class_type, []):
            if scaffold_id not in used_ids:
                target_id = scaffold_id
                break
        if target_id is None:
            while str(next_id) in merged:
                next_id += 1
            target_id = str(next_id)
            next_id += 1
            merged[target_id] = {"class_type": class_type, "inputs": {}}
        merged[target_id] = _merge_node_payload(merged.get(target_id, {}), cand_node)
        used_ids.add(target_id)

    return merged


def _merge_node_payload(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    merged["class_type"] = overlay.get("class_type") or base.get("class_type")
    base_inputs = dict(base.get("inputs") or {}) if isinstance(base.get("inputs"), dict) else {}
    overlay_inputs = overlay.get("inputs") or {}
    if isinstance(overlay_inputs, dict):
        for key, value in overlay_inputs.items():
            if _is_meaningful_input_value(value):
                base_inputs[key] = value
    merged["inputs"] = base_inputs
    return merged


def _is_meaningful_input_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str) and not value.strip():
        return False
    return True


def enrich_candidate_with_recipe(
    candidate: dict[str, Any],
    *,
    prompt: str,
    current_workflow: dict[str, Any],
) -> dict[str, Any]:
    recipe_id = detect_workflow_recipe(prompt)
    if not recipe_id:
        return candidate
    if current_workflow and not is_create_workflow_intent(prompt):
        return candidate

    proposed = candidate.get("workflow") or candidate.get("workflow_api") or {}
    if not isinstance(proposed, dict):
        proposed = {}

    if workflow_needs_recipe_completion(proposed, recipe_id):
        completed = complete_workflow_from_recipe(proposed, recipe_id)
        candidate = dict(candidate)
        candidate["workflow"] = completed
        candidate["assistant_message"] = (
            (candidate.get("assistant_message") or "")
            + f"\n\nApplied the {recipe_id} workflow scaffold so all required nodes are present."
        ).strip()
    return candidate
