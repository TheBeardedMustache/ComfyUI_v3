from __future__ import annotations

import json
import logging
import os
import re
import socket
import sys
import time
import urllib.parse
import uuid
from contextlib import suppress
from typing import Any

import aiohttp
from aiohttp import web

import execution
import folder_paths
import nodes
from comfy_api.internal import _ComfyNodeInternal


COPILOT_EXTENSION_NAME = "comfyui-copilot"


class CopilotManager:
    """Local ComfyUI Copilot routes and web extension registration."""

    def __init__(self, prompt_server):
        self.prompt_server = prompt_server
        self.web_directory = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "web_extensions", "comfyui_copilot")
        )

    def register_extension(self):
        nodes.EXTENSION_WEB_DIRS[COPILOT_EXTENSION_NAME] = self.web_directory

    def add_routes(self, routes: web.RouteTableDef):
        @routes.get("/copilot/status")
        async def status(request):
            return web.json_response(
                {
                    "enabled": True,
                    "local": True,
                    "hosted_backend": False,
                    "extension": COPILOT_EXTENSION_NAME,
                    "routes": [
                        "/api/copilot/chat",
                        "/api/copilot/validate",
                        "/api/copilot/execute",
                        "/api/copilot/download_models",
                        "/api/copilot/node_catalog",
                        "/api/copilot/models",
                        "/api/copilot/hardware",
                    ],
                }
            )

        @routes.get("/copilot/node_catalog")
        async def node_catalog(request):
            query = request.rel_url.query.get("q", "")
            limit = _parse_int(request.rel_url.query.get("limit"), 200)
            full = request.rel_url.query.get("full", "false").lower() == "true"
            catalog = build_node_catalog(query=query, limit=None if full else limit)
            return web.json_response({"nodes": catalog, "total": len(catalog)})

        @routes.get("/copilot/hardware")
        async def hardware(request):
            return web.json_response(build_hardware_context())

        @routes.get("/copilot/models")
        async def models(request):
            settings = _llm_settings_from_request(request, {})
            try:
                model_names = await self._list_models(settings)
                return web.json_response({"models": [{"label": name, "name": name} for name in model_names]})
            except Exception as exc:
                return web.json_response({"error": str(exc), "models": []}, status=400)

        @routes.post("/copilot/validate")
        async def validate(request):
            body = await _read_json(request)
            workflow_api = body.get("workflow_api") or body.get("prompt") or {}
            validation = await self._validate_workflow(workflow_api, strict_topology=True)
            validation["missing_models"] = collect_missing_models(workflow_api, body)
            status_code = 200 if validation["success"] else 400
            return web.json_response(validation, status=status_code)

        @routes.post("/copilot/execute")
        async def execute(request):
            body = await _read_json(request)
            workflow_api = body.get("workflow_api") or body.get("prompt") or {}
            client_id = body.get("client_id")
            try:
                result = await self._queue_prompt(workflow_api, client_id=client_id)
                return web.json_response(result)
            except ValueError as exc:
                return web.json_response({"success": False, "error": str(exc)}, status=400)

        @routes.post("/copilot/download_models")
        async def download_models(request):
            body = await _read_json(request)
            if body.get("approved") is not True:
                return web.json_response(
                    {"success": False, "error": "Model downloads require explicit user approval."},
                    status=403,
                )

            downloads = body.get("downloads")
            if not isinstance(downloads, list):
                return web.json_response({"success": False, "error": "downloads must be a list."}, status=400)

            results = []
            for item in downloads:
                try:
                    results.append(await self._download_model(item))
                except Exception as exc:
                    logging.exception("ComfyUI Copilot model download failed")
                    results.append({"success": False, "error": str(exc), "request": _json_safe(item)})

            with suppress(Exception):
                self.prompt_server.model_file_manager.clear_cache()
            with suppress(Exception):
                from app.assets.scanner import seed_assets

                seed_assets(("models",), enable_logging=False)

            return web.json_response({"success": all(r.get("success") for r in results), "results": results})

        @routes.post("/copilot/chat")
        async def chat(request):
            body = await _read_json(request)
            response = web.StreamResponse(
                status=200,
                reason="OK",
                headers={
                    "Content-Type": "application/json",
                    "X-Content-Type-Options": "nosniff",
                },
            )
            await response.prepare(request)

            async def emit(event_type: str, **payload):
                payload["type"] = event_type
                await response.write((json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"))

            try:
                await emit("status", text="Reading current workflow and installed node catalog.")
                settings = _llm_settings_from_request(request, body)
                prompt = (body.get("prompt") or "").strip()
                if not prompt:
                    raise ValueError("Copilot prompt is empty.")

                current_workflow = body.get("workflow_api") or body.get("prompt_workflow") or {}
                current_ui_workflow = body.get("workflow_ui") or {}
                history = body.get("messages") or []

                await emit("status", text="Asking the local LLM to edit the current graph.")
                candidate = await self._generate_workflow_edit(
                    settings=settings,
                    prompt=prompt,
                    history=history,
                    current_workflow=current_workflow,
                    current_ui_workflow=current_ui_workflow,
                    execution_errors=body.get("execution_errors") or [],
                )

                max_attempts = _parse_int(body.get("max_repair_iterations"), 5)
                validation = {"success": False, "error": "Workflow was not validated"}
                missing_models: list[dict[str, Any]] = []
                for attempt in range(1, max_attempts + 1):
                    workflow_api = candidate.get("workflow") or candidate.get("workflow_api")
                    if not isinstance(workflow_api, dict):
                        raise ValueError("LLM did not return a ComfyUI API-format workflow object.")

                    workflow_api = merge_workflow_into_current(current_workflow, candidate)
                    candidate["workflow"] = workflow_api
                    missing_models = collect_missing_models(workflow_api, candidate)
                    validation = await self._validate_workflow(workflow_api, strict_topology=True)
                    validation["missing_models"] = missing_models
                    if validation["success"] and not missing_models:
                        await emit("status", text=f"Workflow validation passed after {attempt} attempt(s).")
                        break
                    if validation["success"] and missing_models:
                        await emit(
                            "status",
                            text="Graph wiring is valid but required model files are missing.",
                            missing_models=missing_models,
                        )
                        break

                    await emit(
                        "status",
                        text=f"Validation found issues; repair pass {attempt} of {max_attempts}.",
                        validation=summarize_validation_for_client(validation),
                    )
                    if attempt == max_attempts:
                        break
                    candidate = await self._repair_workflow_edit(
                        settings=settings,
                        prompt=prompt,
                        history=history,
                        current_workflow=current_workflow,
                        current_ui_workflow=current_ui_workflow,
                        candidate=candidate,
                        validation=validation,
                        execution_errors=body.get("execution_errors") or [],
                    )

                final_workflow = merge_workflow_into_current(
                    current_workflow,
                    {
                        **candidate,
                        "workflow": candidate.get("workflow") or candidate.get("workflow_api") or {},
                    },
                )
                message = candidate.get("assistant_message") or candidate.get("message") or ""
                if not validation["success"]:
                    issue_summary = summarize_validation_for_client(validation)
                    message = (
                        f"{message}\n\nThe workflow is not complete yet. "
                        f"Issues: {json.dumps(issue_summary, ensure_ascii=False)[:2500]}"
                    ).strip()
                elif missing_models:
                    message = (
                        f"{message}\n\nModel files are still missing; approve downloads to run this workflow."
                    ).strip()

                execute_after_apply = bool(body.get("execute") or candidate.get("run_after_apply"))
                execution_result = None
                if missing_models:
                    await emit(
                        "status",
                        text="The workflow references missing models. Waiting for your approval before downloading.",
                    )
                elif validation["success"] and execute_after_apply:
                    await emit("status", text="Queueing the validated workflow for execution.")
                    execution_result = await self._queue_prompt(final_workflow, client_id=body.get("client_id"))

                await emit(
                    "final",
                    text=message,
                    workflow_api=final_workflow,
                    validation=validation,
                    missing_models=missing_models,
                    execution=execution_result,
                    apply_to_current_graph=True,
                )
            except Exception as exc:
                logging.exception("ComfyUI Copilot chat failed")
                await emit("error", error=str(exc))
            finally:
                await response.write_eof()
            return response

    async def _generate_workflow_edit(
        self,
        *,
        settings: dict[str, str],
        prompt: str,
        history: list[dict[str, Any]],
        current_workflow: dict[str, Any],
        current_ui_workflow: dict[str, Any],
        execution_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        messages = build_copilot_messages(
            prompt=prompt,
            history=history,
            current_workflow=current_workflow,
            current_ui_workflow=current_ui_workflow,
            execution_errors=execution_errors,
        )
        text = await self._call_llm(settings, messages, temperature=0.15, max_tokens=16000)
        return _extract_json_object(text)

    async def _repair_workflow_edit(
        self,
        *,
        settings: dict[str, str],
        prompt: str,
        history: list[dict[str, Any]],
        current_workflow: dict[str, Any],
        current_ui_workflow: dict[str, Any],
        candidate: dict[str, Any],
        validation: dict[str, Any],
        execution_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        messages = build_copilot_messages(
            prompt=prompt,
            history=history,
            current_workflow=current_workflow,
            current_ui_workflow=current_ui_workflow,
            execution_errors=execution_errors,
            repair_context={
                "candidate": candidate,
                "validation": summarize_validation_for_llm(validation),
                "connection_hints": build_connection_hints_for_workflow(
                    candidate.get("workflow") or candidate.get("workflow_api") or {}
                ),
            },
        )
        text = await self._call_llm(settings, messages, temperature=0.05, max_tokens=16000)
        return _extract_json_object(text)

    async def _validate_workflow(self, workflow_api: dict[str, Any], *, strict_topology: bool) -> dict[str, Any]:
        if not isinstance(workflow_api, dict) or not workflow_api:
            return {"success": False, "error": "Workflow is empty or not an object.", "node_errors": {}}

        prompt_id = f"copilot-validate-{uuid.uuid4().hex}"
        valid = await execution.validate_prompt(prompt_id, workflow_api, None)
        topology = lint_workflow_topology(workflow_api)
        connection_issues = lint_workflow_connections(workflow_api)
        success = bool(valid[0]) and (not strict_topology or not topology) and not connection_issues
        return {
            "success": success,
            "error": None if bool(valid[0]) else valid[1],
            "node_errors": valid[3],
            "output_nodes": valid[2] if bool(valid[0]) else [],
            "topology_warnings": topology,
            "connection_issues": connection_issues,
        }

    async def _queue_prompt(self, workflow_api: dict[str, Any], *, client_id: str | None) -> dict[str, Any]:
        validation = await self._validate_workflow(workflow_api, strict_topology=True)
        if not validation["success"]:
            raise ValueError(json.dumps(validation, ensure_ascii=False))

        number = self.prompt_server.number
        self.prompt_server.number += 1
        prompt_id = str(uuid.uuid4())
        extra_data: dict[str, Any] = {"create_time": int(time.time() * 1000)}
        if client_id:
            extra_data["client_id"] = client_id
        self.prompt_server.prompt_queue.put(
            (number, prompt_id, workflow_api, extra_data, validation["output_nodes"], {})
        )
        return {
            "success": True,
            "prompt_id": prompt_id,
            "number": number,
            "node_errors": validation["node_errors"],
        }

    async def _list_models(self, settings: dict[str, str]) -> list[str]:
        provider = settings.get("provider", "openai")
        base_url = settings.get("base_url")
        api_key = settings.get("api_key")
        if provider == "anthropic":
            return [settings.get("model") or "claude-3-5-sonnet-latest"]

        url = _join_url(base_url or "https://api.openai.com/v1", "models")
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = await self._http_json("GET", url, headers=headers)
        models = data.get("data", data.get("models", data if isinstance(data, list) else []))
        names: list[str] = []
        if isinstance(models, list):
            for model in models:
                if isinstance(model, str):
                    names.append(model)
                elif isinstance(model, dict):
                    name = model.get("id") or model.get("name")
                    if name:
                        names.append(name)
        return sorted(set(names))

    async def _download_model(self, item: Any) -> dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError("Each download item must be an object.")
        folder = str(item.get("folder") or "").strip()
        url = str(item.get("url") or item.get("download_url") or "").strip()
        filename = _safe_model_filename(str(item.get("filename") or _filename_from_url(url) or "").strip())
        if not folder or folder not in folder_paths.folder_names_and_paths:
            raise ValueError(f"Unsupported model folder: {folder!r}")
        if not url or not await _is_download_url_allowed(url):
            raise ValueError(f"Refusing unsafe or unsupported download URL: {url!r}")
        if not filename:
            raise ValueError("A safe filename is required for model downloads.")

        destination = _model_destination_path(folder, filename)
        if os.path.exists(destination):
            return {
                "success": True,
                "folder": folder,
                "filename": filename,
                "path": destination,
                "already_present": True,
            }

        temp_path = f"{destination}.copilot-download-{uuid.uuid4().hex}.part"
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        bytes_written = 0
        try:
            session = self.prompt_server.client_session
            if session is not None:
                bytes_written = await _download_url_to_file(session, url, temp_path)
            else:
                timeout = aiohttp.ClientTimeout(total=None)
                async with aiohttp.ClientSession(timeout=timeout) as temp_session:
                    bytes_written = await _download_url_to_file(temp_session, url, temp_path)
            os.replace(temp_path, destination)
        finally:
            if os.path.exists(temp_path):
                with suppress(Exception):
                    os.remove(temp_path)

        return {
            "success": True,
            "folder": folder,
            "filename": filename,
            "path": destination,
            "bytes": bytes_written,
            "already_present": False,
        }

    async def _call_llm(
        self,
        settings: dict[str, str],
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        provider = settings.get("provider", "openai")
        if provider == "anthropic":
            return await self._call_anthropic(settings, messages, temperature=temperature, max_tokens=max_tokens)
        return await self._call_openai_compatible(
            settings, messages, temperature=temperature, max_tokens=max_tokens
        )

    async def _call_openai_compatible(
        self,
        settings: dict[str, str],
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        base_url = settings.get("base_url") or "https://api.openai.com/v1"
        api_key = settings.get("api_key")
        model = settings.get("model") or "gpt-4o-mini"
        if not api_key and not _is_local_base_url(base_url):
            raise ValueError("No LLM API key configured. Add your key in Copilot settings.")

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = await self._http_json("POST", _join_url(base_url, "chat/completions"), headers=headers, json=payload)
        try:
            return data["choices"][0]["message"]["content"] or "{}"
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected LLM response shape: {data}") from exc

    async def _call_anthropic(
        self,
        settings: dict[str, str],
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        base_url = settings.get("base_url") or "https://api.anthropic.com/v1"
        api_key = settings.get("api_key")
        model = settings.get("model") or "claude-3-5-sonnet-latest"
        if not api_key and not _is_local_base_url(base_url):
            raise ValueError("No Anthropic API key configured. Add your key in Copilot settings.")

        system = "\n\n".join(m["content"] for m in messages if m.get("role") == "system")
        anthropic_messages = [
            {"role": "assistant" if m.get("role") == "assistant" else "user", "content": m.get("content", "")}
            for m in messages
            if m.get("role") != "system"
        ]
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if api_key:
            headers["x-api-key"] = api_key
        payload = {
            "model": model,
            "system": system,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = await self._http_json("POST", _join_url(base_url, "messages"), headers=headers, json=payload)
        content = data.get("content") or []
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return "\n".join(text_parts) or "{}"

    async def _http_json(self, method: str, url: str, **kwargs) -> Any:
        session = self.prompt_server.client_session
        if session is not None:
            return await _request_json(session, method, url, **kwargs)
        timeout = aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as temp_session:
            return await _request_json(temp_session, method, url, **kwargs)


COPILOT_SYSTEM_PROMPT = """You are ComfyUI Copilot — the single local agent for ComfyUI.

You edit the workflow currently open in the user's graph. Never open a new workflow tab, never tell the user to paste JSON manually, and never claim the graph is complete until every required input is wired and every link type matches.

Return ONE JSON object only:
{
  "assistant_message": "concise explanation; if issues remain, list them explicitly",
  "workflow": { "node_id": { "class_type": "...", "inputs": { ... } } },
  "removed_node_ids": ["optional node ids to delete from the current graph"],
  "model_downloads": [
    {
      "folder": "checkpoints",
      "filename": "exact_model_filename.safetensors",
      "url": "https://direct-download-url",
      "node_id": "node id that uses it",
      "input_name": "input using it",
      "reason": "why this model is needed"
    }
  ],
  "run_after_apply": false,
  "workflow_complete": false
}

Set workflow_complete to true only when the workflow is fully wired, validates, and matches the user request.

Workflow rules:
- Use ComfyUI API/execution format keyed by string node ids.
- Use only installed class_type values from installed_nodes (core + custom nodes).
- Every required input must be a literal value or a valid link [source_node_id, output_slot_index] (0-based).
- Match link types using installed_nodes.outputs slot types and inputs slot types.
- When editing an existing graph, preserve unrelated nodes: include them in workflow or list removed_node_ids explicitly.
- Return the nodes you changed plus any nodes they connect to; the server merges into the current graph.
- Every non-output node must feed an output node (SaveImage, PreviewImage, etc.). No orphan nodes.
- Prefer installed model filenames from available_models. For missing models, keep filenames and add model_downloads with direct HTTPS URLs.
- Respect hardware_context: lower resolution/batch/steps on low VRAM or CPU-only systems.
- If execution_errors or repair_context.validation are present, fix those issues before claiming completion.
- run_after_apply: true only when the user clearly wants to run/execute and the workflow should be valid.
"""


# Common nodes included in every Copilot context (keeps simple requests small but useful).
ESSENTIAL_NODE_TYPES = frozenset(
    {
        "CheckpointLoaderSimple",
        "CheckpointLoader",
        "KSampler",
        "KSamplerAdvanced",
        "EmptyLatentImage",
        "VAEDecode",
        "VAEEncode",
        "VAELoader",
        "SaveImage",
        "PreviewImage",
        "CLIPTextEncode",
        "CLIPLoader",
        "LoraLoader",
        "LoraLoaderModelOnly",
        "UNETLoader",
        "DualCLIPLoader",
        "LoadImage",
        "ImageScale",
        "ControlNetLoader",
        "ControlNetApply",
        "ControlNetApplyAdvanced",
    }
)

# Rough character budget for the JSON user payload (~30-40k tokens with headroom under 128k total).
COPILOT_CONTEXT_CHAR_BUDGET = 100_000


def build_copilot_messages(
    *,
    prompt: str,
    history: list[dict[str, Any]],
    current_workflow: dict[str, Any],
    current_ui_workflow: dict[str, Any],
    execution_errors: list[dict[str, Any]] | None = None,
    repair_context: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    compact_history = []
    for message in history[-6:]:
        role = message.get("role", "user")
        if role in {"ai", "assistant"}:
            role = "assistant"
        elif role != "user":
            continue
        compact_history.append({"role": role, "content": str(message.get("content", ""))[:900]})

    catalog_limit = 140 if repair_context else 100
    relevant_nodes = build_relevant_node_catalog(
        query=prompt,
        workflow_api=current_workflow,
        repair_context=repair_context,
        limit=catalog_limit,
    )

    slim_repair = _slim_repair_context(repair_context) if repair_context else None
    context = trim_context_for_llm(
        {
            "user_request": prompt,
            "current_workflow_api": compact_workflow_api(current_workflow),
            "current_workflow_ui": summarize_ui_workflow(current_ui_workflow),
            "installed_nodes": relevant_nodes,
            "available_models": build_model_context(limit_per_folder=10),
            "hardware_context": build_hardware_context(),
            "execution_errors": _slim_execution_errors(execution_errors or [])[:4],
            "repair_context": slim_repair,
        }
    )
    user_content = json.dumps(context, ensure_ascii=False)
    if repair_context:
        user_content = (
            "Repair the candidate workflow. Fix every validation and connection issue. "
            "Do not claim workflow_complete until the graph is fully wired.\n"
            + user_content
        )
    return [
        {"role": "system", "content": COPILOT_SYSTEM_PROMPT},
        *compact_history,
        {"role": "user", "content": user_content},
    ]


def build_relevant_node_catalog(
    *,
    query: str,
    workflow_api: dict[str, Any],
    repair_context: dict[str, Any] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    pinned: set[str] = set(ESSENTIAL_NODE_TYPES)
    for node in (workflow_api or {}).values():
        if isinstance(node, dict) and node.get("class_type"):
            pinned.add(str(node["class_type"]))

    if repair_context:
        candidate = repair_context.get("candidate") or {}
        workflow = candidate.get("workflow") or candidate.get("workflow_api") or {}
        for node in workflow.values():
            if isinstance(node, dict) and node.get("class_type"):
                pinned.add(str(node["class_type"]))
        for hint in repair_context.get("connection_hints") or []:
            if isinstance(hint, dict) and hint.get("class_type"):
                pinned.add(str(hint["class_type"]))

    terms = [term.lower() for term in re.findall(r"[a-zA-Z0-9_./-]+", query or "") if len(term) > 2]
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for class_type in nodes.NODE_CLASS_MAPPINGS:
        try:
            info = _node_info(class_type)
        except Exception:
            continue
        entry = _minimal_catalog_entry(class_type, info)
        score = _score_catalog_entry(entry, terms)
        if class_type in pinned:
            score += 1000
        scored.append((score, class_type, entry))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [entry for _, _, entry in scored[:limit]]


def _minimal_catalog_entry(class_type: str, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "class_type": class_type,
        "display_name": info.get("display_name") or class_type,
        "category": str(info.get("category", ""))[:80],
        "inputs": _minimal_inputs(info.get("input", {})),
        "output_slots": _build_output_slots(info),
        "output_node": bool(info.get("output_node", False)),
    }


def _minimal_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for group_name in ("required", "optional"):
        group = inputs.get(group_name) if isinstance(inputs, dict) else {}
        if not isinstance(group, dict):
            continue
        compact[group_name] = {}
        for name, spec in group.items():
            compact[group_name][name] = _minimal_input_spec(spec)
    return compact


def _minimal_input_spec(spec: Any) -> Any:
    if isinstance(spec, (list, tuple)) and spec:
        first = spec[0]
        if isinstance(first, (list, tuple)):
            if len(first) <= 8:
                return {"type": "COMBO", "choices": list(first)}
            return {"type": "COMBO", "sample": list(first[:5]), "choice_count": len(first)}
        if isinstance(first, str):
            return first
        return str(first)
    return _json_safe(spec)


def compact_workflow_api(workflow_api: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(workflow_api, dict) or not workflow_api:
        return {}
    compact: dict[str, Any] = {}
    for node_id, node in workflow_api.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        compact_inputs: dict[str, Any] = {}
        if isinstance(inputs, dict):
            for key, value in inputs.items():
                if isinstance(value, str) and len(value) > 160:
                    compact_inputs[key] = value[:160] + "..."
                else:
                    compact_inputs[key] = value
        compact[str(node_id)] = {
            "class_type": node.get("class_type"),
            "inputs": compact_inputs,
        }
    return compact


def _slim_execution_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slim = []
    for error in errors:
        if not isinstance(error, dict):
            continue
        slim.append(
            {
                "node_id": error.get("node_id"),
                "node_type": error.get("node_type"),
                "exception_message": str(error.get("exception_message") or error.get("message") or "")[:500],
                "exception_type": error.get("exception_type"),
            }
        )
    return slim


def _slim_repair_context(repair_context: dict[str, Any]) -> dict[str, Any]:
    candidate = repair_context.get("candidate") or {}
    workflow = candidate.get("workflow") or candidate.get("workflow_api") or {}
    return {
        "candidate": {
            "assistant_message": candidate.get("assistant_message"),
            "workflow": compact_workflow_api(workflow),
            "removed_node_ids": candidate.get("removed_node_ids") or [],
            "model_downloads": (candidate.get("model_downloads") or [])[:8],
        },
        "validation": repair_context.get("validation"),
        "connection_hints": (repair_context.get("connection_hints") or [])[:40],
    }


def trim_context_for_llm(context: dict[str, Any], char_budget: int = COPILOT_CONTEXT_CHAR_BUDGET) -> dict[str, Any]:
    trimmed = _json_safe(context)
    payload = json.dumps(trimmed, ensure_ascii=False)
    if len(payload) <= char_budget:
        return trimmed

    reduced = dict(trimmed)
    for node_limit in (80, 60, 45, 30):
        nodes_list = reduced.get("installed_nodes")
        if isinstance(nodes_list, list) and len(nodes_list) > node_limit:
            reduced["installed_nodes"] = nodes_list[:node_limit]
        payload = json.dumps(reduced, ensure_ascii=False)
        if len(payload) <= char_budget:
            return reduced

    for folder_limit in (8, 5, 3):
        models = reduced.get("available_models")
        if isinstance(models, dict):
            reduced["available_models"] = {
                folder: files[:folder_limit] for folder, files in models.items() if isinstance(files, list)
            }
        payload = json.dumps(reduced, ensure_ascii=False)
        if len(payload) <= char_budget:
            return reduced

    ui = reduced.get("current_workflow_ui")
    if isinstance(ui, dict):
        reduced["current_workflow_ui"] = {
            "node_count": ui.get("node_count"),
            "nodes": (ui.get("nodes") or [])[:40],
            "links": (ui.get("links") or [])[:60],
        }
    payload = json.dumps(reduced, ensure_ascii=False)
    if len(payload) <= char_budget:
        return reduced

    workflow = reduced.get("current_workflow_api")
    if isinstance(workflow, dict) and len(workflow) > 40:
        keep_ids = list(workflow.keys())[:40]
        reduced["current_workflow_api"] = {node_id: workflow[node_id] for node_id in keep_ids}
    return reduced


def summarize_ui_workflow(workflow_ui: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(workflow_ui, dict):
        return {}
    nodes_ui = workflow_ui.get("nodes")
    links_ui = workflow_ui.get("links")
    summary: dict[str, Any] = {"node_count": 0, "nodes": [], "links": []}
    if isinstance(nodes_ui, list):
        summary["node_count"] = len(nodes_ui)
        summary["nodes"] = [
            {
                "id": node.get("id"),
                "type": node.get("type"),
                "title": node.get("title"),
                "widgets": _summarize_ui_widgets(node.get("widgets")),
            }
            for node in nodes_ui[:80]
            if isinstance(node, dict)
        ]
    if isinstance(links_ui, list):
        summary["links"] = [
            {
                "origin_id": link[1] if isinstance(link, list) and len(link) > 1 else None,
                "origin_slot": link[2] if isinstance(link, list) and len(link) > 2 else None,
                "target_id": link[3] if isinstance(link, list) and len(link) > 3 else None,
                "target_slot": link[4] if isinstance(link, list) and len(link) > 4 else None,
            }
            for link in links_ui[:120]
            if isinstance(link, list)
        ]
    return summary


def _summarize_ui_widgets(widgets: Any) -> list[dict[str, Any]]:
    if not isinstance(widgets, list):
        return []
    compact = []
    for widget in widgets[:30]:
        if not isinstance(widget, dict):
            continue
        value = widget.get("value")
        if isinstance(value, str) and len(value) > 120:
            value = value[:120] + "..."
        compact.append({"name": widget.get("name"), "type": widget.get("type"), "value": value})
    return compact


def merge_workflow_into_current(
    current_workflow: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    proposed = candidate.get("workflow") or candidate.get("workflow_api") or {}
    if not isinstance(proposed, dict):
        raise ValueError("LLM did not return a ComfyUI API-format workflow object.")
    if not current_workflow:
        return {str(node_id): node for node_id, node in proposed.items() if isinstance(node, dict)}

    merged = {str(node_id): node for node_id, node in current_workflow.items() if isinstance(node, dict)}
    for node_id in candidate.get("removed_node_ids") or []:
        merged.pop(str(node_id), None)
    for node_id, node in proposed.items():
        if isinstance(node, dict):
            merged[str(node_id)] = node
    return merged


def build_hardware_context() -> dict[str, Any]:
    with suppress(Exception):
        import comfy.model_management as mm

        device = mm.get_torch_device()
        device_name = mm.get_torch_device_name(device)
        cpu_device = mm.torch.device("cpu")
        ram_total = mm.get_total_memory(cpu_device)
        ram_free = mm.get_free_memory(cpu_device)
        vram_total, torch_vram_total = mm.get_total_memory(device, torch_total_too=True)
        vram_free, torch_vram_free = mm.get_free_memory(device, torch_free_too=True)
        vram_gb = round(vram_total / (1024**3), 2) if vram_total else 0
        vram_free_gb = round(vram_free / (1024**3), 2) if vram_free else 0
        recommendations = []
        if mm.cpu_mode():
            recommendations.append("CPU-only mode: use small resolutions, few steps, and lightweight models.")
        elif vram_gb and vram_gb < 8:
            recommendations.append("Low VRAM: prefer 512-768px, batch size 1, and model offloading nodes.")
        elif vram_gb and vram_gb < 16:
            recommendations.append("Mid VRAM: 1024px is usually fine; avoid huge batches or multiple large checkpoints.")
        else:
            recommendations.append("High VRAM: full-resolution workflows are feasible.")
        return {
            "device_name": device_name,
            "device_type": getattr(device, "type", "unknown"),
            "cpu_only": bool(mm.cpu_mode()),
            "vram_total_gb": vram_gb,
            "vram_free_gb": vram_free_gb,
            "ram_total_gb": round(ram_total / (1024**3), 2) if ram_total else None,
            "ram_free_gb": round(ram_free / (1024**3), 2) if ram_free else None,
            "recommendations": recommendations,
            "argv": [arg for arg in sys.argv if arg.startswith("--")][:12],
        }
    return {"cpu_only": "--cpu" in sys.argv, "recommendations": ["Hardware stats unavailable; assume conservative settings."]}


def summarize_validation_for_llm(validation: dict[str, Any]) -> dict[str, Any]:
    node_errors = validation.get("node_errors") or {}
    flattened = []
    for node_id, issues in node_errors.items():
        if not isinstance(issues, dict):
            continue
        for issue in issues.get("errors", []):
            if isinstance(issue, dict):
                flattened.append(
                    {
                        "node_id": str(node_id),
                        "type": issue.get("type"),
                        "message": issue.get("message"),
                        "details": issue.get("details"),
                        "extra_info": issue.get("extra_info"),
                    }
                )
    return {
        "success": validation.get("success"),
        "error": validation.get("error"),
        "topology_warnings": validation.get("topology_warnings") or [],
        "connection_issues": validation.get("connection_issues") or [],
        "issues": flattened[:40],
        "missing_models": validation.get("missing_models") or [],
    }


def summarize_validation_for_client(validation: dict[str, Any]) -> dict[str, Any]:
    return summarize_validation_for_llm(validation)


def build_connection_hints_for_workflow(workflow_api: dict[str, Any]) -> list[dict[str, Any]]:
    hints = []
    for node_id, node in (workflow_api or {}).items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if class_type not in nodes.NODE_CLASS_MAPPINGS:
            continue
        try:
            info = _node_info(class_type)
        except Exception:
            continue
        hints.append(
            {
                "node_id": str(node_id),
                "class_type": class_type,
                "inputs": _minimal_inputs(info.get("input", {})),
                "output_slots": _build_output_slots(info),
            }
        )
    return hints[:80]


def lint_workflow_connections(workflow_api: dict[str, Any]) -> list[str]:
    if not isinstance(workflow_api, dict):
        return ["workflow is not an object"]

    issues: list[str] = []
    for node_id, node in workflow_api.items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if class_type not in nodes.NODE_CLASS_MAPPINGS:
            continue
        obj_class = nodes.NODE_CLASS_MAPPINGS[class_type]
        try:
            input_specs = obj_class.INPUT_TYPES()
        except Exception:
            continue
        required = input_specs.get("required", {}) if isinstance(input_specs, dict) else {}
        optional = input_specs.get("optional", {}) if isinstance(input_specs, dict) else {}
        inputs = node.get("inputs") or {}
        if not isinstance(inputs, dict):
            continue

        for input_name, spec in {**required, **optional}.items():
            if input_name not in required and input_name not in inputs:
                continue
            if input_name not in inputs:
                if input_name in required:
                    issues.append(f"node {node_id} ({class_type}) missing required input {input_name!r}")
                continue
            value = inputs[input_name]
            if not _is_link(value):
                continue
            if not isinstance(value, list) or len(value) != 2:
                issues.append(f"node {node_id} input {input_name!r} has invalid link {value!r}")
                continue
            source_id, slot_index = str(value[0]), value[1]
            if source_id not in workflow_api:
                issues.append(f"node {node_id} input {input_name!r} links to missing node {source_id}")
                continue
            source = workflow_api[source_id]
            if not isinstance(source, dict):
                continue
            source_type = source.get("class_type")
            if source_type not in nodes.NODE_CLASS_MAPPINGS:
                issues.append(f"node {node_id} input {input_name!r} links to unknown class {source_type!r}")
                continue
            return_types = getattr(nodes.NODE_CLASS_MAPPINGS[source_type], "RETURN_TYPES", ())
            if not isinstance(slot_index, int) or slot_index < 0 or slot_index >= len(return_types):
                issues.append(
                    f"node {node_id} input {input_name!r} uses invalid output slot {slot_index} on node {source_id}"
                )
                continue
            received_type = return_types[slot_index]
            expected_type = spec[0] if isinstance(spec, (list, tuple)) and spec else spec
            from comfy_execution.validation import validate_node_input

            if isinstance(expected_type, str) and isinstance(received_type, str):
                if not validate_node_input(received_type, expected_type):
                    issues.append(
                        f"node {node_id} input {input_name!r} type mismatch: "
                        f"expected {expected_type}, got {received_type} from node {source_id} slot {slot_index}"
                    )
    return issues


def _merge_catalog_entries(primary: list[dict[str, Any]], extra: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = {entry["class_type"] for entry in primary}
    merged = list(primary)
    for entry in extra:
        if entry["class_type"] not in seen:
            merged.append(entry)
            seen.add(entry["class_type"])
    return merged


def _build_output_slots(info: dict[str, Any]) -> list[dict[str, Any]]:
    outputs = info.get("output") or []
    names = info.get("output_name") or outputs
    slots = []
    for index, output_type in enumerate(outputs):
        label = names[index] if index < len(names) else f"output_{index}"
        slots.append({"slot": index, "type": output_type, "name": label})
    return slots


def build_model_context(*, limit_per_folder: int = 10) -> dict[str, list[str]]:
    folders = [
        "checkpoints",
        "loras",
        "vae",
        "clip",
        "unet",
        "diffusion_models",
        "text_encoders",
        "controlnet",
        "upscale_models",
        "embeddings",
    ]
    context: dict[str, list[str]] = {}
    for folder in folders:
        try:
            files = folder_paths.get_filename_list(folder)
        except Exception:
            files = []
        if files:
            context[folder] = files[:limit_per_folder]
    return context


MODEL_INPUT_FOLDER_HINTS = (
    (("ckpt", "checkpoint"), "checkpoints"),
    (("lora",), "loras"),
    (("vae",), "vae"),
    (("controlnet", "control_net"), "controlnet"),
    (("upscale",), "upscale_models"),
    (("unet", "diffusion_model", "diffusion"), "diffusion_models"),
    (("clip",), "clip"),
    (("text_encoder", "text_encoders"), "text_encoders"),
    (("embedding",), "embeddings"),
)


def collect_missing_models(workflow_api: dict[str, Any], candidate: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Find model files referenced by a workflow that are not present locally."""
    folder_files = build_model_context_full()
    declared_downloads = _normalize_declared_downloads((candidate or {}).get("model_downloads"))
    declared_by_key = {
        (_normalize_folder(d.get("folder")), d.get("filename")): d
        for d in declared_downloads
        if d.get("folder") and d.get("filename")
    }
    missing: dict[tuple[str, str], dict[str, Any]] = {}

    for node_id, node in (workflow_api or {}).items():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if class_type not in nodes.NODE_CLASS_MAPPINGS:
            continue
        inputs = node.get("inputs") or {}
        if not isinstance(inputs, dict):
            continue
        try:
            input_specs = _node_info(class_type).get("input", {})
        except Exception:
            input_specs = {}

        for input_name, value in inputs.items():
            if not isinstance(value, str) or not value.strip() or _is_link(value):
                continue
            folder = _infer_model_folder(class_type, input_name, input_specs, folder_files)
            if not folder:
                continue
            filename = _safe_model_filename(value)
            if not filename or _model_file_exists(folder, filename, folder_files):
                continue
            key = (folder, filename)
            declared = declared_by_key.get(key, {})
            missing[key] = {
                "folder": folder,
                "filename": filename,
                "url": declared.get("url") or declared.get("download_url") or "",
                "node_id": str(node_id),
                "class_type": class_type,
                "input_name": input_name,
                "reason": declared.get("reason") or f"{class_type}.{input_name} references a model file that is not installed.",
            }

    for declared in declared_downloads:
        folder = _normalize_folder(declared.get("folder"))
        filename = declared.get("filename")
        if not folder or not filename:
            continue
        if _model_file_exists(folder, filename, folder_files):
            continue
        key = (folder, filename)
        missing.setdefault(
            key,
            {
                "folder": folder,
                "filename": filename,
                "url": declared.get("url") or declared.get("download_url") or "",
                "node_id": str(declared.get("node_id") or ""),
                "class_type": str(declared.get("class_type") or ""),
                "input_name": str(declared.get("input_name") or ""),
                "reason": str(declared.get("reason") or "Declared by Copilot for this workflow."),
            },
        )

    return list(missing.values())


def build_model_context_full() -> dict[str, list[str]]:
    context: dict[str, list[str]] = {}
    for folder in folder_paths.folder_names_and_paths:
        if folder in {"configs", "custom_nodes"}:
            continue
        try:
            context[folder] = folder_paths.get_filename_list(folder)
        except Exception:
            context[folder] = []
    return context


def _normalize_declared_downloads(downloads: Any) -> list[dict[str, Any]]:
    if not isinstance(downloads, list):
        return []
    normalized = []
    for item in downloads:
        if not isinstance(item, dict):
            continue
        folder = _normalize_folder(item.get("folder"))
        filename = _safe_model_filename(str(item.get("filename") or _filename_from_url(str(item.get("url") or "")) or ""))
        if not folder or not filename:
            continue
        normalized.append(
            {
                **item,
                "folder": folder,
                "filename": filename,
                "url": str(item.get("url") or item.get("download_url") or ""),
            }
        )
    return normalized


def _infer_model_folder(
    class_type: str,
    input_name: str,
    input_specs: dict[str, Any],
    folder_files: dict[str, list[str]],
) -> str | None:
    spec = _find_input_spec(input_name, input_specs)
    choices = _input_choices(spec)
    if choices:
        for folder, files in folder_files.items():
            overlap = set(choices) & set(files)
            if overlap:
                return folder

    haystack = f"{class_type} {input_name}".lower()
    for needles, folder in MODEL_INPUT_FOLDER_HINTS:
        if folder in folder_paths.folder_names_and_paths and any(needle in haystack for needle in needles):
            return folder
    return None


def _find_input_spec(input_name: str, input_specs: dict[str, Any]) -> Any:
    if not isinstance(input_specs, dict):
        return None
    for group_name in ("required", "optional"):
        group = input_specs.get(group_name)
        if isinstance(group, dict) and input_name in group:
            return group[input_name]
    return None


def _input_choices(spec: Any) -> list[str]:
    if isinstance(spec, (list, tuple)) and spec:
        first = spec[0]
        if isinstance(first, (list, tuple)):
            return [item for item in first if isinstance(item, str)]
    return []


def _model_file_exists(folder: str, filename: str, folder_files: dict[str, list[str]] | None = None) -> bool:
    files = (folder_files or build_model_context_full()).get(folder, [])
    normalized = filename.replace("\\", "/")
    return normalized in files or os.path.basename(normalized) in {os.path.basename(f) for f in files}


def _normalize_folder(folder: Any) -> str:
    folder_name = str(folder or "").strip()
    if folder_name in folder_paths.folder_names_and_paths:
        return folder_name
    aliases = {
        "checkpoint": "checkpoints",
        "ckpt": "checkpoints",
        "lora": "loras",
        "controlnets": "controlnet",
        "upscale": "upscale_models",
        "upscaler": "upscale_models",
        "embeddings": "embeddings",
    }
    return aliases.get(folder_name.lower(), "")


def _safe_model_filename(filename: str) -> str:
    filename = urllib.parse.unquote((filename or "").replace("\\", "/")).strip()
    if not filename or filename.startswith("/") or ".." in filename.split("/"):
        return ""
    safe_parts = []
    for part in filename.split("/"):
        clean = re.sub(r"[^a-zA-Z0-9._() \-+]", "_", part).strip()
        if not clean:
            return ""
        safe_parts.append(clean)
    return "/".join(safe_parts)


def _filename_from_url(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return ""
    return os.path.basename(urllib.parse.unquote(parsed.path))


def _model_destination_path(folder: str, filename: str) -> str:
    roots = folder_paths.get_folder_paths(folder)
    if not roots:
        raise ValueError(f"No filesystem path is configured for model folder {folder!r}.")
    root = os.path.abspath(roots[0])
    destination = os.path.abspath(os.path.join(root, filename))
    if os.path.commonpath((root, destination)) != root:
        raise ValueError("Model filename escapes the target folder.")
    return destination


def build_node_catalog(query: str = "", limit: int | None = 200) -> list[dict[str, Any]]:
    terms = [term.lower() for term in re.findall(r"[a-zA-Z0-9_./-]+", query or "") if len(term) > 2]
    catalog = []
    for class_type in sorted(nodes.NODE_CLASS_MAPPINGS):
        try:
            info = _node_info(class_type)
        except Exception:
            logging.exception("Failed to read node metadata for %s", class_type)
            continue
        entry = {
            "class_type": class_type,
            "display_name": info.get("display_name") or class_type,
            "category": info.get("category", ""),
            "description": str(info.get("description", ""))[:700],
            "inputs": _compact_inputs(info.get("input", {})),
            "outputs": info.get("output", []),
            "output_names": info.get("output_name", []),
            "output_slots": _build_output_slots(info),
            "output_node": bool(info.get("output_node", False)),
            "python_module": info.get("python_module", ""),
        }
        score = _score_catalog_entry(entry, terms)
        catalog.append((score, entry))

    catalog.sort(key=lambda item: (-item[0], item[1]["class_type"]))
    entries = [entry for _, entry in catalog]
    return entries if limit is None else entries[:limit]


def _node_info(node_class: str) -> dict[str, Any]:
    obj_class = nodes.NODE_CLASS_MAPPINGS[node_class]
    if issubclass(obj_class, _ComfyNodeInternal):
        return _json_safe(obj_class.GET_NODE_INFO_V1())
    input_types = obj_class.INPUT_TYPES()
    info = {
        "input": input_types,
        "input_order": {key: list(value.keys()) for (key, value) in input_types.items()},
        "output": getattr(obj_class, "RETURN_TYPES", []),
        "output_name": getattr(obj_class, "RETURN_NAMES", getattr(obj_class, "RETURN_TYPES", [])),
        "name": node_class,
        "display_name": nodes.NODE_DISPLAY_NAME_MAPPINGS.get(node_class, node_class),
        "description": getattr(obj_class, "DESCRIPTION", ""),
        "python_module": getattr(obj_class, "RELATIVE_PYTHON_MODULE", "nodes"),
        "category": getattr(obj_class, "CATEGORY", "sd"),
        "output_node": bool(getattr(obj_class, "OUTPUT_NODE", False)),
    }
    return _json_safe(info)


def _compact_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for group_name in ("required", "optional"):
        group = inputs.get(group_name) if isinstance(inputs, dict) else {}
        if not isinstance(group, dict):
            continue
        compact[group_name] = {}
        for name, spec in group.items():
            compact[group_name][name] = _compact_input_spec(spec)
    return compact


def _compact_input_spec(spec: Any) -> Any:
    if isinstance(spec, (list, tuple)) and spec:
        first = spec[0]
        metadata = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
        result: dict[str, Any] = {"type": _json_safe(first)}
        if isinstance(first, (list, tuple)) and len(first) > 20:
            result["type"] = list(first[:20])
            result["truncated_values"] = len(first) - 20
        for key in ("default", "min", "max", "step", "tooltip"):
            if key in metadata:
                result[key] = _json_safe(metadata[key])
        return result
    return _json_safe(spec)


def lint_workflow_topology(workflow_api: dict[str, Any]) -> list[str]:
    if not isinstance(workflow_api, dict):
        return ["workflow is not an object"]

    installed = set(nodes.NODE_CLASS_MAPPINGS.keys())
    warnings: list[str] = []
    output_nodes: set[str] = set()
    upstream: dict[str, set[str]] = {str(node_id): set() for node_id in workflow_api}

    for node_id, node in workflow_api.items():
        node_id = str(node_id)
        if not isinstance(node, dict):
            warnings.append(f"node {node_id} is not an object")
            continue
        class_type = node.get("class_type")
        if class_type not in installed:
            warnings.append(f"node {node_id} uses uninstalled class_type {class_type!r}")
            continue
        try:
            info = _node_info(class_type)
            if info.get("output_node"):
                output_nodes.add(node_id)
        except Exception:
            pass
        inputs = node.get("inputs") or {}
        if not isinstance(inputs, dict):
            warnings.append(f"node {node_id} inputs are not an object")
            continue
        for value in inputs.values():
            if _is_link(value):
                source_id = str(value[0])
                if source_id in workflow_api:
                    upstream[node_id].add(source_id)
                else:
                    warnings.append(f"node {node_id} links to missing node {source_id}")

    if not output_nodes:
        warnings.append("workflow has no output node such as SaveImage or PreviewImage")
        return warnings

    reachable: set[str] = set()
    stack = list(output_nodes)
    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        stack.extend(upstream.get(node_id, set()) - reachable)

    orphaned = sorted(set(str(node_id) for node_id in workflow_api) - reachable, key=_natural_node_key)
    if orphaned:
        warnings.append(f"orphan nodes not connected to an output: {', '.join(orphaned[:20])}")
    return warnings


def _score_catalog_entry(entry: dict[str, Any], terms: list[str]) -> int:
    if not terms:
        return 0
    haystack = " ".join(
        [
            entry.get("class_type", ""),
            entry.get("display_name", ""),
            entry.get("category", ""),
            entry.get("description", ""),
            entry.get("python_module", ""),
        ]
    ).lower()
    return sum(1 for term in terms if term in haystack)


def _llm_settings_from_request(request: web.Request, body: dict[str, Any]) -> dict[str, str]:
    provider = (
        body.get("provider")
        or request.headers.get("X-Copilot-Provider")
        or request.headers.get("LLM-Provider")
        or "openai"
    ).lower()
    base_url = (
        body.get("base_url")
        or request.headers.get("X-Copilot-Base-Url")
        or request.headers.get("Openai-Base-Url")
        or ("https://api.anthropic.com/v1" if provider == "anthropic" else "https://api.openai.com/v1")
    )
    if "anthropic.com" in base_url:
        provider = "anthropic"
    return {
        "provider": provider,
        "base_url": base_url.rstrip("/"),
        "api_key": body.get("api_key") or request.headers.get("X-Copilot-Api-Key") or request.headers.get("Openai-Api-Key") or "",
        "model": body.get("model") or request.headers.get("X-Copilot-Model") or "",
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("LLM returned an empty response.")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fenced:
            parsed = json.loads(fenced.group(1))
        else:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise
            parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object.")
    return parsed


def _json_safe(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val, depth + 1) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth + 1) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


async def _read_json(request: web.Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(text=f"Invalid JSON: {exc}") from exc
    if not isinstance(body, dict):
        raise web.HTTPBadRequest(text="Request body must be a JSON object.")
    return body


async def _request_json(session: aiohttp.ClientSession, method: str, url: str, **kwargs) -> Any:
    async with session.request(method, url, **kwargs) as response:
        text = await response.text()
        if response.status >= 400:
            raise ValueError(f"LLM request failed with HTTP {response.status}: {text[:1000]}")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON response: {text[:1000]}") from exc


async def _download_url_to_file(session: aiohttp.ClientSession, url: str, destination: str) -> int:
    async with session.get(url) as response:
        final_url = str(response.url)
        if not await _is_download_url_allowed(final_url):
            raise ValueError(f"Refusing redirect to unsafe URL: {final_url!r}")
        if response.status >= 400:
            text = await response.text()
            raise ValueError(f"Model download failed with HTTP {response.status}: {text[:1000]}")

        bytes_written = 0
        with open(destination, "wb") as file:
            async for chunk in response.content.iter_chunked(1024 * 1024):
                if not chunk:
                    continue
                file.write(chunk)
                bytes_written += len(chunk)
        if bytes_written <= 0:
            raise ValueError("Downloaded model was empty.")
        return bytes_written


async def _is_download_url_allowed(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.strip().lower()
    if host in {"localhost", "0.0.0.0"} or host.endswith(".localhost"):
        return False
    try:
        ip = _ip_address(host)
        if ip is not None:
            return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast)
        for result in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP):
            resolved = _ip_address(result[4][0])
            if resolved is None:
                return False
            if resolved.is_private or resolved.is_loopback or resolved.is_link_local or resolved.is_multicast:
                return False
    except OSError:
        return False
    return True


def _ip_address(host: str):
    try:
        import ipaddress

        return ipaddress.ip_address(host)
    except ValueError:
        return None


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_local_base_url(base_url: str) -> bool:
    return bool(re.search(r"^https?://(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])(?::|/|$)", base_url or ""))


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
    )


def _natural_node_key(node_id: str):
    try:
        return (0, int(node_id))
    except ValueError:
        return (1, node_id)
