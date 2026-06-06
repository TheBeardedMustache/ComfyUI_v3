from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
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
                        "/api/copilot/node_catalog",
                        "/api/copilot/models",
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
                )

                max_attempts = _parse_int(body.get("max_repair_iterations"), 5)
                validation = {"success": False, "error": "Workflow was not validated"}
                for attempt in range(1, max_attempts + 1):
                    workflow_api = candidate.get("workflow") or candidate.get("workflow_api")
                    if not isinstance(workflow_api, dict):
                        raise ValueError("LLM did not return a ComfyUI API-format workflow object.")

                    validation = await self._validate_workflow(workflow_api, strict_topology=True)
                    if validation["success"]:
                        await emit("status", text=f"Workflow validation passed after {attempt} attempt(s).")
                        break

                    await emit(
                        "status",
                        text=f"Validation found issues; repair pass {attempt} of {max_attempts}.",
                        validation=validation,
                    )
                    if attempt == max_attempts:
                        break
                    candidate = await self._repair_workflow_edit(
                        settings=settings,
                        prompt=prompt,
                        candidate=candidate,
                        validation=validation,
                    )

                final_workflow = candidate.get("workflow") or candidate.get("workflow_api") or {}
                message = candidate.get("assistant_message") or candidate.get("message") or ""

                execute_after_apply = bool(body.get("execute"))
                execution_result = None
                if validation["success"] and execute_after_apply:
                    await emit("status", text="Queueing the validated workflow for execution.")
                    execution_result = await self._queue_prompt(final_workflow, client_id=body.get("client_id"))

                await emit(
                    "final",
                    text=message,
                    workflow_api=final_workflow,
                    validation=validation,
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
    ) -> dict[str, Any]:
        messages = build_copilot_messages(
            prompt=prompt,
            history=history,
            current_workflow=current_workflow,
            current_ui_workflow=current_ui_workflow,
        )
        text = await self._call_llm(settings, messages, temperature=0.15, max_tokens=12000)
        return _extract_json_object(text)

    async def _repair_workflow_edit(
        self,
        *,
        settings: dict[str, str],
        prompt: str,
        candidate: dict[str, Any],
        validation: dict[str, Any],
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": COPILOT_REPAIR_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "original_request": prompt,
                        "candidate": candidate,
                        "validation": validation,
                        "installed_nodes": build_node_catalog(limit=700),
                        "available_models": build_model_context(),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        text = await self._call_llm(settings, messages, temperature=0.05, max_tokens=12000)
        return _extract_json_object(text)

    async def _validate_workflow(self, workflow_api: dict[str, Any], *, strict_topology: bool) -> dict[str, Any]:
        if not isinstance(workflow_api, dict) or not workflow_api:
            return {"success": False, "error": "Workflow is empty or not an object.", "node_errors": {}}

        prompt_id = f"copilot-validate-{uuid.uuid4().hex}"
        valid = await execution.validate_prompt(prompt_id, workflow_api, None)
        topology = lint_workflow_topology(workflow_api)
        success = bool(valid[0]) and (not strict_topology or not topology)
        return {
            "success": success,
            "error": None if bool(valid[0]) else valid[1],
            "node_errors": valid[3],
            "output_nodes": valid[2] if bool(valid[0]) else [],
            "topology_warnings": topology,
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


COPILOT_SYSTEM_PROMPT = """You are ComfyUI Copilot running locally inside ComfyUI.

Your job is to edit the workflow currently open in the user's graph. Do not create a new tab, do not tell the user to paste JSON manually, and do not leave partial fixes.

Return ONE JSON object only:
{
  "assistant_message": "short explanation of the edit",
  "workflow": { "...": { "class_type": "...", "inputs": { ... } } },
  "run_after_apply": false
}

Workflow rules:
- The workflow value must be ComfyUI API/execution format, keyed by string node ids.
- Use only installed node class_type values from installed_nodes, including custom nodes.
- Satisfy every required input with either a literal value or a valid [node_id, output_index] link.
- Preserve and modify the current workflow when one is provided. Generate from scratch only when the current graph is empty.
- Do not leave orphan nodes: every non-output node should contribute to a PreviewImage, SaveImage, or another output node.
- Prefer available local model filenames. If no model file is available, choose the closest installed default input value from node metadata.
- Keep the graph focused, runnable, and minimal.
- The frontend will apply the returned API workflow to the currently visible graph and lay it out, so do not include UI-only node positions.
"""


COPILOT_REPAIR_PROMPT = """You repair invalid ComfyUI API workflows.

Return ONE JSON object only with assistant_message and workflow. Use installed node metadata, fix every validation error, remove orphan nodes, preserve the user's requested behavior, and do not invent uninstalled class_type values.
"""


def build_copilot_messages(
    *,
    prompt: str,
    history: list[dict[str, Any]],
    current_workflow: dict[str, Any],
    current_ui_workflow: dict[str, Any],
) -> list[dict[str, str]]:
    compact_history = []
    for message in history[-8:]:
        role = message.get("role", "user")
        if role in {"ai", "assistant"}:
            role = "assistant"
        elif role != "user":
            continue
        compact_history.append({"role": role, "content": str(message.get("content", ""))[:2000]})

    context = {
        "user_request": prompt,
        "current_workflow_api": current_workflow,
        "current_workflow_ui_summary": summarize_ui_workflow(current_ui_workflow),
        "installed_nodes": build_node_catalog(query=prompt, limit=900),
        "available_models": build_model_context(),
    }
    return [
        {"role": "system", "content": COPILOT_SYSTEM_PROMPT},
        *compact_history,
        {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
    ]


def summarize_ui_workflow(workflow_ui: dict[str, Any]) -> dict[str, Any]:
    nodes_ui = workflow_ui.get("nodes") if isinstance(workflow_ui, dict) else None
    if not isinstance(nodes_ui, list):
        return {}
    return {
        "node_count": len(nodes_ui),
        "nodes": [
            {
                "id": node.get("id"),
                "type": node.get("type"),
                "title": node.get("title"),
            }
            for node in nodes_ui[:200]
            if isinstance(node, dict)
        ],
    }


def build_model_context() -> dict[str, list[str]]:
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
            context[folder] = files[:40]
    return context


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
