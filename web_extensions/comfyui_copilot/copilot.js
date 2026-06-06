import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STORAGE_PREFIX = "comfyuiCopilot.";
const state = {
  open: false,
  busy: false,
  messages: [],
  lastWorkflow: null,
  pendingDownloads: [],
};

function setting(name, fallback = "") {
  return localStorage.getItem(STORAGE_PREFIX + name) || fallback;
}

function setSetting(name, value) {
  localStorage.setItem(STORAGE_PREFIX + name, value || "");
}

function headersFromSettings() {
  const headers = {
    "Content-Type": "application/json",
    "X-Copilot-Provider": setting("provider", "openai"),
    "X-Copilot-Base-Url": setting("baseUrl", "https://api.openai.com/v1"),
    "X-Copilot-Model": setting("model", "gpt-4o-mini"),
  };
  const apiKey = setting("apiKey");
  if (apiKey) {
    headers["X-Copilot-Api-Key"] = apiKey;
  }
  return headers;
}

async function currentGraphPayload() {
  if (!app?.graphToPrompt) {
    throw new Error("ComfyUI graph APIs are not ready yet.");
  }
  const graphData = await app.graphToPrompt();
  return {
    workflow_api: graphData.output || {},
    workflow_ui: graphData.workflow || {},
    client_id: api?.clientId || undefined,
  };
}

async function readJsonLineStream(response, onEvent) {
  if (!response.ok) {
    throw new Error(`Copilot request failed: HTTP ${response.status} ${await response.text()}`);
  }
  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error("Copilot response did not include a stream.");
  }
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) {
        onEvent(JSON.parse(line));
      }
    }
  }
  if (buffer.trim()) {
    onEvent(JSON.parse(buffer));
  }
}

async function sendCopilotMessage(prompt, execute = false) {
  const payload = await currentGraphPayload();
  const body = {
    ...payload,
    prompt,
    execute,
    messages: state.messages.slice(-10),
  };
  state.messages.push({ role: "user", content: prompt });
  renderMessages();
  setBusy(true);

  try {
    const response = await fetch("/api/copilot/chat", {
      method: "POST",
      headers: headersFromSettings(),
      body: JSON.stringify(body),
    });
    await readJsonLineStream(response, async (event) => {
      if (event.type === "status") {
        appendTransient(event.text || "Working...");
      } else if (event.type === "error") {
        appendMessage("assistant", `Copilot error: ${event.error}`);
      } else if (event.type === "final") {
        state.lastWorkflow = event.workflow_api || null;
        const text = event.text || "Workflow edit is ready.";
        const missingModels = event.missing_models || event.validation?.missing_models || [];
        appendMessage("assistant", text);
        if (event.workflow_api) {
          await applyWorkflowToCurrentGraph(event.workflow_api);
          appendMessage(
            "assistant",
            missingModels.length
              ? "Applied the edit to the current graph and laid it out. It needs model downloads before it can validate and run."
              : "Applied the validated edit to the current graph and laid it out without overlaps."
          );
        }
        if (missingModels.length) {
          state.pendingDownloads = missingModels;
          appendDownloadApproval(missingModels, Boolean(body.execute));
        } else if (event.validation && !event.validation.success) {
          appendMessage("assistant", `Validation still needs attention: ${JSON.stringify(event.validation)}`);
        }
      }
    });
  } finally {
    setBusy(false);
  }
}

async function validateCurrentGraph() {
  const payload = await currentGraphPayload();
  const response = await fetch("/api/copilot/validate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  appendMessage("assistant", result.success ? "Current workflow validates successfully." : `Validation issues: ${JSON.stringify(result)}`);
}

async function executeCurrentGraph() {
  const payload = await currentGraphPayload();
  const response = await fetch("/api/copilot/execute", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  appendMessage("assistant", result.success ? `Queued workflow ${result.prompt_id}.` : `Could not execute: ${JSON.stringify(result)}`);
}

async function approveModelDownloads(downloads, executeAfterDownload = false) {
  if (!downloads?.length) return;
  setBusy(true);
  try {
    appendMessage("assistant", `Downloading ${downloads.length} approved model file(s)...`);
    const response = await fetch("/api/copilot/download_models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved: true, downloads }),
    });
    const result = await response.json();
    if (!response.ok || !result.success) {
      appendMessage("assistant", `One or more model downloads failed: ${JSON.stringify(result)}`);
      return;
    }
    appendMessage("assistant", `Model downloads complete: ${result.results.map((item) => `${item.folder}/${item.filename}`).join(", ")}`);
    await validateCurrentGraph();
    if (executeAfterDownload) {
      await executeCurrentGraph();
    }
  } finally {
    setBusy(false);
  }
}

async function applyWorkflowToCurrentGraph(workflowApi) {
  if (!workflowApi || typeof workflowApi !== "object") {
    throw new Error("Copilot returned an invalid workflow.");
  }
  if (typeof app.loadApiJson === "function") {
    await app.loadApiJson(workflowApi);
  } else {
    throw new Error("This ComfyUI frontend does not expose loadApiJson().");
  }
  layoutGraphNoOverlap();
  app.graph?.setDirtyCanvas?.(true, true);
}

function layoutGraphNoOverlap() {
  const graph = app.graph;
  const graphNodes = graph?._nodes || graph?.nodes || [];
  if (!graph || !graphNodes.length) return;

  const nodeById = new Map(graphNodes.map((node) => [String(node.id), node]));
  const ranks = new Map(graphNodes.map((node) => [String(node.id), 0]));
  const links = Object.values(graph.links || {});

  for (let iteration = 0; iteration < graphNodes.length + 2; iteration++) {
    let changed = false;
    for (const link of links) {
      const origin = String(link?.origin_id ?? link?.[1] ?? "");
      const target = String(link?.target_id ?? link?.[3] ?? "");
      if (!nodeById.has(origin) || !nodeById.has(target)) continue;
      const nextRank = (ranks.get(origin) || 0) + 1;
      if (nextRank > (ranks.get(target) || 0)) {
        ranks.set(target, nextRank);
        changed = true;
      }
    }
    if (!changed) break;
  }

  const layers = new Map();
  for (const node of graphNodes) {
    const rank = ranks.get(String(node.id)) || 0;
    if (!layers.has(rank)) layers.set(rank, []);
    layers.get(rank).push(node);
  }

  const visible = app.canvas?.visible_area || [0, 0];
  const startX = visible[0] + 80;
  const startY = visible[1] + 80;
  const xSpacing = 360;
  const ySpacing = 90;

  [...layers.keys()].sort((a, b) => a - b).forEach((rank) => {
    const layer = layers.get(rank).sort((a, b) => String(a.id).localeCompare(String(b.id), undefined, { numeric: true }));
    let cursorY = startY;
    for (const node of layer) {
      if (typeof node.computeSize === "function") {
        const computed = node.computeSize();
        if (computed) node.size = computed;
      }
      const height = Array.isArray(node.size) ? node.size[1] : 120;
      node.pos = [startX + rank * xSpacing, cursorY];
      cursorY += Math.max(height, 120) + ySpacing;
    }
  });
}

function appendTransient(text) {
  const last = state.messages[state.messages.length - 1];
  if (last?.role === "status") {
    last.content = text;
  } else {
    state.messages.push({ role: "status", content: text });
  }
  renderMessages();
}

function appendMessage(role, content) {
  state.messages = state.messages.filter((message) => message.role !== "status");
  state.messages.push({ role, content });
  renderMessages();
}

function appendDownloadApproval(downloads, executeAfterDownload) {
  state.messages = state.messages.filter((message) => message.role !== "status");
  state.messages.push({
    role: "download_approval",
    content: "This workflow needs model files that are not installed locally. Review the list and approve downloads if you want Copilot to fetch them into ComfyUI's model folders.",
    downloads,
    executeAfterDownload,
  });
  renderMessages();
}

function setBusy(busy) {
  state.busy = busy;
  const panel = document.getElementById("comfyui-copilot-panel");
  if (panel) {
    panel.dataset.busy = busy ? "true" : "false";
    panel.querySelectorAll("button, textarea, input, select").forEach((el) => {
      if (!el.dataset.alwaysEnabled) el.disabled = busy;
    });
  }
}

function renderMessages() {
  const list = document.getElementById("comfyui-copilot-messages");
  if (!list) return;
  list.replaceChildren();
  for (const message of state.messages) {
    const item = document.createElement("div");
    item.className = `comfyui-copilot-message ${message.role}`;
    if (message.role === "download_approval") {
      renderDownloadApprovalMessage(item, message);
    } else {
      item.textContent = message.content;
    }
    list.appendChild(item);
  }
  list.scrollTop = list.scrollHeight;
}

function renderDownloadApprovalMessage(container, message) {
  const intro = document.createElement("div");
  intro.textContent = message.content;
  container.appendChild(intro);

  const list = document.createElement("ul");
  for (const download of message.downloads || []) {
    const row = document.createElement("li");
    const url = download.url || "No URL provided";
    row.textContent = `${download.folder}/${download.filename} - ${url}${download.reason ? ` (${download.reason})` : ""}`;
    list.appendChild(row);
  }
  container.appendChild(list);

  const actions = document.createElement("div");
  actions.className = "comfyui-copilot-download-actions";

  const approve = document.createElement("button");
  approve.textContent = message.executeAfterDownload ? "Approve downloads + execute" : "Approve downloads";
  approve.onclick = () => {
    const missingUrls = (message.downloads || []).filter((download) => !download.url);
    if (missingUrls.length) {
      appendMessage("assistant", `Cannot download yet; ${missingUrls.length} model(s) have no source URL.`);
      return;
    }
    approveModelDownloads(message.downloads, message.executeAfterDownload).catch((err) => appendMessage("assistant", err.message));
  };
  actions.appendChild(approve);

  const cancel = document.createElement("button");
  cancel.textContent = "Skip downloads";
  cancel.onclick = () => appendMessage("assistant", "Skipped model downloads. The workflow remains on the graph, but it may not execute until the missing models are installed.");
  actions.appendChild(cancel);
  container.appendChild(actions);
}

function createPanel() {
  injectStyles();
  const button = document.createElement("button");
  button.id = "comfyui-copilot-button";
  button.textContent = "Copilot";
  button.title = "Open local ComfyUI Copilot";
  button.dataset.alwaysEnabled = "true";
  button.onclick = () => togglePanel();
  document.body.appendChild(button);

  const panel = document.createElement("div");
  panel.id = "comfyui-copilot-panel";
  panel.innerHTML = `
    <div class="comfyui-copilot-header">
      <strong>Local Copilot</strong>
      <button id="comfyui-copilot-close" data-always-enabled="true">x</button>
    </div>
    <div class="comfyui-copilot-settings">
      <label>Provider
        <select id="comfyui-copilot-provider">
          <option value="openai">OpenAI-compatible</option>
          <option value="anthropic">Anthropic</option>
        </select>
      </label>
      <label>Base URL <input id="comfyui-copilot-base-url" autocomplete="off" /></label>
      <label>Model <input id="comfyui-copilot-model" autocomplete="off" /></label>
      <label>API key <input id="comfyui-copilot-api-key" type="password" autocomplete="off" placeholder="Stored in this browser only" /></label>
    </div>
    <div id="comfyui-copilot-messages"></div>
    <textarea id="comfyui-copilot-input" placeholder="Ask Copilot to edit, fix, validate, or optimize the current graph..."></textarea>
    <div class="comfyui-copilot-actions">
      <button id="comfyui-copilot-send">Edit current graph</button>
      <button id="comfyui-copilot-send-run">Edit + execute</button>
      <button id="comfyui-copilot-validate">Validate</button>
      <button id="comfyui-copilot-execute">Execute</button>
    </div>
  `;
  document.body.appendChild(panel);

  const provider = panel.querySelector("#comfyui-copilot-provider");
  const baseUrl = panel.querySelector("#comfyui-copilot-base-url");
  const model = panel.querySelector("#comfyui-copilot-model");
  const apiKey = panel.querySelector("#comfyui-copilot-api-key");
  provider.value = setting("provider", "openai");
  baseUrl.value = setting("baseUrl", provider.value === "anthropic" ? "https://api.anthropic.com/v1" : "https://api.openai.com/v1");
  model.value = setting("model", provider.value === "anthropic" ? "claude-3-5-sonnet-latest" : "gpt-4o-mini");
  apiKey.value = setting("apiKey");

  provider.onchange = () => {
    setSetting("provider", provider.value);
    if (!baseUrl.value || baseUrl.value.includes("api.openai.com") || baseUrl.value.includes("api.anthropic.com")) {
      baseUrl.value = provider.value === "anthropic" ? "https://api.anthropic.com/v1" : "https://api.openai.com/v1";
    }
    if (!model.value || model.value === "gpt-4o-mini" || model.value === "claude-3-5-sonnet-latest") {
      model.value = provider.value === "anthropic" ? "claude-3-5-sonnet-latest" : "gpt-4o-mini";
    }
    persistSettings();
  };
  [baseUrl, model, apiKey].forEach((el) => el.addEventListener("change", persistSettings));

  panel.querySelector("#comfyui-copilot-close").onclick = () => togglePanel(false);
  panel.querySelector("#comfyui-copilot-send").onclick = () => submitPrompt(false);
  panel.querySelector("#comfyui-copilot-send-run").onclick = () => submitPrompt(true);
  panel.querySelector("#comfyui-copilot-validate").onclick = () => validateCurrentGraph().catch((err) => appendMessage("assistant", err.message));
  panel.querySelector("#comfyui-copilot-execute").onclick = () => executeCurrentGraph().catch((err) => appendMessage("assistant", err.message));

  const input = panel.querySelector("#comfyui-copilot-input");
  input.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      submitPrompt(false);
    }
  });
}

function persistSettings() {
  setSetting("provider", document.getElementById("comfyui-copilot-provider")?.value);
  setSetting("baseUrl", document.getElementById("comfyui-copilot-base-url")?.value);
  setSetting("model", document.getElementById("comfyui-copilot-model")?.value);
  setSetting("apiKey", document.getElementById("comfyui-copilot-api-key")?.value);
}

function submitPrompt(execute) {
  const input = document.getElementById("comfyui-copilot-input");
  const prompt = input.value.trim();
  if (!prompt || state.busy) return;
  persistSettings();
  input.value = "";
  sendCopilotMessage(prompt, execute).catch((err) => {
    appendMessage("assistant", err.message);
    setBusy(false);
  });
}

function togglePanel(force) {
  state.open = typeof force === "boolean" ? force : !state.open;
  const panel = document.getElementById("comfyui-copilot-panel");
  if (panel) panel.classList.toggle("open", state.open);
}

function injectStyles() {
  if (document.getElementById("comfyui-copilot-styles")) return;
  const style = document.createElement("style");
  style.id = "comfyui-copilot-styles";
  style.textContent = `
    #comfyui-copilot-button {
      position: fixed; right: 18px; bottom: 18px; z-index: 10010;
      border: 0; border-radius: 999px; padding: 10px 14px;
      background: #3b82f6; color: white; font-weight: 700; cursor: pointer;
      box-shadow: 0 8px 20px rgba(0,0,0,.3);
    }
    #comfyui-copilot-panel {
      position: fixed; right: 18px; bottom: 66px; z-index: 10010;
      width: 390px; max-width: calc(100vw - 36px); height: min(720px, calc(100vh - 90px));
      display: none; flex-direction: column; gap: 8px; padding: 12px;
      background: var(--comfy-menu-bg, #202124); color: var(--fg-color, #f5f5f5);
      border: 1px solid rgba(255,255,255,.14); border-radius: 12px;
      box-shadow: 0 18px 40px rgba(0,0,0,.45); font: 13px sans-serif;
    }
    #comfyui-copilot-panel.open { display: flex; }
    .comfyui-copilot-header, .comfyui-copilot-actions { display: flex; align-items: center; gap: 8px; }
    .comfyui-copilot-header { justify-content: space-between; }
    .comfyui-copilot-header button, .comfyui-copilot-actions button {
      border: 1px solid rgba(255,255,255,.18); border-radius: 7px; padding: 6px 8px;
      background: rgba(255,255,255,.08); color: inherit; cursor: pointer;
    }
    .comfyui-copilot-actions { flex-wrap: wrap; }
    .comfyui-copilot-settings { display: grid; grid-template-columns: 1fr; gap: 6px; }
    .comfyui-copilot-settings label { display: grid; gap: 3px; color: rgba(255,255,255,.78); }
    .comfyui-copilot-settings input, .comfyui-copilot-settings select, #comfyui-copilot-input {
      width: 100%; box-sizing: border-box; border-radius: 7px; border: 1px solid rgba(255,255,255,.16);
      background: rgba(0,0,0,.25); color: inherit; padding: 7px;
    }
    #comfyui-copilot-messages {
      flex: 1; overflow: auto; display: flex; flex-direction: column; gap: 8px;
      border: 1px solid rgba(255,255,255,.10); border-radius: 8px; padding: 8px; min-height: 140px;
    }
    .comfyui-copilot-message { white-space: pre-wrap; line-height: 1.35; padding: 8px; border-radius: 8px; }
    .comfyui-copilot-message.user { align-self: flex-end; background: rgba(59,130,246,.32); }
    .comfyui-copilot-message.assistant { background: rgba(255,255,255,.08); }
    .comfyui-copilot-message.status { color: #fbbf24; background: rgba(251,191,36,.12); }
    .comfyui-copilot-message.download_approval { background: rgba(251,191,36,.14); border: 1px solid rgba(251,191,36,.35); }
    .comfyui-copilot-message.download_approval ul { margin: 8px 0; padding-left: 18px; word-break: break-word; }
    .comfyui-copilot-download-actions { display: flex; flex-wrap: wrap; gap: 8px; }
    .comfyui-copilot-download-actions button {
      border: 1px solid rgba(255,255,255,.18); border-radius: 7px; padding: 6px 8px;
      background: rgba(255,255,255,.10); color: inherit; cursor: pointer;
    }
    #comfyui-copilot-input { height: 76px; resize: vertical; }
    #comfyui-copilot-panel[data-busy="true"] .comfyui-copilot-actions button { opacity: .55; }
  `;
  document.head.appendChild(style);
}

app.registerExtension({
  name: "Comfy.LocalCopilot",
  setup() {
    createPanel();
  },
});
