import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STORAGE_PREFIX = "comfyuiCopilot.";
const COPILOT_TAB_ID = "comfyui-copilot";

const state = {
  busy: false,
  messages: [],
  lastWorkflow: null,
  pendingDownloads: [],
  lastExecutionErrors: [],
  panelRoot: null,
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
  if (apiKey) headers["X-Copilot-Api-Key"] = apiKey;
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
  if (!reader) throw new Error("Copilot response did not include a stream.");
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line));
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

function wantsExecution(prompt) {
  return /\b(run|execute|queue|render|generate|start)\b/i.test(prompt);
}

async function sendCopilotMessage(prompt) {
  const payload = await currentGraphPayload();
  const body = {
    ...payload,
    prompt,
    execute: wantsExecution(prompt),
    messages: state.messages
      .filter((message) => message.role === "user" || message.role === "assistant")
      .slice(-6)
      .map((message) => ({
        role: message.role,
        content: String(message.content || "").slice(0, 800),
      })),
    execution_errors: state.lastExecutionErrors.slice(-6),
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
    let finalEvent = null;
    await readJsonLineStream(response, async (event) => {
      if (event.type === "status") {
        appendTransient(event.text || "Working...");
      } else if (event.type === "error") {
        appendMessage("assistant", `Error: ${event.error}`);
      } else if (event.type === "final") {
        finalEvent = event;
      }
    });

    if (!finalEvent) return;

    state.lastWorkflow = finalEvent.workflow_api || null;
    const missingModels = finalEvent.missing_models || finalEvent.validation?.missing_models || [];
    let text = finalEvent.text || "Done.";

    if (finalEvent.workflow_api) {
      await applyWorkflowToCurrentGraph(finalEvent.workflow_api);
      if (finalEvent.validation?.success && !missingModels.length) {
        text = `${text}\n\nApplied changes to the current graph.`;
      } else if (missingModels.length) {
        text = `${text}\n\nGraph updated. Approve model downloads below to run it.`;
      } else if (!finalEvent.validation?.success) {
        text = `${text}\n\nPartial changes applied — ask me to fix remaining issues.`;
      } else {
        text = `${text}\n\nApplied changes to the current graph.`;
      }
    }

    appendMessage("assistant", text);

    if (missingModels.length) {
      state.pendingDownloads = missingModels;
      appendDownloadApproval(missingModels, Boolean(body.execute));
    }

    if (finalEvent.execution?.success) {
      appendMessage("assistant", `Queued workflow ${finalEvent.execution.prompt_id}.`);
      state.lastExecutionErrors = [];
    }
  } finally {
    setBusy(false);
  }
}

async function approveModelDownloads(downloads, executeAfterDownload = false) {
  if (!downloads?.length) return;
  setBusy(true);
  try {
    appendMessage("assistant", `Downloading ${downloads.length} model file(s)...`);
    const response = await fetch("/api/copilot/download_models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ approved: true, downloads }),
    });
    const result = await response.json();
    if (!response.ok || !result.success) {
      appendMessage("assistant", `Download failed: ${JSON.stringify(result)}`);
      return;
    }
    appendMessage("assistant", "Downloads complete. You can ask me to run the workflow.");
    if (executeAfterDownload) {
      const payload = await currentGraphPayload();
      const exec = await fetch("/api/copilot/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const execResult = await exec.json();
      if (execResult.success) {
        appendMessage("assistant", `Queued workflow ${execResult.prompt_id}.`);
      }
    }
  } finally {
    setBusy(false);
  }
}

async function applyWorkflowToCurrentGraph(workflowApi) {
  if (!workflowApi || typeof workflowApi !== "object") {
    throw new Error("Copilot returned an invalid workflow.");
  }

  const positions = new Map();
  const graphNodes = app.graph?._nodes || app.graph?.nodes || [];
  for (const node of graphNodes) {
    positions.set(String(node.id), node.pos ? [...node.pos] : null);
  }
  const previousIds = new Set(positions.keys());

  if (typeof app.loadApiJson !== "function") {
    throw new Error("This ComfyUI frontend does not expose loadApiJson().");
  }
  await app.loadApiJson(workflowApi);

  const updatedNodes = app.graph?._nodes || app.graph?.nodes || [];
  const newNodeIds = [];
  for (const node of updatedNodes) {
    const id = String(node.id);
    const saved = positions.get(id);
    if (saved) {
      node.pos = saved;
    } else if (!previousIds.has(id)) {
      newNodeIds.push(node);
    }
  }

  if (newNodeIds.length) {
    layoutNewNodes(newNodeIds, positions);
  }

  app.graph?.setDirtyCanvas?.(true, true);
}

function layoutNewNodes(newNodes, existingPositions) {
  const graph = app.graph;
  const links = Object.values(graph?.links || {});
  const ranks = new Map(newNodes.map((node) => [String(node.id), 0]));

  for (let i = 0; i < newNodes.length + 2; i++) {
    let changed = false;
    for (const link of links) {
      const origin = String(link?.origin_id ?? link?.[1] ?? "");
      const target = String(link?.target_id ?? link?.[3] ?? "");
      if (!ranks.has(origin) || !ranks.has(target)) continue;
      const nextRank = (ranks.get(origin) || 0) + 1;
      if (nextRank > (ranks.get(target) || 0)) {
        ranks.set(target, nextRank);
        changed = true;
      }
    }
    if (!changed) break;
  }

  let anchorX = 120;
  let anchorY = 120;
  for (const pos of existingPositions.values()) {
    if (pos) {
      anchorX = Math.max(anchorX, pos[0] + 420);
      anchorY = Math.min(anchorY, pos[1]);
    }
  }

  const layers = new Map();
  for (const node of newNodes) {
    const rank = ranks.get(String(node.id)) || 0;
    if (!layers.has(rank)) layers.set(rank, []);
    layers.get(rank).push(node);
  }

  const xSpacing = 360;
  const ySpacing = 90;
  [...layers.keys()].sort((a, b) => a - b).forEach((rank) => {
    let cursorY = anchorY;
    const layer = layers.get(rank).sort((a, b) => String(a.id).localeCompare(String(b.id), undefined, { numeric: true }));
    for (const node of layer) {
      if (typeof node.computeSize === "function") {
        const computed = node.computeSize();
        if (computed) node.size = computed;
      }
      const height = Array.isArray(node.size) ? node.size[1] : 120;
      node.pos = [anchorX + rank * xSpacing, cursorY];
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
    content: "Missing model files detected. Approve to download into ComfyUI model folders.",
    downloads,
    executeAfterDownload,
  });
  renderMessages();
}

function setBusy(busy) {
  state.busy = busy;
  const panel = state.panelRoot;
  if (!panel) return;
  const shell = panel.querySelector(".comfyui-copilot-shell");
  if (shell) shell.dataset.busy = busy ? "true" : "false";
  panel.querySelectorAll("button, textarea, input, select").forEach((el) => {
    if (!el.dataset.alwaysEnabled) el.disabled = busy;
  });
}

function renderMessages() {
  const list = state.panelRoot?.querySelector("#comfyui-copilot-messages");
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
    row.textContent = `${download.folder}/${download.filename}${download.reason ? ` — ${download.reason}` : ""}`;
    list.appendChild(row);
  }
  container.appendChild(list);

  const actions = document.createElement("div");
  actions.className = "comfyui-copilot-download-actions";

  const approve = document.createElement("button");
  approve.textContent = message.executeAfterDownload ? "Download + run" : "Download models";
  approve.onclick = () => {
    const missingUrls = (message.downloads || []).filter((download) => !download.url);
    if (missingUrls.length) {
      appendMessage("assistant", `${missingUrls.length} model(s) have no download URL yet. Ask me to find sources.`);
      return;
    }
    approveModelDownloads(message.downloads, message.executeAfterDownload).catch((err) => appendMessage("assistant", err.message));
  };
  actions.appendChild(approve);

  const cancel = document.createElement("button");
  cancel.textContent = "Skip";
  cancel.onclick = () => appendMessage("assistant", "Skipped downloads. The graph is updated but may not run until models are installed.");
  actions.appendChild(cancel);
  container.appendChild(actions);
}

function buildPanelMarkup() {
  return `
    <div class="comfyui-copilot-shell">
      <header class="comfyui-copilot-header">
        <div>
          <strong>ComfyUI Copilot</strong>
          <p class="comfyui-copilot-subtitle">Single agent — builds, fixes, and runs your graph</p>
        </div>
        <button type="button" class="comfyui-copilot-settings-toggle" data-always-enabled="true" title="Settings">⚙</button>
      </header>
      <section class="comfyui-copilot-settings collapsed">
        <label>Provider
          <select id="comfyui-copilot-provider">
            <option value="openai">OpenAI-compatible</option>
            <option value="anthropic">Anthropic</option>
          </select>
        </label>
        <label>Base URL <input id="comfyui-copilot-base-url" autocomplete="off" /></label>
        <label>Model
          <div class="comfyui-copilot-model-row">
            <select id="comfyui-copilot-model"></select>
            <button type="button" id="comfyui-copilot-refresh-models" data-always-enabled="true" title="Refresh model list">↻</button>
          </div>
        </label>
        <label>API key <input id="comfyui-copilot-api-key" type="password" autocomplete="off" placeholder="Stored locally in this browser" /></label>
      </section>
      <div id="comfyui-copilot-messages" class="comfyui-copilot-messages"></div>
      <div class="comfyui-copilot-composer">
        <textarea id="comfyui-copilot-input" placeholder="Ask anything: build a workflow, connect nodes, fix errors, download models, optimize for your GPU..."></textarea>
        <button id="comfyui-copilot-send" type="button">Send</button>
      </div>
    </div>
  `;
}

function defaultModelForProvider(providerName) {
  return providerName === "anthropic" ? "claude-3-5-sonnet-latest" : "gpt-4o-mini";
}

function ensureModelOption(select, modelName) {
  if (!modelName || [...select.options].some((opt) => opt.value === modelName)) return;
  const opt = document.createElement("option");
  opt.value = modelName;
  opt.textContent = modelName;
  select.appendChild(opt);
}

async function loadModelDropdown(options = {}) {
  const select = state.panelRoot?.querySelector("#comfyui-copilot-model");
  if (!select) return;
  const provider = state.panelRoot.querySelector("#comfyui-copilot-provider")?.value || "openai";
  const saved = setting("model", defaultModelForProvider(provider));
  const refreshBtn = state.panelRoot.querySelector("#comfyui-copilot-refresh-models");
  if (refreshBtn) refreshBtn.disabled = true;

  try {
    const response = await fetch("/api/copilot/models", { headers: headersFromSettings() });
    const data = await response.json();
    const models = data.models || [];
    select.replaceChildren();
    if (!models.length) {
      ensureModelOption(select, saved || defaultModelForProvider(provider));
    } else {
      for (const entry of models) {
        const name = entry.name || entry.label;
        if (!name) continue;
        const opt = document.createElement("option");
        opt.value = name;
        opt.textContent = entry.label || name;
        select.appendChild(opt);
      }
      ensureModelOption(select, saved);
    }
    select.value = [...select.options].some((opt) => opt.value === saved)
      ? saved
      : (select.options[0]?.value || saved);
    setSetting("model", select.value);
    if (options.showStatus && data.error) {
      appendMessage("assistant", `Could not refresh models: ${data.error}`);
    }
  } catch (err) {
    ensureModelOption(select, saved || defaultModelForProvider(provider));
    select.value = saved || defaultModelForProvider(provider);
    if (options.showStatus) {
      appendMessage("assistant", `Could not load models: ${err.message}`);
    }
  } finally {
    if (refreshBtn) refreshBtn.disabled = false;
  }
}

function wirePanel(root) {
  state.panelRoot = root;
  const provider = root.querySelector("#comfyui-copilot-provider");
  const baseUrl = root.querySelector("#comfyui-copilot-base-url");
  const model = root.querySelector("#comfyui-copilot-model");
  const apiKey = root.querySelector("#comfyui-copilot-api-key");
  const settings = root.querySelector(".comfyui-copilot-settings");
  const settingsToggle = root.querySelector(".comfyui-copilot-settings-toggle");
  const refreshModels = root.querySelector("#comfyui-copilot-refresh-models");

  provider.value = setting("provider", "openai");
  baseUrl.value = setting("baseUrl", provider.value === "anthropic" ? "https://api.anthropic.com/v1" : "https://api.openai.com/v1");
  apiKey.value = setting("apiKey");
  ensureModelOption(model, setting("model", defaultModelForProvider(provider.value)));
  model.value = setting("model", defaultModelForProvider(provider.value));

  const persistSettings = () => {
    setSetting("provider", provider.value);
    setSetting("baseUrl", baseUrl.value);
    setSetting("model", model.value);
    setSetting("apiKey", apiKey.value);
  };

  provider.onchange = async () => {
    if (!baseUrl.value || baseUrl.value.includes("api.openai.com") || baseUrl.value.includes("api.anthropic.com")) {
      baseUrl.value = provider.value === "anthropic" ? "https://api.anthropic.com/v1" : "https://api.openai.com/v1";
    }
    const nextDefault = defaultModelForProvider(provider.value);
    ensureModelOption(model, nextDefault);
    if (!model.value || model.value === "gpt-4o-mini" || model.value === "claude-3-5-sonnet-latest") {
      model.value = nextDefault;
    }
    persistSettings();
    await loadModelDropdown();
  };
  baseUrl.addEventListener("change", persistSettings);
  apiKey.addEventListener("change", persistSettings);
  model.addEventListener("change", persistSettings);
  refreshModels.onclick = () => loadModelDropdown({ showStatus: true });
  loadModelDropdown();

  settingsToggle.onclick = () => settings.classList.toggle("collapsed");

  root.querySelector("#comfyui-copilot-send").onclick = () => submitPrompt();
  const input = root.querySelector("#comfyui-copilot-input");
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitPrompt();
    }
  });

  renderMessages();
}

function submitPrompt() {
  const input = state.panelRoot?.querySelector("#comfyui-copilot-input");
  const prompt = input?.value?.trim();
  if (!prompt || state.busy) return;
  setSetting("provider", state.panelRoot.querySelector("#comfyui-copilot-provider")?.value);
  setSetting("baseUrl", state.panelRoot.querySelector("#comfyui-copilot-base-url")?.value);
  setSetting("model", state.panelRoot.querySelector("#comfyui-copilot-model")?.value);
  setSetting("apiKey", state.panelRoot.querySelector("#comfyui-copilot-api-key")?.value);
  input.value = "";
  sendCopilotMessage(prompt).catch((err) => {
    appendMessage("assistant", err.message);
    setBusy(false);
  });
}

function mountCopilotPanel(container) {
  injectStyles();
  container.classList.add("comfyui-copilot-host");
  container.innerHTML = buildPanelMarkup();
  wirePanel(container);
}

function unmountCopilotPanel() {
  state.panelRoot = null;
}

function registerSidebarTab() {
  const tab = {
    id: COPILOT_TAB_ID,
    icon: "icon-[lucide--sparkles]",
    title: "Copilot",
    tooltip: "ComfyUI Copilot",
    type: "custom",
    render: (el) => mountCopilotPanel(el),
    destroy: () => unmountCopilotPanel(),
  };

  if (app.extensionManager?.registerSidebarTab) {
    app.extensionManager.registerSidebarTab(tab);
    return true;
  }
  return false;
}

function bindExecutionErrors() {
  if (!api?.addEventListener) return;
  api.addEventListener("execution_error", ({ detail }) => {
    if (!detail) return;
    state.lastExecutionErrors.push(detail);
    if (state.lastExecutionErrors.length > 12) {
      state.lastExecutionErrors = state.lastExecutionErrors.slice(-12);
    }
    const summary = [
      detail.node_type ? `Node: ${detail.node_type}` : null,
      detail.exception_message || detail.message || "Execution failed",
    ].filter(Boolean).join(" — ");
    appendMessage("assistant", `Execution error captured. Tell me to fix it, or ask "fix the error".\n${summary}`);
  });
  api.addEventListener("execution_success", () => {
    state.lastExecutionErrors = [];
  });
}

function injectStyles() {
  if (document.getElementById("comfyui-copilot-styles")) return;
  const style = document.createElement("style");
  style.id = "comfyui-copilot-styles";
  style.textContent = `
    .comfyui-copilot-host {
      height: 100%;
      min-height: 0;
      display: flex;
      flex-direction: column;
      background: var(--comfy-menu-bg, #1a1b1e);
      color: var(--fg-color, #ececec);
      font: 13px/1.45 Inter, system-ui, sans-serif;
    }
    .comfyui-copilot-shell {
      display: flex;
      flex-direction: column;
      height: 100%;
      min-height: 0;
      gap: 10px;
      padding: 12px;
      box-sizing: border-box;
    }
    .comfyui-copilot-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 8px;
    }
    .comfyui-copilot-subtitle {
      margin: 2px 0 0;
      font-size: 11px;
      opacity: 0.72;
    }
    .comfyui-copilot-settings-toggle {
      border: 1px solid rgba(255,255,255,.14);
      background: rgba(255,255,255,.06);
      color: inherit;
      border-radius: 8px;
      width: 32px;
      height: 32px;
      cursor: pointer;
    }
    .comfyui-copilot-settings {
      display: grid;
      gap: 8px;
      padding: 10px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,.1);
      background: rgba(0,0,0,.18);
    }
    .comfyui-copilot-settings.collapsed { display: none; }
    .comfyui-copilot-settings label {
      display: grid;
      gap: 4px;
      font-size: 11px;
      opacity: 0.85;
    }
    .comfyui-copilot-model-row {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 6px;
      align-items: center;
    }
    #comfyui-copilot-refresh-models {
      border: 1px solid rgba(255,255,255,.14);
      background: rgba(255,255,255,.08);
      color: inherit;
      border-radius: 8px;
      width: 34px;
      height: 34px;
      cursor: pointer;
      font-size: 16px;
      line-height: 1;
    }
    .comfyui-copilot-settings input,
    .comfyui-copilot-settings select,
    #comfyui-copilot-input {
      width: 100%;
      box-sizing: border-box;
      border-radius: 8px;
      border: 1px solid rgba(255,255,255,.14);
      background: rgba(0,0,0,.28);
      color: inherit;
      padding: 8px 10px;
    }
    .comfyui-copilot-messages {
      flex: 1;
      min-height: 0;
      overflow: auto;
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 10px;
      border-radius: 10px;
      border: 1px solid rgba(255,255,255,.08);
      background: rgba(0,0,0,.22);
    }
    .comfyui-copilot-message {
      white-space: pre-wrap;
      padding: 10px 12px;
      border-radius: 10px;
      max-width: 100%;
    }
    .comfyui-copilot-message.user {
      align-self: flex-end;
      background: linear-gradient(135deg, rgba(59,130,246,.45), rgba(99,102,241,.35));
    }
    .comfyui-copilot-message.assistant {
      background: rgba(255,255,255,.07);
    }
    .comfyui-copilot-message.status {
      color: #fbbf24;
      background: rgba(251,191,36,.1);
      font-size: 12px;
    }
    .comfyui-copilot-message.download_approval {
      background: rgba(251,191,36,.12);
      border: 1px solid rgba(251,191,36,.28);
    }
    .comfyui-copilot-message.download_approval ul {
      margin: 8px 0;
      padding-left: 18px;
      word-break: break-word;
    }
    .comfyui-copilot-download-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 8px;
    }
    .comfyui-copilot-download-actions button {
      border: 1px solid rgba(255,255,255,.16);
      border-radius: 8px;
      padding: 6px 10px;
      background: rgba(255,255,255,.1);
      color: inherit;
      cursor: pointer;
    }
    .comfyui-copilot-composer {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 8px;
      align-items: end;
    }
    #comfyui-copilot-input {
      min-height: 72px;
      resize: vertical;
    }
    #comfyui-copilot-send {
      border: 0;
      border-radius: 10px;
      padding: 10px 16px;
      font-weight: 600;
      cursor: pointer;
      color: white;
      background: linear-gradient(135deg, #3b82f6, #6366f1);
      height: fit-content;
    }
    .comfyui-copilot-shell[data-busy="true"] #comfyui-copilot-send {
      opacity: 0.55;
      cursor: wait;
    }
  `;
  document.head.appendChild(style);
}

function waitForSidebarRegistration() {
  if (registerSidebarTab()) return;
  const started = Date.now();
  const timer = setInterval(() => {
    if (registerSidebarTab() || Date.now() - started > 30000) {
      clearInterval(timer);
    }
  }, 300);
}

app.registerExtension({
  name: "Comfy.LocalCopilot",
  async setup() {
    bindExecutionErrors();
    waitForSidebarRegistration();
  },
});
