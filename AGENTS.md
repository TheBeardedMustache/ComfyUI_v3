# AGENTS.md

## Cursor Cloud specific instructions

ComfyUI Core is a **single-process Python application**. The web UI, REST API, WebSocket queue, and SQLite database all run inside one `python3 main.py` process. There is no separate frontend dev server or Docker stack.

### Services

| Service | Command | Port |
|---------|---------|------|
| ComfyUI server | `python3 main.py --cpu --listen 127.0.0.1` | 8188 (default) |

Use `--cpu` on cloud VMs without a GPU. Add `--enable-manager` only if ComfyUI-Manager is installed (`pip install -r manager_requirements.txt`).

### Dependency install (manual / first-time)

See CI workflows in `.github/workflows/test-unit.yml` for the canonical install order:

1. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
2. `pip install -r requirements.txt`
3. `pip install -r tests-unit/requirements.txt` (for tests)

Ensure `~/.local/bin` is on `PATH` (pip installs scripts there on this VM).

### `python` vs `python3`

Execution tests spawn the server with the `python` command. Create a shim once if needed:

```bash
ln -sf "$(which python3)" "$HOME/.local/bin/python"
```

### Lint

```bash
pip install ruff
ruff check .
```

Pylint (optional, slower): `pylint comfy_api_nodes` after installing deps + `pip install pylint`.

### Tests

```bash
# Unit tests (no server)
python3 -m pytest tests-unit --ignore=tests-unit/app_test/copilot_manager_test.py

# Execution E2E (spawns its own server; use --port if 8188 is taken)
python3 -m pytest tests/execution -v --skip-timing-checks --port 8189

# Server launch smoke test
python3 main.py --cpu   # verify http://127.0.0.1:8188/ returns 200
```

**Note:** `tests-unit/app_test/copilot_manager_test.py` mocks `sys.modules` at import time and breaks collection when run together with other unit tests. Exclude it when running the full unit suite, or run it in isolation.

Inference tests (`tests/inference`) require SDXL checkpoint files in `models/checkpoints/` and are not needed for basic dev setup.

### ComfyUI Copilot

- Backend: `app/copilot_manager.py`
- Frontend extension: `web_extensions/comfyui_copilot/copilot.js` (sidebar tab via `app.extensionManager.registerSidebarTab`)
- Single agent with server-side validation/repair loop; edits merge into the current graph (positions preserved for existing nodes)
- **MCP tools** (`comfy_mcp/`): Copilot calls `search_nodes`, `get_node_info`, etc. on demand — node catalogs are not dumped into LLM context
- MCP stdio server: `pip install -r mcp_requirements.txt` then `python3 -m comfy_mcp` (see `doc/MCP-SETUP.md`)
- Copilot unit tests: run `tests-unit/app_test/copilot_manager_test.py` in isolation (it mocks `sys.modules` at import time)

### Gotchas

- No GPU in cloud VMs: always pass `--cpu` for local server runs and tests.
- Default SQLite DB is created at `user/comfyui.db` on first startup.
- Frontend is bundled via pip (`comfyui-frontend-package`); no `npm install` or Vite dev server.
- Execution tests default to port 8188; stop any running ComfyUI instance or pass `--port` to avoid conflicts.
