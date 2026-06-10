# ComfyUI MCP Server

ComfyUI includes a stdio MCP server so Cursor, Claude Desktop, and other MCP clients can look up nodes and models **on demand** instead of loading the entire node catalog into LLM context.

ComfyUI Copilot uses the same MCP tool implementations **in-process** (`execute_copilot_tool`) so your selected model in Copilot settings is used directly. Node schemas are fetched one at a time via `search_nodes` / `get_node_info` — the full node catalog is never injected into LLM context.

## Prerequisites

1. ComfyUI running locally (default `http://127.0.0.1:8188`)
2. Python dependencies:

```bash
pip install -r mcp_requirements.txt
```

## Run manually

```bash
export COMFYUI_HOST=http://127.0.0.1:8188
export COMFY_MCP_USE_HTTP=1
python3 -m comfy_mcp
```

## Cursor configuration

Add to your Cursor MCP settings (`.cursor/mcp.json` in the project or user settings):

```json
{
  "mcpServers": {
    "comfyui": {
      "command": "python3",
      "args": ["-m", "comfy_mcp"],
      "cwd": "/absolute/path/to/ComfyUI",
      "env": {
        "COMFYUI_HOST": "http://127.0.0.1:8188",
        "COMFY_MCP_USE_HTTP": "1"
      }
    }
  }
}
```

Replace `cwd` with your ComfyUI checkout path.

## Tools

| Tool | Description |
|------|-------------|
| `search_nodes` | Search installed nodes by keyword |
| `get_node_info` | Input/output schema for a single `class_type` |
| `list_model_files` | List files in a models folder |
| `get_hardware` | GPU/VRAM/CPU stats and recommendations |
| `validate_workflow` | Validate API workflow wiring |

Metadata: `GET /api/copilot/mcp`
