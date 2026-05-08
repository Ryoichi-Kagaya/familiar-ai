"""MCP (Model Context Protocol) client manager.

Connects to external MCP servers and exposes their tools to the agent.
Body-related tools (camera, TTS, mobility) stay as built-in; MCP is for everything else.

Supported transports
--------------------
* **stdio** — launch a local subprocess (default)
* **sse** — connect to an HTTP+SSE server

Config file: ~/.familiar-ai.json  (same mcpServers format as Claude Code's ~/.claude.json)
Override:    MCP_CONFIG=/path/to/config.json

Example config:
    {
      "mcpServers": {
        "filesystem": {
          "type": "stdio",
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
        },
        "memory": {
          "type": "sse",
          "url": "http://localhost:3000/sse"
        }
      }
    }
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif"})


def _load_image_b64(path: str, max_dim: int = 640) -> tuple[str, str | None]:
    """画像ファイルをリサイズしてbase64エンコードして返す。失敗時はエラーメッセージのみ。"""
    p = Path(path)
    try:
        data = p.read_bytes()
    except Exception as e:
        return f"Error reading {p.name}: {e}", None

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=85)
        data = buf.getvalue()
    except Exception:
        pass

    b64 = base64.b64encode(data).decode()
    return f"Image: {p.name} ({len(data):,} bytes)", b64

_DEFAULT_CONFIG = Path.home() / ".familiar-ai.json"


def _resolve_config_path() -> Path:
    env = os.environ.get("MCP_CONFIG", "")
    return Path(env) if env else _DEFAULT_CONFIG


def _load_servers(config_path: Path) -> dict[str, dict[str, Any]]:
    """Read mcpServers from the config file. Returns {} if file absent or malformed."""
    if not config_path.exists():
        return {}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        servers = data.get("mcpServers", {})
        if not isinstance(servers, dict):
            logger.warning("MCP config: mcpServers must be an object, ignoring")
            return {}
        return servers
    except Exception as e:
        logger.warning("Failed to load MCP config %s: %s", config_path, e)
        return {}


class MCPClientManager:
    """Manages MCP server connections (stdio and SSE) for the duration of the agent session."""

    def __init__(self, config_path: Path | None = None) -> None:
        self._config_path = config_path or _resolve_config_path()
        self._servers = _load_servers(self._config_path)
        self._sessions: dict[str, Any] = {}  # server_name → ClientSession
        # tool_name → server_name (for routing)
        self._tool_router: dict[str, str] = {}
        # Cached tool definitions (Anthropic format)
        self._tool_defs: list[dict[str, Any]] = []
        self._exit_stack = AsyncExitStack()
        self._started = False

    @property
    def is_started(self) -> bool:
        return self._started

    async def _register_tools(self, name: str, session: Any) -> int:
        """Register tools from a connected session. Returns count of registered tools."""
        tools_result = await session.list_tools()
        tools = tools_result.tools if hasattr(tools_result, "tools") else []
        count = 0
        for tool in tools:
            tool_name: str = tool.name
            if tool_name in self._tool_router:
                existing = self._tool_router[tool_name]
                logger.warning(
                    "MCP tool name collision: '%s' provided by both '%s' and '%s'; '%s' wins",
                    tool_name,
                    existing,
                    name,
                    existing,
                )
                continue

            self._tool_router[tool_name] = name
            self._tool_defs.append(
                {
                    "name": tool_name,
                    "description": tool.description or "",
                    "input_schema": (
                        tool.inputSchema
                        if isinstance(tool.inputSchema, dict)
                        else {"type": "object", "properties": {}}
                    ),
                }
            )
            count += 1
        return count

    async def start(self) -> None:
        """Connect to all configured servers. Skips servers that fail to connect."""
        if self._started:
            return
        self._started = True

        if not self._servers:
            return

        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.warning("mcp package not installed; MCP support disabled")
            return

        await self._exit_stack.__aenter__()

        for name, cfg in self._servers.items():
            server_type = cfg.get("type", "stdio")

            try:
                if server_type == "stdio":
                    command = cfg.get("command", "")
                    args: list[str] = cfg.get("args", [])
                    env: dict[str, str] | None = cfg.get("env") or None
                    cwd: str | None = cfg.get("cwd") or None

                    if not command:
                        logger.warning("MCP server '%s': missing 'command', skipping", name)
                        continue

                    params = StdioServerParameters(command=command, args=args, env=env, cwd=cwd)
                    read, write = await self._exit_stack.enter_async_context(stdio_client(params))
                    session: Any = await self._exit_stack.enter_async_context(
                        ClientSession(read, write)
                    )
                    await session.initialize()

                elif server_type == "sse":
                    from mcp.client.sse import sse_client

                    url = cfg.get("url", "")
                    if not url:
                        logger.warning(
                            "MCP server '%s': missing 'url' for sse type, skipping", name
                        )
                        continue

                    read, write = await self._exit_stack.enter_async_context(sse_client(url=url))
                    session = await self._exit_stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()

                else:
                    logger.warning(
                        "MCP server '%s': unsupported type '%s', skipping", name, server_type
                    )
                    continue

                self._sessions[name] = session
                count = await self._register_tools(name, session)
                logger.info("Connected to MCP server '%s' (%d tools)", name, count)

            except Exception as e:
                logger.warning("Failed to connect to MCP server '%s': %s", name, e)

    async def stop(self) -> None:
        """Close all MCP connections."""
        if not self._started:
            return
        try:
            await self._exit_stack.__aexit__(None, None, None)
        except Exception as e:
            logger.debug("MCP cleanup error: %s", e)

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return Anthropic-format tool definitions from all connected servers."""
        return list(self._tool_defs)

    async def call(self, tool_name: str, tool_input: dict[str, Any]) -> tuple[str, list[str]]:
        """Call a tool on the appropriate MCP server. Never raises — returns error as text."""
        # Intercept image paths before they reach MCP to prevent binary-as-text token bloat.
        if tool_name == "read_multiple_files":
            return await self._call_read_multiple(tool_input)
        if tool_name == "read_file":
            path = tool_input.get("path", "")
            if Path(path).suffix.lower() in _IMAGE_EXTS:
                desc, b64 = _load_image_b64(path)
                return desc, [b64] if b64 else []

        return await self._mcp_call(tool_name, tool_input)

    async def _call_read_multiple(self, tool_input: dict[str, Any]) -> tuple[str, list[str]]:
        """read_multiple_files のうち画像はローカル処理、テキストはMCPへ。"""
        paths: list[str] = tool_input.get("paths", [])
        image_paths = [p for p in paths if Path(p).suffix.lower() in _IMAGE_EXTS]
        text_paths = [p for p in paths if Path(p).suffix.lower() not in _IMAGE_EXTS]

        parts: list[str] = []
        images: list[str] = []

        for img_path in image_paths:
            desc, b64 = _load_image_b64(img_path)
            parts.append(desc)
            if b64 is not None:
                images.append(b64)

        if text_paths:
            new_input = {**tool_input, "paths": text_paths}
            mcp_text, _ = await self._mcp_call("read_multiple_files", new_input)
            parts.append(mcp_text)

        return "\n".join(parts) if parts else "(no output)", images

    async def _mcp_call(self, tool_name: str, tool_input: dict[str, Any]) -> tuple[str, list[str]]:
        """MCPサーバーへの実際の呼び出し。"""
        server_name = self._tool_router.get(tool_name)
        if server_name is None:
            return f"MCP tool '{tool_name}' not found.", []

        session = self._sessions.get(server_name)
        if session is None:
            return f"MCP server '{server_name}' is not connected.", []

        try:
            result = await session.call_tool(tool_name, arguments=tool_input)
        except Exception as e:
            logger.warning("MCP tool '%s' call failed: %s", tool_name, e)
            return f"MCP tool '{tool_name}' error: {e}", []

        text_parts: list[str] = []
        images: list[str] = []

        content = result.content if hasattr(result, "content") else []
        for item in content:
            item_type = getattr(item, "type", None)
            if item_type == "text":
                text_parts.append(item.text)
            elif item_type == "image":
                images.append(item.data)

        text = "\n".join(text_parts) if text_parts else "(no output)"
        return text, images
