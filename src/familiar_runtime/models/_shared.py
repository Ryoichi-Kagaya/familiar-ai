"""Shared helpers used by multiple model provider adapters."""

from __future__ import annotations

import json
import logging
import re
import uuid

from .base import ToolCall

logger = logging.getLogger(__name__)


# ── Adaptive thinking model detection ──────────────────────────────

_ADAPTIVE_THINKING_MODELS = ("sonnet-4", "opus-4")


def _supports_adaptive_thinking(model: str) -> bool:
    """Return True if the model supports adaptive thinking (Sonnet 4.x / Opus 4.x)."""
    return any(m in model for m in _ADAPTIVE_THINKING_MODELS)


# ── Prompt-based tool calling ──────────────────────────────────────
# Used when the model doesn't support native function calling (most local VLMs).
# Tools are injected into the system prompt; the model outputs <tool_call> JSON.

_TOOLS_PROMPT_HEADER = """\

---
[USING TOOLS]
These are real tools made available to you by familiar-ai. They are not necessarily built into
this model or CLI process: familiar-ai executes each <tool_call> outside the process and returns
the result. Descriptions beginning with "[Connected MCP via familiar-ai: ...]" identify MCP tools
that familiar-ai has already connected. They remain available even if they do not appear in the
CLI's own MCP inventory. Do not claim that such a tool is unavailable without trying it.

You MUST use tools by outputting a <tool_call> block. This is the ONLY way to take actions.

RULE: When you want to use a tool, output EXACTLY this pattern and nothing after it:
<tool_call>{{"name": "...", "input": {{...}}}}</tool_call>

Then STOP. Do not write anything after the closing tag. The result will be given to you next.

CONCRETE EXAMPLES:
{examples}

Available familiar-ai tools ({tool_count}); each entry includes its exact JSON input schema:
{tools_desc}
[/USING TOOLS]
"""

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _build_tools_system(system: str, tools: list[dict]) -> str:
    """Append tool descriptions + usage instructions to a system prompt."""
    if not tools:
        return system

    desc_lines = []
    example_lines = []
    for t in tools:
        input_schema = t.get("input_schema", {})
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object", "properties": {}}
        props = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        schema_json = json.dumps(input_schema, ensure_ascii=False, separators=(",", ":"))
        desc_lines.append(f"- {t['name']}: {t['description']}\n  input_schema: {schema_json}")

        example_input: dict = {}
        for k in required:
            prop = props.get(k, {})
            ptype = prop.get("type", "string")
            enum = prop.get("enum")
            if enum:
                example_input[k] = enum[0]
            elif ptype == "integer":
                example_input[k] = prop.get("default", 30)
            else:
                example_input[k] = f"<{k}>"
        example_json = json.dumps({"name": t["name"], "input": example_input}, ensure_ascii=False)
        example_lines.append(f"<tool_call>{example_json}</tool_call>")

    tools_desc = "\n".join(desc_lines)
    examples = "\n".join(example_lines)
    return system + _TOOLS_PROMPT_HEADER.format(
        tool_count=len(tools), tools_desc=tools_desc, examples=examples
    )


def _parse_tool_calls_from_text(text: str) -> list[ToolCall]:
    """Extract <tool_call> JSON blocks from model output."""
    tool_calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            data = json.loads(match.group(1).strip())
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=data["name"],
                    input=data.get("input", {}),
                )
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning("Failed to parse tool_call: %s", match.group(1))
    return tool_calls
