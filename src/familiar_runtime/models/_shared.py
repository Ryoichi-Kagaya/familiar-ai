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
_TOOL_CALL_START_RE = re.compile(r"<tool_call>\s*", re.IGNORECASE)
_TOOL_CALL_CLOSE_RE = re.compile(
    r"\s*(?:</tool_call>|</parameter>\s*</invoke>)",
    re.IGNORECASE,
)
_TOOL_CONTROL_TAG_RE = re.compile(
    r"</?(?:tool_call|parameter|invoke)>|<invoke\b[^>]*>|<parameter\b[^>]*>",
    re.IGNORECASE,
)


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


def _extract_tool_calls_from_text(text: str) -> tuple[list[ToolCall], list[tuple[int, int]]]:
    """Extract tool-call JSON and its spans, tolerating malformed closing XML.

    Some prompt-driven models correctly emit the JSON object but finish it with
    another tool protocol's ``</parameter></invoke>`` tags (or omit a closing
    tag entirely).  ``JSONDecoder.raw_decode`` lets the JSON object itself be
    the authoritative boundary instead of trusting model-generated XML.
    """
    decoder = json.JSONDecoder()
    tool_calls: list[ToolCall] = []
    spans: list[tuple[int, int]] = []
    for match in _TOOL_CALL_START_RE.finditer(text):
        try:
            data, json_end = decoder.raw_decode(text, match.end())
            if not isinstance(data, dict):
                raise ValueError("tool_call JSON must be an object")
            name = data.get("name")
            tool_input = data.get("input", {})
            if not isinstance(name, str) or not name.strip():
                raise ValueError("tool_call name must be a non-empty string")
            if not isinstance(tool_input, dict):
                raise ValueError("tool_call input must be an object")

            close_match = _TOOL_CALL_CLOSE_RE.match(text, json_end)
            block_end = close_match.end() if close_match else json_end
            tool_calls.append(
                ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=name,
                    input=tool_input,
                )
            )
            spans.append((match.start(), block_end))
            if close_match and "</tool_call>" not in close_match.group(0).lower():
                logger.warning("Recovered tool_call %s from malformed closing tags", name)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse tool_call near offset %d: %s", match.start(), exc)
    return tool_calls, spans


def _parse_tool_calls_from_text(text: str) -> list[ToolCall]:
    """Extract prompt-driven tool calls from model output."""
    tool_calls, _ = _extract_tool_calls_from_text(text)
    return tool_calls


def _strip_tool_calls_from_text(text: str, spans: list[tuple[int, int]] | None = None) -> str:
    """Remove parsed or malformed tool protocol fragments from visible text."""
    if spans is None:
        _, spans = _extract_tool_calls_from_text(text)

    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    clean = "".join(parts)

    # Never expose an unparseable tool request. Preserve any useful prose that
    # preceded it, but discard the broken control block from its opening tag.
    broken_start = _TOOL_CALL_START_RE.search(clean)
    if broken_start:
        clean = clean[: broken_start.start()]
    clean = _TOOL_CONTROL_TAG_RE.sub("", clean)
    return clean.strip()


def _canonical_tool_call_text(tool_calls: list[ToolCall]) -> str:
    """Serialize parsed tool calls back into the one protocol we ask for."""
    return "\n".join(
        "<tool_call>"
        + json.dumps({"name": call.name, "input": call.input}, ensure_ascii=False)
        + "</tool_call>"
        for call in tool_calls
    )
