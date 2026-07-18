"""Anthropic Claude backend implementation."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any, cast

from ._shared import _supports_adaptive_thinking
from .base import ModelTurnResult, ToolCall
from .content import UserTurn, compact_image_blocks

logger = logging.getLogger(__name__)


def _detect_image_media_type(b64_data: str) -> str:
    """Detect image MIME type from base64-encoded bytes via magic bytes."""
    import base64

    try:
        padding = (4 - len(b64_data[:16]) % 4) % 4
        header = base64.b64decode(b64_data[:16] + "=" * padding)[:12]
    except Exception:
        return "image/jpeg"
    if header[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if header[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "image/jpeg"


class AnthropicBackend:
    """Backend using the official Anthropic SDK."""

    def __init__(
        self,
        api_key: str,
        model: str,
        thinking_mode: str = "auto",
        thinking_budget: int = 10000,
        thinking_effort: str = "high",
    ) -> None:
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.thinking_mode = thinking_mode
        self.thinking_budget = thinking_budget
        self.thinking_effort = thinking_effort

    def _build_thinking_params(self) -> dict:
        """Return thinking kwargs for the Anthropic API call.

        Per official docs (https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking):
        - adaptive thinking: no beta header needed (GA feature since Opus/Sonnet 4.6)
        - adaptive mode automatically enables interleaved thinking
        - extended mode on Sonnet 4.6 needs interleaved-thinking-2025-05-14 beta for interleaved support
        - extended mode on Opus 4.6 does NOT support interleaved thinking even with beta header

        Returns a dict that may contain:
          - "thinking": thinking config
          - "output_config": effort level (adaptive mode only)
          - "betas": list of beta header strings (extended mode on Sonnet 4.6 only)
        """
        mode = self.thinking_mode
        if mode == "auto":
            mode = "adaptive" if _supports_adaptive_thinking(self.model) else "disabled"

        if mode == "adaptive":
            params: dict = {"thinking": {"type": "adaptive"}}
            if self.thinking_effort != "high":
                params["output_config"] = {"effort": self.thinking_effort}
            return params

        if mode == "extended":
            params = {"thinking": {"type": "enabled", "budget_tokens": self.thinking_budget}}
            if "sonnet-4" in self.model:
                params["betas"] = ["interleaved-thinking-2025-05-14"]
            return params

        return {}

    # ── message factories ─────────────────────────────────────────

    def make_user_message(self, content: str | list | UserTurn) -> dict:
        if isinstance(content, UserTurn):
            blocks: list[dict[str, Any]] = [{"type": "text", "text": content.text}]
            blocks.extend(
                self.make_image_block(image.base64_data, image.media_type)
                for image in content.images
            )
            content = blocks
        return {"role": "user", "content": content}

    def make_image_block(self, b64: str, media_type: str = "image/jpeg") -> dict:
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        }

    def make_assistant_message(self, result: ModelTurnResult, raw_content: Any) -> dict:  # noqa: ARG002
        return {"role": "assistant", "content": raw_content}

    def make_tool_results(
        self,
        tool_calls: list[ToolCall],
        results: Sequence[tuple[str, str | list[str] | None]],
    ) -> list[dict]:
        """Returns a one-element list containing the Anthropic tool_result user message."""
        content: list[dict[str, Any]] = []
        for tc, (text, image) in zip(tool_calls, results):
            result_content: list[dict[str, Any]] = [{"type": "text", "text": text}]
            imgs: list[str] = image if isinstance(image, list) else ([image] if image else [])
            for img in imgs:
                result_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": _detect_image_media_type(img),
                            "data": img,
                        },
                    }
                )
            content.append({"type": "tool_result", "tool_use_id": tc.id, "content": result_content})
        msgs: list[dict[str, Any]] = [{"role": "user", "content": content}]
        return msgs

    # ── API calls ─────────────────────────────────────────────────

    def _convert_tools(self, tool_defs: list[dict]) -> list[dict]:
        return tool_defs  # already in Anthropic format

    def _flatten_messages(self, messages: list) -> list[dict]:
        """Expand nested lists (from make_tool_results) into a flat message list."""
        flat: list[dict] = []
        for msg in messages:
            if isinstance(msg, list):
                flat.extend(msg)
            else:
                flat.append(msg)
        return flat

    @staticmethod
    def compact_images(messages: list[dict], keep_last: int = 3) -> list[dict]:
        """Strip base64 image data from old tool results, keeping the last `keep_last`.

        Human-like forgetting: the text description of what was seen is preserved;
        only the raw pixel data (base64) is dropped from older turns.

        Inspired by Claude Code's Dk() microcompact (KEEP_LAST=3).
        """
        return compact_image_blocks(messages, keep_last=keep_last)

    @staticmethod
    def _build_system_param(system: str | tuple[str, str]) -> str | list[dict]:
        """Convert system prompt to Anthropic API format, adding cache_control when possible.

        If system is a (stable, variable) tuple, the stable block gets
        cache_control so it is reused across turns within the 5-minute window.
        If system is a plain string (e.g. from tests or other callers), pass as-is.
        """
        if not isinstance(system, tuple):
            return system
        stable, variable = system
        blocks: list[dict] = []
        if stable:
            blocks.append({"type": "text", "text": stable, "cache_control": {"type": "ephemeral"}})
        if variable:
            blocks.append({"type": "text", "text": variable})
        if len(blocks) == 1 and "cache_control" not in blocks[0]:
            return blocks[0]["text"]
        return blocks

    async def stream_turn(
        self,
        system: str | tuple[str, str],
        messages: list,
        tools: list[dict],
        max_tokens: int,
        on_text: Callable[[str], None] | None,
    ) -> tuple[ModelTurnResult, Any]:
        """Stream one agent turn. Returns (result, raw_content_for_assistant_message)."""
        from anthropic.types import MessageParam, ToolParam

        thinking_params = self._build_thinking_params()
        betas = thinking_params.pop("betas", [])

        sys_param = self._build_system_param(system)
        stream_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": sys_param,
            "tools": cast(list[ToolParam], self._convert_tools(tools)),
            "messages": cast(list[MessageParam], self._flatten_messages(messages)),
        }
        if betas:
            stream_kwargs["extra_headers"] = {"anthropic-beta": ",".join(betas)}
        if "thinking" in thinking_params:
            stream_kwargs["thinking"] = thinking_params["thinking"]
        if "output_config" in thinking_params:
            stream_kwargs["output_config"] = thinking_params["output_config"]
        flat_messages = self._flatten_messages(messages)
        flat_messages = self.compact_images(flat_messages)
        stream_kwargs["messages"] = cast(list[MessageParam], flat_messages)
        async with self.client.messages.stream(**stream_kwargs) as stream:  # type: ignore[arg-type]
            async for chunk in stream.text_stream:
                if on_text:
                    on_text(chunk)
            response = await stream.get_final_message()

        text = "".join(b.text for b in response.content if hasattr(b, "text"))
        tool_calls = [
            ToolCall(id=b.id, name=b.name, input=b.input)
            for b in response.content
            if b.type == "tool_use"
        ]
        stop = "end_turn" if response.stop_reason == "end_turn" else "tool_use"
        in_tok = getattr(response.usage, "input_tokens", 0) if response.usage else 0
        out_tok = getattr(response.usage, "output_tokens", 0) if response.usage else 0
        return (
            ModelTurnResult(
                stop_reason=stop,
                text=text,
                tool_calls=tool_calls,
                input_tokens=in_tok,
                output_tokens=out_tok,
            ),
            response.content,
        )

    async def complete(self, prompt: str, max_tokens: int) -> str:
        """Simple completion (no tools, no streaming) for utility calls."""
        try:
            logger.debug("complete() calling %s with %d chars", self.model, len(prompt))
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            from anthropic.types import TextBlock

            first = resp.content[0] if resp.content else None
            result = first.text.strip() if isinstance(first, TextBlock) else ""
            if not result:
                logger.warning(
                    "complete() empty response from %s: content=%s, stop=%s",
                    self.model,
                    resp.content,
                    resp.stop_reason,
                )
            return result
        except Exception as e:
            logger.warning("complete() failed: %s", e)
            return ""
