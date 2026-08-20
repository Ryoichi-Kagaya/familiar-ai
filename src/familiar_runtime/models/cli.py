"""CLI backend (shell out to any CLI LLM tool via stdin/stdout)."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import tempfile
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._shared import (
    _canonical_tool_call_text,
    _extract_tool_calls_from_text,
    _strip_tool_calls_from_text,
    _build_tools_system,
)
from .base import ModelTurnResult, ToolCall
from .content import UserTurn, compact_image_blocks

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _CLIInvocation:
    """Text plus optional provider-reported usage from one CLI invocation."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(slots=True)
class _ClaudeSessionState:
    """App-side mirror used to decide whether a Claude session can be resumed."""

    enabled: bool
    session_id: str | None = None
    context_key: str | None = None
    messages: list[Any] = field(default_factory=list)

    def reset(self) -> None:
        self.session_id = None
        self.context_key = None
        self.messages.clear()

    def can_resume(self, context_key: str, messages: list[Any]) -> bool:
        return (
            self.session_id is not None
            and self.context_key == context_key
            and len(messages) >= len(self.messages)
            and messages[: len(self.messages)] == self.messages
        )

    def remember(self, context_key: str, messages: list[Any], raw: dict[str, Any]) -> None:
        self.context_key = context_key
        # History entries are append-only. A shallow snapshot avoids copying
        # large image payloads; image compaction replaces entries wholesale,
        # which deliberately fails the next prefix check and rebuilds context.
        self.messages = [*messages, raw]


def _estimate_cli_tokens(text: str) -> int:
    """Estimate tokens for CLI backends that do not report model usage.

    Latin text averages roughly four characters per token, while Japanese and
    other non-ASCII scripts are commonly much denser. Counting each non-ASCII
    code point as one token keeps the compaction trigger deliberately
    conservative without adding a provider-specific tokenizer dependency.
    """
    if not text:
        return 0
    ascii_chars = sum(1 for char in text if ord(char) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return (ascii_chars + 3) // 4 + non_ascii_chars


class CLIBackend:
    """Backend that shells out to any CLI LLM tool via stdin/stdout.

    Tool calling uses prompt injection + <tool_call> tag parsing (same mechanism
    as OpenAICompatibleBackend with tools_mode="prompt"). Generic commands are
    text-only and reject image turns explicitly; Claude Code image transport is
    implemented by :class:`ClaudeCodeCLIBackend` below.

    Config::

        PLATFORM=cli
        MODEL=claude -p               # Claude Code — prompt goes via stdin
        MODEL=ollama run gemma3:27b   # stdin-based (no {} needed)
        MODEL=llm -m gpt-4o {}        # Simon Willison's llm CLI

    If the command contains ``{}``, the serialised prompt is injected there as a
    positional argument. Otherwise the prompt is written to **stdin**. Claude Code
    is handled specially and always receives prompts via stdin so large conversations
    cannot exceed the operating system's command-line size limit.
    """

    def __init__(self, command: list[str], *, timeout_seconds: float = 120.0) -> None:
        self._cmd = command
        self._timeout_seconds = timeout_seconds

    # ── message factories ─────────────────────────────────────────

    def make_image_block(self, b64: str, media_type: str = "image/jpeg") -> dict:  # noqa: ARG002
        return {"type": "text", "text": "[image]"}

    def make_user_message(self, content: str | list | UserTurn) -> dict:
        if isinstance(content, UserTurn):
            if content.images:
                raise RuntimeError("This CLI model command does not support image input.")
            content = content.text
        if isinstance(content, list):
            text = "\n".join(
                item["text"]
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
            return {"role": "user", "content": text}
        return {"role": "user", "content": content}

    def make_assistant_message(self, result: ModelTurnResult, raw_content: Any) -> dict:  # noqa: ARG002
        return raw_content

    def make_tool_results(
        self,
        tool_calls: list[ToolCall],
        results: Sequence[tuple[str, str | list[str] | None]],
    ) -> list[dict]:
        parts = [f"[Tool result: {tc.name}]\n{text}" for tc, (text, _) in zip(tool_calls, results)]
        return [{"role": "user", "content": "\n\n".join(parts)}]

    # ── conversation serialisation ────────────────────────────────

    def _fmt_msg(self, msg: dict) -> str:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        if isinstance(content, list):
            text = "\n".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") in ("text",)
            )
        else:
            text = str(content)
        prefix = "User" if role == "user" else "Assistant"
        return f"{prefix}:\n{text}"

    def _serialize(self, system: str | tuple[str, str], messages: list, tools: list[dict]) -> str:
        if isinstance(system, tuple):
            system = "\n\n---\n\n".join(s for s in system if s)
        parts: list[str] = []
        augmented = _build_tools_system(system, tools)
        if augmented:
            parts.append(f"<system>\n{augmented}\n</system>")

        for msg in messages:
            if isinstance(msg, list):
                for m in msg:
                    parts.append(self._fmt_msg(m))
            elif isinstance(msg, dict):
                parts.append(self._fmt_msg(msg))

        parts.append("Assistant:")
        return "\n\n".join(parts)

    # ── subprocess I/O ────────────────────────────────────────────

    @staticmethod
    async def _stop_process(proc: asyncio.subprocess.Process) -> None:
        """Terminate a child process, escalating to kill if it does not exit."""
        if proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()

    async def _run(
        self,
        prompt: str,
        *,
        command: list[str] | None = None,
        cwd: Path | None = None,
    ) -> str:
        """Run the CLI command with the prompt.

        If ``{}`` appears anywhere in the command, the prompt is injected
        there as a positional argument (e.g. ``claude -p {}``).
        Otherwise the prompt is written to stdin (e.g. ``ollama run model``).
        """
        # Strip CLAUDECODE so nested `claude -p` invocations are allowed
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        selected_command = command or self._cmd
        use_arg = "{}" in selected_command
        if use_arg:
            cmd = [prompt if tok == "{}" else tok for tok in selected_command]
            stdin_data: bytes | None = None
        else:
            cmd = selected_command
            stdin_data = prompt.encode("utf-8")

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE
                if stdin_data is not None
                else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=str(cwd) if cwd is not None else None,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"CLI command not found: {cmd[0]}") from exc
        except OSError as exc:
            raise RuntimeError(f"Failed to start CLI command {cmd[0]}: {exc}") from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(stdin_data), timeout=self._timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            await asyncio.shield(self._stop_process(proc))
            raise RuntimeError(
                f"CLI backend timed out after {self._timeout_seconds:g} seconds."
            ) from exc
        except asyncio.CancelledError:
            await asyncio.shield(self._stop_process(proc))
            raise

        stdout_text = stdout.decode("utf-8", errors="replace").strip()
        stderr_text = stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            detail = stderr_text or stdout_text or "no error output"
            logger.warning("CLI backend failed (%d): %s", proc.returncode, detail[:300])
            raise RuntimeError(f"CLI backend exited with code {proc.returncode}: {detail[:300]}")
        if not stdout_text:
            raise RuntimeError("CLI backend returned an empty response.")
        return stdout_text

    # ── backend interface ─────────────────────────────────────────

    @staticmethod
    def _normalize_turn(
        prompt: str,
        invocation: _CLIInvocation,
        on_text: Callable[[str], None] | None,
    ) -> tuple[ModelTurnResult, dict[str, Any]]:
        """Parse prompt-tool markup and attach measured or estimated usage."""
        text = invocation.text
        tool_calls, spans = _extract_tool_calls_from_text(text)
        clean_text = _strip_tool_calls_from_text(text, spans)
        stop = "tool_use" if tool_calls else "end_turn"
        if on_text and clean_text:
            on_text(clean_text)
        raw_text = _canonical_tool_call_text(tool_calls) if tool_calls else clean_text
        raw: dict[str, Any] = {"role": "assistant", "content": raw_text}
        return (
            ModelTurnResult(
                stop_reason=stop,
                text=clean_text,
                tool_calls=tool_calls,
                input_tokens=(
                    invocation.input_tokens
                    if invocation.input_tokens is not None
                    else _estimate_cli_tokens(prompt)
                ),
                output_tokens=(
                    invocation.output_tokens
                    if invocation.output_tokens is not None
                    else _estimate_cli_tokens(text)
                ),
            ),
            raw,
        )

    async def stream_turn(
        self,
        system: str | tuple[str, str],
        messages: list,
        tools: list[dict],
        max_tokens: int,  # noqa: ARG002
        on_text: Callable[[str], None] | None,
    ) -> tuple[ModelTurnResult, Any]:
        prompt = self._serialize(system, messages, tools)
        text = await self._run(prompt)
        return self._normalize_turn(prompt, _CLIInvocation(text), on_text)

    async def complete(self, prompt: str, max_tokens: int) -> str:  # noqa: ARG002
        return await self._run(prompt)


class ClaudeCodeCLIBackend(CLIBackend):
    """Claude Code print-mode adapter with image input via its ``Read`` tool.

    Claude Code's text/JSON input transport does not expose a stable binary
    image channel.  For image turns we stage validated bytes in an isolated
    temporary working directory, enable only ``Read``, and tell Claude to read
    those paths. Prompts always travel over stdin; a legacy ``{}`` placeholder in
    the configured command is removed before spawning the process.
    """

    _MEDIA_SUFFIXES = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }

    _SESSION_CONFLICT_FLAGS = {
        "--no-session-persistence",
        "--continue",
        "-c",
        "--resume",
        "-r",
        "--session-id",
        "--fork-session",
    }

    def __init__(self, command: list[str], *, timeout_seconds: float = 120.0) -> None:
        super().__init__(command, timeout_seconds=timeout_seconds)
        self._session = _ClaudeSessionState(
            enabled=not any(
                token in self._SESSION_CONFLICT_FLAGS
                or token.startswith(("--no-session-persistence=", "--resume=", "--session-id="))
                for token in command
            )
        )

    @property
    def manages_context(self) -> bool:
        """Whether Claude Code owns the live history and auto-compaction."""
        return self._session.enabled

    def reset_session(self) -> None:
        """Forget the resumable Claude session without touching app history."""
        self._session.reset()

    async def _run(
        self,
        prompt: str,
        *,
        command: list[str] | None = None,
        cwd: Path | None = None,
    ) -> str:
        """Send Claude prompts over stdin to avoid argv size limits."""
        selected_command = command or self._cmd
        stdin_command = [token for token in selected_command if token != "{}"]
        return await super()._run(prompt, command=stdin_command, cwd=cwd)

    @staticmethod
    def _without_option(command: list[str], names: set[str]) -> list[str]:
        """Remove value-taking CLI options in either ``--flag x`` or ``--flag=x`` form."""
        result: list[str] = []
        index = 0
        while index < len(command):
            token = command[index]
            if token in names:
                index += 2
                continue
            if any(token.startswith(f"{name}=") for name in names):
                index += 1
                continue
            result.append(token)
            index += 1
        return result

    @classmethod
    def _session_command(cls, command: list[str], session_id: str, *, resume: bool) -> list[str]:
        """Build a JSON print-mode command for a new or resumed managed session."""
        result = [token for token in command if token != "{}"]
        result = cls._without_option(
            result,
            {"--output-format", "--session-id", "--resume", "-r"},
        )
        result.extend(["--output-format", "json"])
        result.extend(["--resume" if resume else "--session-id", session_id])
        return result

    @classmethod
    def _one_shot_command(cls, command: list[str]) -> list[str]:
        """Keep utility completions out of the managed conversation session."""
        result = [token for token in command if token != "{}"]
        if "--no-session-persistence" not in result:
            result.append("--no-session-persistence")
        return result

    async def complete(self, prompt: str, max_tokens: int) -> str:  # noqa: ARG002
        return await self._run(prompt, command=self._one_shot_command(self._cmd))

    @staticmethod
    def _usage_tokens(payload: dict[str, Any], key: str) -> int:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return 0
        value = usage.get(key, 0)
        return int(value) if isinstance(value, int | float) else 0

    async def _run_managed(
        self,
        prompt: str,
        *,
        command: list[str],
        cwd: Path | None,
        resume: bool,
    ) -> _CLIInvocation:
        """Run one Claude Code query and recover its exact JSON usage metadata."""
        session_id = self._session.session_id if resume else str(uuid.uuid4())
        if session_id is None:  # defensive: ``resume`` is only true with an active id
            raise RuntimeError("Claude Code managed session is not initialized.")
        selected = self._session_command(command, session_id, resume=resume)
        raw = await self._run(prompt, command=selected, cwd=cwd)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Claude Code returned invalid JSON session output.") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("result"), str):
            raise RuntimeError("Claude Code JSON session output did not contain a result.")

        returned_session_id = payload.get("session_id")
        if isinstance(returned_session_id, str) and returned_session_id:
            session_id = returned_session_id
        self._session.session_id = session_id

        # Claude reports cache reads/creation separately from uncached input.
        # Their sum is the useful approximation of the current context size.
        input_tokens = sum(
            self._usage_tokens(payload, key)
            for key in (
                "input_tokens",
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
            )
        )
        output_tokens = self._usage_tokens(payload, "output_tokens")
        return _CLIInvocation(
            payload["result"].strip(),
            input_tokens or None,
            output_tokens or None,
        )

    def make_user_message(self, content: str | list | UserTurn) -> dict:
        if isinstance(content, UserTurn):
            blocks: list[dict[str, Any]] = [{"type": "text", "text": content.text}]
            blocks.extend(
                self.make_image_block(image.base64_data, image.media_type)
                for image in content.images
            )
            return {"role": "user", "content": blocks}
        return super().make_user_message(content)

    def make_image_block(self, b64: str, media_type: str = "image/jpeg") -> dict:
        if media_type not in self._MEDIA_SUFFIXES:
            raise RuntimeError(f"Unsupported Claude Code image type: {media_type}")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": b64},
        }

    def make_tool_results(
        self,
        tool_calls: list[ToolCall],
        results: Sequence[tuple[str, str | list[str] | None]],
    ) -> list[dict]:
        """Preserve images returned by tools such as ``see``.

        The generic CLI transport is text-only, but Claude Code can inspect
        staged image files through its Read tool. Represent tool-result images
        with the same blocks used for user attachments so ``stream_turn`` can
        stage them before invoking Claude.
        """
        if not any(image for _, image in results):
            return super().make_tool_results(tool_calls, results)

        parts: list[dict[str, Any]] = []
        for tool_call, (text, image) in zip(tool_calls, results):
            parts.append(
                {
                    "type": "text",
                    "text": f"[Tool result: {tool_call.name}]\n{text}",
                }
            )
            images = image if isinstance(image, list) else ([image] if image else [])
            parts.extend(self.make_image_block(item) for item in images)
        return [{"role": "user", "content": parts}]

    @staticmethod
    def _merge_tool_name(value: str, tool_name: str) -> str:
        if value == "default":
            return value
        names = [name.strip() for name in value.split(",") if name.strip()]
        if tool_name not in names:
            names.append(tool_name)
        return ",".join(names)

    @classmethod
    def _command_with_read(cls, command: list[str]) -> list[str]:
        """Enable and pre-approve Read while preserving other configured tools."""

        result = list(command)
        found_tools = False
        found_allowed = False
        index = 0
        while index < len(result):
            token = result[index]
            if token in {"--tools"} and index + 1 < len(result):
                result[index + 1] = cls._merge_tool_name(result[index + 1], "Read")
                found_tools = True
                index += 2
                continue
            if token.startswith("--tools="):
                result[index] = "--tools=" + cls._merge_tool_name(token.partition("=")[2], "Read")
                found_tools = True
            if token in {"--allowedTools", "--allowed-tools"} and index + 1 < len(result):
                result[index + 1] = cls._merge_tool_name(result[index + 1], "Read")
                found_allowed = True
                index += 2
                continue
            if token.startswith(("--allowedTools=", "--allowed-tools=")):
                flag, _, value = token.partition("=")
                result[index] = f"{flag}={cls._merge_tool_name(value, 'Read')}"
                found_allowed = True
            index += 1

        insert_at = result.index("{}") if "{}" in result else len(result)
        additions: list[str] = []
        if not found_tools:
            additions.extend(["--tools", "Read"])
        if not found_allowed:
            additions.extend(["--allowedTools", "Read"])
        result[insert_at:insert_at] = additions
        return result

    @classmethod
    def _stage_image(cls, block: dict[str, Any], directory: Path, index: int) -> Path:
        source = block.get("source", {})
        media_type = str(source.get("media_type", ""))
        suffix = cls._MEDIA_SUFFIXES.get(media_type)
        if suffix is None:
            raise RuntimeError(f"Unsupported Claude Code image type: {media_type}")
        try:
            raw = base64.b64decode(str(source.get("data", "")), validate=True)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Invalid image data for Claude Code.") from exc
        path = directory / f"image-{index}{suffix}"
        path.write_bytes(raw)
        return path

    def _fmt_msg_with_images(
        self,
        msg: dict[str, Any],
        directory: Path,
        next_index: int,
    ) -> tuple[str, int]:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        if not isinstance(content, list):
            return self._fmt_msg(msg), next_index

        lines: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                lines.append(str(item.get("text", "")))
            elif item.get("type") == "image":
                path = self._stage_image(item, directory, next_index)
                lines.append(
                    f"[Attached image {next_index}: {path}\n"
                    "Use the Read tool on this exact path before answering.]"
                )
                next_index += 1
        prefix = "User" if role == "user" else "Assistant"
        return f"{prefix}:\n" + "\n".join(lines), next_index

    def _serialize_with_images(
        self,
        system: str | tuple[str, str],
        messages: list,
        tools: list[dict],
        directory: Path,
    ) -> str:
        if isinstance(system, tuple):
            system = "\n\n---\n\n".join(part for part in system if part)
        parts: list[str] = []
        augmented = _build_tools_system(system, tools)
        if augmented:
            parts.append(f"<system>\n{augmented}\n</system>")

        image_index = 1
        for msg in messages:
            batch = msg if isinstance(msg, list) else [msg]
            for item in batch:
                if not isinstance(item, dict):
                    continue
                rendered, image_index = self._fmt_msg_with_images(item, directory, image_index)
                parts.append(rendered)
        parts.append("Assistant:")
        return "\n\n".join(parts)

    @staticmethod
    def _context_parts(system: str | tuple[str, str]) -> tuple[str, str]:
        if isinstance(system, tuple):
            return system
        return system, ""

    @classmethod
    def _context_key(cls, system: str | tuple[str, str], tools: list[dict]) -> str:
        """Key the context Claude must retain verbatim for delta-only turns."""
        stable, _ = cls._context_parts(system)
        return json.dumps(
            [stable, tools],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def _serialize_continuation(
        self,
        system: str | tuple[str, str],
        messages: list,
    ) -> str:
        """Serialize only state and messages not already held by Claude Code."""
        _, variable = self._context_parts(system)
        parts: list[str] = []
        if variable:
            parts.append(f"<turn-context>\n{variable}\n</turn-context>")
        for entry in messages:
            batch = entry if isinstance(entry, list) else [entry]
            parts.extend(self._fmt_msg(item) for item in batch if isinstance(item, dict))
        parts.append("Assistant:")
        return "\n\n".join(parts)

    @staticmethod
    def _has_images(messages: list[Any]) -> bool:
        return any(
            isinstance(message, dict)
            and isinstance(message.get("content"), list)
            and any(
                isinstance(block, dict) and block.get("type") == "image"
                for block in message["content"]
            )
            for entry in messages
            for message in (entry if isinstance(entry, list) else [entry])
        )

    async def _invoke_image_turn(
        self,
        system: str | tuple[str, str],
        messages: list[Any],
        tools: list[dict],
    ) -> tuple[str, _CLIInvocation]:
        """Run an image turn in a disposable directory and session."""
        self.reset_session()
        with tempfile.TemporaryDirectory(prefix="familiar-ai-images-") as temp_dir:
            directory = Path(temp_dir)
            prompt = self._serialize_with_images(system, messages, tools, directory)
            command = self._one_shot_command(self._command_with_read(self._cmd))
            text = await self._run(prompt, command=command, cwd=directory)
        return prompt, _CLIInvocation(text)

    async def _invoke_managed_text_turn(
        self,
        system: str | tuple[str, str],
        messages: list[Any],
        tools: list[dict],
    ) -> tuple[str, _CLIInvocation, str]:
        """Run a text turn from either a matching delta or a rebuilt transcript."""
        context_key = self._context_key(system, tools)
        resume = self._session.can_resume(context_key, messages)
        if resume:
            delta = messages[len(self._session.messages) :]
            prompt = self._serialize_continuation(system, delta)
        else:
            self.reset_session()
            prompt = self._serialize(system, messages, tools)
        invocation = await self._run_managed(
            prompt,
            command=self._cmd,
            cwd=None,
            resume=resume,
        )
        return prompt, invocation, context_key

    async def stream_turn(
        self,
        system: str | tuple[str, str],
        messages: list,
        tools: list[dict],
        max_tokens: int,  # noqa: ARG002
        on_text: Callable[[str], None] | None,
    ) -> tuple[ModelTurnResult, Any]:
        messages = compact_image_blocks(messages)
        has_images = self._has_images(messages)
        if not self._session.enabled and not has_images:
            return await super().stream_turn(system, messages, tools, max_tokens, on_text)

        if has_images:
            # Image paths live under an invocation-scoped cwd. Do not bind a
            # resumable session to that soon-to-be-deleted directory; process
            # the image as a one-shot and rebuild the text session next time.
            prompt, invocation = await self._invoke_image_turn(system, messages, tools)
            context_key = None
        else:
            prompt, invocation, context_key = await self._invoke_managed_text_turn(
                system, messages, tools
            )

        result, raw = self._normalize_turn(prompt, invocation, on_text)
        if context_key is not None:
            self._session.remember(context_key, messages, raw)
        return result, raw
