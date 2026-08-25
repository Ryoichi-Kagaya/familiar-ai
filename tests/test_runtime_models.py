"""Smoke tests for familiar_runtime.models package.

Exercises provider serialization paths (make_user_message, make_tool_results)
and shared helpers without making any API calls.  Each provider import is
deferred inside the test so a missing optional SDK fails that test only.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_base_protocol_and_dataclasses_importable() -> None:
    from familiar_runtime.models import ModelBackend, ModelTurnResult, ToolCall

    tc = ToolCall(id="call_x", name="ping", input={"k": "v"})
    res = ModelTurnResult(stop_reason="end_turn", text="hi")
    assert tc.name == "ping"
    assert res.stop_reason == "end_turn"
    # Protocol is importable and runtime_checkable-like (just ensure class exists).
    assert ModelBackend.__name__ == "ModelBackend"


def test_shared_helpers_exposed() -> None:
    from familiar_runtime.models import (
        _build_tools_system,
        _parse_tool_calls_from_text,
        _supports_adaptive_thinking,
    )

    assert _supports_adaptive_thinking("claude-sonnet-4-6") is True
    assert _supports_adaptive_thinking("claude-haiku-3-5") is False

    augmented = _build_tools_system(
        "You are helpful.",
        [
            {
                "name": "ping",
                "description": "Reply pong.",
                "input_schema": {
                    "type": "object",
                    "properties": {"message": {"type": "string", "description": "Payload"}},
                    "required": ["message"],
                },
            }
        ],
    )
    assert "[USING TOOLS]" in augmented
    assert "ping" in augmented
    assert "real tools made available to you by familiar-ai" in augmented
    assert "CLI's own MCP inventory" in augmented
    assert 'input_schema: {"type":"object","properties":{"message"' in augmented

    calls = _parse_tool_calls_from_text(
        'sure: <tool_call>{"name": "ping", "input": {"x": 1}}</tool_call>'
    )
    assert len(calls) == 1
    assert calls[0].name == "ping"
    assert calls[0].input == {"x": 1}


def test_user_message_metadata_includes_local_timestamp_and_weekday() -> None:
    from familiar_runtime.models import format_user_message_text

    sent_at = datetime(2026, 7, 23, 21, 5, 6, tzinfo=timezone(timedelta(hours=9)))

    assert format_user_message_text("こんばんは", sent_at=sent_at) == (
        "[familiar-ai message metadata: "
        "sent_at=2026-07-23T21:05:06+09:00; weekday=Thursday]\n"
        "こんばんは"
    )


def test_prompt_tool_call_recovers_mixed_closing_protocol() -> None:
    from familiar_runtime.models import (
        _parse_tool_calls_from_text,
        _strip_tool_calls_from_text,
    )

    text = '<tool_call>{"name": "get_status", "input": {}}</parameter>\n</invoke>'
    calls = _parse_tool_calls_from_text(text)

    assert [(call.name, call.input) for call in calls] == [("get_status", {})]
    assert _strip_tool_calls_from_text(text) == ""


def test_prompt_tool_call_fragments_never_become_visible_text() -> None:
    from familiar_runtime.models import _strip_tool_calls_from_text

    text = '先に確認するで。\n<tool_call>{"name": "get_status", "input": '
    assert _strip_tool_calls_from_text(text) == "先に確認するで。"


def test_backend_compat_aliases_match_runtime() -> None:
    """Legacy ``familiar_agent.backend`` imports must alias the runtime types."""
    from familiar_agent import backend as legacy
    from familiar_runtime.models import ModelTurnResult, ToolCall

    assert legacy.ToolCall is ToolCall
    assert legacy.TurnResult is ModelTurnResult
    assert legacy.ModelTurnResult is ModelTurnResult


def test_anthropic_backend_serializes_without_sdk_calls() -> None:
    pytest.importorskip("anthropic")
    from familiar_runtime.models import AnthropicBackend, ModelTurnResult, ToolCall

    backend = AnthropicBackend(api_key="test", model="claude-sonnet-4-6")
    user_msg = backend.make_user_message("hello")
    assert user_msg == {"role": "user", "content": "hello"}

    tc = ToolCall(id="t1", name="see", input={})
    results = backend.make_tool_results([tc], [("captured", "BASE64==")])
    assert results[0]["role"] == "user"
    parts = results[0]["content"]
    assert parts[0]["type"] == "tool_result"
    inner = parts[0]["content"]
    assert inner[0]["text"] == "captured"
    assert inner[1]["type"] == "image"
    assert inner[1]["source"]["data"] == "BASE64=="

    assistant = backend.make_assistant_message(
        ModelTurnResult(stop_reason="end_turn", text="ok"), raw_content=["raw"]
    )
    assert assistant == {"role": "assistant", "content": ["raw"]}


def test_anthropic_thinking_params_adaptive_vs_extended() -> None:
    pytest.importorskip("anthropic")
    from familiar_runtime.models import AnthropicBackend

    adaptive = AnthropicBackend(api_key="x", model="claude-sonnet-4-6", thinking_mode="auto")
    params = adaptive._build_thinking_params()
    assert params["thinking"]["type"] == "adaptive"

    extended = AnthropicBackend(
        api_key="x", model="claude-sonnet-4-6", thinking_mode="extended", thinking_budget=8000
    )
    params = extended._build_thinking_params()
    assert params["thinking"] == {"type": "enabled", "budget_tokens": 8000}
    assert "interleaved-thinking-2025-05-14" in params["betas"]


def test_openai_compat_backend_native_and_prompt_tool_results() -> None:
    pytest.importorskip("openai")
    from familiar_runtime.models import OpenAICompatibleBackend, ToolCall

    native = OpenAICompatibleBackend(
        api_key="k", model="gpt-4o-mini", base_url="https://api.openai.com/v1", tools_mode="native"
    )
    tc = ToolCall(id="tc1", name="search", input={"q": "x"})
    native_results = native.make_tool_results([tc], [("ok", None)])
    assert native_results[0]["role"] == "tool"
    assert native_results[0]["tool_call_id"] == "tc1"

    prompt = OpenAICompatibleBackend(
        api_key="k", model="local", base_url="http://localhost:11434/v1", tools_mode="prompt"
    )
    prompt_results = prompt.make_tool_results([tc], [("ok", "IMG")])
    assert prompt_results[0]["role"] == "user"
    parts = prompt_results[0]["content"]
    assert any(p.get("type") == "image_url" for p in parts)


def test_kimi_backend_emits_reasoning_is_true() -> None:
    pytest.importorskip("openai")
    from familiar_runtime.models import KimiBackend

    backend = KimiBackend(api_key="k", model="kimi-k2.5")
    assert backend.emits_reasoning is True


@pytest.mark.parametrize(
    "base_url,expected",
    [
        ("https://api.kimi.com/coding/v1", True),
        ("https://API.KIMI.COM/v1", True),
        ("https://api.moonshot.ai/v1", True),
        ("https://api.moonshot.cn/v1", True),
        ("https://api.openai.com/v1", False),
        ("http://localhost:11434/v1", False),
    ],
)
def test_openai_compat_backend_detects_kimi_reasoning(base_url: str, expected: bool) -> None:
    pytest.importorskip("openai")
    from familiar_runtime.models import OpenAICompatibleBackend

    backend = OpenAICompatibleBackend(api_key="k", model="m", base_url=base_url)
    assert backend.emits_reasoning is expected


@pytest.mark.asyncio
async def test_openai_compat_captures_streaming_usage_and_guards_empty_choices() -> None:
    """Ollama/OpenAI-compat emit a final usage-only chunk (empty choices) under
    stream_options.include_usage. The backend must fold that usage into the
    result AND not IndexError on the choiceless chunk."""
    pytest.importorskip("openai")
    from types import SimpleNamespace

    from familiar_runtime.models import OpenAICompatibleBackend

    backend = OpenAICompatibleBackend(
        api_key="", model="gemma4:latest", base_url="http://localhost:11434/v1", tools_mode="prompt"
    )

    def _delta_chunk(text):
        choice = SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=None)
        return SimpleNamespace(choices=[choice], usage=None)

    def _usage_chunk(pt, ct):
        return SimpleNamespace(
            choices=[],  # the usage-only chunk carries no choices
            usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct),
        )

    class _FakeStream:
        def __init__(self, chunks):
            self._chunks = chunks

        def __aiter__(self):
            async def gen():
                for c in self._chunks:
                    yield c

            return gen()

    async def _fake_create(**kwargs):
        assert kwargs.get("stream_options") == {"include_usage": True}
        return _FakeStream([_delta_chunk("Hi "), _delta_chunk("there"), _usage_chunk(176, 78)])

    backend.client.chat.completions.create = _fake_create  # type: ignore[assignment]

    captured: list[str] = []
    result, _ = await backend.stream_turn(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        max_tokens=50,
        on_text=captured.append,
    )
    assert result.text == "Hi there"
    assert result.input_tokens == 176
    assert result.output_tokens == 78
    assert "".join(captured) == "Hi there"


def test_strip_think_blocks() -> None:
    pytest.importorskip("openai")
    from familiar_runtime.models.openai_compat import _strip_think_blocks

    # Closed span removed, reply kept
    assert _strip_think_blocks("<think>reasoning...</think>\nこんにちは。") == "こんにちは。"
    # Multiple spans
    assert _strip_think_blocks("<think>a</think>hi<think>b</think> there") == "hi there"
    # Unclosed span: model burned the budget reasoning — drop the tail
    assert _strip_think_blocks("<think>never ends") == ""
    assert _strip_think_blocks("prefix <think>never ends") == "prefix"
    # No markers: byte-identical (no stray strip)
    assert _strip_think_blocks("  plain text  ") == "  plain text  "


@pytest.mark.asyncio
async def test_openai_compat_prompt_mode_scrubs_think_before_tool_parse() -> None:
    """A <tool_call> the model merely contemplated inside <think> must not run;
    the committed tool call after the reasoning span still parses."""
    pytest.importorskip("openai")
    from types import SimpleNamespace

    from familiar_runtime.models import OpenAICompatibleBackend

    backend = OpenAICompatibleBackend(
        api_key="", model="gemma4:latest", base_url="http://localhost:11434/v1", tools_mode="prompt"
    )

    def _delta_chunk(text):
        choice = SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=None)
        return SimpleNamespace(choices=[choice], usage=None)

    class _FakeStream:
        def __init__(self, chunks):
            self._chunks = chunks

        def __aiter__(self):
            async def gen():
                for c in self._chunks:
                    yield c

            return gen()

    contemplated = '<think>maybe <tool_call>{"name": "walk", "arguments": {}}</tool_call>?</think>'
    committed = '<tool_call>{"name": "say", "arguments": {"text": "hi"}}</tool_call>'

    async def _fake_create(**kwargs):
        return _FakeStream([_delta_chunk(contemplated), _delta_chunk(committed)])

    backend.client.chat.completions.create = _fake_create  # type: ignore[assignment]

    result, raw = await backend.stream_turn(
        system="s",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        max_tokens=50,
        on_text=None,
    )
    assert [tc.name for tc in result.tool_calls] == ["say"]
    assert "<think>" not in raw["content"]


def test_kimi_backend_make_tool_results_includes_image_block() -> None:
    pytest.importorskip("openai")
    from familiar_runtime.models import KimiBackend, ToolCall

    backend = KimiBackend(api_key="k", model="kimi-k2.5")
    tc = ToolCall(id="t", name="see", input={})
    msgs = backend.make_tool_results([tc], [("seen", "BASE64")])
    assert msgs[0]["role"] == "tool"
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"][0]["type"] == "image_url"


def test_glm_backend_make_tool_results_includes_image_block() -> None:
    pytest.importorskip("openai")
    from familiar_runtime.models import GLMBackend, ToolCall

    backend = GLMBackend(api_key="k", model="glm-4.6v")
    tc = ToolCall(id="t", name="see", input={})
    msgs = backend.make_tool_results([tc], [("seen", "BASE64")])
    assert msgs[0]["role"] == "tool"
    assert msgs[1]["content"][0]["type"] == "image_url"


def test_gemini_backend_user_and_tool_messages_use_parts() -> None:
    pytest.importorskip("google.genai")
    from familiar_runtime.models import GeminiBackend, ToolCall

    backend = GeminiBackend(api_key="k", model="gemini-2.5-flash")
    user_msg = backend.make_user_message("hello")
    assert user_msg["role"] == "user"
    assert user_msg["parts"][0]["text"] == "hello"

    tc = ToolCall(id="t", name="see", input={})
    results = backend.make_tool_results([tc], [("seen", "IMG")])
    assert results[0]["role"] == "user"
    function_part = results[0]["parts"][0]
    assert "function_response" in function_part
    inline_part = results[0]["parts"][1]
    assert inline_part["inline_data"]["mime_type"] == "image/jpeg"


def test_cli_backend_serialises_messages_and_tool_results() -> None:
    from familiar_runtime.models import CLIBackend, ToolCall

    backend = CLIBackend(["ollama", "run", "gemma3:27b"])
    msg = backend.make_user_message(
        [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]
    )
    assert "first" in msg["content"]
    assert "second" in msg["content"]

    tc = ToolCall(id="t", name="ping", input={})
    results = backend.make_tool_results([tc], [("pong", None)])
    assert "Tool result: ping" in results[0]["content"]

    serialised = backend._serialize(
        "be helpful",
        [{"role": "user", "content": "hi"}],
        [
            {
                "name": "ping",
                "description": "Reply pong.",
                "input_schema": {"properties": {}, "required": []},
            }
        ],
    )
    assert "<system>" in serialised
    assert "[USING TOOLS]" in serialised
    assert serialised.endswith("Assistant:")


@pytest.mark.asyncio
async def test_cli_backend_estimates_multilingual_token_usage() -> None:
    from familiar_runtime.models import CLIBackend

    backend = CLIBackend(["model"])
    japanese = "あ" * 60_001

    with patch.object(backend, "_run", new=AsyncMock(return_value="了解したで")):
        result, _ = await backend.stream_turn(
            "system",
            [backend.make_user_message(japanese)],
            [],
            100,
            None,
        )

    assert result.input_tokens > 60_000
    assert result.output_tokens >= len("了解したで")


@pytest.mark.asyncio
async def test_cli_backend_normalizes_recovered_tool_call() -> None:
    from familiar_runtime.models import CLIBackend

    backend = CLIBackend(["model"])
    malformed = '<tool_call>{"name": "get_status", "input": {}}</parameter>\n</invoke>'
    streamed: list[str] = []

    with patch.object(backend, "_run", new=AsyncMock(return_value=malformed)):
        result, raw = await backend.stream_turn("system", [], [], 100, streamed.append)

    assert result.stop_reason == "tool_use"
    assert [(call.name, call.input) for call in result.tool_calls] == [("get_status", {})]
    assert result.text == ""
    assert raw["content"] == '<tool_call>{"name": "get_status", "input": {}}</tool_call>'
    assert streamed == []


def test_provider_backends_serialize_user_turn_images() -> None:
    pytest.importorskip("anthropic")
    from familiar_runtime.models import AnthropicBackend, ImageAttachment, UserTurn

    backend = AnthropicBackend(api_key="k", model="test")
    message = backend.make_user_message(
        UserTurn(
            text="look",
            images=(ImageAttachment(b"png", "image/png", "sample.png"),),
        )
    )

    assert message["content"][0] == {"type": "text", "text": "look"}
    assert message["content"][1]["source"]["media_type"] == "image/png"


def test_gemini_serializes_user_turn_without_content_key_assumption() -> None:
    pytest.importorskip("google.genai")
    from familiar_runtime.models import GeminiBackend, ImageAttachment, UserTurn

    backend = GeminiBackend(api_key="k", model="test")
    message = backend.make_user_message(
        UserTurn(text="look", images=(ImageAttachment(b"png", "image/png"),))
    )

    assert message["parts"][0] == {"text": "look"}
    assert message["parts"][1]["inline_data"]["mime_type"] == "image/png"


def test_generic_cli_backend_rejects_images_explicitly() -> None:
    from familiar_runtime.models import CLIBackend, ImageAttachment, UserTurn

    backend = CLIBackend(["ollama", "run", "model"])

    with pytest.raises(RuntimeError, match="does not support image input"):
        backend.make_user_message(
            UserTurn(text="look", images=(ImageAttachment(b"image", "image/jpeg"),))
        )


def test_create_backend_dispatches_by_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_backend should return the right provider for each PLATFORM value."""
    pytest.importorskip("anthropic")
    pytest.importorskip("openai")
    from familiar_agent.backend import create_backend
    from familiar_runtime.models import (
        AnthropicBackend,
        CLIBackend,
        GLMBackend,
        KimiBackend,
        OpenAICompatibleBackend,
    )

    class FakeConfig:
        def __init__(self, platform: str, model: str | None = None) -> None:
            self.platform = platform
            self.model = model
            self.api_key = "test"
            self.base_url = "https://api.openai.com/v1"
            self.tools_mode = "native"
            self.thinking_mode = "disabled"
            self.thinking_budget = 0
            self.thinking_effort = "high"

    assert isinstance(create_backend(FakeConfig("anthropic")), AnthropicBackend)
    # Need BASE_URL unset for openai factory.
    monkeypatch.delenv("BASE_URL", raising=False)
    monkeypatch.delenv("TOOLS_MODE", raising=False)
    assert isinstance(create_backend(FakeConfig("openai")), OpenAICompatibleBackend)
    assert isinstance(create_backend(FakeConfig("kimi")), KimiBackend)
    assert isinstance(create_backend(FakeConfig("glm")), GLMBackend)
    cli_backend = create_backend(FakeConfig("cli", model="echo {}"))
    assert isinstance(cli_backend, CLIBackend)


def test_create_backend_uses_safe_claude_default_for_cli() -> None:
    from familiar_agent.backend import create_backend
    from familiar_runtime.models import ClaudeCodeCLIBackend

    config = MagicMock(platform="cli", model="")

    backend = create_backend(config)

    assert isinstance(backend, ClaudeCodeCLIBackend)
    assert backend._cmd == [
        "claude",
        "-p",
        "--safe-mode",
        "--tools",
        "",
        "--no-session-persistence",
        "--system-prompt",
        "",
    ]
    assert backend.manages_context is False


def test_create_backend_preserves_explicit_managed_claude_opt_in() -> None:
    from familiar_agent.backend import create_backend
    from familiar_runtime.models import ClaudeCodeCLIBackend

    config = MagicMock(
        platform="cli",
        model='claude -p --tools "" --autocompact auto',
    )

    backend = create_backend(config)

    assert isinstance(backend, ClaudeCodeCLIBackend)
    assert backend._cmd == ["claude", "-p", "--tools", "", "--autocompact", "auto"]
    assert backend.manages_context is True


def test_claude_cli_command_enables_only_read_for_image_turns() -> None:
    from familiar_runtime.models import ClaudeCodeCLIBackend

    command = ["claude", "-p", "--safe-mode", "--tools", "", "{}"]

    assert ClaudeCodeCLIBackend._command_with_read(command) == [
        "claude",
        "-p",
        "--safe-mode",
        "--tools",
        "Read",
        "--allowedTools",
        "Read",
        "{}",
    ]
    assert command[4] == ""


@pytest.mark.asyncio
async def test_claude_cli_stages_images_for_read_and_cleans_them_up() -> None:
    from familiar_runtime.models import ClaudeCodeCLIBackend, ImageAttachment, UserTurn

    backend = ClaudeCodeCLIBackend(
        ["claude", "-p", "--tools", "", "--no-session-persistence", "{}"]
    )
    messages = [
        backend.make_user_message(
            UserTurn(
                text="what is this?",
                images=(ImageAttachment(b"image-bytes", "image/png"),),
            )
        )
    ]
    captured_path: Path | None = None

    async def fake_run(prompt: str, *, command=None, cwd=None) -> str:
        nonlocal captured_path
        assert command == [
            "claude",
            "-p",
            "--tools",
            "Read",
            "--no-session-persistence",
            "--allowedTools",
            "Read",
        ]
        assert cwd is not None
        captured_path = Path(cwd) / "image-1.png"
        assert captured_path.read_bytes() == b"image-bytes"
        assert str(captured_path) in prompt
        assert "Use the Read tool" in prompt
        return "seen"

    with patch.object(backend, "_run", side_effect=fake_run):
        result, _ = await backend.stream_turn("system", messages, [], 100, None)

    assert result.text == "seen"
    assert captured_path is not None
    assert not captured_path.exists()


@pytest.mark.asyncio
async def test_claude_cli_stages_tool_result_images_for_read() -> None:
    from familiar_runtime.models import ClaudeCodeCLIBackend, ToolCall

    backend = ClaudeCodeCLIBackend(
        ["claude", "-p", "--tools", "", "--no-session-persistence", "{}"]
    )
    messages = [
        backend.make_tool_results(
            [ToolCall(id="t", name="see", input={})],
            [("You see the current view.", "aW1hZ2UtYnl0ZXM=")],
        )
    ]

    async def fake_run(prompt: str, *, command=None, cwd=None) -> str:
        assert cwd is not None
        image_path = Path(cwd) / "image-1.jpg"
        assert image_path.read_bytes() == b"image-bytes"
        assert "[Tool result: see]" in prompt
        assert str(image_path) in prompt
        assert "Use the Read tool" in prompt
        return "I can see it."

    with patch.object(backend, "_run", side_effect=fake_run):
        result, _ = await backend.stream_turn("system", messages, [], 100, None)

    assert result.text == "I can see it."


@pytest.mark.asyncio
async def test_claude_cli_text_turn_keeps_original_command_path() -> None:
    from familiar_runtime.models import ClaudeCodeCLIBackend

    backend = ClaudeCodeCLIBackend(
        ["claude", "-p", "--tools", "", "--no-session-persistence", "{}"]
    )

    with patch.object(backend, "_run", new=AsyncMock(return_value="reply")) as run:
        await backend.stream_turn("system", [backend.make_user_message("hello")], [], 100, None)

    run.assert_awaited_once()
    assert run.call_args.kwargs == {}


@pytest.mark.asyncio
async def test_claude_cli_managed_session_resumes_delta_and_rebuilds_on_context_change() -> None:
    from familiar_runtime.models import ClaudeCodeCLIBackend

    backend = ClaudeCodeCLIBackend(["claude", "-p"])
    calls: list[tuple[str, list[str]]] = []
    usage_keys = (
        "input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "output_tokens",
    )
    usages = iter(
        dict(zip(usage_keys, values)) for values in ((10, 20, 30, 4), (5, 0, 60, 3), (0, 0, 0, 1))
    )

    async def fake_run(prompt: str, *, command=None, cwd=None) -> str:
        assert cwd is None
        calls.append((prompt, command))
        return json.dumps(
            {"result": f"reply {len(calls)}", "session_id": command[-1], "usage": next(usages)}
        )

    messages = []
    tools = [{"name": "ping", "description": "ping", "input_schema": {"type": "object"}}]

    async def turn(state: str, user: str, turn_tools: list[dict]):
        messages.append(backend.make_user_message(user))
        result, raw = await backend.stream_turn(
            ("stable rules", f"turn state {state}"), messages, turn_tools, 100, None
        )
        messages.append(raw)
        return result

    with patch.object(backend, "_run", side_effect=fake_run):
        first = await turn("one", "hello", tools)
        second = await turn("two", "next question", tools)
        await turn("three", "third question", [{**tools[0], "name": "new_tool"}])

    first_prompt, first_command = calls[0]
    second_prompt, second_command = calls[1]
    third_prompt, third_command = calls[2]
    assert all(text in first_prompt for text in ("stable rules", "hello"))
    assert "--session-id" in first_command
    assert first_command[-4:-2] == ["--output-format", "json"]
    assert all(text not in second_prompt for text in ("stable rules", "hello", "reply 1"))
    assert all(text in second_prompt for text in ("turn state two", "next question"))
    assert second_command[-2:] == ["--resume", first_command[-1]]
    assert "--session-id" in third_command
    assert third_command[-1] != first_command[-1]
    assert all(text in third_prompt for text in ("stable rules", "hello"))
    usage = (first.input_tokens, first.output_tokens, second.input_tokens, second.output_tokens)
    assert usage == (60, 4, 65, 3)


@pytest.mark.asyncio
async def test_claude_cli_complete_is_not_added_to_managed_session() -> None:
    from familiar_runtime.models import ClaudeCodeCLIBackend

    backend = ClaudeCodeCLIBackend(["claude", "-p", "--tools", ""])
    with patch.object(backend, "_run", new=AsyncMock(return_value="utility reply")) as run:
        reply = await backend.complete("utility prompt", 20)

    assert reply == "utility reply"
    assert run.call_args.kwargs["command"][-1] == "--no-session-persistence"
    assert backend._session.session_id is None


@pytest.mark.asyncio
async def test_cli_backend_injects_prompt_and_strips_claudecode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from familiar_runtime.models import CLIBackend

    proc = MagicMock(returncode=0)
    proc.communicate = AsyncMock(return_value=(b"reply\n", b""))
    monkeypatch.setenv("CLAUDECODE", "nested")

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as spawn:
        reply = await CLIBackend(["claude", "-p", "{}"])._run("hello")

    assert reply == "reply"
    assert spawn.call_args.args == ("claude", "-p", "hello")
    assert "CLAUDECODE" not in spawn.call_args.kwargs["env"]
    proc.communicate.assert_awaited_once_with(None)


@pytest.mark.asyncio
async def test_claude_cli_backend_sends_legacy_placeholder_prompt_via_stdin() -> None:
    from familiar_runtime.models import ClaudeCodeCLIBackend

    proc = MagicMock(returncode=0)
    proc.communicate = AsyncMock(return_value=(b"reply\n", b""))

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)) as spawn:
        reply = await ClaudeCodeCLIBackend(["claude", "-p", "{}"])._run("x" * 300_000)

    assert reply == "reply"
    assert spawn.call_args.args == ("claude", "-p")
    assert spawn.call_args.kwargs["stdin"] is asyncio.subprocess.PIPE
    proc.communicate.assert_awaited_once_with(b"x" * 300_000)


@pytest.mark.asyncio
async def test_cli_backend_raises_on_nonzero_exit() -> None:
    from familiar_runtime.models import CLIBackend

    proc = MagicMock(returncode=2)
    proc.communicate = AsyncMock(return_value=(b"", b"authentication failed"))

    with (
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
        pytest.raises(RuntimeError, match="code 2: authentication failed"),
    ):
        await CLIBackend(["claude", "-p", "{}"])._run("hello")


@pytest.mark.asyncio
async def test_cli_backend_terminates_process_on_timeout() -> None:
    from familiar_runtime.models import CLIBackend

    async def communicate(_stdin):
        await asyncio.Future()

    proc = MagicMock(returncode=None)
    proc.communicate = communicate
    proc.wait = AsyncMock(return_value=0)

    with (
        patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)),
        pytest.raises(RuntimeError, match="timed out after 0.01 seconds"),
    ):
        await CLIBackend(["claude", "-p", "{}"], timeout_seconds=0.01)._run("hello")

    proc.terminate.assert_called_once_with()
    proc.wait.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_cli_backend_terminates_process_when_cancelled() -> None:
    from familiar_runtime.models import CLIBackend

    started = asyncio.Event()
    never_finishes = asyncio.Future()

    async def communicate(_stdin):
        started.set()
        return await never_finishes

    proc = MagicMock(returncode=None)
    proc.communicate = communicate
    proc.wait = AsyncMock(return_value=0)

    with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)):
        task = asyncio.create_task(CLIBackend(["claude", "-p", "{}"])._run("hello"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    proc.terminate.assert_called_once_with()
    proc.wait.assert_awaited_once_with()
