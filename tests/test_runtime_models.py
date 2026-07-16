"""Smoke tests for familiar_runtime.models package.

Exercises provider serialization paths (make_user_message, make_tool_results)
and shared helpers without making any API calls.  Each provider import is
deferred inside the test so a missing optional SDK fails that test only.
"""

from __future__ import annotations

import asyncio
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
                "input_schema": {"properties": {}, "required": []},
            }
        ],
    )
    assert "[USING TOOLS]" in augmented
    assert "ping" in augmented

    calls = _parse_tool_calls_from_text(
        'sure: <tool_call>{"name": "ping", "input": {"x": 1}}</tool_call>'
    )
    assert len(calls) == 1
    assert calls[0].name == "ping"
    assert calls[0].input == {"x": 1}


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

    config = MagicMock(platform="cli", model="")

    backend = create_backend(config)

    assert backend._cmd == [
        "claude",
        "-p",
        "--safe-mode",
        "--tools",
        "",
        "--no-session-persistence",
        "--system-prompt",
        "",
        "{}",
    ]


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
