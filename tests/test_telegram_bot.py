"""Presentation-boundary and shared-agent tests for Telegram replies."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from familiar_agent.config import AgentConfig, TelegramConfig
from familiar_agent.telegram_bot import (
    TelegramChannel,
    _run_telegram_agent_turn,
    _sanitize_telegram_text,
    _telegram_user_text,
)
from familiar_runtime.models import UserTurn


def test_telegram_context_distinguishes_chat_reply_and_stackchan_speech() -> None:
    text = _telegram_user_text("かがや", "どう？")

    assert "Telegram text chat" in text
    assert "`say` is intentionally unavailable" in text
    assert "`speak` is a separate StackChan" in text
    assert text.endswith("[かがや]: どう？")


def test_telegram_context_includes_recent_channel_history() -> None:
    text = _telegram_user_text(
        "かがや",
        "返事するね",
        "[Recent Telegram conversation — oldest first]\nAgent (Telegram): さっき送ったよ",
    )

    assert "Agent (Telegram): さっき送ったよ" in text
    assert text.endswith("[かがや]: 返事するね")


def test_telegram_sanitizer_preserves_voice_failure_narration_for_observability() -> None:
    text = (
        "（声のツールが今回使えへんみたいやー今回は声に出せんかった。）\n\n"
        "StackChanの状態はconnected=falseやったで。"
    )

    assert _sanitize_telegram_text(text) == text


def test_telegram_sanitizer_suppresses_turn_control_and_tool_protocol() -> None:
    assert _sanitize_telegram_text("Turn complete.)") == ""
    assert (
        _sanitize_telegram_text(
            '<tool_call>{"name": "get_status", "input": {}}</parameter>\n</invoke>'
        )
        == ""
    )


def test_telegram_sanitizer_removes_audio_direction_tags() -> None:
    assert _sanitize_telegram_text("[warmly] おはよう。[laughs]") == "おはよう。"


def test_telegram_sanitizer_preserves_ordinary_parenthetical_text() -> None:
    text = "（これは補足やで）本文を続ける。"
    assert _sanitize_telegram_text(text) == text


@pytest.mark.asyncio
async def test_telegram_turn_uses_surface_owned_agent_and_desires() -> None:
    agent = AsyncMock()
    agent.run.return_value = "  同じわたしからの返事  "
    desires = object()
    user_turn = UserTurn(text="Telegramからこんにちは")
    on_image = object()

    result = await _run_telegram_agent_turn(
        agent,
        desires,
        user_turn,
        profile_id="default",
        on_image=on_image,
    )

    assert result == "同じわたしからの返事"
    agent.run.assert_awaited_once_with(
        user_turn,
        on_image=on_image,
        desires=desires,
        user_id="default",
        turn_source="telegram",
        excluded_tools=frozenset({"say", "send_telegram_message"}),
    )


def test_telegram_channel_requires_explicit_inbound_activation() -> None:
    config = AgentConfig(telegram=TelegramConfig(token="token", inbound_enabled=False))

    assert TelegramChannel.from_config(config, object(), object()) is None


@pytest.mark.asyncio
async def test_telegram_channel_owns_polling_task_lifecycle(monkeypatch) -> None:
    started = asyncio.Event()

    async def fake_run(token, agent, desires, *, allowed_ids) -> None:
        assert token == "token"
        assert agent is expected_agent
        assert desires is expected_desires
        assert allowed_ids == {123, 456}
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr("familiar_agent.telegram_bot.run_telegram_bot", fake_run)
    expected_agent = object()
    expected_desires = object()
    config = AgentConfig(
        telegram=TelegramConfig(
            token="token",
            allowed_ids_raw="123,456",
            inbound_enabled=True,
        )
    )
    channel = TelegramChannel.from_config(config, expected_agent, expected_desires)
    assert channel is not None

    channel.start()
    await started.wait()
    assert channel.is_running

    await channel.stop()
    assert not channel.is_running
