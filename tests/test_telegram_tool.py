"""Tests for outbound Telegram messaging as an agent tool."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from familiar_agent.config import TelegramConfig
from familiar_agent.tools.telegram import TelegramTool, TelegramTransport
from familiar_agent.user_profile import UserRegistry
from familiar_capabilities import TelegramCapability


def _linked_registry(tmp_path: Path) -> UserRegistry:
    registry = UserRegistry(tmp_path / "users")
    registry.create("default", "コウタ")
    registry.link_telegram("default", 12345)
    registry.create("friend", "友だち")
    registry.link_telegram("friend", 67890)
    return registry


def test_telegram_config_allowed_ids_ignores_invalid_entries() -> None:
    config = TelegramConfig(allowed_ids_raw="123, nope, 456, -7, ")

    assert config.allowed_ids == {123, 456}


def test_telegram_capability_exposes_linked_profiles(tmp_path: Path) -> None:
    registry = _linked_registry(tmp_path)
    tool = TelegramTool(
        TelegramConfig(token="token"),
        registry,
        current_user_id=lambda: "default",
        allowed_ids=set(),
    )

    capability = TelegramCapability(tool)
    [spec] = capability.specs()
    assert spec.name == "send_telegram_message"
    assert spec.category == "messaging"
    assert spec.input_schema["properties"]["user_id"]["enum"] == ["default", "friend"]


@pytest.mark.asyncio
async def test_send_telegram_message_defaults_to_current_user(tmp_path: Path) -> None:
    registry = _linked_registry(tmp_path)
    sent: list[tuple[int, str]] = []

    async def sender(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    tool = TelegramTool(
        TelegramConfig(token="token"),
        registry,
        current_user_id=lambda: "default",
        allowed_ids=set(),
        sender=sender,
    )

    result, image = await tool.call("send_telegram_message", {"text": "  ただいま  "})

    assert result == "Telegram message sent to コウタ (default)."
    assert image is None
    assert sent == [(12345, "ただいま")]


@pytest.mark.asyncio
async def test_send_telegram_message_can_target_linked_profile(tmp_path: Path) -> None:
    registry = _linked_registry(tmp_path)
    sent: list[tuple[int, str]] = []

    async def sender(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    tool = TelegramTool(
        TelegramConfig(token="token"),
        registry,
        current_user_id=lambda: "default",
        allowed_ids={67890},
        sender=sender,
    )

    result, _ = await tool.call("send_telegram_message", {"user_id": "friend", "text": "着いたよ"})

    assert result == "Telegram message sent to 友だち (friend)."
    assert sent == [(67890, "着いたよ")]


@pytest.mark.asyncio
async def test_send_telegram_message_rejects_unlinked_or_disallowed_profile(tmp_path: Path) -> None:
    registry = _linked_registry(tmp_path)
    registry.create("unlinked", "未登録")
    sent: list[tuple[int, str]] = []

    async def sender(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    tool = TelegramTool(
        TelegramConfig(token="token"),
        registry,
        current_user_id=lambda: "default",
        allowed_ids={67890},
        sender=sender,
    )

    unlinked, _ = await tool.call("send_telegram_message", {"user_id": "unlinked", "text": "test"})
    disallowed, _ = await tool.call("send_telegram_message", {"text": "test"})
    unknown, _ = await tool.call("send_telegram_message", {"user_id": "new-user", "text": "test"})

    assert "has no linked Telegram account" in unlinked
    assert "not in TELEGRAM_ALLOWED_IDS" in disallowed
    assert "does not exist" in unknown
    assert "new-user" not in {profile.id for profile in registry.list_users()}
    assert sent == []


@pytest.mark.asyncio
async def test_send_telegram_message_splits_at_telegram_limit(tmp_path: Path) -> None:
    registry = _linked_registry(tmp_path)
    sent: list[tuple[int, str]] = []

    async def sender(chat_id: int, text: str) -> None:
        sent.append((chat_id, text))

    tool = TelegramTool(
        TelegramConfig(token="token"),
        registry,
        current_user_id=lambda: "default",
        allowed_ids=set(),
        sender=sender,
    )
    text = "あ" * 4100

    result, _ = await tool.call("send_telegram_message", {"text": text})

    assert result.startswith("Telegram message sent")
    assert sent == [(12345, "あ" * 4096), (12345, "あ" * 4)]


@pytest.mark.asyncio
async def test_send_telegram_message_reports_partial_delivery_without_error_details(
    tmp_path: Path,
) -> None:
    registry = _linked_registry(tmp_path)
    calls = 0

    async def sender(chat_id: int, text: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("secret transport detail")

    tool = TelegramTool(
        TelegramConfig(token="token"),
        registry,
        current_user_id=lambda: "default",
        allowed_ids=set(),
        sender=sender,
    )

    result, _ = await tool.call("send_telegram_message", {"text": "あ" * 4100})

    assert "partially sent" in result
    assert "1/2 chunks" in result
    assert "Do not resend" in result
    assert "secret transport detail" not in result


@pytest.mark.asyncio
async def test_telegram_transport_reuses_and_closes_initialized_bot(monkeypatch) -> None:
    instances = []

    class FakeBot:
        def __init__(self, *, token: str) -> None:
            self.token = token
            self.initialize_calls = 0
            self.shutdown_calls = 0
            self.messages: list[tuple[int, str]] = []
            instances.append(self)

        async def initialize(self) -> None:
            self.initialize_calls += 1

        async def send_message(self, *, chat_id: int, text: str) -> None:
            self.messages.append((chat_id, text))

        async def shutdown(self) -> None:
            self.shutdown_calls += 1

    monkeypatch.setitem(sys.modules, "telegram", SimpleNamespace(Bot=FakeBot))
    transport = TelegramTransport("token")

    await transport.send(123, "one")
    await transport.send(123, "two")
    await transport.close()

    assert len(instances) == 1
    assert instances[0].initialize_calls == 1
    assert instances[0].messages == [(123, "one"), (123, "two")]
    assert instances[0].shutdown_calls == 1
