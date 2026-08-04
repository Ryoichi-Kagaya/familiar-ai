"""Outbound Telegram messaging tool.

The Telegram bot front-end owns inbound updates and ordinary replies.  This
tool is deliberately outbound-only so other familiar surfaces (GUI, TUI,
REPL, and autonomous turns) can contact a user who explicitly linked their
Telegram account to a familiar user profile.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from ..config import TelegramConfig
from ..user_profile import UserProfile, UserRegistry

_MAX_MESSAGE_LENGTH = 4096
_MAX_TOOL_TEXT_LENGTH = _MAX_MESSAGE_LENGTH * 4

TelegramSender = Callable[[int, str], Awaitable[None]]
CurrentUserId = Callable[[], str]


class TelegramTransport:
    """Lazy, reusable outbound Bot client shared by all tool calls."""

    def __init__(self, token: str) -> None:
        self.token = token
        self._bot: Any = None
        self._lock = asyncio.Lock()

    async def _get_bot(self):
        if self._bot is not None:
            return self._bot
        try:
            from telegram import Bot
        except ImportError as exc:
            raise RuntimeError(
                "python-telegram-bot is not installed; install familiar-ai[telegram]"
            ) from exc
        bot = Bot(token=self.token)
        await bot.initialize()
        self._bot = bot
        return bot

    async def send(self, chat_id: int, text: str) -> None:
        async with self._lock:
            bot = await self._get_bot()
            await bot.send_message(chat_id=chat_id, text=text)

    async def close(self) -> None:
        async with self._lock:
            bot, self._bot = self._bot, None
            if bot is not None:
                await bot.shutdown()


class TelegramTool:
    """Send text to Telegram accounts linked through :class:`UserRegistry`."""

    def __init__(
        self,
        config: TelegramConfig,
        registry: UserRegistry,
        *,
        current_user_id: CurrentUserId,
        allowed_ids: set[int] | None = None,
        sender: TelegramSender | None = None,
        transport: TelegramTransport | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        self._current_user_id = current_user_id
        self._allowed_ids = config.allowed_ids if allowed_ids is None else set(allowed_ids)
        self._transport = transport or TelegramTransport(config.token)
        self._sender = sender or self._transport.send

    def _linked_profiles(self) -> list[UserProfile]:
        return [
            profile for profile in self._registry.list_users() if profile.telegram_id is not None
        ]

    def get_tool_definitions(self) -> list[dict]:
        linked_ids = [profile.id for profile in self._linked_profiles()]
        recipient_help = (
            f" Linked profile IDs: {', '.join(linked_ids)}."
            if linked_ids
            else " No Telegram-linked profile is currently available."
        )
        user_id_schema: dict[str, object] = {
            "type": "string",
            "description": (
                "Familiar user profile ID to contact. Omit it to use the current user."
            ),
        }
        if linked_ids:
            user_id_schema["enum"] = linked_ids

        return [
            {
                "name": "send_telegram_message",
                "description": (
                    "Send a text message through Telegram to a user who linked their account. "
                    "Use this for an explicitly requested message or a useful proactive "
                    "follow-up/reminder. Do not use it merely to repeat the visible reply in "
                    "the current chat." + recipient_help
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The complete user-visible message to send.",
                            "maxLength": _MAX_TOOL_TEXT_LENGTH,
                        },
                        "user_id": user_id_schema,
                    },
                    "required": ["text"],
                },
            }
        ]

    def _resolve_profile(self, user_id: str) -> UserProfile | None:
        return next(
            (profile for profile in self._registry.list_users() if profile.id == user_id), None
        )

    async def call(self, name: str, tool_input: dict) -> tuple[str, None]:
        if name != "send_telegram_message":
            return f"Unknown Telegram tool: {name}", None

        text = tool_input.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return "Telegram message was not sent: text must not be empty.", None
        text = text.strip()
        if len(text) > _MAX_TOOL_TEXT_LENGTH:
            return (
                f"Telegram message was not sent: text exceeds {_MAX_TOOL_TEXT_LENGTH} characters.",
                None,
            )

        raw_user_id = tool_input.get("user_id")
        if raw_user_id is None:
            user_id = self._current_user_id()
        elif isinstance(raw_user_id, str) and raw_user_id.strip():
            user_id = raw_user_id.strip()
        else:
            return "Telegram message was not sent: user_id must be a profile ID.", None

        profile = self._resolve_profile(user_id)
        if profile is None:
            return f"Telegram message was not sent: user profile '{user_id}' does not exist.", None
        if profile.telegram_id is None:
            return (
                f"Telegram message was not sent: profile '{user_id}' has no linked Telegram account.",
                None,
            )
        if self._allowed_ids and profile.telegram_id not in self._allowed_ids:
            return (
                f"Telegram message was not sent: profile '{user_id}' is not in "
                "TELEGRAM_ALLOWED_IDS.",
                None,
            )

        sent_chunks = 0
        chunk_count = (len(text) + _MAX_MESSAGE_LENGTH - 1) // _MAX_MESSAGE_LENGTH
        try:
            for start in range(0, len(text), _MAX_MESSAGE_LENGTH):
                await self._sender(profile.telegram_id, text[start : start + _MAX_MESSAGE_LENGTH])
                sent_chunks += 1
        except Exception as exc:  # noqa: BLE001 - tool errors must return to the agent
            error_name = type(exc).__name__
            if sent_chunks:
                return (
                    f"Telegram message was partially sent to {profile.name} "
                    f"({sent_chunks}/{chunk_count} chunks; {error_name}). Do not resend the "
                    "whole message because that would duplicate delivered text.",
                    None,
                )
            return f"Telegram message failed for {profile.name} ({error_name}).", None

        return f"Telegram message sent to {profile.name} ({profile.id}).", None
