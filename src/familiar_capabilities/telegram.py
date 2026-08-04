"""Outbound Telegram capability adapter for the generic runtime."""

from __future__ import annotations

from familiar_agent.tools.telegram import TelegramTool
from familiar_runtime.tools.legacy import LegacyToolProvider


class TelegramCapability(LegacyToolProvider):
    """Expose outbound Telegram messaging through the ToolProvider protocol."""

    def __init__(self, tool: TelegramTool) -> None:
        super().__init__(
            tool,
            names={"send_telegram_message"},
            category="messaging",
            tags={"neighbor", "task", "messaging"},
        )
