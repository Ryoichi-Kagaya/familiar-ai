"""Single-flight coordination for turns arriving from multiple channels."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from familiar_runtime.models import UserTurn


@dataclass(slots=True)
class TurnRequest:
    """Everything needed to execute one embodied-agent turn."""

    user_input: str | UserTurn
    source: str = "local"
    user_id: str | None = None
    on_action: Callable[[str, dict], None] | None = None
    on_text: Callable[[str], None] | None = None
    on_image: Callable[[str], None] | None = None
    on_phase: Callable[[str], None] | None = None
    on_tool_result: Callable[[str, dict, str], None] | None = None
    desires: Any = None
    inner_voice: str = ""
    interrupt_queue: Any = None
    excluded_tools: frozenset[str] | None = None
    user_images: list[str] | None = None


TurnRunner = Callable[[TurnRequest], Awaitable[str]]
UserSwitcher = Callable[[str], Awaitable[str]]


class TurnCoordinator:
    """Serialize channel turns while allowing same-task nested turn scopes."""

    def __init__(self, *, runner: TurnRunner, switch_user: UserSwitcher) -> None:
        self._runner = runner
        self._switch_user = switch_user
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[Any] | None = None
        self._depth = 0
        self._sources: list[str] = []

    @property
    def is_busy(self) -> bool:
        return self._owner is not None

    @property
    def active_source(self) -> str | None:
        return self._sources[-1] if self._sources else None

    @asynccontextmanager
    async def scope(self, source: str) -> AsyncIterator[None]:
        """Enter the shared turn scope; nested use by the owner is re-entrant."""
        task = asyncio.current_task()
        if task is not None and self._owner is task:
            self._depth += 1
            self._sources.append(source)
            try:
                yield
            finally:
                self._sources.pop()
                self._depth -= 1
            return

        await self._lock.acquire()
        self._owner = task
        self._depth = 1
        self._sources.append(source)
        try:
            yield
        finally:
            self._sources.pop()
            self._depth = 0
            self._owner = None
            self._lock.release()

    async def run(self, request: TurnRequest) -> str:
        async with self.scope(request.source):
            if request.user_id:
                await self._switch_user(request.user_id)
            return await self._runner(request)
