"""Cross-surface single-flight behavior for EmbodiedAgent turns."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from familiar_agent.agent import EmbodiedAgent
from familiar_agent.turn_coordinator import TurnCoordinator, TurnRequest


@pytest.mark.asyncio
async def test_run_serializes_turns_and_switches_user_inside_lock() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent.switch_user = AsyncMock(return_value="user")  # type: ignore[method-assign]

    first_started = asyncio.Event()
    release_first = asyncio.Event()
    running = 0
    max_running = 0

    async def run_request(request) -> str:
        nonlocal running, max_running
        running += 1
        max_running = max(max_running, running)
        if request.user_input == "first":
            first_started.set()
            await release_first.wait()
        running -= 1
        return str(request.user_input)

    agent._run_request = run_request  # type: ignore[method-assign]

    first = asyncio.create_task(agent.run("first", user_id="default"))
    await first_started.wait()
    second = asyncio.create_task(agent.run("second", user_id="mom"))
    await asyncio.sleep(0)

    assert not second.done()
    assert agent.switch_user.await_args_list[0].args == ("default",)

    release_first.set()
    assert await first == "first"
    assert await second == "second"
    assert max_running == 1
    assert [call.args for call in agent.switch_user.await_args_list] == [("default",), ("mom",)]
    assert agent.active_turn_source is None


@pytest.mark.asyncio
async def test_turn_coordinator_scope_is_reentrant_for_autonomous_social_turn() -> None:
    seen_sources: list[str | None] = []

    async def switch_user(_user_id: str) -> str:
        return "user"

    async def run_request(_request: TurnRequest) -> str:
        seen_sources.append(coordinator.active_source)
        async with coordinator.scope("nested-social-turn"):
            seen_sources.append(coordinator.active_source)
        seen_sources.append(coordinator.active_source)
        return "done"

    coordinator = TurnCoordinator(runner=run_request, switch_user=switch_user)

    result = await coordinator.run(TurnRequest("hello", source="autonomous-action"))

    assert result == "done"
    assert seen_sources == ["autonomous-action", "nested-social-turn", "autonomous-action"]
    assert coordinator.is_busy is False
