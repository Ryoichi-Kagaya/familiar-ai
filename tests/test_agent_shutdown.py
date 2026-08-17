"""Resource-shutdown behavior for :class:`EmbodiedAgent`."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.agent import EmbodiedAgent


@pytest.mark.asyncio
async def test_close_releases_each_resource_once_and_continues_after_failures() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)

    camera = MagicMock()
    agent._cameras = {"main": camera, "alias": camera}
    agent._camera = camera

    agent._drain_background_tasks = AsyncMock()
    agent._write_today_narrative = AsyncMock()
    agent.backend = MagicMock()
    agent._utility_backend = agent.backend

    memory_worker = MagicMock()
    memory_worker.stop = AsyncMock(side_effect=RuntimeError("worker failed"))
    agent._memory_worker = memory_worker
    inner_loop = MagicMock()
    inner_loop.stop = AsyncMock()
    agent._inner_loop = inner_loop

    self_state = MagicMock()
    self_state.flush.side_effect = RuntimeError("state failed")
    agent._self_state = self_state
    agent._attention_schema = MagicMock()

    mcp = MagicMock()
    mcp.stop = AsyncMock()
    agent._mcp = mcp
    telegram_transport = MagicMock()
    telegram_transport.close = AsyncMock()
    agent._telegram_transport = telegram_transport

    agent._memory = MagicMock()
    delegation_runner = MagicMock()
    delegation_runner.shutdown = AsyncMock()
    agent._delegation_runner = delegation_runner

    person_model = MagicMock()
    person_model.close.side_effect = RuntimeError("person model failed")
    agent._person_model = person_model
    agent._commitment_store = MagicMock()

    await agent.close()

    camera.close.assert_called_once_with()
    agent._drain_background_tasks.assert_awaited_once_with()
    agent._write_today_narrative.assert_awaited_once_with()
    memory_worker.stop.assert_awaited_once_with()
    inner_loop.stop.assert_awaited_once_with()
    self_state.flush.assert_called_once_with()
    agent._attention_schema.flush.assert_called_once_with()
    mcp.stop.assert_awaited_once_with()
    telegram_transport.close.assert_awaited_once_with()
    agent._memory.close.assert_called_once_with()
    delegation_runner.shutdown.assert_awaited_once_with()
    person_model.close.assert_called_once_with()
    agent._commitment_store.close.assert_called_once_with()


@pytest.mark.asyncio
async def test_close_refreshes_day_summary_with_dedicated_utility_backend() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent._cameras = {}
    agent._camera = None
    agent._drain_background_tasks = AsyncMock()
    agent._write_today_narrative = AsyncMock()
    agent.backend = MagicMock()
    agent._utility_backend = MagicMock()
    agent._memory = MagicMock()
    agent._generate_day_summary = AsyncMock()
    agent._memory_worker = None
    agent._inner_loop = None
    agent._self_state = None
    agent._attention_schema = None
    agent._mcp = None
    agent._telegram_transport = None
    agent._delegation_runner = None
    agent._person_model = None
    agent._commitment_store = None

    await agent.close()

    [summary_date] = agent._generate_day_summary.await_args.args
    agent._memory.delete_day_summaries_for_date.assert_called_once_with(summary_date)
