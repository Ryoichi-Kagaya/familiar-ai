"""Wiring tests for FamiliarApp._reminder_tick (TUI proactive reminders)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.tui import FamiliarApp
from familiar_agent.drive_executor import DriveActionResult
from familiar_runtime.commitments import SQLiteCommitmentStore


def _stub(store, *, proactive: bool = True, agent_running: bool = False):
    app = SimpleNamespace(
        agent=SimpleNamespace(
            config=SimpleNamespace(proactive_reminders=proactive, auto_desire=False),
            _commitment_store=store,
            _heartbeat=None,
        ),
        _agent_running=agent_running,
        _input_queue=asyncio.Queue(),
        _last_interaction=time.time() - 3600,
        _run_agent=AsyncMock(),
    )
    return app


@pytest.mark.asyncio
async def test_tui_reminder_tick_fires_and_marks(tmp_path):
    store = SQLiteCommitmentStore(tmp_path / "c.db")
    store.create(summary="stretch break", due_at=time.time() - 60)
    app = _stub(store)

    await FamiliarApp._reminder_tick(app)

    app._run_agent.assert_awaited_once()
    assert "stretch break" in app._run_agent.await_args.kwargs["inner_voice"]
    assert store.list_open()[0].reminder_count == 1
    store.close()


@pytest.mark.asyncio
async def test_tui_reminder_tick_respects_toggle(tmp_path):
    store = SQLiteCommitmentStore(tmp_path / "c.db")
    store.create(summary="stretch break", due_at=time.time() - 60)
    app = _stub(store, proactive=False)

    await FamiliarApp._reminder_tick(app)

    app._run_agent.assert_not_awaited()
    store.close()


@pytest.mark.asyncio
async def test_tui_reminder_tick_skips_while_running(tmp_path):
    store = SQLiteCommitmentStore(tmp_path / "c.db")
    store.create(summary="stretch break", due_at=time.time() - 60)
    app = _stub(store, agent_running=True)

    await FamiliarApp._reminder_tick(app)

    app._run_agent.assert_not_awaited()
    assert store.list_open()[0].reminder_count == 0
    store.close()


@pytest.mark.asyncio
async def test_tui_mid_turn_snooze_survives_mark(tmp_path):
    """mark_reminded fires BEFORE the turn, so an in-turn snooze reset is final."""
    store = SQLiteCommitmentStore(tmp_path / "c.db")
    c = store.create(summary="stretch break", due_at=time.time() - 60)
    app = _stub(store)

    async def _snoozes_during_turn(text, inner_voice=""):
        # The model responds to the reminder by snoozing the commitment.
        store.snooze(c.id, until=time.time() + 3600)

    app._run_agent = _snoozes_during_turn

    await FamiliarApp._reminder_tick(app)

    got = store.get(c.id)
    assert got.reminder_count == 0  # snooze reset not clobbered
    assert got.last_reminded_at is None
    store.close()


@pytest.mark.asyncio
async def test_tui_idle_desire_dispatches_without_logging_murmur():
    executor = MagicMock()
    executor.dispatch = AsyncMock(return_value=DriveActionResult(fired=True, desire_name="reflect"))
    desires = MagicMock()
    desires.get_dominant.return_value = ("reflect", 0.9)
    app = SimpleNamespace(
        agent=SimpleNamespace(config=SimpleNamespace(auto_desire=True)),
        desires=desires,
        _drive_executor=executor,
        _agent_running=False,
        _input_queue=asyncio.Queue(),
        _last_interaction=time.time() - 3600,
        _run_agent=AsyncMock(),
        _log_system=MagicMock(),
    )

    await FamiliarApp._desire_tick(app)

    executor.dispatch.assert_awaited_once()
    call = executor.dispatch.await_args
    assert call.args == ("reflect",)
    assert callable(call.kwargs["run_social_turn"])
    app._run_agent.assert_not_awaited()
    app._log_system.assert_not_called()
    desires.set_schedule_multiplier.assert_called_once_with(1.0)
    desires.satisfy.assert_not_called()
    assert desires.curiosity_target is None


@pytest.mark.asyncio
async def test_tui_idle_desire_sleeps_during_quiet_hours():
    executor = MagicMock()
    executor.dispatch = AsyncMock(return_value=DriveActionResult(fired=True, desire_name="reflect"))
    desires = MagicMock()
    desires.get_dominant.return_value = ("reflect", 0.9)
    heartbeat = MagicMock()
    heartbeat.routine_state.return_value = SimpleNamespace(quiet_hours=True)
    app = SimpleNamespace(
        agent=SimpleNamespace(config=SimpleNamespace(auto_desire=True), _heartbeat=heartbeat),
        desires=desires,
        _drive_executor=executor,
        _agent_running=False,
        _input_queue=asyncio.Queue(),
        _last_interaction=time.time() - 3600,
        _run_agent=AsyncMock(),
    )

    await FamiliarApp._desire_tick(app)

    desires.set_schedule_multiplier.assert_called_once_with(0.0)
    desires.get_dominant.assert_not_called()
    executor.dispatch.assert_not_awaited()
    app._run_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_tui_social_desire_uses_normal_ui_turn():
    async def dispatch(desire_name: str, **kwargs):
        assert desire_name == "greet_companion"
        await kwargs["run_social_turn"]("say hello")
        return DriveActionResult(fired=True, desire_name=desire_name)

    desires = MagicMock()
    desires.get_dominant.return_value = ("greet_companion", 0.9)
    app = SimpleNamespace(
        agent=SimpleNamespace(config=SimpleNamespace(auto_desire=True)),
        desires=desires,
        _drive_executor=SimpleNamespace(dispatch=dispatch),
        _agent_running=False,
        _input_queue=asyncio.Queue(),
        _last_interaction=time.time() - 3600,
        _run_agent=AsyncMock(),
    )

    await FamiliarApp._desire_tick(app)

    app._run_agent.assert_awaited_once_with("", inner_voice="say hello")


@pytest.mark.asyncio
async def test_tui_process_queue_requeues_input_during_autonomous_turn(tmp_path):
    """Input dequeued while an autonomous turn runs is re-queued, not run concurrently."""
    queue: asyncio.Queue = asyncio.Queue()
    run_calls: list[str] = []

    app = SimpleNamespace(_agent_running=False, _input_queue=queue)

    async def _run_agent(text):
        run_calls.append(text)

    app._run_agent = _run_agent

    task = asyncio.create_task(FamiliarApp._process_queue(app))
    await asyncio.sleep(0.01)  # parked in get()

    app._agent_running = True  # an autonomous turn (reminder tick) starts
    await queue.put("hello")  # user speaks during the turn
    await asyncio.sleep(0.12)  # woken consumer must re-queue, not run

    assert run_calls == []
    app._agent_running = False  # autonomous turn finishes
    await asyncio.sleep(0.2)

    assert run_calls == ["hello"]
    await queue.put(None)
    await asyncio.wait_for(task, timeout=2)
