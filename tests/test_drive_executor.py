"""Tests for DriveActionExecutor — desire effect routing."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.desires import DesireSystem
from familiar_agent.drive_executor import ABSENCE_THRESHOLD, DriveActionExecutor
from familiar_neighbor.mind.desires import DriveEffect


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def desires(tmp_path: Path) -> DesireSystem:
    return DesireSystem(state_path=tmp_path / "desires.json", companion_name="Kota")


@pytest.fixture
def mock_agent() -> MagicMock:
    agent = MagicMock()
    agent._camera = None
    agent._tts = None
    agent._memory = AsyncMock()
    agent._memory.save_async = AsyncMock(return_value=True)
    agent._memory_worker = AsyncMock()
    agent._memory_worker.run_once = AsyncMock(return_value=0)
    agent._utility_backend = AsyncMock()
    agent._utility_backend.complete = AsyncMock(return_value="一文のつぶやき。")
    agent._self_narrative = MagicMock()
    agent._self_narrative.write = MagicMock()
    agent.run = AsyncMock()
    return agent


@pytest.fixture
def executor(mock_agent: MagicMock, desires: DesireSystem) -> DriveActionExecutor:
    return DriveActionExecutor(mock_agent, desires)


# ---------------------------------------------------------------------------
# DriveEffect classification
# ---------------------------------------------------------------------------


def test_look_around_is_silent_action(desires: DesireSystem) -> None:
    spec = desires._drive_specs["look_around"]
    assert spec.effect_type == DriveEffect.SILENT_ACTION


def test_explore_is_silent_action(desires: DesireSystem) -> None:
    assert desires._drive_specs["explore"].effect_type == DriveEffect.SILENT_ACTION


def test_consolidate_is_silent_action(desires: DesireSystem) -> None:
    assert desires._drive_specs["consolidate"].effect_type == DriveEffect.SILENT_ACTION


def test_reflect_is_silent_action(desires: DesireSystem) -> None:
    assert desires._drive_specs["reflect"].effect_type == DriveEffect.SILENT_ACTION


def test_curiosity_is_silent_action(desires: DesireSystem) -> None:
    assert desires._drive_specs["curiosity"].effect_type == DriveEffect.SILENT_ACTION


def test_share_memory_is_expressive_solo(desires: DesireSystem) -> None:
    assert desires._drive_specs["share_memory"].effect_type == DriveEffect.EXPRESSIVE_SOLO


def test_play_is_expressive_solo(desires: DesireSystem) -> None:
    assert desires._drive_specs["play"].effect_type == DriveEffect.EXPRESSIVE_SOLO


def test_attachment_is_social_initiation(desires: DesireSystem) -> None:
    assert desires._drive_specs["attachment"].effect_type == DriveEffect.SOCIAL_INITIATION


def test_greet_companion_is_social_initiation(desires: DesireSystem) -> None:
    assert desires._drive_specs["greet_companion"].effect_type == DriveEffect.SOCIAL_INITIATION


def test_repair_is_social_initiation(desires: DesireSystem) -> None:
    assert desires._drive_specs["repair"].effect_type == DriveEffect.SOCIAL_INITIATION


def test_worry_companion_is_absent_care(desires: DesireSystem) -> None:
    assert desires._drive_specs["worry_companion"].effect_type == DriveEffect.ABSENT_CARE


def test_care_is_absent_care(desires: DesireSystem) -> None:
    assert desires._drive_specs["care"].effect_type == DriveEffect.ABSENT_CARE


def test_rest_is_gate(desires: DesireSystem) -> None:
    assert desires._drive_specs["rest"].effect_type == DriveEffect.GATE


def test_self_protect_is_gate(desires: DesireSystem) -> None:
    assert desires._drive_specs["self_protect"].effect_type == DriveEffect.GATE


def test_worry_companion_min_interval_is_4h(desires: DesireSystem) -> None:
    assert desires._drive_specs["worry_companion"].min_interval_seconds == 14400


def test_care_min_interval_is_2h(desires: DesireSystem) -> None:
    assert desires._drive_specs["care"].min_interval_seconds == 7200


# ---------------------------------------------------------------------------
# Companion presence check
# ---------------------------------------------------------------------------


def test_companion_present_when_recent_interaction(
    executor: DriveActionExecutor,
) -> None:
    recent = time.time() - 60  # 1 minute ago
    assert executor._companion_present(recent) is True


def test_companion_absent_after_threshold(
    executor: DriveActionExecutor,
) -> None:
    old = time.time() - ABSENCE_THRESHOLD - 1
    assert executor._companion_present(old) is False


# ---------------------------------------------------------------------------
# GATE drives
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gate_drive_does_not_fire(
    executor: DriveActionExecutor, mock_agent: MagicMock, desires: DesireSystem
) -> None:
    before = time.time()
    result = await executor.dispatch("rest", last_interaction_time=time.time())
    assert result.fired is False
    assert result.reason == "gate_drive"
    assert desires._last_attempted["rest"] >= before
    mock_agent.run.assert_not_called()


@pytest.mark.asyncio
async def test_self_protect_gate_does_not_fire(
    executor: DriveActionExecutor, mock_agent: MagicMock
) -> None:
    result = await executor.dispatch("self_protect", last_interaction_time=time.time())
    assert result.fired is False
    assert result.reason == "gate_drive"


# ---------------------------------------------------------------------------
# SOCIAL_INITIATION — absent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_social_drive_does_not_fire_when_companion_absent(
    executor: DriveActionExecutor, mock_agent: MagicMock
) -> None:
    old_time = time.time() - ABSENCE_THRESHOLD - 10
    result = await executor.dispatch("greet_companion", last_interaction_time=old_time)
    assert result.fired is False
    assert result.reason == "companion_absent"
    mock_agent.run.assert_not_called()


@pytest.mark.asyncio
async def test_attachment_does_not_fire_when_absent(
    executor: DriveActionExecutor, mock_agent: MagicMock
) -> None:
    old_time = time.time() - ABSENCE_THRESHOLD - 10
    result = await executor.dispatch("attachment", last_interaction_time=old_time)
    assert result.fired is False


# ---------------------------------------------------------------------------
# SOCIAL_INITIATION — present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_social_drive_fires_when_companion_present(
    executor: DriveActionExecutor,
    mock_agent: MagicMock,
    desires: DesireSystem,
) -> None:
    recent = time.time() - 60
    desires.boost("greet_companion", 1.0)
    result = await executor.dispatch("greet_companion", last_interaction_time=recent)
    assert result.fired is True
    mock_agent.run.assert_called_once()


@pytest.mark.asyncio
async def test_social_drive_can_use_ui_turn_runner(
    executor: DriveActionExecutor,
    mock_agent: MagicMock,
    desires: DesireSystem,
) -> None:
    recent = time.time() - 60
    desires.boost("greet_companion", 1.0)
    run_social_turn = AsyncMock()

    result = await executor.dispatch(
        "greet_companion",
        last_interaction_time=recent,
        run_social_turn=run_social_turn,
    )

    assert result.fired is True
    run_social_turn.assert_awaited_once_with(desires._drive_specs["greet_companion"].prompt_text)
    mock_agent.run.assert_not_called()


# ---------------------------------------------------------------------------
# SILENT_ACTION — consolidate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consolidate_calls_memory_worker(
    executor: DriveActionExecutor, mock_agent: MagicMock
) -> None:
    result = await executor.dispatch("consolidate", last_interaction_time=time.time())
    assert result.fired is True
    mock_agent._memory_worker.run_once.assert_called_once()


# ---------------------------------------------------------------------------
# SILENT_ACTION — reflect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reflect_writes_to_narrative(
    executor: DriveActionExecutor, mock_agent: MagicMock
) -> None:
    result = await executor.dispatch("reflect", last_interaction_time=time.time())
    assert result.fired is True
    mock_agent._self_narrative.write.assert_called_once()
    call_kwargs = mock_agent._self_narrative.write.call_args
    assert (
        call_kwargs.kwargs.get("trigger") == "reflect_drive"
        or call_kwargs[1].get("trigger") == "reflect_drive"
        or "reflect_drive" in str(call_kwargs)
    )


@pytest.mark.asyncio
async def test_reflect_saves_to_memory(
    executor: DriveActionExecutor, mock_agent: MagicMock
) -> None:
    await executor.dispatch("reflect", last_interaction_time=time.time())
    mock_agent._memory.save_async.assert_called()


# ---------------------------------------------------------------------------
# SILENT_ACTION — look_around without camera
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_look_around_no_camera_does_not_fire(
    executor: DriveActionExecutor, mock_agent: MagicMock, desires: DesireSystem
) -> None:
    mock_agent._camera = None
    before = time.time()
    result = await executor.dispatch("look_around", last_interaction_time=time.time())
    assert result.fired is False
    assert result.reason == "no_camera"
    assert desires._last_attempted["look_around"] >= before


# ---------------------------------------------------------------------------
# EXPRESSIVE_SOLO — companion absent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_share_memory_solo_when_absent(
    executor: DriveActionExecutor, mock_agent: MagicMock
) -> None:
    old_time = time.time() - ABSENCE_THRESHOLD - 10
    result = await executor.dispatch("share_memory", last_interaction_time=old_time)
    assert result.fired is True
    mock_agent.run.assert_not_called()
    mock_agent._memory.save_async.assert_called()
    mock_agent._self_narrative.write.assert_called()


@pytest.mark.asyncio
async def test_share_memory_solo_generated_text_is_non_empty(
    executor: DriveActionExecutor, mock_agent: MagicMock
) -> None:
    old_time = time.time() - ABSENCE_THRESHOLD - 10
    result = await executor.dispatch("share_memory", last_interaction_time=old_time)
    assert len(result.generated_text) > 0


# ---------------------------------------------------------------------------
# EXPRESSIVE_SOLO — companion present → social turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_share_memory_uses_social_turn_when_present(
    executor: DriveActionExecutor, mock_agent: MagicMock
) -> None:
    recent = time.time() - 60
    result = await executor.dispatch("share_memory", last_interaction_time=recent)
    assert result.fired is True
    mock_agent.run.assert_called_once()


# ---------------------------------------------------------------------------
# ABSENT_CARE — companion absent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worry_absent_records_emotion_and_resets(
    executor: DriveActionExecutor, mock_agent: MagicMock, desires: DesireSystem
) -> None:
    old_time = time.time() - ABSENCE_THRESHOLD - 10
    desires.boost("worry_companion", 0.9)

    result = await executor.dispatch("worry_companion", last_interaction_time=old_time)

    assert result.fired is True
    assert result.reason == "absent_care_recorded"
    # Full reset — level should be 0
    assert desires.level("worry_companion") == 0.0
    mock_agent._memory.save_async.assert_called()
    mock_agent._self_narrative.write.assert_called()


@pytest.mark.asyncio
async def test_worry_absent_cooldown_is_set(
    executor: DriveActionExecutor, desires: DesireSystem, mock_agent: MagicMock
) -> None:
    old_time = time.time() - ABSENCE_THRESHOLD - 10
    desires.boost("worry_companion", 0.9)
    before = time.time()
    await executor.dispatch("worry_companion", last_interaction_time=old_time)
    assert desires._last_fired.get("worry_companion", 0) >= before


@pytest.mark.asyncio
async def test_care_absent_records_emotion(
    executor: DriveActionExecutor, mock_agent: MagicMock, desires: DesireSystem
) -> None:
    old_time = time.time() - ABSENCE_THRESHOLD - 10
    desires.boost("care", 0.9)
    result = await executor.dispatch("care", last_interaction_time=old_time)
    assert result.fired is True
    assert desires.level("care") == 0.0


# ---------------------------------------------------------------------------
# ABSENT_CARE — companion present → LLM turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worry_uses_social_turn_when_present(
    executor: DriveActionExecutor, mock_agent: MagicMock, desires: DesireSystem
) -> None:
    recent = time.time() - 60
    desires.boost("worry_companion", 0.9)
    result = await executor.dispatch("worry_companion", last_interaction_time=recent)
    assert result.fired is True
    mock_agent.run.assert_called_once()


# ---------------------------------------------------------------------------
# Unknown drive
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_drive_returns_not_fired(
    executor: DriveActionExecutor,
) -> None:
    result = await executor.dispatch("nonexistent_drive", last_interaction_time=time.time())
    assert result.fired is False
    assert result.reason == "unknown_drive"
