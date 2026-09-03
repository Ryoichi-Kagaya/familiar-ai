from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest

from familiar_agent.agent import EmbodiedAgent
from familiar_agent.tools.memory import MemoryTool, ObservationMemory
from familiar_agent.user_context import allowed_cross_user_ids, cross_user_memory_scope, user_scope
from familiar_agent.user_profile import UserRegistry
from familiar_runtime.models import UserTurn


@pytest.fixture
def observation_memory(
    tmp_path: Path,
    stub_embedding_model: None,
) -> Iterator[ObservationMemory]:
    memory = ObservationMemory(db_path=str(tmp_path / "memory.db"))
    try:
        yield memory
    finally:
        memory.close()


def test_conversation_history_is_partitioned_by_user() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent._messages_by_user = {}  # noqa: SLF001
    agent._current_user = SimpleNamespace(id="default")  # noqa: SLF001

    with user_scope("default"):
        agent.messages.append({"role": "user", "content": "本人の話"})
    with user_scope("honoruru"):
        assert agent.messages == []
        agent.messages.append({"role": "user", "content": "姉の話"})
    with user_scope("default"):
        assert [message["content"] for message in agent.messages] == ["本人の話"]


def test_interrupt_drain_leaves_another_users_turn_in_queue() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent._messages_by_user = {}  # noqa: SLF001
    agent._current_user = SimpleNamespace(id="default")  # noqa: SLF001
    queue: asyncio.Queue[UserTurn] = asyncio.Queue()
    queue.put_nowait(UserTurn("本人の追伸", user_id="default"))
    queue.put_nowait(UserTurn("姉の次の発言", user_id="honoruru"))

    with user_scope("default"):
        drained = agent._drain_interrupt_queue(queue)  # noqa: SLF001

    assert [turn.text for turn in drained] == ["本人の追伸"]
    assert queue.get_nowait().user_id == "honoruru"


def test_observations_and_threads_are_partitioned_by_user(
    observation_memory: ObservationMemory,
) -> None:
    with user_scope("default"):
        assert observation_memory.save("本人だけの旅行", kind="conversation")
        assert observation_memory.open_unfinished_business("本人の続き")
        observation_memory.upsert_semantic_fact("favorite_trip", "本人は山が好き")

    with user_scope("honoruru"):
        assert observation_memory.recall("旅行") == []
        assert observation_memory.list_unfinished_business() == []
        assert observation_memory.recall_semantic_facts("山") == []
        assert observation_memory.save("姉だけの旅行", kind="conversation")

    with user_scope("default"):
        recalled = observation_memory.recall("旅行")
        assert [item["summary"] for item in recalled] == ["本人だけの旅行"]
        assert [item["summary"] for item in observation_memory.list_unfinished_business()] == [
            "本人の続き"
        ]


def test_unattributed_memory_can_be_relearned_for_a_registered_user(
    observation_memory: ObservationMemory,
) -> None:
    assert observation_memory.save("昔いっしょに琵琶湖へ行った", kind="conversation")
    candidate = observation_memory.recall_unattributed("琵琶湖", n=1)[0]
    assert candidate["ownership"] == "unverified"
    observation_memory.upsert_semantic_fact(
        "trip:biwako",
        "琵琶湖へ行ったことがある",
        source_memory_id=candidate["memory_id"],
    )

    with user_scope("default"):
        attributed, memory_id = observation_memory.attribute_memory_owner(
            candidate["memory_id"],
            "honoruru",
            allowed_user_ids={"default", "honoruru"},
        )
        assert attributed is True
        assert memory_id == candidate["memory_id"]
        assert observation_memory.recall("琵琶湖") == []

    with user_scope("honoruru"):
        assert observation_memory.recall("琵琶湖")[0]["summary"] == "昔いっしょに琵琶湖へ行った"
        assert observation_memory.recall_semantic_facts("琵琶湖")[0]["key"] == "trip:biwako"
        assert observation_memory.recall_unattributed("琵琶湖") == []


@pytest.mark.asyncio
async def test_memory_tool_attributes_only_to_registered_profiles(
    observation_memory: ObservationMemory,
) -> None:
    tool = MemoryTool(observation_memory)
    tool.set_user_profiles(lambda: [("default", "カガヤ"), ("honoruru", "ほのるる")])
    memory_id, saved = observation_memory.save_with_id("姉と海へ行った", kind="conversation")
    assert saved and memory_id
    with user_scope("default"):
        result, _ = await tool.call(
            "attribute_memory_owner",
            {"memory_id": memory_id[:8], "owner_user_id": "honoruru"},
        )
    assert "ほのるる" in result

    definitions = {item["name"]: item for item in tool.get_tool_definitions()}
    owner_schema = definitions["attribute_memory_owner"]["input_schema"]["properties"]
    assert owner_schema["owner_user_id"]["enum"] == ["default", "honoruru"]


@pytest.mark.asyncio
async def test_agent_grants_cross_user_recall_for_only_the_matching_turn(tmp_path: Path) -> None:
    class CapturingCoordinator:
        async def run(self, request) -> str:
            assert request.user_input == "姉何してる？"
            assert allowed_cross_user_ids() == frozenset({"honoruru"})
            return "ok"

    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent._current_user = SimpleNamespace(id="default")  # noqa: SLF001
    agent._user_registry = UserRegistry(users_dir=tmp_path)  # noqa: SLF001
    agent._user_registry.create("default", "カガヤ")  # noqa: SLF001
    agent._user_registry.create(  # noqa: SLF001
        "honoruru", "ほのるる", aliases_by_user={"default": ["姉"]}
    )
    agent._turn_coordinator = CapturingCoordinator()  # type: ignore[assignment]  # noqa: SLF001

    assert await agent.run("姉何してる？") == "ok"
    assert allowed_cross_user_ids() == frozenset()


@pytest.mark.asyncio
async def test_cross_user_recall_is_available_only_inside_explicit_turn_grant(
    observation_memory: ObservationMemory,
) -> None:
    tool = MemoryTool(observation_memory)
    tool.set_user_profiles(lambda: [("default", "カガヤ"), ("honoruru", "ほのるる")])
    with user_scope("honoruru"):
        assert observation_memory.save("今日は美術館を見てきた", kind="conversation")

    with user_scope("default"):
        assert "recall_user_memory" not in {
            definition["name"] for definition in tool.get_tool_definitions()
        }
        denied, _ = await tool.call(
            "recall_user_memory", {"user_id": "honoruru", "query": "美術館"}
        )
        assert denied.startswith("Error:")

        with cross_user_memory_scope({"honoruru"}):
            definitions = {item["name"]: item for item in tool.get_tool_definitions()}
            target_schema = definitions["recall_user_memory"]["input_schema"]["properties"]
            assert target_schema["user_id"]["enum"] == ["honoruru"]
            result, _ = await tool.call(
                "recall_user_memory", {"user_id": "honoruru", "query": "美術館"}
            )
            assert "ほのるる (honoruru)" in result
            assert "今日は美術館を見てきた" in result
            assert observation_memory.recall("美術館") == []
            wrong_target, _ = await tool.call(
                "recall_user_memory", {"user_id": "default", "query": "美術館"}
            )
            assert wrong_target.startswith("Error:")


def test_unattributed_recall_cadence_is_per_user() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent._messages_by_user = {}  # noqa: SLF001
    agent._current_user = SimpleNamespace(id="default")  # noqa: SLF001
    agent._unattributed_recall_counts_by_user = {}  # noqa: SLF001

    with user_scope("default"):
        assert [agent._should_surface_unattributed_memory() for _ in range(6)] == [  # noqa: SLF001
            True,
            False,
            False,
            False,
            False,
            True,
        ]
    with user_scope("honoruru"):
        assert agent._should_surface_unattributed_memory() is True  # noqa: SLF001


def test_brief_correction_keeps_only_memory_attribution_tool() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent._messages_by_user = {  # noqa: SLF001
        "default": [{"role": "user", "content": "owner:unverified id:deadbeef"}]
    }
    agent._current_user = SimpleNamespace(id="default")  # noqa: SLF001
    all_tools = [
        {"name": "say"},
        {"name": "recall"},
        {"name": "attribute_memory_owner"},
    ]
    with patch.object(EmbodiedAgent, "_all_tool_defs", new_callable=PropertyMock) as tool_defs:
        tool_defs.return_value = all_tools
        selected = agent._tool_defs_for_turn(brief_reply_mode=True)  # noqa: SLF001

    assert [tool["name"] for tool in selected] == ["say", "attribute_memory_owner"]


def test_brief_person_question_keeps_cross_user_recall_tool() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent._messages_by_user = {}  # noqa: SLF001
    agent._current_user = SimpleNamespace(id="default")  # noqa: SLF001
    all_tools = [
        {"name": "say"},
        {"name": "recall"},
        {"name": "recall_user_memory"},
    ]
    with (
        cross_user_memory_scope({"honoruru"}),
        patch.object(EmbodiedAgent, "_all_tool_defs", new_callable=PropertyMock) as tool_defs,
    ):
        tool_defs.return_value = all_tools
        selected = agent._tool_defs_for_turn(brief_reply_mode=True)  # noqa: SLF001

    assert [tool["name"] for tool in selected] == ["say", "recall_user_memory"]
