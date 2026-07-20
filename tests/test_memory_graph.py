from __future__ import annotations

from pathlib import Path

import pytest

from familiar_agent.tools.memory import ObservationMemory


@pytest.fixture
def memory(tmp_path: Path, stub_embedding_model: None) -> ObservationMemory:
    store = ObservationMemory(db_path=str(tmp_path / "memory.db"))
    try:
        yield store
    finally:
        store.close()


def test_memory_episodes_and_associative_recall_work(memory: ObservationMemory) -> None:
    first_id, ok1 = memory.save_with_id("雨の散歩のことを覚えておく", kind="conversation")
    second_id, ok2 = memory.save_with_id("散歩中に古い喫茶店を見つけた", kind="observation")
    assert ok1 is True and ok2 is True
    assert first_id is not None and second_id is not None

    episode_id = memory.create_episode("雨の日の散歩", summary="雨の散歩と喫茶店の記憶")
    assert episode_id is not None
    assert memory.append_to_episode(episode_id, first_id) is True
    assert memory.append_to_episode(episode_id, second_id) is True
    assert memory.link_memories(first_id, second_id, link_type="related") is True

    recalled = memory.recall_divergent("雨 散歩", n=4)
    assert recalled
    assert any(item.get("memory_id") == second_id for item in recalled)
    assert any(item.get("episode_id") == episode_id for item in recalled)

    working = memory.refresh_working_memory("雨 散歩", n=4)
    assert working
    assert memory.get_working_memory()


def test_conflicting_semantic_facts_create_revisions(memory: ObservationMemory) -> None:
    with memory._db_lock:  # noqa: SLF001
        db = memory._ensure_connected()  # noqa: SLF001
        memory._upsert_semantic_fact_locked(
            db, "favorite_drink", "Coffee is the favorite", confidence=0.7
        )  # noqa: SLF001
        memory._upsert_semantic_fact_locked(
            db, "favorite_drink", "Tea is the favorite", confidence=0.8
        )  # noqa: SLF001
        db.commit()

    revisions = memory.recall_revisions(entity_type="semantic_fact", entity_key="favorite_drink")
    assert revisions
    assert revisions[0]["previous_text"] == "Coffee is the favorite"
    assert revisions[0]["new_text"] == "Tea is the favorite"


def test_link_memories_accepts_surfaced_8char_prefixes(memory: ObservationMemory) -> None:
    """remember/recall surface ids truncated to 8 chars; linking with those
    prefixes must create a REAL link (regression: dangling links were silently
    inserted for unknown ids)."""
    first_id, ok1 = memory.save_with_id("青いカメラが届いた", kind="observation")
    second_id, ok2 = memory.save_with_id("カメラの設定を終えた", kind="observation")
    assert ok1 and ok2

    assert memory.link_memories(first_id[:8], second_id[:8], link_type="leads_to") is True
    linked = memory.get_linked_memories(first_id)
    assert any(item.get("id") == second_id for item in linked)


def test_link_memories_rejects_unknown_target(memory: ObservationMemory) -> None:
    first_id, _ = memory.save_with_id("孤立した記憶", kind="observation")
    assert memory.link_memories(first_id, "deadbeef") is False
    assert memory.get_linked_memories(first_id) == []
