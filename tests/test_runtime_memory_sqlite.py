"""End-to-end smoke tests for SQLiteMemoryStore around ObservationMemory."""

from __future__ import annotations

from pathlib import Path

import pytest

from familiar_runtime.memory import (
    MemoryStore,
    ObservationRecord,
    RecallResult,
    SQLiteMemoryStore,
)


@pytest.fixture
def memory_db(tmp_path: Path, stub_embedding_model: None):
    """Provide a fresh ObservationMemory bound to a tmp_path SQLite file.

    The adapter tests cover persistence and translation, not embedding model
    quality, so the shared lightweight embedding fixture avoids model loading.
    """
    from familiar_agent.tools.memory import ObservationMemory

    db_path = tmp_path / "observations.db"
    store = ObservationMemory(db_path=str(db_path))
    try:
        yield store
    finally:
        store.close()


@pytest.mark.asyncio
async def test_sqlite_adapter_satisfies_protocol_and_round_trips(memory_db) -> None:
    adapter: MemoryStore = SQLiteMemoryStore(memory_db)
    obs_id = await adapter.save_observation(
        "the morning sky was orange",
        metadata={"kind": "observation", "emotion": "moved", "direction": "outside"},
    )
    assert obs_id

    recent = await adapter.recent(limit=5)
    assert any(record.id == obs_id for record in recent)
    matched = next(record for record in recent if record.id == obs_id)
    assert "morning sky" in matched.text
    assert matched.metadata["kind"] == "observation"


@pytest.mark.asyncio
async def test_recall_returns_records_with_scores(memory_db) -> None:
    adapter = SQLiteMemoryStore(memory_db)
    await adapter.save_observation(
        "evening walk under the streetlights", metadata={"kind": "observation"}
    )
    await adapter.save_observation(
        "morning coffee on the balcony", metadata={"kind": "observation"}
    )
    hits = await adapter.recall("morning", limit=3)
    assert isinstance(hits, list)
    for hit in hits:
        assert isinstance(hit, RecallResult)
        assert isinstance(hit.record, ObservationRecord)
        assert hit.score >= 0.0


@pytest.mark.asyncio
async def test_unknown_kind_and_emotion_fall_back_to_defaults(memory_db) -> None:
    """Pass garbage in metadata; adapter should clamp to known values, not error."""
    adapter = SQLiteMemoryStore(memory_db)
    obs_id = await adapter.save_observation(
        "test note",
        metadata={"kind": "totally_unknown_kind", "emotion": "??"},
    )
    assert obs_id
    recent = await adapter.recent(limit=1)
    assert recent[0].metadata["kind"] == "observation"
    assert recent[0].metadata["emotion"] == "neutral"


def test_record_from_row_helper_handles_dict_rows() -> None:
    """Ensure recall dicts are converted into ObservationRecord cleanly."""
    from familiar_runtime.memory.sqlite_store import _record_from_row

    row = {
        "memory_id": "abc",
        "summary": "hello",
        "timestamp": "2026-05-28T01:00:00",
        "kind": "observation",
        "emotion": "happy",
    }
    record = _record_from_row(row)
    assert record.id == "abc"
    assert record.text == "hello"
    assert record.metadata["kind"] == "observation"
    assert record.metadata["emotion"] == "happy"


def test_record_from_row_rejects_non_mapping_inputs() -> None:
    from familiar_runtime.memory.sqlite_store import _record_from_row

    with pytest.raises(TypeError):
        _record_from_row(object())
