"""Global pytest configuration for familiar-ai."""

from __future__ import annotations

import os

import pytest


# ObservationMemory prewarms heavy embedding threads on init. In the full test
# suite that causes many concurrent daemon loads, noisy progress bars, and
# unstable shutdown behavior. Disable it globally unless a test explicitly opts in.
os.environ.setdefault("FAMILIAR_EMBEDDING_PREWARM", "0")


@pytest.fixture
def stub_embedding_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep storage tests focused on persistence instead of model inference.

    Tests that use this fixture still exercise vector serialization and semantic
    recall, but avoid loading the production sentence-transformer. Embedding
    model behavior has dedicated coverage in ``test_memory_prewarm.py``.
    """
    from familiar_agent.tools.memory import _EmbeddingModel

    def constant_vectors(_model: _EmbeddingModel, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setattr(_EmbeddingModel, "encode_document", constant_vectors)
    monkeypatch.setattr(_EmbeddingModel, "encode_query", constant_vectors)
