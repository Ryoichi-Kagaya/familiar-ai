from pathlib import Path

import pytest

from familiar_agent.user_profile import UserRegistry
from familiar_agent.user_references import resolve_cross_user_memory_targets


@pytest.fixture
def user_registry(tmp_path: Path) -> UserRegistry:
    registry = UserRegistry(users_dir=tmp_path)
    registry.create("default", "カガヤ")
    registry.create(
        "honoruru",
        "ほのるる",
        aliases_by_user={"default": ["姉", "お姉ちゃん"]},
    )
    registry.create("mom", "母")
    return registry


@pytest.mark.parametrize(
    ("text", "current_user_id", "expected"),
    [
        ("姉何してる？", "default", {"honoruru"}),
        ("母は最近どう？", "default", {"mom"}),
        ("姉と旅行した", "default", set()),
        ("最近どう？", "default", set()),
        ("姉は最近どう？", "mom", set()),
        ("what moment?", "default", set()),
        ("What is honoruru doing?", "default", {"honoruru"}),
        ("What is honorururu doing?", "default", set()),
    ],
)
def test_resolve_cross_user_memory_targets(
    user_registry: UserRegistry,
    text: str,
    current_user_id: str,
    expected: set[str],
) -> None:
    assert resolve_cross_user_memory_targets(
        text,
        current_user_id,
        user_registry.list_users(),
    ) == frozenset(expected)
