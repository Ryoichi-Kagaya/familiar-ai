"""Task-local user identity for conversation and memory operations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from collections.abc import Iterator


LEGACY_USER_ID = "__legacy__"

_active_user_id: ContextVar[str | None] = ContextVar(
    "familiar_active_user_id",
    default=None,
)
_cross_user_memory_ids: ContextVar[frozenset[str]] = ContextVar(
    "familiar_cross_user_memory_ids",
    default=frozenset(),
)


def current_user_id(*, fallback: str = LEGACY_USER_ID) -> str:
    """Return the user bound to the current async task."""
    return _active_user_id.get() or fallback


@contextmanager
def user_scope(user_id: str) -> Iterator[None]:
    """Bind a user to this task and any child tasks it creates."""
    token = _active_user_id.set(user_id.strip() or LEGACY_USER_ID)
    try:
        yield
    finally:
        _active_user_id.reset(token)


def allowed_cross_user_ids() -> frozenset[str]:
    """Return other users whose memory may be searched in this turn."""
    return _cross_user_memory_ids.get()


@contextmanager
def cross_user_memory_scope(user_ids: set[str] | frozenset[str]) -> Iterator[None]:
    """Temporarily grant memory search for explicitly referenced users."""
    normalized = frozenset(user_id.strip() for user_id in user_ids if user_id.strip())
    token = _cross_user_memory_ids.set(normalized)
    try:
        yield
    finally:
        _cross_user_memory_ids.reset(token)
