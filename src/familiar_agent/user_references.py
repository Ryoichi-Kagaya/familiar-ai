"""Resolve explicit references to registered users in a conversation turn."""

from __future__ import annotations

import re
from collections.abc import Iterable

from .user_profile import UserProfile

_INQUIRY_RE = re.compile(
    r"(?:[?？]|何|なに|どう|どこ|いつ|誰|元気|最近|近頃|様子|予定|"
    r"してる|している|してた|知ってる|覚えてる|聞いた|教えて|"
    r"(?i:\b(?:what|how|where|when|who|recently|doing|know|remember)\b))"
)
_ASCII_REFERENCE_RE = re.compile(r"[A-Za-z0-9_-]+")


def resolve_cross_user_memory_targets(
    text: str,
    current_user_id: str,
    profiles: Iterable[UserProfile],
) -> frozenset[str]:
    """Return users explicitly named in a question by another registered user."""
    normalized = text.strip()
    if not normalized or _INQUIRY_RE.search(normalized) is None:
        return frozenset()

    return frozenset(
        profile.id
        for profile in profiles
        if profile.id != current_user_id
        and any(
            _reference_matches(reference, normalized)
            for reference in profile.references_from(current_user_id)
        )
    )


def _reference_matches(reference: str, text: str) -> bool:
    """Match ASCII identifiers as tokens and non-ASCII names as ordinary text."""
    if not reference:
        return False
    if _ASCII_REFERENCE_RE.fullmatch(reference):
        return (
            re.search(
                rf"(?i)(?<![a-z0-9_-]){re.escape(reference)}(?![a-z0-9_-])",
                text,
            )
            is not None
        )
    return reference in text
