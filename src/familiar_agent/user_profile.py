"""User profile management for multi-user support."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

_FAI_DIR = Path.home() / ".familiar_ai"
_USERS_DIR = _FAI_DIR / "users"
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def _default_display_name() -> str:
    return os.environ.get("COMPANION_NAME", "").strip() or "ユーザー"


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]", "_", name.lower().strip())
    slug = re.sub(r"_+", "_", slug).strip("_") or "user"
    return slug[:32]


@dataclass
class UserProfile:
    id: str
    name: str
    dir: Path
    telegram_id: int | None = None
    aliases: tuple[str, ...] = ()
    aliases_by_user: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def references_from(self, user_id: str) -> tuple[str, ...]:
        """Return global and relationship-relative names used by one speaker."""
        scoped = self.aliases_by_user.get(user_id, ())
        return (self.name, self.id, *self.aliases, *scoped)

    @property
    def mental_state_path(self) -> Path:
        return self.dir / "mental_state.jsonl"

    @property
    def self_narrative_path(self) -> Path:
        return self.dir / "self_narrative.jsonl"


class UserRegistry:
    """Manages user profiles via ~/.familiar_ai/users/users.json."""

    _REGISTRY_FILE = "users.json"

    def __init__(self, users_dir: Path | None = None) -> None:
        self._users_dir = users_dir or _USERS_DIR
        self._users_dir.mkdir(parents=True, exist_ok=True)
        self._json_path = self._users_dir / self._REGISTRY_FILE

    # ── internal helpers ──────────────────────────────────────────────

    def _read(self) -> list[dict]:
        if not self._json_path.exists():
            return []
        try:
            data = json.loads(self._json_path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _write(self, entries: list[dict]) -> None:
        self._json_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _ensure_dir(self, user_id: str) -> Path:
        d = self._users_dir / user_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _to_profile(self, entry: dict) -> UserProfile:
        uid = entry["id"]
        raw_aliases = entry.get("aliases", [])
        aliases = (
            tuple(
                alias.strip() for alias in raw_aliases if isinstance(alias, str) and alias.strip()
            )
            if isinstance(raw_aliases, list)
            else ()
        )
        raw_scoped_aliases = entry.get("aliases_by_user", {})
        aliases_by_user = (
            {
                str(viewer_id): tuple(
                    alias.strip()
                    for alias in viewer_aliases
                    if isinstance(alias, str) and alias.strip()
                )
                for viewer_id, viewer_aliases in raw_scoped_aliases.items()
                if isinstance(viewer_aliases, list)
            }
            if isinstance(raw_scoped_aliases, dict)
            else {}
        )
        return UserProfile(
            id=uid,
            name=entry.get("name") or _default_display_name(),
            dir=self._ensure_dir(uid),
            telegram_id=entry.get("telegram_id"),
            aliases=aliases,
            aliases_by_user=aliases_by_user,
        )

    # ── public API ────────────────────────────────────────────────────

    def list_users(self) -> list[UserProfile]:
        return [self._to_profile(e) for e in self._read()]

    def get(self, user_id: str) -> UserProfile:
        """Return profile for user_id, adding it to users.json if absent."""
        user_id = user_id.strip()
        if not _SLUG_RE.match(user_id):
            user_id = _slugify(user_id)
        entries = self._read()
        for e in entries:
            if e["id"] == user_id:
                return self._to_profile(e)
        # Not found — register with default name
        entry: dict = {"id": user_id, "name": _default_display_name()}
        entries.append(entry)
        self._write(entries)
        return self._to_profile(entry)

    def create(
        self,
        user_id: str,
        name: str,
        *,
        aliases: list[str] | tuple[str, ...] | None = None,
        aliases_by_user: Mapping[str, list[str] | tuple[str, ...]] | None = None,
    ) -> UserProfile:
        """Add or update a user entry with an explicit display name."""
        user_id = user_id.strip()
        name = name.strip()
        normalized_aliases = list(
            dict.fromkeys(alias.strip() for alias in aliases or () if alias.strip())
        )
        normalized_scoped_aliases = {
            str(viewer_id): list(
                dict.fromkeys(alias.strip() for alias in viewer_aliases if alias.strip())
            )
            for viewer_id, viewer_aliases in (aliases_by_user or {}).items()
        }
        entries = self._read()
        for e in entries:
            if e["id"] == user_id:
                e["name"] = name
                if aliases is not None:
                    e["aliases"] = normalized_aliases
                if aliases_by_user is not None:
                    e["aliases_by_user"] = normalized_scoped_aliases
                self._write(entries)
                return self._to_profile(e)
        entry: dict = {"id": user_id, "name": name}
        if aliases is not None:
            entry["aliases"] = normalized_aliases
        if aliases_by_user is not None:
            entry["aliases_by_user"] = normalized_scoped_aliases
        entries.append(entry)
        self._write(entries)
        return self._to_profile(entry)

    def get_by_telegram_id(self, telegram_id: int) -> UserProfile | None:
        """Return the profile linked to this Telegram user ID, or None."""
        for e in self._read():
            if e.get("telegram_id") == telegram_id:
                return self._to_profile(e)
        return None

    def link_telegram(self, user_id: str, telegram_id: int) -> UserProfile:
        """Associate a Telegram user ID with an existing user profile."""
        user_id = user_id.strip()
        entries = self._read()
        # Clear any existing mapping for this telegram_id first
        for e in entries:
            if e.get("telegram_id") == telegram_id and e["id"] != user_id:
                e.pop("telegram_id", None)
        for e in entries:
            if e["id"] == user_id:
                e["telegram_id"] = telegram_id
                self._write(entries)
                return self._to_profile(e)
        # user_id not yet registered — create it
        entry: dict = {"id": user_id, "name": _default_display_name(), "telegram_id": telegram_id}
        entries.append(entry)
        self._write(entries)
        return self._to_profile(entry)

    def get_active(self) -> UserProfile:
        return self.get("default")
