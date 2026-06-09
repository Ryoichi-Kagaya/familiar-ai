"""User profile management for multi-user support."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_FAI_DIR = Path.home() / ".familiar_ai"
_USERS_DIR = _FAI_DIR / "users"
_USERS_JSON = _USERS_DIR / "users.json"
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]", "_", name.lower().strip())
    slug = re.sub(r"_+", "_", slug).strip("_") or "user"
    return slug[:32]


@dataclass
class UserProfile:
    id: str
    name: str
    dir: Path

    @property
    def mental_state_path(self) -> Path:
        return self.dir / "mental_state.jsonl"

    @property
    def self_narrative_path(self) -> Path:
        return self.dir / "self_narrative.jsonl"


class UserRegistry:
    """Manages user profiles via ~/.familiar_ai/users/users.json."""

    def __init__(self, users_dir: Path | None = None) -> None:
        self._users_dir = users_dir or _USERS_DIR
        self._users_dir.mkdir(parents=True, exist_ok=True)
        self._json_path = self._users_dir / "users.json"

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
        return UserProfile(
            id=uid,
            name=entry.get("name") or "ユーザー",
            dir=self._ensure_dir(uid),
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
        entry: dict = {"id": user_id, "name": "ユーザー"}
        entries.append(entry)
        self._write(entries)
        return self._to_profile(entry)

    def create(self, user_id: str, name: str) -> UserProfile:
        """Add or update a user entry with an explicit display name."""
        user_id = user_id.strip()
        name = name.strip()
        entries = self._read()
        for e in entries:
            if e["id"] == user_id:
                e["name"] = name
                self._write(entries)
                return self._to_profile(e)
        entry: dict = {"id": user_id, "name": name}
        entries.append(entry)
        self._write(entries)
        return self._to_profile(entry)

    def active_id(self) -> str:
        active_file = self._users_dir / "active.txt"
        if active_file.exists():
            val = active_file.read_text(encoding="utf-8").strip()
            if val:
                return val
        return "default"

    def set_active(self, user_id: str) -> None:
        (self._users_dir / "active.txt").write_text(user_id, encoding="utf-8")

    def get_active(self) -> UserProfile:
        return self.get(self.active_id())
