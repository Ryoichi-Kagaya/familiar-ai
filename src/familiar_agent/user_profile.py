"""User profile management for multi-user support."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_FAI_DIR = Path.home() / ".familiar_ai"
_USERS_DIR = _FAI_DIR / "users"
_ACTIVE_FILE = _USERS_DIR / "active.txt"
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
    """Manages user profiles stored under ~/.familiar_ai/users/."""

    def __init__(self, users_dir: Path | None = None) -> None:
        self._users_dir = users_dir or _USERS_DIR
        self._users_dir.mkdir(parents=True, exist_ok=True)

    def list_users(self) -> list[UserProfile]:
        profiles = []
        for d in sorted(self._users_dir.iterdir()):
            if d.is_dir() and _SLUG_RE.match(d.name):
                profiles.append(self._load_profile(d))
        return profiles

    def get(self, user_id: str) -> UserProfile:
        """Return profile for user_id, creating it if it doesn't exist."""
        user_id = user_id.strip()
        if not _SLUG_RE.match(user_id):
            user_id = _slugify(user_id)
        user_dir = self._users_dir / user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        return self._load_profile(user_dir)

    def create(self, user_id: str, name: str) -> UserProfile:
        """Explicitly create a user profile with a display name."""
        profile = self.get(user_id)
        name_file = profile.dir / "name.txt"
        name_file.write_text(name.strip(), encoding="utf-8")
        return UserProfile(id=profile.id, name=name.strip(), dir=profile.dir)

    def active_id(self) -> str:
        active_file = self._users_dir / "active.txt"
        if active_file.exists():
            val = active_file.read_text(encoding="utf-8").strip()
            if val:
                return val
        return "default"

    def set_active(self, user_id: str) -> None:
        active_file = self._users_dir / "active.txt"
        active_file.write_text(user_id, encoding="utf-8")

    def get_active(self) -> UserProfile:
        return self.get(self.active_id())

    def _load_profile(self, user_dir: Path) -> UserProfile:
        name_file = user_dir / "name.txt"
        if name_file.exists():
            name = name_file.read_text(encoding="utf-8").strip()
        else:
            name = user_dir.name
        return UserProfile(id=user_dir.name, name=name, dir=user_dir)
