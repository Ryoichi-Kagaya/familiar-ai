"""Startup/bootstrap helpers for familiar-ai."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from .setup import migrate_legacy_env_file

_FAI_DIR = Path.home() / ".familiar_ai"


def _migrate_legacy_user_files(companion_name: str = "") -> None:
    """Move top-level mental_state/self_narrative logs to users/default/ on first run."""
    default_dir = _FAI_DIR / "users" / "default"
    default_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("mental_state.jsonl", "self_narrative.jsonl"):
        src = _FAI_DIR / fname
        dst = default_dir / fname
        if src.exists() and not dst.exists():
            src.rename(dst)
    active_txt = _FAI_DIR / "users" / "active.txt"
    if not active_txt.exists():
        active_txt.write_text("default", encoding="utf-8")
    name_txt = default_dir / "name.txt"
    if not name_txt.exists() and companion_name:
        name_txt.write_text(companion_name, encoding="utf-8")


def resolve_env_path() -> Path:
    """Resolve the app `.env` path.

    Prefer the repository root when running from source, otherwise fall back to
    the current working directory.
    """
    root_env = Path(__file__).resolve().parents[2] / ".env"
    if root_env.exists():
        return root_env
    return Path.cwd() / ".env"


@dataclass
class AppBootstrap:
    env_path: Path
    configured: bool
    needs_setup: bool
    legacy_config_detected: bool
    migrated: bool = False
    messages: list[str] = field(default_factory=list)


def load_app_bootstrap(env_path: Path | None = None) -> AppBootstrap:  # noqa: C901
    """Load `.env`, migrate legacy Anthropic keys, and report startup status."""
    path = env_path or resolve_env_path()

    legacy_detected = False
    migrated = False
    messages: list[str] = []

    if path.exists():
        raw = path.read_text(encoding="utf-8")
        legacy_detected = "ANTHROPIC_API_KEY=" in raw or "ANTHROPIC_MODEL=" in raw
        migrated, messages = migrate_legacy_env_file(path)
        load_dotenv(path, override=True)

    companion_name = (os.environ.get("COMPANION_NAME") or "").strip()
    _migrate_legacy_user_files(companion_name)

    api_key = (os.environ.get("API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    configured = bool(api_key)
    return AppBootstrap(
        env_path=path,
        configured=configured,
        needs_setup=not configured,
        legacy_config_detected=legacy_detected,
        migrated=migrated,
        messages=messages,
    )
