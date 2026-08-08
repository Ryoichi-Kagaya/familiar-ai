"""Startup/bootstrap helpers for familiar-ai."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

from .setup import migrate_legacy_env_file

_FAI_DIR = Path.home() / ".familiar_ai"
ACTIVE_CAMERA_PROFILE_ENV = "FAMILIAR_ACTIVE_CAMERA_PROFILE"

_CAMERA_PROFILE_KEYS = {
    "CAMERA_HOST",
    "CAMERA_USERNAME",
    "CAMERA_PASSWORD",
    "CAMERA_ONVIF_PORT",
    "CAMERA_PREVIEW",
    "CAMERA_PTZ_HOST",
    "CAMERA_PTZ_USERNAME",
    "CAMERA_PTZ_PASSWORD",
    "CAMERA_PTZ_PORT",
    "CAMERA_RTSP_PATH",
    "CAMERA_TAPO_PASSWORD",
    "CAMERA_TAPO_HASH",
    "GO2RTC_URL",
    "GO2RTC_STREAM",
    "STT_INPUT",
    "TTS_OUTPUT",
}

# These predate the canonical CAMERA_* names but are still accepted by CameraConfig.
# Clear them when switching profiles so an omitted profile value cannot silently fall
# back to the primary camera from the main .env.
_LEGACY_CAMERA_KEYS = {
    "TAPO_CAMERA_HOST",
    "TAPO_USERNAME",
    "TAPO_PASSWORD",
    "TAPO_ONVIF_PORT",
}


def _migrate_legacy_user_files() -> None:
    """Move top-level mental_state/self_narrative logs to users/default/ on first run."""
    default_dir = _FAI_DIR / "users" / "default"
    default_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("mental_state.jsonl", "self_narrative.jsonl"):
        src = _FAI_DIR / fname
        dst = default_dir / fname
        if src.exists() and not dst.exists():
            src.rename(dst)


def resolve_env_path() -> Path:
    """Resolve the app `.env` path.

    Prefer the repository root when running from source, otherwise fall back to
    the current working directory.
    """
    root_env = Path(__file__).resolve().parents[2] / ".env"
    if root_env.exists():
        return root_env
    return Path.cwd() / ".env"


def load_camera_profile(path: Path) -> tuple[Path, list[str]]:
    """Overlay camera-only settings from a dotenv profile.

    The main ``.env`` remains authoritative for model credentials, persona, memory,
    and autonomy settings. This makes it safe to switch physical cameras without
    duplicating or replacing the familiar's primary configuration.
    """
    resolved = path.expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Camera profile not found: {resolved}")

    values = dotenv_values(resolved)
    for key in _CAMERA_PROFILE_KEYS | _LEGACY_CAMERA_KEYS:
        os.environ.pop(key, None)

    loaded: list[str] = []
    for key in sorted(_CAMERA_PROFILE_KEYS):
        value = values.get(key)
        if value is not None:
            os.environ[key] = value
            loaded.append(key)
    os.environ[ACTIVE_CAMERA_PROFILE_ENV] = str(resolved)
    return resolved, loaded


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

    _migrate_legacy_user_files()

    api_key = (os.environ.get("API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    platform = os.environ.get("PLATFORM", "anthropic").strip().lower()
    configured = platform == "cli" or bool(api_key)
    return AppBootstrap(
        env_path=path,
        configured=configured,
        needs_setup=not configured,
        legacy_config_detected=legacy_detected,
        migrated=migrated,
        messages=messages,
    )
