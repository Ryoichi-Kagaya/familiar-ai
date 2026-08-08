from __future__ import annotations

import os
from pathlib import Path

import pytest

from familiar_agent.bootstrap import (
    ACTIVE_CAMERA_PROFILE_ENV,
    load_app_bootstrap,
    load_camera_profile,
)


def _clear_runtime_env(monkeypatch) -> None:
    for key in ("PLATFORM", "API_KEY", "MODEL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL"):
        monkeypatch.delenv(key, raising=False)


def test_load_app_bootstrap_needs_setup_when_env_missing(tmp_path: Path, monkeypatch) -> None:
    _clear_runtime_env(monkeypatch)

    state = load_app_bootstrap(tmp_path / ".env")

    assert state.env_path == tmp_path / ".env"
    assert state.configured is False
    assert state.needs_setup is True
    assert state.legacy_config_detected is False


def test_load_app_bootstrap_accepts_cli_without_api_key(tmp_path: Path, monkeypatch) -> None:
    _clear_runtime_env(monkeypatch)
    env_path = tmp_path / ".env"
    env_path.write_text("PLATFORM=cli\n", encoding="utf-8")

    state = load_app_bootstrap(env_path)

    assert state.configured is True
    assert state.needs_setup is False


def test_load_app_bootstrap_migrates_legacy_anthropic_env(tmp_path: Path, monkeypatch) -> None:
    _clear_runtime_env(monkeypatch)
    env_path = tmp_path / ".env"
    env_path.write_text("ANTHROPIC_API_KEY=sk-ant-old\nANTHROPIC_MODEL=claude-haiku\n")

    state = load_app_bootstrap(env_path)

    content = env_path.read_text(encoding="utf-8")
    assert state.configured is True
    assert state.needs_setup is False
    assert state.legacy_config_detected is True
    assert state.migrated is True
    assert "API_KEY=sk-ant-old" in content
    assert "MODEL=claude-haiku" in content
    assert "PLATFORM=anthropic" in content


def test_load_camera_profile_overlays_only_camera_settings(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "keep-main-key")
    monkeypatch.setenv("CAMERA_HOST", "192.168.1.10")
    profile_path = tmp_path / "camera.env"
    profile_path.write_text(
        "CAMERA_HOST=192.168.50.20\n"
        "CAMERA_PASSWORD=remote-secret\n"
        "TTS_OUTPUT=remote\n"
        "API_KEY=must-not-load\n",
        encoding="utf-8",
    )

    resolved, loaded = load_camera_profile(profile_path)

    assert resolved == profile_path.resolve()
    assert loaded == ["CAMERA_HOST", "CAMERA_PASSWORD", "TTS_OUTPUT"]
    assert os.environ["CAMERA_HOST"] == "192.168.50.20"
    assert os.environ["CAMERA_PASSWORD"] == "remote-secret"
    assert os.environ["TTS_OUTPUT"] == "remote"
    assert os.environ["API_KEY"] == "keep-main-key"
    assert os.environ[ACTIVE_CAMERA_PROFILE_ENV] == str(profile_path.resolve())


def test_load_camera_profile_clears_primary_camera_values_not_in_profile(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CAMERA_PTZ_HOST", "192.168.1.11")
    monkeypatch.setenv("CAMERA_TAPO_PASSWORD", "primary-cloud-secret")
    monkeypatch.setenv("TAPO_CAMERA_HOST", "192.168.1.10")
    profile_path = tmp_path / "camera.env"
    profile_path.write_text("CAMERA_HOST=192.168.50.20\n", encoding="utf-8")

    load_camera_profile(profile_path)

    assert os.environ["CAMERA_HOST"] == "192.168.50.20"
    assert "CAMERA_PTZ_HOST" not in os.environ
    assert "CAMERA_TAPO_PASSWORD" not in os.environ
    assert "TAPO_CAMERA_HOST" not in os.environ


def test_load_camera_profile_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Camera profile not found"):
        load_camera_profile(tmp_path / "missing.env")
