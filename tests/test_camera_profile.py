from __future__ import annotations

from pathlib import Path

import pytest

from familiar_agent.main import _camera_profile_from_argv


def test_camera_profile_argument_accepts_separate_path() -> None:
    assert _camera_profile_from_argv(["--gui", "--camera-profile", "remote.env"]) == Path(
        "remote.env"
    )


def test_camera_profile_argument_accepts_equals_path() -> None:
    assert _camera_profile_from_argv(["--camera-profile=remote.env"]) == Path("remote.env")


@pytest.mark.parametrize("argv", [["--camera-profile"], ["--camera-profile="]])
def test_camera_profile_argument_requires_path(argv: list[str]) -> None:
    with pytest.raises(ValueError, match="requires a file path"):
        _camera_profile_from_argv(argv)
