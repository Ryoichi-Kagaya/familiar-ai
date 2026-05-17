"""Unit tests for CodingTool._save_image."""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from familiar_agent.tools.coding import CodingTool


def _make_tool(workdir: str) -> CodingTool:
    config = MagicMock()
    config.workdir = workdir
    config.bash_enabled = False
    return CodingTool(config)


def test_save_image_writes_bytes(tmp_path: Path) -> None:
    tool = _make_tool(str(tmp_path))
    raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
    data = base64.b64encode(raw).decode()

    result = tool._save_image("out.png", data)

    dest = tmp_path / "out.png"
    assert dest.exists()
    assert dest.read_bytes() == raw
    assert "out.png" in result


def test_save_image_creates_parent_dirs(tmp_path: Path) -> None:
    tool = _make_tool(str(tmp_path))
    raw = b"JFIF"
    data = base64.b64encode(raw).decode()

    tool._save_image("sub/dir/img.jpg", data)

    assert (tmp_path / "sub" / "dir" / "img.jpg").exists()


def test_save_image_invalid_base64(tmp_path: Path) -> None:
    tool = _make_tool(str(tmp_path))
    result = tool._save_image("out.png", "!!!not_base64!!!")
    assert "invalid base64" in result.lower()
