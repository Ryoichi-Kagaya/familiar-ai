"""Tests for CameraTool — OpenCV mocked, no real hardware required."""

from __future__ import annotations

import base64
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_camera_tool(host: str = "192.168.1.100"):
    """Create a CameraTool with the capture thread patched out."""
    from familiar_agent.tools.camera import CameraTool

    with patch.object(CameraTool, "start"):
        cam = CameraTool.__new__(CameraTool)
        cam.host = host
        cam.username = "admin"
        cam.password = "password"
        cam.port = 2020
        cam.preview = False
        cam.ptz_host = host
        cam.ptz_username = "admin"
        cam.ptz_password = "password"
        cam.ptz_port = 2020
        cam._cam_onvif = None
        cam._ptz = None
        cam._profile_token = None
        cam._ptz_connect_failed_at = 0.0
        cam._cap = None
        cam._last_frame = None
        cam._running = False
        cam._thread = None
        cam._lock = threading.Lock()
    return cam


def _make_fake_frame(height: int = 480, width: int = 640) -> "np.ndarray":
    """Create a fake BGR numpy frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def _make_mock_onvif_cam(profile_token: str = "profile_1"):
    """Return a MagicMock that mimics onvif-zeep-async >=4.x (all sync methods)."""
    mock_profile = MagicMock()
    mock_profile.token = profile_token

    mock_media = MagicMock()
    mock_media.GetProfiles.return_value = [mock_profile]

    mock_ptz = MagicMock()

    mock_cam = MagicMock()
    mock_cam.update_xaddrs = MagicMock(return_value=None)  # sync, returns None
    mock_cam.create_media_service.return_value = mock_media
    mock_cam.create_ptz_service.return_value = mock_ptz

    return mock_cam, mock_ptz, mock_profile.token


# ---------------------------------------------------------------------------
# Tests: capture()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_capture_returns_none_when_no_frame():
    """capture() returns (None, None) when no frame has been grabbed yet."""
    cam = _make_camera_tool()
    cam._last_frame = None

    b64, path = await cam.capture()

    assert b64 is None
    assert path is None


@pytest.mark.asyncio
async def test_capture_returns_base64_jpeg_on_valid_frame():
    """capture() encodes a valid frame to base64 JPEG."""
    cam = _make_camera_tool()
    cam._last_frame = _make_fake_frame()

    b64, path = await cam.capture()

    assert b64 is not None
    # Should be valid base64
    decoded = base64.b64decode(b64)
    # JPEG magic bytes: FF D8 FF
    assert decoded[:2] == b"\xff\xd8"


@pytest.mark.asyncio
async def test_capture_saves_file_to_disk(tmp_path):
    """capture() writes the JPEG to disk and returns the path."""
    import familiar_agent.tools.camera as camera_module

    cam = _make_camera_tool()
    cam._last_frame = _make_fake_frame()

    original_dir = camera_module.CAPTURE_DIR
    camera_module.CAPTURE_DIR = tmp_path
    try:
        b64, path = await cam.capture()
    finally:
        camera_module.CAPTURE_DIR = original_dir

    assert path is not None
    assert b64 is not None
    assert (tmp_path / path.split("/")[-1]).exists()


@pytest.mark.asyncio
async def test_capture_resizes_large_frame():
    """capture() resizes frames taller than 640px."""
    cam = _make_camera_tool()
    cam._last_frame = _make_fake_frame(height=1080, width=1920)

    b64, _ = await cam.capture()

    assert b64 is not None
    # Decode and verify the JPEG was produced (resize didn't crash)
    decoded = base64.b64decode(b64)
    assert decoded[:2] == b"\xff\xd8"


# ---------------------------------------------------------------------------
# Tests: call()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_see_returns_image_on_success():
    """call('see', {}) returns (description, base64_image) on success."""
    cam = _make_camera_tool()

    async def _fake_capture():
        return "fakeb64", "/tmp/capture_123.jpg"

    cam.capture = _fake_capture

    result, img = await cam.call("see", {})

    assert img == ["fakeb64"]
    assert "saved to" in result


@pytest.mark.asyncio
async def test_call_see_returns_error_when_capture_fails():
    """call('see', {}) returns failure message when capture returns (None, None)."""
    cam = _make_camera_tool()

    async def _fake_capture():
        return None, None

    cam.capture = _fake_capture

    result, img = await cam.call("see", {})

    assert img == []
    assert "failed" in result.lower()


@pytest.mark.asyncio
async def test_call_look_delegates_to_move():
    """call('look', ...) delegates to move() and returns its result."""
    cam = _make_camera_tool()

    async def _fake_move(direction, degrees=30):
        return f"Moved {direction} by {degrees}°"

    cam.move = _fake_move

    result, img = await cam.call("look", {"direction": "left", "degrees": 45})

    assert "left" in result
    assert "45" in result
    assert img == []


@pytest.mark.asyncio
async def test_call_unknown_tool_returns_error():
    """call() with an unrecognized tool name returns an error string."""
    cam = _make_camera_tool()

    result, img = await cam.call("nonexistent", {})

    assert "Unknown" in result or "nonexistent" in result


# ---------------------------------------------------------------------------
# Tests: _ensure_connected()  — onvif-zeep-async >=4.x sync API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_connected_success():
    """_ensure_connected() wires up _cam_onvif, _ptz, and _profile_token."""
    cam = _make_camera_tool()
    mock_cam, mock_ptz, token = _make_mock_onvif_cam("profile_1")

    with patch("familiar_agent.tools.camera.ONVIFCamera", return_value=mock_cam):
        result = await cam._ensure_connected()

    assert result is True
    assert cam._cam_onvif is mock_cam
    assert cam._ptz is mock_ptz
    assert cam._profile_token == "profile_1"
    assert cam._ptz_connect_failed_at == 0.0


@pytest.mark.asyncio
async def test_ensure_connected_already_connected():
    """_ensure_connected() returns True immediately if already connected."""
    cam = _make_camera_tool()
    cam._cam_onvif = MagicMock()

    with patch("familiar_agent.tools.camera.ONVIFCamera") as mock_cls:
        result = await cam._ensure_connected()

    mock_cls.assert_not_called()
    assert result is True


@pytest.mark.asyncio
async def test_ensure_connected_failure_sets_cooldown():
    """_ensure_connected() records failure time when all ports fail."""
    cam = _make_camera_tool()
    mock_cam = MagicMock()
    mock_cam.update_xaddrs.side_effect = ConnectionError("connection refused")

    with patch("familiar_agent.tools.camera.ONVIFCamera", return_value=mock_cam):
        result = await cam._ensure_connected()

    assert result is False
    assert cam._ptz_connect_failed_at > 0.0
    assert cam._cam_onvif is None


@pytest.mark.asyncio
async def test_ensure_connected_respects_cooldown():
    """_ensure_connected() skips all retries within the 60 s cooldown window."""
    cam = _make_camera_tool()
    cam._ptz_connect_failed_at = time.monotonic()  # just failed

    with patch("familiar_agent.tools.camera.ONVIFCamera") as mock_cls:
        result = await cam._ensure_connected()

    mock_cls.assert_not_called()
    assert result is False


@pytest.mark.asyncio
async def test_ensure_connected_retries_after_cooldown_expires():
    """_ensure_connected() retries once the cooldown window has passed."""
    cam = _make_camera_tool()
    cam._ptz_connect_failed_at = time.monotonic() - 61.0  # expired

    mock_cam, mock_ptz, _ = _make_mock_onvif_cam()

    with patch("familiar_agent.tools.camera.ONVIFCamera", return_value=mock_cam):
        result = await cam._ensure_connected()

    assert result is True


@pytest.mark.asyncio
async def test_ensure_connected_skips_usb_host():
    """_ensure_connected() returns False for integer (USB) camera sources."""
    cam = _make_camera_tool(host="0")
    cam.ptz_host = "0"

    with patch("familiar_agent.tools.camera.ONVIFCamera") as mock_cls:
        result = await cam._ensure_connected()

    mock_cls.assert_not_called()
    assert result is False


# ---------------------------------------------------------------------------
# Tests: move()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_move_right_sends_negative_pan():
    """move('right', 30) sends a negative x pan to RelativeMove."""
    cam = _make_camera_tool()
    cam._cam_onvif = MagicMock()
    cam._ptz = MagicMock()
    cam._ptz.RelativeMove = MagicMock(return_value=None)
    cam._profile_token = "profile_1"

    result = await cam.move("right", 30)

    assert "right" in result
    assert "30" in result
    called_params = cam._ptz.RelativeMove.call_args[0][0]
    assert called_params["Translation"]["PanTilt"]["x"] < 0


@pytest.mark.asyncio
async def test_move_left_sends_positive_pan():
    """move('left', 30) sends a positive x pan to RelativeMove."""
    cam = _make_camera_tool()
    cam._cam_onvif = MagicMock()
    cam._ptz = MagicMock()
    cam._ptz.RelativeMove = MagicMock(return_value=None)
    cam._profile_token = "profile_1"

    result = await cam.move("left", 30)

    assert "left" in result
    called_params = cam._ptz.RelativeMove.call_args[0][0]
    assert called_params["Translation"]["PanTilt"]["x"] > 0


@pytest.mark.asyncio
async def test_move_resets_connection_on_failure():
    """move() clears _cam_onvif so the next call retries the connection."""
    cam = _make_camera_tool()
    cam._cam_onvif = MagicMock()
    cam._ptz = MagicMock()
    cam._ptz.RelativeMove = MagicMock(side_effect=RuntimeError("PTZ error"))
    cam._profile_token = "profile_1"

    result = await cam.move("left", 30)

    assert "failed" in result.lower()
    assert cam._cam_onvif is None


@pytest.mark.asyncio
async def test_move_returns_unsupported_when_ptz_unavailable():
    """move() returns a clear message when PTZ connection cannot be established."""
    cam = _make_camera_tool()
    mock_cam = MagicMock()
    mock_cam.update_xaddrs.side_effect = ConnectionError("refused")

    with patch("familiar_agent.tools.camera.ONVIFCamera", return_value=mock_cam):
        result = await cam.move("right", 30)

    assert "not supported" in result.lower()


# ---------------------------------------------------------------------------
# Tests: PTZ connection params
# ---------------------------------------------------------------------------


def test_ptz_params_fall_back_to_stream_url_credentials():
    cam = _make_camera_tool("rtsp://stream-user:stream-pass@192.168.1.206/live0")
    cam.username = ""
    cam.password = ""
    cam.ptz_host = cam.host
    cam.ptz_username = ""
    cam.ptz_password = ""

    host, username, password, port = cam._get_ptz_connection_params()

    assert host == "192.168.1.206"
    assert username == "stream-user"
    assert password == "stream-pass"
    assert port == 2020


def test_ptz_params_prefer_explicit_overrides():
    cam = _make_camera_tool("rtsp://stream-user:stream-pass@192.168.1.206/live0")
    cam.ptz_host = "192.168.1.145"
    cam.ptz_username = "ptz-user"
    cam.ptz_password = "ptz-pass"
    cam.ptz_port = 8899

    host, username, password, port = cam._get_ptz_connection_params()

    assert host == "192.168.1.145"
    assert username == "ptz-user"
    assert password == "ptz-pass"
    assert port == 8899
