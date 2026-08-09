"""Tests for TTSTool — ElevenLabs API and audio playback mocked."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helper: TTSTool without real __init__ side-effects
# ---------------------------------------------------------------------------


def _make_tts(api_key: str = "fake-key", voice_id: str = "fake-voice"):
    from familiar_agent.tools.tts import TTSTool

    with patch("familiar_agent.tools.tts._ensure_go2rtc"):
        tool = TTSTool(api_key=api_key, voice_id=voice_id, output="local")
    return tool


# ---------------------------------------------------------------------------
# Tests: get_tool_definitions()
# ---------------------------------------------------------------------------


def test_get_tool_definitions_returns_say():
    tool = _make_tts()
    defs = tool.get_tool_definitions()
    assert len(defs) == 1
    assert defs[0]["name"] == "say"


# ---------------------------------------------------------------------------
# Tests: call()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_say_invokes_say_method():
    """call('say', ...) delegates to say() and returns its result."""
    tool = _make_tts()
    tool.say = AsyncMock(return_value="Said: hello")

    result, img = await tool.call("say", {"text": "hello"})

    assert result == "Said: hello"
    assert img == []
    tool.say.assert_awaited_once_with("hello")


@pytest.mark.asyncio
async def test_call_unknown_tool_returns_error():
    tool = _make_tts()
    result, img = await tool.call("nonexistent", {})
    assert "Unknown" in result or "nonexistent" in result


# ---------------------------------------------------------------------------
# Tests: say() — API + playback mocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_say_calls_elevenlabs_api():
    """say() POSTs to ElevenLabs with the correct API key and text."""
    tool = _make_tts(api_key="test-api-key")

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read = AsyncMock(return_value=b"fake_mp3_data")
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("aiohttp.ClientSession", return_value=mock_session),
        patch("familiar_agent.tools.tts._play_local", new=AsyncMock(return_value=True)),
        patch("builtins.open", MagicMock()),
        patch("os.unlink"),
        patch("tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        tmp_file = MagicMock()
        tmp_file.__enter__ = MagicMock(return_value=tmp_file)
        tmp_file.__exit__ = MagicMock(return_value=False)
        tmp_file.name = "/tmp/fake.mp3"
        mock_tmp.return_value = tmp_file

        await tool.say("hello world")

    # Verify ElevenLabs API was called
    mock_session.post.assert_called_once()
    call_args = mock_session.post.call_args
    assert "xi-api-key" in call_args[1]["headers"] or "xi-api-key" in call_args.kwargs.get(
        "headers", {}
    )


@pytest.mark.asyncio
async def test_say_truncates_long_text():
    """say() truncates text longer than 200 characters."""
    tool = _make_tts()
    long_text = "x" * 300

    truncated_texts = []

    async def fake_say_inner(text, output=None):
        truncated_texts.append(text)
        return "Said: ..."

    # Patch at the API call level instead
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read = AsyncMock(return_value=b"fake_mp3_data")
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    posted_payloads = []

    def capture_post(url, json=None, headers=None):
        if json:
            posted_payloads.append(json)
        return mock_response

    mock_session.post = MagicMock(side_effect=capture_post)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("aiohttp.ClientSession", return_value=mock_session),
        patch("familiar_agent.tools.tts._play_local", new=AsyncMock(return_value=True)),
        patch("os.unlink"),
        patch("tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        tmp_file = MagicMock()
        tmp_file.__enter__ = MagicMock(return_value=tmp_file)
        tmp_file.__exit__ = MagicMock(return_value=False)
        tmp_file.name = "/tmp/fake.mp3"
        mock_tmp.return_value = tmp_file

        await tool.say(long_text)

    assert posted_payloads, "API was never called"
    sent_text = posted_payloads[0]["text"]
    assert len(sent_text) <= 200
    assert sent_text.endswith("...")


@pytest.mark.asyncio
async def test_say_returns_error_on_api_failure():
    """say() returns an error string when the ElevenLabs API returns non-200."""
    tool = _make_tts()

    mock_response = MagicMock()
    mock_response.status = 429
    mock_response.text = AsyncMock(return_value="Rate limit exceeded")
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("aiohttp.ClientSession", return_value=mock_session):
        result = await tool.say("hello")

    assert "429" in result or "failed" in result.lower()


@pytest.mark.asyncio
async def test_remote_failure_falls_back_to_local_playback():
    """A broken camera backchannel must not leave successful synthesis silent."""
    tool = _make_tts()
    tool.output = "remote"

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.headers = {"Content-Type": "audio/pcm"}
    mock_response.read = AsyncMock(return_value=b"\x00\x00")
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("aiohttp.ClientSession", return_value=mock_session),
        patch(
            "familiar_agent.tools.tts._play_via_go2rtc",
            return_value=(False, "can't find consumer"),
        ),
        patch(
            "familiar_agent.tools.tts._play_local", new=AsyncMock(return_value=True)
        ) as play_local,
        patch("familiar_agent.tools.tts._write_pcm_as_wav", return_value="/tmp/fake.wav"),
        patch("os.unlink"),
    ):
        result = await tool.say("hello")

    assert "via local fallback" in result
    play_local.assert_awaited_once_with("/tmp/fake.wav", volume=1.0)


def test_go2rtc_autoconfig_adds_tapo_backchannel(monkeypatch, tmp_path):
    from familiar_agent.tools import tts

    config_path = tmp_path / "go2rtc.yaml"
    monkeypatch.setattr(tts, "_GO2RTC_CACHE", tmp_path)
    monkeypatch.setattr(tts, "_GO2RTC_CONFIG", config_path)
    monkeypatch.setenv("CAMERA_HOST", "192.0.2.10")
    monkeypatch.setenv("CAMERA_USERNAME", "camera user")
    monkeypatch.setenv("CAMERA_PASSWORD", "local/pass")
    monkeypatch.setenv("CAMERA_TAPO_PASSWORD", "cloud pass")
    monkeypatch.delenv("CAMERA_TAPO_HASH", raising=False)

    tts._write_go2rtc_config("tapo_cam")

    generated = config_path.read_text()
    assert "rtsp://camera%20user:local%2Fpass@192.0.2.10/stream1" in generated
    assert "tapo://cloud%20pass@192.0.2.10" in generated


def test_go2rtc_autoconfig_preserves_manual_config_without_tapo_env(monkeypatch, tmp_path):
    from familiar_agent.tools import tts

    config_path = tmp_path / "go2rtc.yaml"
    manual = "streams:\n  custom: tapo://manual-credential@192.0.2.20\n"
    config_path.write_text(manual)
    monkeypatch.setattr(tts, "_GO2RTC_CACHE", tmp_path)
    monkeypatch.setattr(tts, "_GO2RTC_CONFIG", config_path)
    monkeypatch.setenv("CAMERA_HOST", "192.0.2.10")
    monkeypatch.setenv("CAMERA_USERNAME", "camera")
    monkeypatch.setenv("CAMERA_PASSWORD", "password")
    monkeypatch.delenv("CAMERA_TAPO_PASSWORD", raising=False)
    monkeypatch.delenv("CAMERA_TAPO_HASH", raising=False)

    tts._write_go2rtc_config("tapo_cam")

    assert config_path.read_text() == manual


def test_go2rtc_autoconfig_formats_prehashed_tapo_password(monkeypatch, tmp_path):
    from familiar_agent.tools import tts

    config_path = tmp_path / "go2rtc.yaml"
    monkeypatch.setattr(tts, "_GO2RTC_CACHE", tmp_path)
    monkeypatch.setattr(tts, "_GO2RTC_CONFIG", config_path)
    monkeypatch.setenv("CAMERA_HOST", "192.0.2.10")
    monkeypatch.delenv("CAMERA_USERNAME", raising=False)
    monkeypatch.delenv("CAMERA_PASSWORD", raising=False)
    monkeypatch.delenv("CAMERA_TAPO_PASSWORD", raising=False)
    monkeypatch.setenv("CAMERA_TAPO_HASH", "ABCDEF123")

    tts._write_go2rtc_config("tapo_cam")

    assert "tapo://admin:ABCDEF123@192.0.2.10" in config_path.read_text()


@pytest.mark.asyncio
async def test_say_notifies_voice_guard_on_success():
    tool = _make_tts()
    tool._voice_guard = MagicMock()

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read = AsyncMock(return_value=b"fake_mp3_data")
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("aiohttp.ClientSession", return_value=mock_session),
        patch("familiar_agent.tools.tts._play_local", new=AsyncMock(return_value=True)),
        patch("os.unlink"),
        patch("tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        tmp_file = MagicMock()
        tmp_file.__enter__ = MagicMock(return_value=tmp_file)
        tmp_file.__exit__ = MagicMock(return_value=False)
        tmp_file.name = "/tmp/fake.mp3"
        mock_tmp.return_value = tmp_file

        await tool.say("hello world")

    tool._voice_guard.on_tts_start.assert_called_once_with("hello world")
    tool._voice_guard.on_tts_end.assert_called_once_with("hello world", played=True)


@pytest.mark.asyncio
async def test_say_serializes_concurrent_calls():
    """Concurrent say() calls must be serialized (lock prevents overlap)."""
    tool = _make_tts()

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read = AsyncMock(return_value=b"fake")
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=False)

    mock_session = MagicMock()
    mock_session.post = MagicMock(return_value=mock_response)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("aiohttp.ClientSession", return_value=mock_session),
        patch("familiar_agent.tools.tts._play_local", new=AsyncMock(return_value=True)),
        patch("os.unlink"),
        patch("tempfile.NamedTemporaryFile") as mock_tmp,
    ):
        tmp_file = MagicMock()
        tmp_file.__enter__ = MagicMock(return_value=tmp_file)
        tmp_file.__exit__ = MagicMock(return_value=False)
        tmp_file.name = "/tmp/fake.mp3"
        mock_tmp.return_value = tmp_file

        # Launch two say() calls concurrently
        results = await asyncio.gather(
            tool.say("first"),
            tool.say("second"),
        )

    # Both should succeed (no exception)
    assert len(results) == 2


# ---------------------------------------------------------------------------
# Tests: say() — volume == 0 (audio delegated to device)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_say_volume_zero_returns_said_snippet():
    """When volume is 0 say() returns 'Said: <text> (audio handled by device)'."""
    tool = _make_tts()
    tool.volume = 0.0
    tool._voice_guard = MagicMock()

    result = await tool.say("hello world")
    assert result == "Said: hello world (audio handled by device)"
    tool._voice_guard.on_tts_start.assert_called_once_with("hello world")
    tool._voice_guard.on_tts_end.assert_called_once_with("hello world", played=True)


@pytest.mark.asyncio
async def test_say_volume_zero_truncates_long_text():
    """Long text is truncated to 50 chars + '...' in the volume-0 message."""
    tool = _make_tts()
    tool.volume = 0.0
    tool._voice_guard = MagicMock()
    long_text = "a" * 60

    result = await tool.say(long_text)
    assert result == f"Said: {'a' * 50}... (audio handled by device)"
    tool._voice_guard.on_tts_start.assert_called_once_with(long_text)
    tool._voice_guard.on_tts_end.assert_called_once_with(long_text, played=True)
