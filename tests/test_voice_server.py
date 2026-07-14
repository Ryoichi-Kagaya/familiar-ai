"""Tests for VoiceServer emotion mapping and request handling."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from aiohttp.test_utils import TestClient, TestServer

from familiar_agent.voice_server import VoiceServer
from familiar_agent.agent import affect_to_emotion
from familiar_agent.mental_state import AffectiveState
from familiar_agent.voice_guard import VoiceLoopGuard


# ── affect_to_emotion ────────────────────────────────────────────────────────


def test_emotion_none_returns_neutral():
    assert affect_to_emotion(None) == "neutral"


@pytest.mark.parametrize(
    "valence,arousal,expected",
    [
        (0.5, 0.1, "happy"),  # positive valence → happy regardless of arousal
        (0.15, 0.6, "happy"),
        (0.11, 0.0, "happy"),
        (0.05, 0.8, "neutral"),  # within neutral band
        (0.0, 0.5, "neutral"),
        (-0.05, 0.2, "neutral"),
        (-0.5, 0.6, "angry"),  # negative + high arousal
        (-0.2, 0.35, "angry"),  # boundary: a == 0.35 → angry
        (-0.5, 0.3, "sad"),  # negative + low arousal
        (-0.15, 0.0, "sad"),
    ],
)
def test_emotion_mapping(valence: float, arousal: float, expected: str):
    affect = AffectiveState(valence=valence, arousal=arousal)
    assert affect_to_emotion(affect) == expected


# ── VoiceServer.handle_voice_turn ─────────────────────────────────────────────


def _make_server(valence: float = 0.3, use_say_tool: bool = False) -> VoiceServer:
    agent = MagicMock()
    agent._last_affect = AffectiveState(valence=valence)

    async def _fake_run(text, on_action=None, desires=None, **_kw):
        if on_action and use_say_tool:
            # Simulate model calling say() tool
            on_action("say", {"text": "hello"})
        return "hello"

    agent.run = AsyncMock(side_effect=_fake_run)
    desires = MagicMock()
    return VoiceServer(agent=agent, desires=desires)


@pytest.mark.asyncio
async def test_valid_request_returns_text_and_emotion():
    vs = _make_server(valence=0.3)
    async with TestClient(TestServer(vs.build_app())) as client:
        resp = await client.post("/voice_turn", json={"text": "こんにちは"})
        assert resp.status == 200
        data = await resp.json()
        assert data["text"] == "hello"
        assert data["emotion"] == "happy"


@pytest.mark.asyncio
async def test_say_tool_takes_priority_over_final_text():
    """say() chunks from on_action take priority over the agent's final text."""
    vs = _make_server(valence=0.3, use_say_tool=True)
    async with TestClient(TestServer(vs.build_app())) as client:
        resp = await client.post("/voice_turn", json={"text": "こんにちは"})
        assert resp.status == 200
        data = await resp.json()
        # say() tool chunk ("hello") is returned, not raw final_text
        assert data["text"] == "hello"


@pytest.mark.asyncio
async def test_invalid_json_returns_400():
    vs = _make_server()
    async with TestClient(TestServer(vs.build_app())) as client:
        resp = await client.post(
            "/voice_turn",
            data="not json",
            headers={"Content-Type": "application/json"},
        )
        assert resp.status == 400
        data = await resp.json()
        assert "error" in data


@pytest.mark.asyncio
async def test_empty_text_returns_400():
    vs = _make_server()
    async with TestClient(TestServer(vs.build_app())) as client:
        resp = await client.post("/voice_turn", json={"text": ""})
        assert resp.status == 400


@pytest.mark.asyncio
async def test_missing_text_field_returns_400():
    vs = _make_server()
    async with TestClient(TestServer(vs.build_app())) as client:
        resp = await client.post("/voice_turn", json={"message": "hi"})
        assert resp.status == 400


@pytest.mark.asyncio
async def test_no_last_affect_falls_back_to_neutral():
    vs = _make_server()
    del vs._agent._last_affect
    async with TestClient(TestServer(vs.build_app())) as client:
        resp = await client.post("/voice_turn", json={"text": "test"})
        assert resp.status == 200
        data = await resp.json()
        assert data["emotion"] == "neutral"


@pytest.mark.asyncio
async def test_health_endpoint_returns_ok_without_agent_call():
    vs = _make_server()
    async with TestClient(TestServer(vs.build_app())) as client:
        resp = await client.get("/health")
        assert resp.status == 200
        data = await resp.json()
        assert data == {"status": "ok"}
    vs._agent.run.assert_not_called()


@pytest.mark.asyncio
async def test_voice_turn_suppresses_self_echo(monkeypatch):
    """Recent agent speech echoed back by realtime STT should be dropped."""
    guard = VoiceLoopGuard(suppression_window_secs=1.0)
    guard.on_tts_start("test")
    guard.on_tts_end("test", played=True)

    monkeypatch.setattr(
        "familiar_agent.voice_server.get_shared_voice_guard",
        lambda: guard,
    )

    vs = _make_server()
    async with TestClient(TestServer(vs.build_app())) as client:
        resp = await client.post("/voice_turn", json={"text": "test"})
        assert resp.status == 200
        data = await resp.json()
        assert data["text"] == ""
        assert data["emotion"] == "neutral"
    vs._agent.run.assert_not_called()
