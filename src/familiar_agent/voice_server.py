"""Minimal aiohttp HTTP server exposing familiar-ai as a voice_turn endpoint.

Start with ``familiar --voice-server`` (or ``--voice-server --port 8090``).
The stackchan-mcp gateway calls this endpoint for each transcribed utterance
it receives from the device in xiaozhi AI agent mode.

Endpoint
--------
POST /voice_turn
  Request:  {"text": "<transcribed speech>"}
  Response: {"text": "<agent reply>", "emotion": "<happy|neutral|sad|angry>"}
"""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import signal

from aiohttp import web

from .agent import EmbodiedAgent
from .config import AgentConfig
from .desires import DesireSystem
from .mental_state import AffectiveState

logger = logging.getLogger(__name__)


def _find_pid_using_port(port: int) -> int | None:
    """Return the PID listening on *port* by parsing /proc/net/tcp, or None."""
    hex_port = f"{port:04X}"
    try:
        with open("/proc/net/tcp") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 10 or parts[3] != "0A":  # 0A = LISTEN
                    continue
                _, lport = parts[1].split(":")
                if lport.upper() != hex_port:
                    continue
                inode = int(parts[9])
                for pid_str in os.listdir("/proc"):
                    if not pid_str.isdigit():
                        continue
                    fd_dir = f"/proc/{pid_str}/fd"
                    try:
                        for fd in os.listdir(fd_dir):
                            try:
                                if f"socket:[{inode}]" in os.readlink(f"{fd_dir}/{fd}"):
                                    return int(pid_str)
                            except OSError:
                                pass
                    except OSError:
                        pass
    except OSError:
        pass
    return None


async def _release_port(port: int) -> bool:
    """Kill the process holding *port*. Returns True once the port is free."""
    pid = _find_pid_using_port(port)
    if pid is None:
        return True
    logger.warning("port %d held by PID %d — sending SIGTERM", port, pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    for _ in range(15):
        await asyncio.sleep(0.2)
        if _find_pid_using_port(port) is None:
            return True
    logger.warning("PID %d did not exit cleanly — sending SIGKILL", pid)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    await asyncio.sleep(0.5)
    return _find_pid_using_port(port) is None


def _affect_to_emotion(affect: AffectiveState | None) -> str:
    """Map familiar-ai appraisal state to a xiaozhi emotion string.

    Valence drives the positive/negative split; arousal separates angry from sad.
    Arousal from joy alone peaks around 0.14 per pattern match, so the old 0.5
    threshold made "happy" unreachable in practice.

      valence > 0.1              → happy
      valence < -0.1, arousal >= 0.35 → angry
      valence < -0.1, arousal  < 0.35 → sad
      otherwise                  → neutral
    """
    if affect is None:
        return "neutral"
    v: float = affect.valence
    a: float = affect.arousal
    if v > 0.1:
        return "happy"
    if v < -0.1 and a >= 0.35:
        return "angry"
    if v < -0.1:
        return "sad"
    return "neutral"


class VoiceServer:
    """aiohttp server wrapping an :class:`EmbodiedAgent` turn loop."""

    def __init__(self, agent: EmbodiedAgent, desires: DesireSystem) -> None:
        self._agent = agent
        self._desires = desires
        self._lock = asyncio.Lock()

    async def handle_voice_turn(self, request: web.Request) -> web.Response:
        """Handle ``POST /voice_turn``."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)

        text: str = body.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return web.json_response({"error": "'text' must be a non-empty string"}, status=400)

        # Serialize concurrent requests so the agent state machine
        # doesn't interleave two turns.
        async with self._lock:
            collected: list[str] = []

            def _on_text(chunk: str) -> None:
                collected.append(chunk)

            try:
                await self._agent.run(
                    text,
                    on_text=_on_text,
                    desires=self._desires,
                )
            except Exception as exc:
                logger.error("voice_turn: agent.run failed: %s", exc, exc_info=True)
                return web.json_response({"error": str(exc)}, status=500)

            reply = "".join(collected).strip()
            affect: AffectiveState | None = getattr(self._agent, "_last_affect", None)
            emotion = _affect_to_emotion(affect)

        logger.info("voice_turn: text=%r reply=%r emotion=%s", text[:60], reply[:60], emotion)
        return web.json_response({"text": reply, "emotion": emotion})

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/voice_turn", self.handle_voice_turn)
        return app


async def run_voice_server(host: str = "0.0.0.0", port: int = 8090) -> None:
    """Start the voice server and block until interrupted."""
    config = AgentConfig()
    agent = EmbodiedAgent(config)
    desires = DesireSystem(companion_name=config.companion_name)

    server = VoiceServer(agent, desires)
    app = server.build_app()

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    try:
        await site.start()
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        logger.warning("port %d already in use — attempting to replace existing instance", port)
        if not await _release_port(port):
            raise RuntimeError(
                f"port {port} is still in use after trying to free it; cannot start voice server"
            ) from exc
        await site.start()

    logger.info("VoiceServer listening on http://%s:%d/voice_turn", host, port)
    print(f"familiar voice server: http://{host}:{port}/voice_turn")

    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            pass  # Windows

    await stop_event.wait()
    try:
        await runner.cleanup()
    finally:
        await agent.close()
