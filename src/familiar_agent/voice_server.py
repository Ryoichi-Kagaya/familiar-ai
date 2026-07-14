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

from .agent import EmbodiedAgent, affect_to_emotion
from .config import AgentConfig
from .desires import DesireSystem
from .mental_state import AffectiveState
from .voice_guard import get_shared_voice_guard

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

        # Suppress TTS -> realtime STT self-echo that leaks through the gateway.
        # The shared guard is updated by TTSTool in this process, so recent agent
        # speech can be recognised and dropped before it re-enters the agent loop.
        guard = get_shared_voice_guard()
        decision = guard.check_transcript(text)
        if decision.blocked:
            logger.info(
                "voice_turn: suppressed self-echo (%s): %r",
                decision.reason,
                text[:60],
            )
            return web.json_response({"text": "", "emotion": "neutral"})

        # Serialize concurrent requests so the agent state machine
        # doesn't interleave two turns.
        async with self._lock:
            say_chunks: list[str] = []

            def _on_action(name: str, args: dict) -> None:
                if name == "say":
                    t = args.get("text", "")
                    if isinstance(t, str) and t:
                        say_chunks.append(t)

            try:
                final_text = (
                    await self._agent.run(
                        text,
                        on_action=_on_action,
                        desires=self._desires,
                        excluded_tools=frozenset({"listen"}),
                    )
                    or ""
                ).strip()
            except Exception as exc:
                logger.error("voice_turn: agent.run failed: %s", exc, exc_info=True)
                return web.json_response({"error": str(exc)}, status=500)

            # Prefer say() calls (the model's spoken output); fall back to the
            # final text response for models that write replies without say().
            if say_chunks:
                reply = " ".join(say_chunks).strip()
            elif final_text and final_text != "(no response)":
                reply = final_text
            else:
                reply = ""

            affect: AffectiveState | None = getattr(self._agent, "_last_affect", None)
            emotion = affect_to_emotion(affect)

        logger.info("voice_turn: text=%r reply=%r emotion=%s", text[:60], reply[:60], emotion)
        return web.json_response({"text": reply, "emotion": emotion})

    async def handle_health(self, request: web.Request) -> web.Response:
        """Handle ``GET /health`` — lightweight liveness probe, no agent call."""
        return web.json_response({"status": "ok"})

    def build_app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/health", self.handle_health)
        app.router.add_post("/voice_turn", self.handle_voice_turn)
        return app


async def run_voice_server(host: str = "0.0.0.0", port: int = 8090) -> None:
    """Start the voice server and block until interrupted."""
    config = AgentConfig()
    agent = EmbodiedAgent(config)
    desires = DesireSystem(companion_name=config.companion_name)

    # The device (StackChan) handles TTS.  Mute any local ElevenLabs playback
    # so audio comes only from the device; the say() tool stays registered so
    # the model produces proper spoken output via on_action capture.
    if agent._tts is not None:
        agent._tts.volume = 0.0

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
        # The failed site.start() already registered the old site with the runner.
        # Create a fresh TCPSite so _reg_site() doesn't raise "already registered".
        site = web.TCPSite(runner, host, port)
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
