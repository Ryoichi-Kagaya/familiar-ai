"""Batch STT debug script — run with: uv run python debug/debug_batch_stt.py

Tests:
1. Config resolution (STT_INPUT, ELEVENLABS_API_KEY, CAMERA_HOST)
2. Local mic availability (sounddevice)
3. RTSP stream availability (PyAV, if CAMERA_HOST is set)
4. Short record + transcribe round-trip (press Enter to stop recording)
"""

from __future__ import annotations

import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────

ELEVENLABS_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "ja")
STT_INPUT = os.environ.get("STT_INPUT", "auto")

CAM_HOST = os.environ.get("CAMERA_HOST", "")
CAM_USER = os.environ.get("CAMERA_USERNAME", "admin")
CAM_PASS = os.environ.get("CAMERA_PASSWORD", "")
RTSP_URL = f"rtsp://{CAM_USER}:{CAM_PASS}@{CAM_HOST}:554/stream1" if CAM_HOST else ""


def sep(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def ok(msg: str) -> None:
    print(f"  ✅  {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠️   {msg}")


def err(msg: str) -> None:
    print(f"  ❌  {msg}")


# ── 1. config ─────────────────────────────────────────────────────────────────

sep("1. Config")
print(f"  STT_INPUT        = {STT_INPUT!r}")
print(f"  STT_LANGUAGE     = {STT_LANGUAGE!r}")
print(f"  ELEVENLABS_API_KEY present = {bool(ELEVENLABS_KEY)}")
print(f"  CAMERA_HOST      = {CAM_HOST!r}")
print(f"  RTSP_URL         = {RTSP_URL!r}" if RTSP_URL else "  RTSP_URL         = (not configured)")

if not ELEVENLABS_KEY:
    err("ELEVENLABS_API_KEY is not set — transcription will fail")
    sys.exit(1)

if STT_INPUT == "remote" and not RTSP_URL:
    err("STT_INPUT=remote but CAMERA_HOST is not set")
    sys.exit(1)


# ── 2. local mic ──────────────────────────────────────────────────────────────

sep("2. Local mic (sounddevice)")
if STT_INPUT == "remote":
    warn("STT_INPUT=remote — skipping local mic check")
else:
    try:
        import sounddevice as sd

        dev = sd.query_devices(kind="input")
        ok(f"Default input device: {dev['name']!r}  ({int(dev['default_samplerate'])} Hz)")
    except Exception as exc:
        err(f"sounddevice error: {exc}")
        if STT_INPUT == "local":
            sys.exit(1)


# ── 3. RTSP stream ────────────────────────────────────────────────────────────

sep("3. RTSP stream (PyAV)")
if not RTSP_URL:
    warn("CAMERA_HOST not set — skipping RTSP check")
elif STT_INPUT == "local":
    warn("STT_INPUT=local — skipping RTSP check")
else:
    try:
        import av  # type: ignore[import]

        container = av.open(RTSP_URL, options={"rtsp_transport": "tcp"})
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if audio_streams:
            s = audio_streams[0]
            ok(
                f"RTSP audio stream found: codec={s.codec_context.name}  rate={s.codec_context.sample_rate} Hz"
            )
        else:
            warn("RTSP stream opened but has no audio track")
        container.close()
    except ImportError:
        warn("PyAV (av) not installed — install with: uv add av")
    except Exception as exc:
        err(f"RTSP connection failed: {exc}")
        if STT_INPUT == "remote":
            sys.exit(1)


# ── 4. record + transcribe ────────────────────────────────────────────────────

sep("4. Record + transcribe round-trip")

# Inline STTTool to avoid full agent startup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from familiar_agent.tools.stt import STTTool  # noqa: E402


RECORD_SECS = int(os.environ.get("STT_DEBUG_SECS", "4"))


async def run_record_test() -> None:
    tool = STTTool(ELEVENLABS_KEY, STT_LANGUAGE, RTSP_URL, STT_INPUT)
    stop = asyncio.Event()

    source_label = {"local": "PC mic", "remote": "RTSP camera", "auto": "PC mic (auto)"}.get(
        STT_INPUT, STT_INPUT
    )
    print(f"\n  Source : {source_label}")
    print(f"  Duration: {RECORD_SECS}s  (override with STT_DEBUG_SECS=N)")
    print("  Say something now…\n")

    async def auto_stop() -> None:
        await asyncio.sleep(RECORD_SECS)
        stop.set()

    _, text = await asyncio.gather(auto_stop(), tool.record_and_transcribe(stop))

    if text:
        ok(f"Transcript: {text!r}")
    else:
        warn("Transcript is empty (no speech detected, or API returned nothing)")


asyncio.run(run_record_test())

sep("Done")
