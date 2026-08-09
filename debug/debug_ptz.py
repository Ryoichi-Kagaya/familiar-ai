"""PTZ debug script — run with: uv run python debug/debug_ptz.py

Tests both pan (left/right) and tilt (up/down), and dumps PTZ capabilities
and current position so you can diagnose why tilt might not move.
"""

from __future__ import annotations

import asyncio
import os
import socket
import sys

from dotenv import load_dotenv

load_dotenv()

HOST = os.environ.get("CAMERA_HOST", "")
USER = os.environ.get("CAMERA_USERNAME", "admin")
PASS = os.environ.get("CAMERA_PASSWORD", "")
PTZ_HOST = os.environ.get("CAMERA_PTZ_HOST", "") or HOST
PTZ_USER = os.environ.get("CAMERA_PTZ_USERNAME", "") or USER
PTZ_PASS = os.environ.get("CAMERA_PTZ_PASSWORD", "") or PASS
PTZ_PORT = int(os.environ.get("CAMERA_PTZ_PORT", os.environ.get("CAMERA_ONVIF_PORT", "2020")))

PORTS_TO_TRY = [PTZ_PORT] + [p for p in (2020, 8080, 80) if p != PTZ_PORT]


def sep(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def ng(msg: str) -> None:
    print(f"  [NG]  {msg}")


def info(msg: str) -> None:
    print(f"  [--]  {msg}")


def warn(msg: str) -> None:
    print(f"  [!!]  {msg}")


# ── 1. Config dump ────────────────────────────────────────────
sep("1. Config (.env)")
info(f"CAMERA_HOST       = {HOST!r}")
info(f"PTZ_HOST          = {PTZ_HOST!r}")
info(f"PTZ_USER          = {PTZ_USER!r}")
info(f"PTZ_PASS          = {'***' if PTZ_PASS else '(empty)'}")
info(f"PTZ_PORT (base)   = {PTZ_PORT}")
info(f"Ports to try      = {PORTS_TO_TRY}")

if not HOST:
    print("\n[FATAL] CAMERA_HOST not set in .env. Aborting.")
    sys.exit(1)

# ── 2. TCP reachability ───────────────────────────────────────
sep("2. TCP reachability")
reachable_ports: list[int] = []
for port in PORTS_TO_TRY:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(3)
    result = sock.connect_ex((PTZ_HOST, port))
    sock.close()
    if result == 0:
        ok(f"port {port} is OPEN")
        reachable_ports.append(port)
    else:
        ng(f"port {port} is CLOSED/FILTERED (errno={result})")

if not reachable_ports:
    print("\n[FATAL] No ONVIF ports are reachable. Check camera IP and network.")
    sys.exit(1)


# ── 3. ONVIF connect ─────────────────────────────────────────
async def try_onvif(host: str, port: int, user: str, pwd: str) -> tuple[bool, object | None, str]:
    import onvif
    from onvif import ONVIFCamera

    onvif_dir = os.path.dirname(onvif.__file__)
    wsdl_dir = os.path.join(onvif_dir, "wsdl")
    if not os.path.isdir(wsdl_dir):
        wsdl_dir = os.path.join(os.path.dirname(onvif_dir), "wsdl")

    try:
        cam = ONVIFCamera(host, port, user, pwd, wsdl_dir=wsdl_dir)
        # onvif-zeep-async >=4.x: methods are synchronous — run in thread pool
        await asyncio.to_thread(cam.update_xaddrs)
        media = cam.create_media_service()
        profiles = await asyncio.to_thread(media.GetProfiles)
        token = profiles[0].token if profiles else "Profile_1"
        ptz = cam.create_ptz_service()
        return True, (cam, ptz, token), ""
    except Exception as e:
        return False, None, str(e)


async def run_onvif_checks() -> tuple[object | None, object | None, str | None]:
    sep("3. ONVIF connect")
    for port in reachable_ports:
        info(f"Trying ONVIF on port {port} ...")
        success, result, err = await try_onvif(PTZ_HOST, port, PTZ_USER, PTZ_PASS)
        if success:
            cam, ptz, token = result  # type: ignore[misc]
            ok(f"ONVIF connected on port {port}  (profile token={token!r})")
            return cam, ptz, token
        else:
            ng(f"port {port} ONVIF failed: {err}")

    print("\n[FATAL] ONVIF connection failed on all reachable ports.")
    print("  Possible causes:")
    print("  - Wrong username/password for the camera local account")
    print("    (Tapo requires a LOCAL account created in the app, not the TP-Link cloud account)")
    print("  - ONVIF/local access is disabled in camera settings")
    print("  - The camera model does not support ONVIF PTZ")
    return None, None, None


# ── 4. PTZ capabilities ──────────────────────────────────────
async def check_ptz_capabilities(ptz: object, token: str) -> None:
    sep("4. PTZ capabilities")

    # GetConfigurations: shows pan/tilt/zoom limits
    try:
        configs = await asyncio.to_thread(ptz.GetConfigurations)  # type: ignore[union-attr]
        for cfg in configs:
            info(
                f"Config name: {getattr(cfg, 'Name', '?')!r}  token: {getattr(cfg, 'token', '?')!r}"
            )
            limits = getattr(cfg, "PanTiltLimits", None)
            if limits:
                r = getattr(limits, "Range", None)
                if r:
                    xr = getattr(r, "XRange", None)
                    yr = getattr(r, "YRange", None)
                    if xr:
                        info(
                            f"  Pan  (x) range : Min={getattr(xr, 'Min', '?')}  Max={getattr(xr, 'Max', '?')}"
                        )
                    if yr:
                        info(
                            f"  Tilt (y) range : Min={getattr(yr, 'Min', '?')}  Max={getattr(yr, 'Max', '?')}"
                        )
                        y_min = getattr(yr, "Min", None)
                        y_max = getattr(yr, "Max", None)
                        if (
                            y_min is not None
                            and y_max is not None
                            and float(y_min) == float(y_max) == 0.0
                        ):
                            warn("Tilt range is 0–0: this camera may NOT support tilt!")
            else:
                warn("No PanTiltLimits in configuration (tilt may be unsupported)")
        ok("GetConfigurations succeeded")
    except Exception as e:
        ng(f"GetConfigurations failed: {e}")

    # GetStatus: current pan/tilt position
    try:
        status = await asyncio.to_thread(
            ptz.GetStatus,  # type: ignore[union-attr]
            {"ProfileToken": token},
        )
        pos = getattr(status, "Position", None)
        if pos:
            pt = getattr(pos, "PanTilt", None)
            if pt:
                info(
                    f"Current position — Pan(x)={getattr(pt, 'x', '?')}  Tilt(y)={getattr(pt, 'y', '?')}"
                )
            else:
                warn("Position.PanTilt not present in GetStatus response")
        ok("GetStatus succeeded")
    except Exception as e:
        ng(f"GetStatus failed: {e}")

    # GetNodes: exposes supported move modes
    try:
        nodes = await asyncio.to_thread(ptz.GetNodes)  # type: ignore[union-attr]
        for node in nodes:
            info(f"Node: {getattr(node, 'Name', '?')!r}")
            supported = getattr(node, "SupportedPTZSpaces", None)
            if supported:
                rel_spaces = getattr(supported, "RelativePanTiltTranslationSpace", [])
                if rel_spaces:
                    ok(
                        f"  RelativeMove PanTilt spaces: {[getattr(s, 'URI', s) for s in rel_spaces]}"
                    )
                else:
                    warn(
                        "  No RelativePanTiltTranslationSpace — RelativeMove may not be supported!"
                    )
        ok("GetNodes succeeded")
    except Exception as e:
        ng(f"GetNodes failed: {e}")


# ── 5. Pan test ──────────────────────────────────────────────
async def try_pan_move(ptz: object, token: str) -> None:
    sep("5. Pan test (right 10°, then left 10° to return)")
    # ONVIF convention: positive x = right, negative x = left
    # camera.py inverts this (left=positive, right=negative) — pan reportedly works,
    # so this camera likely uses the inverted convention.
    info("Using camera.py sign convention: right=-0.056, left=+0.056")
    try:
        await asyncio.to_thread(
            ptz.RelativeMove,  # type: ignore[union-attr]
            {"ProfileToken": token, "Translation": {"PanTilt": {"x": -10 / 180.0, "y": 0.0}}},
        )
        ok("RelativeMove pan RIGHT (-x) succeeded")
        await asyncio.sleep(1.0)
        await asyncio.to_thread(
            ptz.RelativeMove,  # type: ignore[union-attr]
            {"ProfileToken": token, "Translation": {"PanTilt": {"x": 10 / 180.0, "y": 0.0}}},
        )
        ok("RelativeMove pan LEFT (+x, return) succeeded")
        await asyncio.sleep(1.0)
    except Exception as e:
        ng(f"Pan RelativeMove failed: {e}")


# ── 6. Tilt test ─────────────────────────────────────────────
async def try_tilt_move(ptz: object, token: str) -> None:
    sep("6. Tilt test — four variants to find which sign works")

    # camera.py uses: up → y = -degrees/90, down → y = +degrees/90
    # Standard ONVIF:  up → y = +degrees/90, down → y = -degrees/90
    # We test both so you can see which actually moves the camera.

    moves: list[tuple[str, float]] = [
        ("camera.py 'up'   (y = -0.111, should tilt UP)", -10 / 90.0),
        ("camera.py 'down' (y = +0.111, should tilt DOWN)", +10 / 90.0),
    ]

    for label, y_val in moves:
        info(f"Sending y={y_val:.4f}  [{label}]")
        try:
            await asyncio.to_thread(
                ptz.RelativeMove,  # type: ignore[union-attr]
                {"ProfileToken": token, "Translation": {"PanTilt": {"x": 0.0, "y": y_val}}},
            )
            ok(f"RelativeMove succeeded (y={y_val:.4f}) — did camera move?")
        except Exception as e:
            ng(f"RelativeMove failed (y={y_val:.4f}): {e}")
        await asyncio.sleep(1.5)

    # After both moves the net displacement should be ~0 (returned to start).
    # If neither moved, the camera likely reports OK but ignores tilt.
    print()
    warn("Check physically: did the camera tilt at all during step 6?")
    warn("If it moved on 'down' but not 'up', camera.py signs are correct.")
    warn("If it moved on 'up' but not 'down', signs are inverted (bug in camera.py).")
    warn("If neither moved, tilt is hardware-unsupported or requires a different move type.")


# ── 7. AbsoluteMove tilt fallback ────────────────────────────
async def try_absolute_tilt(ptz: object, token: str) -> None:
    sep("7. AbsoluteMove tilt fallback (some cameras ignore RelativeMove tilt)")
    info("Sending AbsoluteMove to tilt position y=0.3 (up), then y=0.0 (center)")
    try:
        await asyncio.to_thread(
            ptz.AbsoluteMove,  # type: ignore[union-attr]
            {"ProfileToken": token, "Position": {"PanTilt": {"x": 0.0, "y": 0.3}}},
        )
        ok("AbsoluteMove y=0.3 succeeded — did camera tilt up?")
        await asyncio.sleep(1.5)
        await asyncio.to_thread(
            ptz.AbsoluteMove,  # type: ignore[union-attr]
            {"ProfileToken": token, "Position": {"PanTilt": {"x": 0.0, "y": 0.0}}},
        )
        ok("AbsoluteMove y=0.0 (center) succeeded")
    except Exception as e:
        ng(f"AbsoluteMove failed: {e}")
        info("AbsoluteMove not supported — only RelativeMove is available.")


async def main() -> None:
    cam, ptz, token = await run_onvif_checks()
    if ptz is None or token is None:
        return

    await check_ptz_capabilities(ptz, token)
    await try_pan_move(ptz, token)
    await try_tilt_move(ptz, token)
    await try_absolute_tilt(ptz, token)

    sep("Done — review [!!] warnings above for tilt diagnosis")


asyncio.run(main())
