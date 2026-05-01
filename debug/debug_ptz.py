"""PTZ debug script — run with: uv run python debug_ptz.py"""

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
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


def ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def ng(msg: str) -> None:
    print(f"  [NG]  {msg}")


def info(msg: str) -> None:
    print(f"  [--]  {msg}")


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


# ── 4. PTZ move ──────────────────────────────────────────────
async def try_ptz_move(ptz: object, token: str) -> None:
    sep("4. PTZ move test (right 10°, then left 10°)")
    try:
        await asyncio.to_thread(
            ptz.RelativeMove,  # type: ignore[union-attr]
            {"ProfileToken": token, "Translation": {"PanTilt": {"x": -10 / 180.0, "y": 0.0}}},
        )
        ok("RelativeMove RIGHT succeeded")
        await asyncio.sleep(0.8)
        await asyncio.to_thread(
            ptz.RelativeMove,  # type: ignore[union-attr]
            {"ProfileToken": token, "Translation": {"PanTilt": {"x": 10 / 180.0, "y": 0.0}}},
        )
        ok("RelativeMove LEFT (return) succeeded")
    except Exception as e:
        ng(f"RelativeMove failed: {e}")
        print("  Possible cause: camera does not expose PTZ service even though ONVIF is reachable.")


async def main() -> None:
    cam, ptz, token = await run_onvif_checks()
    if ptz is None or token is None:
        return
    await try_ptz_move(ptz, token)
    sep("Done")


asyncio.run(main())
