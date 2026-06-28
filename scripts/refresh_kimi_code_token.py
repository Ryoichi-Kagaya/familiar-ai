#!/usr/bin/env python3
"""Refresh API_KEY in .env from Kimi Code CLI's OAuth credentials.

Kimi Code CLI stores a short-lived OAuth access token in
~/.kimi-code/credentials/kimi-code.json. familiar-ai can reuse that token
to call the same managed model endpoint (https://api.kimi.com/coding/v1).

This script refreshes the access token directly against Kimi's OAuth token
endpoint, writes the new token back to the credentials file, and updates
.env so familiar-ai keeps working without manual intervention.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

KIMI_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
KIMI_TOKEN_ENDPOINT = "https://auth.kimi.com/api/oauth/token"
DEFAULT_MARGIN_SECONDS = 300  # refresh if within 5 min of expiry


def _device_headers(home_dir: Path) -> dict[str, str]:
    """Build the same device headers Kimi Code CLI sends."""
    device_id_path = home_dir / "device_id"
    device_id = device_id_path.read_text().strip() if device_id_path.exists() else ""
    return {
        "X-Msh-Device-Name": socket.gethostname(),
        "X-Msh-Device-Model": f"{platform.system()} {platform.release()} {platform.machine()}",
        "X-Msh-Device-Id": device_id,
    }


def _refresh_token(refresh_token: str, home_dir: Path) -> dict:
    """Exchange a refresh_token for a new access_token from Kimi OAuth."""
    data = urllib.parse.urlencode({
        "client_id": KIMI_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }).encode()

    req = urllib.request.Request(
        KIMI_TOKEN_ENDPOINT,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            **_device_headers(home_dir),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        raise SystemExit(f"Token refresh failed (HTTP {e.code}): {body}") from e


def _load_credentials(cred_path: Path) -> dict:
    if not cred_path.exists():
        raise SystemExit(
            f"Kimi Code CLI credentials not found: {cred_path}\n"
            "Please run `kimi login` first."
        )
    return json.loads(cred_path.read_text())


def _save_credentials(cred_path: Path, wire: dict) -> None:
    """Write token in the same shape Kimi Code CLI expects."""
    cred_path.parent.mkdir(parents=True, exist_ok=True)
    cred_path.write_text(json.dumps(wire, indent=2) + "\n")
    cred_path.chmod(0o600)


def _update_env(env_path: Path, access_token: str) -> None:
    if not env_path.exists():
        raise SystemExit(".env not found in current directory.")

    text = env_path.read_text()
    text = re.sub(
        r"^[ \t]*#?[ \t]*API_KEY\s*=\s*.*$",
        f"API_KEY={access_token}",
        text,
        flags=re.MULTILINE,
    )
    env_path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh familiar-ai's API_KEY from Kimi Code CLI OAuth credentials."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh even if the current token is still valid.",
    )
    args = parser.parse_args()

    home_dir = Path.home() / ".kimi-code"
    cred_path = home_dir / "credentials" / "kimi-code.json"
    env_path = Path(".env")

    cred = _load_credentials(cred_path)
    refresh_token = cred.get("refresh_token")
    if not refresh_token:
        raise SystemExit("No refresh_token found in credentials. Run `kimi login`.")

    expires_at = cred.get("expires_at", 0)
    now = int(time.time())
    margin = int(os.environ.get("KIMI_REFRESH_MARGIN_SECONDS", str(DEFAULT_MARGIN_SECONDS)))

    if not args.force and expires_at and (expires_at - now) > margin:
        print(f"Access token still valid for {expires_at - now}s; skipping refresh.")
        _update_env(env_path, cred["access_token"])
        print("Confirmed API_KEY is up to date.")
        return

    print("Refreshing Kimi Code CLI access token...")
    wire = _refresh_token(refresh_token, home_dir)

    if not wire.get("access_token") or not wire.get("refresh_token"):
        raise SystemExit("OAuth response missing access_token or refresh_token.")

    # Build credentials file payload in the same wire format.
    new_cred = {
        "access_token": wire["access_token"],
        "refresh_token": wire["refresh_token"],
        "expires_at": int(time.time()) + int(wire["expires_in"]),
        "scope": wire.get("scope", "kimi-code"),
        "token_type": wire.get("token_type", "Bearer"),
        "expires_in": int(wire["expires_in"]),
    }

    _save_credentials(cred_path, new_cred)
    _update_env(env_path, new_cred["access_token"])

    print("Token refreshed successfully.")
    print(f"  New expiry: {new_cred['expires_at']} ({new_cred['expires_in']}s)")


if __name__ == "__main__":
    main()
