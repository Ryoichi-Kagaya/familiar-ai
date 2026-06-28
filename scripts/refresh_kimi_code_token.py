#!/usr/bin/env python3
"""Refresh API_KEY in .env from Kimi Code CLI's OAuth credentials.

Kimi Code CLI stores a short-lived access token in
~/.kimi-code/credentials/kimi-code.json. familiar-ai can reuse that token
to call the same managed model endpoint (https://api.kimi.com/coding/v1).
Run this script whenever the token expires.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def main() -> None:
    cred_path = Path.home() / ".kimi-code" / "credentials" / "kimi-code.json"
    if not cred_path.exists():
        raise SystemExit(
            f"Kimi Code CLI credentials not found: {cred_path}\n"
            "Please run `kimi login` first."
        )

    cred = json.loads(cred_path.read_text())
    token = cred.get("access_token")
    if not token:
        raise SystemExit("No access_token found in credentials. Run `kimi login`.")

    env_path = Path(".env")
    if not env_path.exists():
        raise SystemExit(".env not found in current directory.")

    text = env_path.read_text()
    text = re.sub(
        r"^[ \t]*#?[ \t]*API_KEY\s*=\s*.*$",
        f"API_KEY={token}",
        text,
        flags=re.MULTILINE,
    )
    env_path.write_text(text)
    print("Refreshed API_KEY from Kimi Code CLI credentials.")


if __name__ == "__main__":
    main()
