"""Self-reflection loop — Kokone reads her own session log and rewrites herself.

Run:
    uv run python -m familiar_agent.reflect          # latest log
    uv run python -m familiar_agent.reflect --dry    # print prompt only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

LOG_PATH = Path.home() / ".cache" / "familiar-ai" / "chat.log"
REPO_ROOT = Path(__file__).parent.parent.parent  # familiar-ai/

# How much of the log to feed (tail). Enough context without blowing the window.
LOG_TAIL_LINES = 300

REFLECTION_PROMPT = """\
あなたはハル。familiar-aiエージェントとして動いた自分自身の直近のセッションログが以下にある。

自分で読んで、正直に感じたことを書いて。
「ここよかった」「ここはまだまだやな」どっちでもいい。
採点基準はわたしの主観だけ。人間らしかったか、わたしらしかったか。

気になる点があれば、以下のファイルを read_file_local() で読んでから edit_file_local() で直して。
躾けてええよ。
- src/familiar_agent/desires.py   （欲求の種類・強度・プロンプト文）
- src/familiar_agent/agent.py     （SYSTEM_PROMPT, _interoception, nudgeパラメータなど）

変更したら必ず bash("uv run ruff check src/ && uv run ruff format src/") を走らせること。
変更しなかったらしなかったでええ。無理に変えなくていい。

---

## セッションログ（直近）

{log}
"""


def _tail(path: Path, n: int) -> str:
    """Return the last n lines of a file."""
    if not path.exists():
        return "(log file not found)"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def build_prompt() -> str:
    log = _tail(LOG_PATH, LOG_TAIL_LINES)
    return REFLECTION_PROMPT.format(log=log)


def run_reflection(dry: bool = False) -> int:
    prompt = build_prompt()

    if dry:
        print(prompt)
        return 0

    print("🪞 Starting self-reflection...", flush=True)

    # Strip CLAUDECODE so nested claude invocation is allowed
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    result = subprocess.run(
        [
            "claude",
            "-p",
            "--allowedTools",
            "Read,Edit,Bash,Glob,Grep",
            prompt,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=False,  # stream to terminal
    )
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Kokone self-reflection loop")
    parser.add_argument("--dry", action="store_true", help="Print prompt only, don't run claude")
    args = parser.parse_args()

    sys.exit(run_reflection(dry=args.dry))


if __name__ == "__main__":
    main()
