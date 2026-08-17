"""Small durable transcript for Telegram conversations.

The main agent history is shared by several presentation surfaces and is also
intentionally forked for autonomous turns.  Telegram therefore needs its own
lightweight transcript so a reply can still be understood after an outbound
message was sent by an autonomous turn (or after a restart).
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Literal

TelegramRole = Literal["user", "assistant"]


class TelegramHistory:
    """Persist a bounded, per-chat Telegram transcript as JSONL.

    The file is rewritten as a bounded snapshot after each change.  This keeps
    the implementation simple and prevents an append-only log from growing
    forever even though only recent messages are used.
    """

    def __init__(self, path: Path | None = None, *, max_messages: int = 40) -> None:
        self._path = path or (Path.home() / ".familiar_ai" / "telegram_history.jsonl")
        self._max_messages = max(2, max_messages)
        self._lock = Lock()
        self._messages: dict[str, list[dict[str, str]]] = {}
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            lines = self._path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        for line in lines:
            try:
                entry = json.loads(line)
            except (TypeError, ValueError):
                continue
            if not isinstance(entry, dict):
                continue
            chat_id = entry.get("chat_id")
            role = entry.get("role")
            text = entry.get("text")
            if chat_id is None or not isinstance(text, str):
                continue
            if role == "clear":
                self._messages[str(chat_id)] = []
                continue
            if role not in {"user", "assistant"}:
                continue
            self._messages.setdefault(str(chat_id), []).append({"role": role, "text": text})
        for entries in self._messages.values():
            del entries[: -self._max_messages]

    def _persist_locked(self) -> None:
        """Write the current bounded snapshot; the caller owns ``_lock``."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._path.with_name(f".{self._path.name}.tmp")
        lines = []
        for chat_id, entries in self._messages.items():
            for entry in entries:
                lines.append(
                    json.dumps(
                        {"chat_id": chat_id, **entry},
                        ensure_ascii=False,
                    )
                )
        temp_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        temp_path.replace(self._path)

    def _record(self, chat_id: int | str, role: TelegramRole, text: str) -> None:
        text = text.strip()
        if not text:
            return
        key = str(chat_id)
        with self._lock:
            entries = self._messages.setdefault(key, [])
            entries.append({"role": role, "text": text})
            del entries[: -self._max_messages]
            try:
                self._persist_locked()
            except OSError:
                # The in-memory transcript is still useful for this process;
                # disk persistence must never turn a delivered message into a
                # failed Telegram tool call.
                return

    def record_user(self, chat_id: int | str, text: str) -> None:
        """Record text received from a Telegram user."""
        self._record(chat_id, "user", text)

    def record_agent(self, chat_id: int | str, text: str) -> None:
        """Record text successfully sent by the agent to Telegram."""
        self._record(chat_id, "assistant", text)

    def clear(self, chat_id: int | str) -> None:
        """Forget one chat, including after the next process restart."""
        key = str(chat_id)
        with self._lock:
            self._messages.pop(key, None)
            try:
                self._persist_locked()
            except OSError:
                return

    def render(self, chat_id: int | str, *, max_chars: int = 6000) -> str:
        """Return recent transcript text suitable for an inbound prompt."""
        with self._lock:
            entries = list(self._messages.get(str(chat_id), ()))
        if not entries:
            return ""

        lines = ["[Recent Telegram conversation — oldest first]"]
        for entry in entries:
            speaker = "User" if entry["role"] == "user" else "Agent"
            lines.append(f"{speaker} (Telegram): {entry['text']}")
        rendered = "\n".join(lines)
        return rendered[-max_chars:]
