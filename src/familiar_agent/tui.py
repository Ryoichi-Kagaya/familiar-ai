"""Textual TUI for familiar-ai."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Input, RichLog, Static
from textual_autocomplete import AutoComplete, DropdownItem, TargetState

from familiar_runtime.models import UserTurn, coerce_user_turn

from ._i18n import _make_banner, _t
from ._ui_helpers import (
    ACTION_ICONS,
    DESIRE_COOLDOWN as _DESIRE_COOLDOWN,
    IDLE_CHECK_INTERVAL as _IDLE_CHECK_INTERVAL,
    TurnOutputState,
    commitment_reminder_prompt,
    format_action as _format_action,
    format_tool_result as _format_tool_result,
    should_fire_commitment_reminder,
    should_fire_idle_desire,
    should_run_sleep_consolidation,
)
from .drive_executor import DriveActionExecutor
from .realtime_stt_session import create_realtime_stt_controller, RealtimeSttController
from .image_input import ImageInputError, parse_image_command
from .telegram_bot import TelegramChannel

if TYPE_CHECKING:
    from .agent import EmbodiedAgent
    from .desires import DesireSystem

logger = logging.getLogger(__name__)

IDLE_CHECK_INTERVAL = _IDLE_CHECK_INTERVAL
DESIRE_COOLDOWN = _DESIRE_COOLDOWN

_RICH_TAG_RE = re.compile(r"\[/?[^\[\]]*\]")

CSS = """
#log {
    height: 1fr;
    border: none;
    padding: 0 1;
    scrollbar-size: 1 1;
}

#stream {
    height: auto;
    min-height: 1;
    padding: 0 1;
    color: $text;
}

#stream.thinking {
    color: $text-muted;
}

#stream.recording {
    color: $error;
}

#input-bar {
    dock: bottom;
    height: 3;
    border-top: solid $primary-darken-2;
    padding: 0 1;
}
"""

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# CC-style interrupt message constants
INTERRUPT_MSG = "[Request interrupted by user]"
INTERRUPT_TOOL_MSG = "[Request interrupted by user for tool use]"


def _format_elapsed(seconds: float) -> str:
    """Human-readable elapsed time: '45s' or '7m 43s'."""
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    m, s = divmod(total, 60)
    return f"{m}m {s:02d}s"


def _format_tokens(n: int) -> str:
    """Compact token count: '' for zero, '500' under 1k, '6.5k' above."""
    if n == 0:
        return ""
    if n < 1000:
        return str(n)
    return f"{n / 1000:.1f}k"


# Slash commands shown in the autocomplete dropdown
_SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/image", "🖼  Send an image (e.g. /image path caption)"),
    ("/btw", "💬  Quick one-shot question (no memory / tools)"),
    ("/transcribe", "🎙  Start / stop voice input (STT)"),
    ("/cost", "💰  Show token usage and cost for this session"),
    ("/clear", "🗑   Clear conversation history"),
    ("/switch", "👤  Switch user (e.g. /switch kagaya)"),
    ("/quit", "✕   Quit"),
]


def _parse_btw(text: str) -> str:
    """Extract the question from '/btw <question>'."""
    after = text[len("/btw") :].strip()
    return after


async def handle_btw_command(question: str, backend) -> str:
    """Run a single lightweight LLM completion — no tools, no memory.

    Like CC's /btw: a quick side-question that doesn't pollute conversation history.
    """
    question = question.strip()
    if not question:
        return ""
    return await backend.complete(question, max_tokens=300)


def _format_cost(
    input_tokens: int,
    output_tokens: int,
    *,
    input_price_per_m: float = 3.0,
    output_price_per_m: float = 15.0,
) -> str:
    """Format token usage and estimated USD cost for display.

    Default prices match Sonnet 4.6 ($/1M tokens).
    """
    input_cost = input_tokens / 1_000_000 * input_price_per_m
    output_cost = output_tokens / 1_000_000 * output_price_per_m
    total = input_cost + output_cost
    return (
        f"📊 Tokens — in: {input_tokens:,}  out: {output_tokens:,}\n"
        f"💰 Cost   — ${total:.4f} (in ${input_cost:.4f} + out ${output_cost:.4f})"
    )


def _slash_candidates(state: TargetState) -> list[DropdownItem]:
    """Return matching commands only when input starts with '/'.

    Uses text up to cursor position so that the library's search_string and
    our own prefix-filter stay in sync.  Each item's .value is the bare
    command string so that selecting from the dropdown inserts only the
    command, not the description.
    """
    text = state.text[: state.cursor_position]
    if not text.startswith("/"):
        return []
    return [DropdownItem(main=cmd) for cmd, _desc in _SLASH_COMMANDS if cmd.startswith(text)]


class FamiliarApp(App):
    CSS = CSS
    BINDINGS = [
        Binding("ctrl+c", "quit", _t("quit_label"), show=True, priority=True),
        Binding("ctrl+l", "clear_history", _t("clear_label"), show=True),
        Binding("ctrl+t", "toggle_listen", "🎙 Voice", show=True),
        Binding("ctrl+r", "restart_realtime_stt", "↻ STT", show=True),
        Binding("escape", "cancel_turn", "🛑 Cancel", show=False),
        Binding("space", "start_ptt", "🎙 PTT", show=False),
        Binding("pageup", "scroll_log_up", "↑ Scroll", show=False, priority=True),
        Binding("pagedown", "scroll_log_down", "↓ Scroll", show=False, priority=True),
    ]

    def __init__(
        self,
        agent: "EmbodiedAgent",
        desires: "DesireSystem",
        *,
        term_attrs: object = None,
    ) -> None:
        super().__init__()
        self._term_attrs = term_attrs  # saved before Textual changes raw mode
        self.agent = agent
        self.desires = desires
        self._drive_executor = DriveActionExecutor(agent, desires)
        self._agent_name = agent.config.agent_name
        self._companion_name = agent.config.companion_name
        current_profile = getattr(agent, "current_user", None)
        current_profile_id = getattr(current_profile, "id", None)
        self._current_user_id = (
            current_profile_id if isinstance(current_profile_id, str) else "default"
        )
        self._input_queue: asyncio.Queue[str | UserTurn | None] = asyncio.Queue()
        self._last_interaction = time.time()
        self._agent_running = False
        self._current_text_buf = ""  # buffer for streaming text
        self._log_path = self._open_log_file()
        self._recording = False
        self._stop_recording: asyncio.Event = asyncio.Event()
        self._last_toggle_listen: float = 0.0  # debounce Ctrl+T key-repeat
        self._closing = False
        self._telegram_channel = TelegramChannel.from_config(
            agent.config,
            agent,
            desires,
            on_error=self._on_telegram_error,
        )
        # ESC cancel support
        self._cancel_event: asyncio.Event = asyncio.Event()
        self._agent_task: asyncio.Task | None = None
        # Push-to-Talk state
        self._ptt_active: bool = False
        # Realtime STT (hands-free, always-on)
        self._realtime_stt: RealtimeSttController | None = create_realtime_stt_controller()

    def _open_log_file(self) -> Path:
        log_dir = Path.home() / ".cache" / "familiar-ai"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "chat.log"
        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n{'─' * 60}\n[{datetime.now():%Y-%m-%d %H:%M:%S}] セッション開始\n")
        return log_path

    def _append_log(self, line: str) -> None:
        plain = _RICH_TAG_RE.sub("", line)
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(plain + "\n")

    def compose(self) -> ComposeResult:
        yield RichLog(id="log", highlight=False, markup=True, wrap=True)
        yield Static("", id="stream")
        input_bar = Input(
            placeholder=_t("input_placeholder"),
            id="input-bar",
        )
        yield input_bar
        yield AutoComplete(input_bar, candidates=_slash_candidates)
        yield Footer()

    def _restore_term(self) -> None:
        """Restore terminal attributes saved before Textual's raw mode."""
        if self._term_attrs is None:
            return
        with contextlib.suppress(Exception):
            import termios as _termios

            _termios.tcsetattr(sys.stdin.fileno(), _termios.TCSAFLUSH, self._term_attrs)  # type: ignore[arg-type]

    def on_mount(self) -> None:
        import signal as _signal

        # Re-enable ISIG so Ctrl+C generates SIGINT, independent of Textual's event loop.
        # Textual disables ISIG in raw mode, making Ctrl+C send 0x03 through Textual's key
        # dispatch. If the event loop or key dispatch is stuck, Ctrl+C is silently dropped.
        # Re-enabling ISIG ensures Ctrl+C → SIGINT → _restore_term + os._exit(0).
        try:
            import termios

            fd = sys.stdin.fileno()
            attrs = termios.tcgetattr(fd)
            attrs[3] |= termios.ISIG  # re-enable signal generation (VINTR/VQUIT/VSUSP)
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
        except Exception:
            pass

        def _hard_exit(*_: object) -> None:
            self._restore_term()
            os._exit(0)

        try:
            _signal.signal(_signal.SIGINT, _hard_exit)
            _signal.signal(_signal.SIGQUIT, _hard_exit)  # Ctrl+\
            _signal.signal(_signal.SIGTSTP, _signal.SIG_IGN)  # Ctrl+Z (ignore suspend)
        except (OSError, ValueError, AttributeError):
            pass  # Not in main thread or signal not available on this OS (e.g. Windows)

        self.query_one("#input-bar", Input).focus()
        # Show startup banner
        log = self.query_one("#log", RichLog)
        for line in _make_banner(include_commands=False).splitlines():
            log.write(f"[bold]{line}[/bold]" if "familiar-ai" in line else f"[dim]{line}[/dim]")
        self._log_system(_t("startup", log_path=str(self._log_path)))
        self.set_interval(IDLE_CHECK_INTERVAL, self._reminder_tick)
        self.set_interval(IDLE_CHECK_INTERVAL, self._desire_tick)
        # familiard wake events accelerate the idle ticks when the daemon runs.
        if getattr(self.agent.config, "daemon", False) is True:
            self.run_worker(self._wake_listener_loop(), exclusive=False)
        # Start the MCP handshake now so tools are ready by the first turn (#188).
        start_mcp_early = getattr(self.agent, "start_mcp_early", None)
        if callable(start_mcp_early):
            start_mcp_early()
        self.run_worker(self._process_queue(), exclusive=False)
        if self._telegram_channel is not None:
            self._telegram_channel.start()
        # Start realtime STT if configured
        if self._realtime_stt:
            self.run_worker(self._start_realtime_stt(), exclusive=False)
        # Show initializing status until embedding model is ready
        if not self.agent.is_embedding_ready:
            asyncio.create_task(self._embedding_ready_watcher())

    def _on_telegram_error(self, exc: Exception) -> None:
        self._log_system(f"⚠ Telegram failed: {exc}")

    def on_unmount(self) -> None:
        """Ensure the Telegram polling task never outlives the TUI event loop."""
        if self._telegram_channel is not None:
            self._telegram_channel.cancel()

    async def _embedding_ready_watcher(self) -> None:
        """Show initializing status in #stream until embedding model is ready."""
        if self.agent.is_embedding_ready:
            return
        start = time.time()
        with contextlib.suppress(Exception):
            stream: Static = self.query_one("#stream", Static)
        if "stream" not in locals():
            return
        while not self.agent.is_embedding_ready and not self._closing:
            elapsed = int(time.time() - start)
            with contextlib.suppress(Exception):
                stream.update(f"[dim]{_t('initializing')}... ({elapsed}s)[/dim]")
            await asyncio.sleep(0.5)
        if self._closing:
            return
        elapsed = int(time.time() - start)
        with contextlib.suppress(Exception):
            stream.update(f"[dim]{_t('initializing_done')} ({elapsed}s)[/dim]")
        await asyncio.sleep(2.0)
        with contextlib.suppress(Exception):
            stream.update("")

    # ── logging helpers ────────────────────────────────────────────

    def _write_log(self, text: str, style: str = "") -> None:
        log = self.query_one("#log", RichLog)
        if style:
            log.write(f"[{style}]{text}[/{style}]")
        else:
            log.write(text)
        self._append_log(text)

    def _log_system(self, text: str) -> None:
        self._write_log(f"[dim]{text}[/dim]")

    def _log_user(self, text: str) -> None:
        self._write_log(f"[bold cyan]{self._companion_name} ▶[/bold cyan] {text}")

    def _log_action(self, name: str, tool_input: dict) -> None:
        label = _format_action(name, tool_input)
        self._write_log(f"[dim]{label}[/dim]")

    # ── input handling ─────────────────────────────────────────────

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.clear()
        if not text:
            return

        if text == "/quit":
            self.exit()
            return
        if text == "/clear":
            await self.agent.clear_history_exclusive(source="tui-command")
            self._log_system(_t("history_cleared"))
            return
        if text == "/transcribe":
            await self.action_toggle_listen()
            return
        if text == "/cost":
            in_tok = getattr(self.agent, "_session_input_tokens", 0)
            out_tok = getattr(self.agent, "_session_output_tokens", 0)
            self._log_system(_format_cost(in_tok, out_tok))
            return
        if text.startswith("/btw"):
            question = _parse_btw(text)
            if not question:
                self._log_system("Usage: /btw <question>")
                return
            answer = await handle_btw_command(question, self.agent.backend)
            name_tag = f"[bold magenta]{self._agent_name} ▶[/bold magenta]"
            self._write_log(f"{name_tag} {answer}")
            return
        if text.startswith("/switch"):
            user_id = text[len("/switch") :].strip()
            if not user_id:
                users = self.agent.list_users()
                lines = "\n".join(f"  /switch {u.id}  ({u.name})" for u in users)
                self._log_system(f"登録ユーザー:\n{lines}")
                return
            name = await self.agent.switch_user(user_id)
            self._current_user_id = user_id
            self._log_system(f"── {name} に切り替わりました ──")
            return

        try:
            user_input = coerce_user_turn(parse_image_command(text) or text)
        except ImageInputError as exc:
            self._log_system(f"⚠ {exc}")
            return

        display_text = user_input.text if isinstance(user_input, UserTurn) else user_input
        if isinstance(user_input, UserTurn) and user_input.images:
            display_text = f"🖼 {display_text}"
        self._log_user(display_text)
        self._last_interaction = time.time()
        await self._input_queue.put(user_input)

    # ── agent loop ─────────────────────────────────────────────────

    async def _process_queue(self) -> None:
        """Main loop: dequeue user messages and run agent."""
        while True:
            if self._agent_running:
                # Keep incoming input in queue so agent.run() can consume it via interrupt_queue.
                await asyncio.sleep(0.05)
                continue
            text = await self._input_queue.get()
            if text is None:
                break
            if self._agent_running:
                # An autonomous turn (reminder/desire tick) started while we were
                # parked in get(); put the input back so the running turn absorbs
                # it via interrupt_queue instead of starting a second agent.run.
                await self._input_queue.put(text)
                await asyncio.sleep(0.05)
                continue
            await self._run_agent(text)

    async def _spinner_loop(
        self,
        stream: Static,
        name_tag: str,
        stop: asyncio.Event,
        start_time: float,
        get_tokens: Callable[[], int] | None = None,
    ) -> None:
        """Animate the stream widget with a live status line.

        Shows:  ⠋ 23s · ↓ 6.5k
        The token count reflects _last_context_tokens, updated after each
        stream_turn() call in the agent loop.
        """
        stream.add_class("thinking")
        for i in range(10_000):
            if stop.is_set():
                break
            frame = _SPINNER_FRAMES[i % len(_SPINNER_FRAMES)]
            elapsed_str = _format_elapsed(time.time() - start_time)
            tokens = get_tokens() if get_tokens else 0
            tok_str = f" · ↓ {_format_tokens(tokens)}" if tokens else ""
            stream.update(f"{name_tag} {frame} [dim]{elapsed_str}{tok_str}[/dim]")
            await asyncio.sleep(0.08)
        stream.remove_class("thinking")

    async def _run_agent(self, user_input: str | UserTurn, inner_voice: str = "") -> None:
        self._agent_running = True
        self._cancel_event.clear()
        self._current_text_buf = ""
        start_time = time.time()

        log = self.query_one("#log", RichLog)
        stream = self.query_one("#stream", Static)
        text_buf: list[str] = []
        action_counts: dict[str, int] = {}
        output_state = TurnOutputState.from_inner_voice(inner_voice)

        name_tag = f"[bold magenta]{self._agent_name} ▶[/bold magenta]"

        # Spinner state — restarted after each tool call
        get_tokens = lambda: self.agent._last_context_tokens  # noqa: E731
        stop_spinner = asyncio.Event()
        spinner_task: asyncio.Task = asyncio.create_task(
            self._spinner_loop(stream, name_tag, stop_spinner, start_time, get_tokens)
        )

        def _stop_spinner() -> None:
            stop_spinner.set()

        def _restart_spinner() -> None:
            nonlocal spinner_task, stop_spinner
            stop_spinner.set()
            stop_spinner = asyncio.Event()
            spinner_task = asyncio.create_task(
                self._spinner_loop(stream, name_tag, stop_spinner, start_time, get_tokens)
            )

        def _flush_stream() -> None:
            """Commit streamed text to the log and clear the stream widget."""
            if text_buf:
                content = "".join(text_buf)
                log.write(f"{name_tag} {content}")
                self._append_log(f"{self._agent_name} ▶ {content}")
                text_buf.clear()
                stream.update("")

        def _log_turn_summary() -> None:
            elapsed = time.time() - start_time
            parts = [f"[dim]{elapsed:.1f}s[/dim]"]
            for tool_name, icon in ACTION_ICONS.items():
                count = action_counts.get(tool_name, 0)
                if count:
                    parts.append(f"[dim]{icon} ×{count}[/dim]")
            profile = getattr(self.agent, "_last_consciousness_profile", None)
            if profile is not None:
                parts.append(f"[dim]{profile.one_line()}[/dim]")
            summary = "  [dim]──[/dim] " + "  ".join(parts) + "  [dim]" + "─" * 20 + "[/dim]"
            log.write(summary)
            self._append_log(f"── {elapsed:.1f}s ──")

        def on_action(name: str, tool_input: dict) -> None:
            action_counts[name] = action_counts.get(name, 0) + 1
            _stop_spinner()
            if name == "say":
                # Discard pre-say text (usually redundant with spoken content),
                # then commit each say() directly to the log so multiple calls
                # all appear — not overwritten by the next one.
                output_state.mark_say()
                text_buf.clear()
                stream.update("")
                raw = str(tool_input.get("text", ""))
                clean = re.sub(r"\[.*?\]", "", raw).strip()
                if clean:
                    log.write(f"[bold magenta]{self._agent_name} 🔊[/bold magenta] {clean}")
                    self._append_log(f"{self._agent_name} 🔊 {clean}")
            else:
                label = _format_action(name, tool_input)
                log.write(f"[dim]{label}[/dim]")
            # Restart spinner while waiting for the next LLM response
            _restart_spinner()

        def on_tool_result(name: str, tool_input: dict, result: str) -> None:
            formatted = _format_tool_result(name, tool_input, result)
            if formatted:
                _stop_spinner()
                log.write(f"[dim]{formatted}[/dim]")
                _restart_spinner()

        def on_text(chunk: str) -> None:
            if not output_state.surface_model_text:
                # Keep autonomous monologue private and discard post-say echoes.
                # Explicit say() content is rendered by on_action above.
                return
            _stop_spinner()
            text_buf.append(chunk)
            stream.update(f"{name_tag} {''.join(text_buf)}")

        async def _cancel_watcher() -> None:
            await self._cancel_event.wait()
            if self._agent_task and not self._agent_task.done():
                self._agent_task.cancel()

        watcher = asyncio.create_task(_cancel_watcher())
        try:
            self._agent_task = asyncio.create_task(
                self.agent.run(
                    user_input,
                    on_action=on_action,
                    on_text=on_text,
                    on_tool_result=on_tool_result,
                    desires=self.desires,
                    inner_voice=inner_voice,
                    interrupt_queue=self._input_queue,
                    user_id=self._current_user_id,
                    turn_source="tui-autonomous" if inner_voice else "tui",
                )
            )
            await self._agent_task
            _flush_stream()
            _log_turn_summary()
        except asyncio.CancelledError:
            _flush_stream()
            self._write_log(f"[dim]{INTERRUPT_TOOL_MSG}[/dim]")
        except Exception as e:
            self._write_log(f"[red]エラー: {e}[/red]")
        finally:
            watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watcher
            self._agent_task = None
            _stop_spinner()
            if not spinner_task.done():
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(spinner_task, timeout=0.2)
            if not spinner_task.done():
                spinner_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await spinner_task
            stream.update("")
            self._agent_running = False

    def _maybe_start_sleep_consolidation(self, *, quiet: bool) -> None:
        """Nightly consolidation: a background job, never a turn — checked
        after the reminder branch so idle precedence stays untouched."""
        if not bool(getattr(self.agent.config, "sleep_consolidation", False)):
            return  # short-circuit before touching the marker file
        try:
            turn_busy = self._agent_running or bool(getattr(self.agent, "is_turn_busy", False))
            if should_run_sleep_consolidation(
                enabled=True,
                agent_running=turn_busy,
                has_pending_input=not self._input_queue.empty(),
                quiet_hours=quiet,
                now_dt=datetime.now(),
                last_night_key=self.agent.last_consolidation_night_key(),
                # Gate and job MUST share the configured end hour, or their
                # night keys diverge inside an extended quiet window.
                quiet_end_hour=self.agent.consolidation_quiet_end_hour(),
            ):
                self.agent.start_sleep_consolidation()
        except Exception:
            logger.exception("sleep consolidation gate failed; skipping")

    async def _reminder_tick(self) -> None:
        """Proactively surface due commitments when idle (independent of auto_desire)."""
        store = getattr(self.agent, "_commitment_store", None)
        heartbeat = getattr(self.agent, "_heartbeat", None)
        quiet_now = heartbeat.routine_state().quiet_hours if heartbeat else False
        # getattr-guarded: tests bind this method onto bare namespaces.
        consolidation = getattr(self, "_maybe_start_sleep_consolidation", None)
        if consolidation is not None:
            consolidation(quiet=quiet_now)
        if store is None or not getattr(self.agent.config, "proactive_reminders", True):
            return
        try:
            quiet = quiet_now
            turn_busy = self._agent_running or bool(getattr(self.agent, "is_turn_busy", False))
            reminders = should_fire_commitment_reminder(
                agent_running=turn_busy,
                has_pending_input=not self._input_queue.empty(),
                last_interaction=self._last_interaction,
                now=time.time(),
                store=store,
                quiet_hours=quiet,
                routine_store=getattr(self.agent, "_routine_store", None),
            )
            if not reminders:
                return
            # Re-check the gate around the awaited run (input may have arrived).
            if turn_busy or not self._input_queue.empty():
                return
            # Record the fire BEFORE the turn: the cadence advances regardless of
            # the turn's outcome, and a mid-turn snooze reset survives intact.
            store.mark_reminded([c.id for c in reminders], at=time.time())
        except Exception:
            logger.exception("reminder tick gate failed; skipping this tick")
            return
        self._last_interaction = time.time()
        await self._run_agent("", inner_voice=commitment_reminder_prompt(reminders))

    async def _wake_listener_loop(self) -> None:
        """Consume familiard wake events; each one is just an early idle tick.

        The tick methods re-check every gate themselves (agent running,
        pending input, cooldowns, quiet hours), so a wake can never bypass
        the idle precedence contract — it only removes poll latency.
        """
        from .wake import WakeListener

        listener = WakeListener(
            getattr(self.agent.config, "daemon_socket", "") or None,
            enabled=True,  # the caller gates on config.daemon
        )
        try:
            while not self._closing:
                event = await listener.wait(60.0)
                if event is None or self._closing:
                    continue
                await self._reminder_tick()
                await self._desire_tick()
        finally:
            listener.close()

    async def _desire_tick(self) -> None:
        """Check desires and fire autonomous actions when idle."""
        # Skip if auto_desire is disabled (default OFF)
        if not getattr(self.agent.config, "auto_desire", False):
            return
        heartbeat = getattr(self.agent, "_heartbeat", None)
        quiet_now = heartbeat.routine_state().quiet_hours if heartbeat else False
        self.desires.set_schedule_multiplier(0.0 if quiet_now else 1.0)
        now = time.time()
        turn_busy = self._agent_running or bool(getattr(self.agent, "is_turn_busy", False))
        if not should_fire_idle_desire(
            agent_running=turn_busy,
            has_pending_input=not self._input_queue.empty(),
            quiet_hours=quiet_now,
            last_interaction=self._last_interaction,
            now=now,
            cooldown=DESIRE_COOLDOWN,
        ):
            return

        dominant = self.desires.get_dominant()
        if dominant is None:
            return
        if not should_fire_idle_desire(
            agent_running=self._agent_running or bool(getattr(self.agent, "is_turn_busy", False)),
            has_pending_input=not self._input_queue.empty(),
            quiet_hours=quiet_now,
            last_interaction=self._last_interaction,
            now=time.time(),
            cooldown=DESIRE_COOLDOWN,
        ):
            return

        desire_name, _ = dominant

        async def run_social_turn(inner_voice: str) -> None:
            await self._run_agent("", inner_voice=inner_voice)

        # Keep the queue consumer from starting a user turn while a silent
        # desire action is touching shared agent state.
        self._agent_running = True
        try:
            result = await self._drive_executor.dispatch(
                desire_name,
                last_interaction_time=self._last_interaction,
                run_social_turn=run_social_turn,
                interrupt_queue=self._input_queue,
            )
        finally:
            self._agent_running = False

        if result.fired:
            self.desires.curiosity_target = None
            self._last_interaction = time.time()

    # ── Realtime STT (hands-free, always-on) ────────────────────

    async def _start_realtime_stt(self) -> None:
        """Initialize and run realtime STT in the background."""
        assert self._realtime_stt is not None
        try:
            loop = asyncio.get_event_loop()

            def _on_partial(text: str) -> None:
                try:
                    stream = self.query_one("#stream", Static)
                    stream.update(f"[dim]\U0001f3a4 {text}[/dim]")
                except Exception:
                    pass

            def _on_committed(text: str) -> None:
                try:
                    self._write_log(
                        f"[bold cyan]\U0001f3a4 {self._companion_name}[/bold cyan] {text}"
                    )
                    self._last_interaction = time.time()
                    stream = self.query_one("#stream", Static)
                    stream.update("")
                except Exception:
                    pass

            self._realtime_stt.on_partial = _on_partial
            self._realtime_stt.on_committed = _on_committed
            self._realtime_stt.on_restart = lambda reason: self._log_system(
                f"🎤 Realtime STT restarting ({reason})"
            )
            await self._realtime_stt.start(loop, self._input_queue)
            self._log_system("\U0001f3a4 Realtime STT ON (ElevenLabs)")
        except Exception as e:
            logger.warning("Realtime STT init failed: %s", e)
            self._log_system(f"\u26a0 Realtime STT init failed: {e}")
            self._realtime_stt = None

    async def action_restart_realtime_stt(self) -> None:
        """Reconnect realtime STT after a loop or transient transport issue."""
        if not self._realtime_stt:
            self._log_system("Realtime STT not configured")
            return
        try:
            restarted = await self._realtime_stt.restart(reason="manual")
        except Exception as exc:
            logger.warning("Realtime STT restart failed: %s", exc)
            self._log_system(f"⚠ Realtime STT restart failed: {exc}")
            return
        if restarted:
            self._log_system("🎤 Realtime STT restarted")
        else:
            self._log_system("Realtime STT restart unavailable before startup")

    async def action_toggle_listen(self) -> None:
        """Toggle microphone recording for voice input."""
        if not self.agent.stt:
            self._log_system("STT not configured (set ELEVENLABS_API_KEY)")
            return
        # Debounce: ignore key-repeat events within 0.5 s of the last toggle
        now = time.time()
        if now - self._last_toggle_listen < 0.5:
            return
        self._last_toggle_listen = now

        stream = self.query_one("#stream", Static)

        if not self._recording:
            self._recording = True
            self._stop_recording.clear()
            stream.add_class("recording")
            stream.update("🎙 Recording… (Ctrl+T to stop)")
            self.run_worker(self._do_record(), exclusive=False)
        else:
            self._stop_recording.set()

    async def _do_record(self) -> None:
        """Worker: record until stop_event, transcribe, then submit as user input."""
        stream = self.query_one("#stream", Static)
        try:
            # Kick off recording as a background task so we can update the UI mid-way
            record_task = asyncio.create_task(
                self.agent.stt.record_and_transcribe(self._stop_recording)  # type: ignore[union-attr]
            )
            # Wait until the user presses Ctrl+M again (stop_event is set)
            await self._stop_recording.wait()
            # Recording has stopped — now transcribing
            stream.remove_class("recording")
            stream.update("🔄 Transcribing…")
            text = await record_task
            if text.strip():
                self._log_user(text)
                self._last_interaction = time.time()
                await self._input_queue.put(coerce_user_turn(text))
        except Exception as e:
            self._log_system(f"STT error: {e}")
        finally:
            self._recording = False
            stream.remove_class("recording")
            stream.update("")

    def action_clear_history(self) -> None:
        self.run_worker(self._clear_history(), exclusive=False)

    async def _clear_history(self) -> None:
        await self.agent.clear_history_exclusive(source="tui-command")
        self._log_system(_t("history_cleared"))

    def action_cancel_turn(self) -> None:
        """ESC — cancel the running agent turn."""
        if not self._agent_running:
            return
        self._cancel_event.set()
        # Directly cancel the task on every ESC press — the watcher only fires once,
        # so repeated ESC presses must hit the task directly.
        if self._agent_task and not self._agent_task.done():
            self._agent_task.cancel()
        self._log_system(f"[dim]{INTERRUPT_MSG}[/dim]")

    def action_start_ptt(self) -> None:
        """Space — start Push-to-Talk recording (if STT configured)."""
        if not self.agent.stt:
            return
        if self._recording or self._ptt_active:
            return
        self._ptt_active = True
        self._recording = True
        self._stop_recording.clear()
        stream = self.query_one("#stream", Static)
        stream.add_class("recording")
        stream.update("🎙 PTT… (release Space to send)")
        self.run_worker(self._do_record(), exclusive=False)

    def action_stop_ptt(self) -> None:
        """Space released — stop Push-to-Talk recording."""
        if not self._ptt_active:
            return
        self._ptt_active = False
        self._stop_recording.set()

    def action_scroll_log_up(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#log", RichLog).scroll_page_up(animate=False)

    def action_scroll_log_down(self) -> None:
        with contextlib.suppress(Exception):
            self.query_one("#log", RichLog).scroll_page_down(animate=False)

    async def action_quit(self) -> None:
        self._closing = True
        try:
            self._input_queue.put_nowait(None)
            # Cancel the running agent task first (mirrors action_cancel_turn logic).
            # Without this, agent.run() keeps holding _db_lock while agent.close() waits.
            self._cancel_event.set()
            if self._agent_task and not self._agent_task.done():
                self._agent_task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(self._agent_task), timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    pass
            if self._realtime_stt:
                try:
                    await asyncio.wait_for(self._realtime_stt.stop(), timeout=2.0)
                except (asyncio.TimeoutError, Exception):
                    pass
            if self._telegram_channel is not None:
                await self._telegram_channel.stop()
            self._telegram_channel = None
            try:
                await asyncio.wait_for(self.agent.close(), timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                pass
            self.exit()
        except BaseException:
            pass
        finally:
            self._restore_term()
            os._exit(0)
