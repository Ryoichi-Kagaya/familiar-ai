"""Telegram bot front-end for familiar-ai.

Start with:
    familiar --telegram

Required env var:
    TELEGRAM_BOT_TOKEN   — BotFather token

Optional env var:
    TELEGRAM_ALLOWED_IDS — comma-separated Telegram user IDs that are allowed
                           to chat. If unset, any user can interact.
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import re
from collections.abc import Callable
from typing import Any

from familiar_runtime.models import ImageAttachment, UserTurn

from .config import AgentConfig, TelegramConfig
from .image_input import ImageInputError, normalize_image_bytes
from .telegram_history import TelegramHistory
from .user_profile import UserRegistry

logger = logging.getLogger(__name__)

_MAX_MSG_LEN = 4096  # Telegram hard limit
_TELEGRAM_CHANNEL_CONTEXT = """\
[Telegram text chat]
- Your final reply is delivered to this Telegram chat. Reply in the normal final channel;
  `say` is intentionally unavailable on inbound Telegram turns because it controls room audio.
- `speak` is a separate StackChan room-device action. Do not claim that a voice or reply tool is
  unavailable unless you called that tool in this turn and its result explicitly failed.
- Never include tool protocol, private stage directions, or turn-control messages in the reply.
"""
_AUDIO_DIRECTION_RE = re.compile(r"\[(?=[A-Za-z])[^\]\n]{1,40}\]")
_TOOL_BLOCK_START_RE = re.compile(r"<tool_call\b", re.IGNORECASE)
_TOOL_CONTROL_TAG_RE = re.compile(
    r"</?(?:tool_call|parameter|invoke)>|<invoke\b[^>]*>|<parameter\b[^>]*>",
    re.IGNORECASE,
)
_TURN_CONTROL_RE = re.compile(
    r"^\s*[（(]?\s*(?:turn\s+complete|ターン完了)[.!。]?\s*[）)]*\s*$",
    re.IGNORECASE,
)


def _telegram_user_text(
    speaker_name: str,
    user_input: str,
    history_context: str = "",
) -> str:
    """Add channel semantics that distinguish chat replies from room speech."""
    history = f"\n{history_context}\n" if history_context else ""
    return f"{_TELEGRAM_CHANNEL_CONTEXT}{history}\n[{speaker_name}]: {user_input}"


def _sanitize_telegram_text(text: str) -> str:
    """Remove model control syntax while preserving observable reply content."""
    clean = _AUDIO_DIRECTION_RE.sub("", text).strip()

    # A broken prompt-driven tool call must never become chat content. The
    # runtime parser normally consumes it; this is the presentation boundary's
    # final defence.
    tool_start = _TOOL_BLOCK_START_RE.search(clean)
    if tool_start:
        clean = clean[: tool_start.start()]
    clean = _TOOL_CONTROL_TAG_RE.sub("", clean).strip()

    if _TURN_CONTROL_RE.fullmatch(clean):
        return ""

    clean = re.sub(r"\n{3,}", "\n\n", clean).strip()
    return "" if _TURN_CONTROL_RE.fullmatch(clean) else clean


async def _run_telegram_agent_turn(
    agent,
    desires,
    user_turn: UserTurn,
    *,
    profile_id: str | None,
    on_image,
) -> str:
    """Run an inbound Telegram turn on the surface-owned shared agent."""
    return (
        await agent.run(
            user_turn,
            on_image=on_image,
            desires=desires,
            user_id=profile_id,
            turn_source="telegram",
            # The visible reply is delivered by this presentation boundary.
            # Removing outbound Telegram prevents recursive/double sends while
            # keeping the tool available to GUI/TUI/REPL and autonomous turns.
            excluded_tools=frozenset({"say", "send_telegram_message"}),
        )
        or ""
    ).strip()


async def run_telegram_bot(
    token: str,
    agent,
    desires,
    *,
    allowed_ids: set[int] | None = None,
) -> None:
    """Run the Telegram bot using long-polling until cancelled."""
    try:
        from telegram import Update
        from telegram.ext import (
            Application,
            CommandHandler,
            ContextTypes,
            MessageHandler,
            filters,
        )
    except ImportError as exc:
        raise RuntimeError(
            "python-telegram-bot is not installed. Run: uv pip install 'familiar-ai[telegram]'"
        ) from exc

    allowed_ids = set(allowed_ids or ())
    if allowed_ids:
        logger.info("Telegram: allowed user IDs: %s", allowed_ids)
    else:
        logger.info("Telegram: no ID allowlist — any user can chat")

    registry = UserRegistry()
    history = getattr(agent, "telegram_history", None)
    if not isinstance(history, TelegramHistory):
        history = None

    def _is_allowed(user_id: int) -> bool:
        return not allowed_ids or user_id in allowed_ids

    def _resolve_speaker(tg_user) -> tuple[str, str | None]:
        """Return (display_name, user_profile_id_or_None) for a Telegram user."""
        profile = registry.get_by_telegram_id(tg_user.id)
        if profile:
            return profile.name, profile.id
        # Fall back to Telegram first_name; no profile switch
        return (tg_user.first_name or str(tg_user.id)), None

    async def _cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None or not _is_allowed(update.effective_user.id):
            return
        if update.message is None:
            return
        await update.message.reply_text("こんにちは！何でも話しかけてください。")

    async def _cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None or not _is_allowed(update.effective_user.id):
            return
        if update.message is None:
            return
        if history is not None and update.effective_user is not None:
            history.clear(update.effective_user.id)
        clear_exclusive = getattr(agent, "clear_history_exclusive", None)
        if callable(clear_exclusive):
            await clear_exclusive(source="telegram-command")
        else:
            agent.clear_history()
        await update.message.reply_text("会話履歴をリセットしました。")

    async def _cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None or not _is_allowed(update.effective_user.id):
            return
        if update.message is None:
            return
        await update.message.reply_text(
            "/start            — 挨拶\n"
            "/clear            — 会話履歴をリセット\n"
            "/register <id>    — このTelegramアカウントをユーザープロファイルに紐付け\n"
            "/whoami           — 現在の紐付け状況を確認\n"
            "/help             — このメッセージ"
        )

    async def _cmd_register(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Link the sender's Telegram ID to a users.json profile by slug."""
        if update.effective_user is None or not _is_allowed(update.effective_user.id):
            return
        if update.message is None:
            return
        args = context.args or []
        if not args:
            await update.message.reply_text(
                "使い方: /register <user_id>\n例: /register default  または  /register honoruru"
            )
            return
        user_id_slug = args[0].strip().lower()
        tg_id = update.effective_user.id
        try:
            profile = registry.link_telegram(user_id_slug, tg_id)
        except Exception:
            logger.exception("Failed to link Telegram ID %s to profile %s", tg_id, user_id_slug)
            await update.message.reply_text("登録に失敗しました。")
            return
        logger.info("Telegram: linked %d → profile '%s' (%s)", tg_id, profile.id, profile.name)
        await update.message.reply_text(
            f"紐付けました: Telegram ID {tg_id} → {profile.name} ({profile.id})"
        )

    async def _cmd_whoami(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None or not _is_allowed(update.effective_user.id):
            return
        if update.message is None:
            return
        tg_id = update.effective_user.id
        profile = registry.get_by_telegram_id(tg_id)
        if profile:
            await update.message.reply_text(f"Telegram ID {tg_id} → {profile.name} ({profile.id})")
        else:
            await update.message.reply_text(
                f"Telegram ID {tg_id} はまだ紐付けされていません。\n"
                "/register <user_id> で登録してください。"
            )

    async def _run_turn(
        update: Update,
        user_input: str,
        images: tuple[ImageAttachment, ...] = (),
    ) -> None:
        """Execute one agent turn and send the response back."""
        tg_user = update.effective_user
        if tg_user is None:
            return
        speaker_name, profile_id = _resolve_speaker(tg_user)
        chat_id = tg_user.id
        history_context = history.render(chat_id) if history is not None else ""
        user_turn = UserTurn(
            text=_telegram_user_text(speaker_name, user_input, history_context), images=images
        )
        if history is not None:
            history.record_user(chat_id, user_input)

        await update.message.chat.send_action("typing")  # type: ignore[union-attr]

        captured_images: list[str] = []

        def on_image(b64: str) -> None:
            captured_images.append(b64)

        try:
            final_text = await _run_telegram_agent_turn(
                agent,
                desires,
                user_turn,
                profile_id=profile_id,
                on_image=on_image,
            )
        except Exception:
            logger.exception("Agent error during Telegram turn")
            await update.message.reply_text(  # type: ignore[union-attr]
                "エラーが発生しました。ログを確認してください。"
            )
            return

        for b64 in captured_images:
            try:
                raw = base64.b64decode(b64)
                bio = io.BytesIO(raw)
                bio.name = "capture.jpg"
                await update.message.reply_photo(bio)  # type: ignore[union-attr]
            except Exception:
                logger.exception("Failed to send camera capture to Telegram")

        clean_final = _sanitize_telegram_text(final_text)
        if clean_final != final_text.strip():
            logger.info(
                "Telegram sanitized final output: raw=%r clean=%r",
                final_text[:500],
                clean_final[:500],
            )
        if clean_final and clean_final != "(no response)":
            response = clean_final
            response_source = "final_text"
        else:
            response = ""
            response_source = "empty"

        if not response:
            logger.warning(
                "Telegram reply suppressed after sanitization (final=%r)",
                final_text[:300],
            )
            return

        logger.info("Telegram reply (%s): %r", response_source, response[:500])

        for i in range(0, len(response), _MAX_MSG_LEN):
            chunk = response[i : i + _MAX_MSG_LEN]
            await update.message.reply_text(chunk)  # type: ignore[union-attr]
            if history is not None:
                history.record_agent(chat_id, chunk)

        desires.satisfy("greet_companion")

    async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None or not _is_allowed(update.effective_user.id):
            logger.warning("Rejected message from Telegram user %s", update.effective_user)
            return
        if update.message is None:
            return

        user_input = (update.message.text or "").strip()
        if not user_input:
            return

        await _run_turn(update, user_input)

    async def _handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None or not _is_allowed(update.effective_user.id):
            logger.warning("Rejected photo from Telegram user %s", update.effective_user)
            return
        if update.message is None:
            return

        # Largest available size is last in the list
        photos = update.message.photo
        if not photos:
            return
        photo = photos[-1]

        try:
            file = await context.bot.get_file(photo.file_id)
            raw = await file.download_as_bytearray()
            attachment = normalize_image_bytes(bytes(raw), filename="telegram-photo.jpg")
        except ImageInputError as exc:
            await update.message.reply_text(str(exc))
            return
        except Exception:
            logger.exception("Failed to download Telegram photo")
            await update.message.reply_text("画像のダウンロードに失敗しました。")
            return

        caption = (update.message.caption or "").strip()
        user_input = caption if caption else "この画像を見て。"

        await _run_turn(update, user_input, images=(attachment,))

    async def _handle_image_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None or not _is_allowed(update.effective_user.id):
            logger.warning("Rejected image document from Telegram user %s", update.effective_user)
            return
        if update.message is None or update.message.document is None:
            return

        document = update.message.document
        if not (document.mime_type or "").startswith("image/"):
            return
        try:
            file = await context.bot.get_file(document.file_id)
            raw = await file.download_as_bytearray()
            attachment = normalize_image_bytes(bytes(raw), filename=document.file_name)
        except ImageInputError as exc:
            await update.message.reply_text(str(exc))
            return
        except Exception:
            logger.exception("Failed to download Telegram image document")
            await update.message.reply_text("画像のダウンロードに失敗しました。")
            return

        caption = (update.message.caption or "").strip()
        user_input = caption if caption else "この画像を見て。"
        await _run_turn(update, user_input, images=(attachment,))

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(CommandHandler("clear", _cmd_clear))
    app.add_handler(CommandHandler("help", _cmd_help))
    app.add_handler(CommandHandler("register", _cmd_register))
    app.add_handler(CommandHandler("whoami", _cmd_whoami))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, _handle_photo))
    app.add_handler(MessageHandler(filters.Document.IMAGE, _handle_image_document))

    await app.initialize()
    await app.start()
    assert app.updater is not None
    await app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Telegram bot started. Ctrl+C to stop.")
    print("[telegram] bot is running. Ctrl+C to stop.")

    try:
        await asyncio.Event().wait()
    finally:
        assert app.updater is not None
        await app.updater.stop()
        await app.stop()
        await app.shutdown()


TelegramErrorHandler = Callable[[Exception], None]


class TelegramChannel:
    """Own the inbound polling lifecycle for one presentation surface."""

    def __init__(
        self,
        *,
        token: str,
        allowed_ids: set[int],
        agent: Any,
        desires: Any,
        on_error: TelegramErrorHandler | None = None,
    ) -> None:
        self._token = token
        self._allowed_ids = set(allowed_ids)
        self._agent = agent
        self._desires = desires
        self._on_error = on_error
        self._task: asyncio.Task[None] | None = None

    @classmethod
    def from_config(
        cls,
        config: AgentConfig,
        agent: Any,
        desires: Any,
        *,
        on_error: TelegramErrorHandler | None = None,
    ) -> TelegramChannel | None:
        telegram = getattr(config, "telegram", None)
        if (
            not isinstance(telegram, TelegramConfig)
            or not telegram.inbound_enabled
            or not telegram.token
        ):
            return None
        return cls(
            token=telegram.token,
            allowed_ids=telegram.allowed_ids,
            agent=agent,
            desires=desires,
            on_error=on_error,
        )

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.is_running:
            return
        self._task = asyncio.create_task(self._run(), name="telegram-channel")

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        try:
            await run_telegram_bot(
                self._token,
                self._agent,
                self._desires,
                allowed_ids=self._allowed_ids,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - boundary reports channel failure
            logger.exception("Telegram background channel failed")
            if self._on_error is not None:
                self._on_error(exc)
