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
import os

from .user_profile import UserRegistry

logger = logging.getLogger(__name__)

_MAX_MSG_LEN = 4096  # Telegram hard limit


def _parse_allowed_ids() -> set[int]:
    raw = os.environ.get("TELEGRAM_ALLOWED_IDS", "").strip()
    if not raw:
        return set()
    ids: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if part.isdigit():
            ids.add(int(part))
    return ids


async def run_telegram_bot(token: str, agent, desires) -> None:
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
            "python-telegram-bot is not installed. "
            "Run: uv pip install 'familiar-ai[telegram]'"
        ) from exc

    allowed_ids = _parse_allowed_ids()
    if allowed_ids:
        logger.info("Telegram: allowed user IDs: %s", allowed_ids)
    else:
        logger.info("Telegram: no ID allowlist — any user can chat")

    registry = UserRegistry()

    # Serialize all agent turns so history stays consistent
    _turn_lock = asyncio.Lock()

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
        args = (context.args or [])
        if not args:
            await update.message.reply_text(
                "使い方: /register <user_id>\n"
                "例: /register default  または  /register honoruru"
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
            await update.message.reply_text(
                f"Telegram ID {tg_id} → {profile.name} ({profile.id})"
            )
        else:
            await update.message.reply_text(
                f"Telegram ID {tg_id} はまだ紐付けされていません。\n"
                "/register <user_id> で登録してください。"
            )

    async def _run_turn(
        update: Update,
        user_input: str,
        images_b64: list[str] | None = None,
    ) -> None:
        """Execute one agent turn and send the response back."""
        tg_user = update.effective_user
        speaker_name, profile_id = _resolve_speaker(tg_user) if tg_user else ("unknown", None)
        prefixed_input = f"[{speaker_name}]: {user_input}"

        await update.message.chat.send_action("typing")  # type: ignore[union-attr]

        chunks: list[str] = []
        captured_images: list[str] = []

        def on_text(chunk: str) -> None:
            chunks.append(chunk)

        def on_image(b64: str) -> None:
            captured_images.append(b64)

        async with _turn_lock:
            if profile_id:
                await agent.switch_user(profile_id)
            try:
                await agent.run(
                    prefixed_input,
                    on_text=on_text,
                    on_image=on_image,
                    desires=desires,
                    user_images=images_b64,
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

        response = "".join(chunks).strip()
        if not response:
            return

        for i in range(0, len(response), _MAX_MSG_LEN):
            await update.message.reply_text(response[i : i + _MAX_MSG_LEN])  # type: ignore[union-attr]

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
            b64 = base64.b64encode(raw).decode()
        except Exception:
            logger.exception("Failed to download Telegram photo")
            await update.message.reply_text("画像のダウンロードに失敗しました。")
            return

        caption = (update.message.caption or "").strip()
        user_input = caption if caption else "この画像を見て。"

        await _run_turn(update, user_input, images_b64=[b64])

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(CommandHandler("clear", _cmd_clear))
    app.add_handler(CommandHandler("help", _cmd_help))
    app.add_handler(CommandHandler("register", _cmd_register))
    app.add_handler(CommandHandler("whoami", _cmd_whoami))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, _handle_photo))

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
