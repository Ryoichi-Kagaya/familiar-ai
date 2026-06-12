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
import logging
import os

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

    # Serialize all agent turns so history stays consistent
    _turn_lock = asyncio.Lock()

    def _is_allowed(user_id: int) -> bool:
        return not allowed_ids or user_id in allowed_ids

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
            "/start — 挨拶\n"
            "/clear — 会話履歴をリセット\n"
            "/help  — このメッセージ"
        )

    async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if update.effective_user is None or not _is_allowed(update.effective_user.id):
            logger.warning("Rejected message from Telegram user %s", update.effective_user)
            return
        if update.message is None:
            return

        user_input = (update.message.text or "").strip()
        if not user_input:
            return

        await update.message.chat.send_action("typing")

        chunks: list[str] = []

        def on_text(chunk: str) -> None:
            chunks.append(chunk)

        async with _turn_lock:
            try:
                await agent.run(user_input, on_text=on_text, desires=desires)
            except Exception:
                logger.exception("Agent error during Telegram turn")
                await update.message.reply_text("エラーが発生しました。ログを確認してください。")
                return

        response = "".join(chunks).strip()
        if not response:
            return

        # Split into ≤4096-char chunks (Telegram hard limit)
        for i in range(0, len(response), _MAX_MSG_LEN):
            await update.message.reply_text(response[i : i + _MAX_MSG_LEN])

        desires.satisfy("greet_companion")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(CommandHandler("clear", _cmd_clear))
    app.add_handler(CommandHandler("help", _cmd_help))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))

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
