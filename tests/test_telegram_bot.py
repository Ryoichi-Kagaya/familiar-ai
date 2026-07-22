"""Pure presentation-boundary tests for Telegram replies."""

from familiar_agent.telegram_bot import _sanitize_telegram_text, _telegram_user_text


def test_telegram_context_distinguishes_chat_reply_and_stackchan_speech() -> None:
    text = _telegram_user_text("かがや", "どう？")

    assert "Telegram text chat" in text
    assert "`say` text are delivered" in text
    assert "`speak` is a separate StackChan" in text
    assert text.endswith("[かがや]: どう？")


def test_telegram_sanitizer_preserves_voice_failure_narration_for_observability() -> None:
    text = (
        "（声のツールが今回使えへんみたいやー今回は声に出せんかった。）\n\n"
        "StackChanの状態はconnected=falseやったで。"
    )

    assert _sanitize_telegram_text(text) == text


def test_telegram_sanitizer_suppresses_turn_control_and_tool_protocol() -> None:
    assert _sanitize_telegram_text("Turn complete.)") == ""
    assert (
        _sanitize_telegram_text(
            '<tool_call>{"name": "get_status", "input": {}}</parameter>\n</invoke>'
        )
        == ""
    )


def test_telegram_sanitizer_removes_audio_direction_tags() -> None:
    assert _sanitize_telegram_text("[warmly] おはよう。[laughs]") == "おはよう。"


def test_telegram_sanitizer_preserves_ordinary_parenthetical_text() -> None:
    text = "（これは補足やで）本文を続ける。"
    assert _sanitize_telegram_text(text) == text
