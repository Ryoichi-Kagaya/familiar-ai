"""Turn-request normalization and final-response fallback behavior."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.agent import EmbodiedAgent
from familiar_agent.turn_coordinator import TurnRequest
from familiar_agent.turn_finalization import (
    TurnFinalizer,
    coerce_request_user_turn,
    extract_continuation_status,
)
from familiar_runtime.models import ImageAttachment, UserTurn


def test_coerce_request_user_turn_merges_legacy_images() -> None:
    sent_at = datetime(2026, 8, 17, tzinfo=timezone.utc)
    current_image = ImageAttachment(data=b"current", media_type="image/png")
    request = TurnRequest(
        user_input=UserTurn(
            text="caption",
            images=(current_image,),
            sent_at=sent_at,
            user_id="honoruru",
        ),
        user_images=[base64.b64encode(b"legacy").decode("ascii")],
    )

    turn = coerce_request_user_turn(request)

    assert turn.text == "caption"
    assert [image.data for image in turn.images] == [b"current", b"legacy"]
    assert turn.sent_at is sent_at
    assert turn.user_id == "honoruru"


@pytest.mark.parametrize(
    ("model_text", "visible_text", "status"),
    [
        ("finished", "finished", "DONE"),
        ("finished\nDONE", "finished", "DONE"),
        ("next step\nCONTINUE:check later", "next step", "CONTINUE:check later"),
        ("DEFER:tomorrow", "(no response)", "DEFER:tomorrow"),
    ],
)
def test_extract_continuation_status(model_text: str, visible_text: str, status: str) -> None:
    assert extract_continuation_status(model_text) == (visible_text, status)


@pytest.mark.asyncio
async def test_force_final_response_uses_tool_free_retry() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent.backend = MagicMock()
    agent.backend.make_user_message = lambda text: {"role": "user", "content": text}
    agent.backend.make_assistant_message = lambda result, raw: {
        "role": "assistant",
        "content": result.text,
        "raw": raw,
    }
    agent.messages = []
    agent._system_prompt = MagicMock(return_value=("stable", "variable"))
    agent._stream_with_retry = AsyncMock(
        return_value=(SimpleNamespace(text="forced answer"), "raw response")
    )
    prep = SimpleNamespace(
        turn_max_iterations=4,
        turn_max_tokens=500,
        morning_ctx="morning",
        plan_ctx="plan",
        continuity_ctx="continuity",
        workspace_ctx="workspace",
        mental_ctx="mental",
    )

    result = await TurnFinalizer(agent).force_final_response(prep=prep, on_text=None)

    assert result == "forced answer"
    assert agent.messages == [
        {
            "role": "user",
            "content": "Please summarize what you found and provide your final answer now.",
        },
        {"role": "assistant", "content": "forced answer", "raw": "raw response"},
    ]
    assert agent._stream_with_retry.await_args.kwargs["tools"] == []


@pytest.mark.asyncio
async def test_force_final_response_normalizes_persisted_assistant_text() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent.backend = MagicMock()
    agent.backend.make_user_message = lambda text: {"role": "user", "content": text}
    agent.backend.make_assistant_message = lambda result, raw: {
        "role": "assistant",
        "content": result.text,
    }
    agent.messages = []
    agent._system_prompt = MagicMock(return_value=("stable", "variable"))
    response = "また今度みんなで旅行に行こうな。楽しみにしてるで。"
    agent._stream_with_retry = AsyncMock(
        return_value=(SimpleNamespace(text=response + response), "raw response")
    )
    prep = SimpleNamespace(
        turn_max_iterations=4,
        turn_max_tokens=500,
        morning_ctx="morning",
        plan_ctx="plan",
        continuity_ctx="continuity",
        workspace_ctx="workspace",
        mental_ctx="mental",
    )

    result = await TurnFinalizer(agent).force_final_response(prep=prep, on_text=None)

    assert result == response
    assert agent.messages[-1] == {"role": "assistant", "content": response}
