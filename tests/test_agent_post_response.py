"""Post-response pipeline orchestration and deferred-context behavior."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from familiar_agent.agent import EmbodiedAgent
from familiar_agent.post_response import PostResponsePipeline


@pytest.mark.asyncio
async def test_post_response_pipeline_runs_stages_in_order_and_flushes() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    pipeline = PostResponsePipeline(agent, generate_plan_fn=AsyncMock(return_value=""))
    order: list[str] = []

    pipeline._process_observation = AsyncMock(
        side_effect=lambda **kwargs: order.append("observation")
    )

    async def persist_conversation(**kwargs) -> str:
        order.append("conversation")
        return "tender"

    pipeline._persist_conversation = AsyncMock(side_effect=persist_conversation)
    pipeline._update_relationship = MagicMock(
        side_effect=lambda **kwargs: order.append("relationship")
    )

    async def persist_curiosity(**kwargs) -> str:
        order.append("curiosity")
        return "window light"

    pipeline._persist_curiosity = AsyncMock(side_effect=persist_curiosity)
    agent._capture_companion_thread = AsyncMock(
        side_effect=lambda *args: order.append("companion_thread")
    )
    pipeline._update_continuity = MagicMock(side_effect=lambda **kwargs: order.append("continuity"))
    agent._maybe_adapt_values = AsyncMock(side_effect=lambda **kwargs: order.append("values"))
    agent._maybe_update_identity = AsyncMock(side_effect=lambda **kwargs: order.append("identity"))
    pipeline._refresh_deferred_context = AsyncMock(
        side_effect=lambda **kwargs: order.append("deferred_context")
    )
    agent._self_state = MagicMock()
    agent._self_state.flush = MagicMock(side_effect=lambda: order.append("flush"))

    await pipeline.run(
        user_input="どう見えた？",
        final_text="窓の光が気になった。",
        camera_used=True,
        observation_action_name="look",
        observation_action_input={"direction": "left"},
        companion_mood="engaged",
        is_desire_turn=False,
        desires=MagicMock(),
    )

    assert order == [
        "observation",
        "conversation",
        "relationship",
        "curiosity",
        "companion_thread",
        "continuity",
        "values",
        "identity",
        "deferred_context",
        "flush",
    ]


@pytest.mark.asyncio
async def test_post_response_pipeline_stops_after_failure_but_still_flushes() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    pipeline = PostResponsePipeline(agent, generate_plan_fn=AsyncMock(return_value=""))
    pipeline._process_observation = AsyncMock()
    pipeline._persist_conversation = AsyncMock(side_effect=RuntimeError("memory unavailable"))
    pipeline._update_relationship = MagicMock()
    agent._self_state = MagicMock()

    await pipeline.run(
        user_input="hello",
        final_text="hi",
        camera_used=False,
        observation_action_name=None,
        observation_action_input=None,
        companion_mood="engaged",
        is_desire_turn=False,
        desires=None,
    )

    pipeline._update_relationship.assert_not_called()
    agent._self_state.flush.assert_called_once_with()


@pytest.mark.asyncio
async def test_deferred_turn_context_updates_cache_and_frustration_drive() -> None:
    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent.backend = MagicMock()
    agent._utility_backend = agent.backend
    agent._gather_workspace_context = AsyncMock(return_value="workspace")
    agent._infer_companion_mood = AsyncMock(return_value="frustrated")
    agent._online_temporal_context = AsyncMock(return_value="temporal")
    desires = MagicMock()
    pipeline = PostResponsePipeline(agent, generate_plan_fn=AsyncMock(return_value=""))

    await pipeline._refresh_deferred_context(
        user_input="今日はしんどい",
        is_desire_turn=False,
        desires=desires,
    )

    assert agent._cached_plan_ctx == ""
    assert agent._cached_workspace_ctx == "workspace"
    assert agent._cached_companion_mood == "frustrated"
    assert agent._cached_temporal_ctx == "temporal"
    desires.boost.assert_called_once_with("worry_companion", 0.3)
