"""Tests for the EmbodiedAgent ReAct loop (run() method)."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from familiar_agent.backend import ToolCall, TurnResult
from familiar_agent.desires import DesireSystem
from familiar_agent.exploration import ExplorationTracker


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _turn(stop: str, text: str = "", tool_calls: list | None = None) -> TurnResult:
    return TurnResult(
        stop_reason=stop,
        text=text,
        tool_calls=tool_calls or [],
        input_tokens=100,
        output_tokens=50,
    )


def _make_agent(*, with_tts: bool = False, with_camera: bool = False, with_mcp: bool = False):
    """Minimal EmbodiedAgent with all heavy dependencies mocked out."""
    from familiar_agent.agent import EmbodiedAgent

    agent = EmbodiedAgent.__new__(EmbodiedAgent)
    agent.config = MagicMock()
    agent.config.max_tokens = 1000
    agent.config.agent_name = "Kokone"
    agent.config.companion_name = "Kouta"
    # MagicMock attrs are truthy — pin flag-gated subsystems to their real
    # defaults so tests exercise the same paths a default install does.
    agent.config.consciousness_profile = False
    agent.config.reality_gate = False
    agent.config.inner_dense = False
    agent.config.inner_min_interval = 5.0
    agent.config.sleep_consolidation = False
    agent.config.dream_mode = False
    agent.config.experience_ledger = False

    agent._turn_count = 0
    agent._session_input_tokens = 0
    agent._session_output_tokens = 0
    agent._last_context_tokens = 0
    agent._post_compact = False
    agent._background_tasks = set()
    agent._cached_plan_ctx = ""
    agent._cached_workspace_ctx = ""
    agent._cached_temporal_ctx = None
    agent._cached_companion_mood = "engaged"
    agent._started_at = 0.0
    agent.messages = []
    agent._me_md = ""

    # Backend: make_tool_results must accept (tool_calls, results) and return a list
    backend = MagicMock()
    backend.emits_reasoning = False
    backend.complete = AsyncMock(return_value="")
    backend.make_user_message = lambda t: {"role": "user", "content": t}
    backend.make_assistant_message = lambda result, raw: {
        "role": "assistant",
        "content": result.text,
    }
    backend.make_tool_results = MagicMock(return_value=[{"role": "tool", "content": "ok"}])
    agent.backend = backend
    agent._utility_backend = backend

    # Memory
    mem = MagicMock()
    mem.is_embedding_ready = MagicMock(return_value=True)
    mem.recall_async = AsyncMock(return_value=[])
    mem.recent_feelings_async = AsyncMock(return_value=[])
    mem.recall_self_model_async = AsyncMock(return_value=[])
    mem.recall_curiosities_async = AsyncMock(return_value=[])
    mem.recall_day_summaries_async = AsyncMock(return_value=[])
    mem.recall_semantic_facts_async = AsyncMock(return_value=[])
    mem.recall_behavior_policies_async = AsyncMock(return_value=[])
    mem.format_for_context = MagicMock(return_value="")
    mem.format_feelings_for_context = MagicMock(return_value="")
    mem.format_day_summaries_for_context = MagicMock(return_value="")
    mem.format_semantic_facts_for_context = MagicMock(return_value="")
    mem.format_behavior_policies_for_context = MagicMock(return_value="")
    mem.format_self_model_for_context = MagicMock(return_value="")
    mem.format_curiosities_for_context = MagicMock(return_value="")
    mem.save_async = AsyncMock()
    mem.adjust_semantic_fact_confidence_async = AsyncMock(return_value=None)
    mem.adjust_behavior_policy_confidence_async = AsyncMock(return_value=None)
    mem.get_dates_with_observations = MagicMock(return_value=[])
    mem.get_dates_with_summaries = MagicMock(return_value=[])
    mem.as_coalition_async = AsyncMock(return_value=None)
    agent._memory = mem

    mem_tool = MagicMock()
    mem_tool.get_tool_definitions = MagicMock(return_value=[])
    mem_tool.call = AsyncMock(return_value=("remembered", None))
    agent._memory_tool = mem_tool

    tom = MagicMock()
    tom.get_tool_definitions = MagicMock(return_value=[])
    tom.call = AsyncMock(return_value=("tom result", None))
    agent._tom_tool = tom

    coding = MagicMock()
    coding.get_tool_definitions = MagicMock(return_value=[])
    coding.call = AsyncMock(return_value=("code result", None))
    agent._coding = coding

    art_critique = MagicMock()
    art_critique.get_tool_definitions = MagicMock(return_value=[])
    art_critique.call = AsyncMock(return_value=("art result", None))
    agent._art_critique_tool = art_critique

    agent._camera = None
    agent._camera_gui_priority = False
    agent._mobility = None
    agent._mcp = None

    if with_tts:
        tts_tool = MagicMock()
        tts_tool.get_tool_definitions = MagicMock(return_value=[{"name": "say"}])
        tts_tool.call = AsyncMock(return_value=("spoken", None))
        agent._tts = tts_tool
    else:
        agent._tts = None

    if with_camera:
        cam = MagicMock()
        cam.get_tool_definitions = MagicMock(return_value=[{"name": "see"}, {"name": "look"}])
        cam.call = AsyncMock(return_value=("I see a room", "base64img"))
        agent._camera = cam

    if with_mcp:
        mcp_client = MagicMock()
        mcp_client.get_tool_definitions = MagicMock(return_value=[])
        mcp_client.call = AsyncMock(return_value=("mcp result", None))
        mcp_client.is_started = True
        agent._mcp = mcp_client

    agent._exploration = ExplorationTracker()
    agent._scene = None

    from familiar_agent.self_narrative import SelfNarrative
    from familiar_agent.relationship import RelationshipTracker
    from familiar_agent.workspace import GlobalWorkspace
    from familiar_agent.prediction import PredictionEngine
    from familiar_agent.attention_schema import AttentionSchema
    import time as _time

    agent._self_narrative = SelfNarrative()
    agent._relationship = RelationshipTracker()
    agent._self_state = MagicMock()
    agent._self_state.snapshot = MagicMock(return_value={"unresolved_tension": 0.2})
    agent._workspace = GlobalWorkspace()
    agent._prediction = PredictionEngine()
    agent._attention_schema = AttentionSchema()
    agent._dmn = MagicMock()
    agent._dmn.wander = AsyncMock(return_value=None)
    agent._meta_monitor = MagicMock()
    agent._meta_monitor.as_coalition = MagicMock(return_value=None)
    agent._meta_monitor.record_step = MagicMock()
    agent._memory_worker = MagicMock()
    agent._memory_worker.is_running = True
    agent._mood = "neutral"
    agent._mood_intensity = 0.0

    # Inner loop (Phase 2): live tick state; the loop itself is not started.
    from familiar_agent.inner_loop import InnerLoopConfig, TrainOfThought
    from collections import deque

    agent._turn_active = False
    agent._desires = None
    agent._inner_monologue = deque(maxlen=8)
    agent._train_of_thought = TrainOfThought()
    agent._inner_tick_count = 0
    agent._inner_escalated_at = {}
    agent._inner_loop_config = InnerLoopConfig()
    agent._inner_backend = None
    agent._last_micro_thought_at = 0.0

    # Per-turn cognition pipeline (PR3 runtime reorg).  __new__ skipped
    # the EmbodiedAgent.__init__ that normally wires the hook, so attach
    # one here so agent.run() can delegate prepare_turn / commit_after_end_turn.
    from familiar_neighbor.embodied_hook import EmbodiedAgentHook

    agent._hook = EmbodiedAgentHook(agent)
    agent._mood_set_at = _time.time()

    return agent


def _heavy_patches() -> dict:
    """Create isolated mocks for sub-calls outside ReAct-loop test scope."""
    return {
        "familiar_agent.agent.EmbodiedAgent._morning_reconstruction": AsyncMock(return_value=""),
        "familiar_agent.agent.EmbodiedAgent._infer_companion_mood": AsyncMock(
            return_value="engaged"
        ),
        "familiar_agent.agent.EmbodiedAgent._infer_emotion": AsyncMock(return_value="neutral"),
        "familiar_agent.agent.EmbodiedAgent._summarize_exchange": AsyncMock(return_value="summary"),
        "familiar_agent.agent.EmbodiedAgent._online_temporal_context": AsyncMock(return_value=None),
        # These run() unit tests exercise turn behavior, not workspace competition.
        # Avoid real asyncio.to_thread calls so repeated turns stay isolated from
        # executor scheduling and teardown behavior in constrained test runners.
        "familiar_agent.agent.EmbodiedAgent._gather_workspace_context": AsyncMock(return_value=""),
        "familiar_agent.agent.EmbodiedAgent._run_post_response_pipeline": AsyncMock(),
        "familiar_agent.agent.EmbodiedAgent._update_self_model": AsyncMock(),
        "familiar_agent.agent.EmbodiedAgent._maybe_update_self_narrative": AsyncMock(),
        "familiar_agent.agent.EmbodiedAgent._maybe_adapt_values": AsyncMock(),
        "familiar_agent.agent.EmbodiedAgent.extract_curiosity": AsyncMock(return_value=None),
        "familiar_agent.agent.generate_plan": AsyncMock(return_value=""),
        "familiar_agent.agent.check_plan_blocked": AsyncMock(return_value=False),
    }


@contextmanager
def _patch_heavy(extra: dict | None = None):
    """Temporarily suppress heavy async sub-calls, with optional overrides."""
    patches = _heavy_patches()
    if extra:
        patches.update(extra)
    with ExitStack() as stack:
        for target, new in patches.items():
            stack.enter_context(patch(target, new))
        yield


# ---------------------------------------------------------------------------
# Tests: basic single-turn end_turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_end_turn_returns_text():
    """run() with immediate end_turn returns the model's text."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="Hello!"), "Hello!"))

    with _patch_heavy():
        result = await agent.run("こんにちは")

    assert result == "Hello!"


@pytest.mark.asyncio
async def test_brief_greeting_turn_uses_only_say_and_skips_heavy_prep():
    agent = _make_agent(with_tts=True, with_camera=True)
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="おはよう。"), "おはよう。")
    )

    morning_mock = AsyncMock(return_value="morning context")
    companion_mood_mock = AsyncMock(return_value="engaged")
    workspace_mock = AsyncMock(return_value="[workspace]")
    patches = _heavy_patches()
    patches["familiar_agent.agent.EmbodiedAgent._morning_reconstruction"] = morning_mock
    patches["familiar_agent.agent.EmbodiedAgent._infer_companion_mood"] = companion_mood_mock
    patches["familiar_agent.agent.EmbodiedAgent._gather_workspace_context"] = workspace_mock

    with _patch_heavy(patches):
        result = await agent.run("おはよう")

    assert result == "おはよう。"
    stream_kwargs = agent.backend.stream_turn.await_args.kwargs
    assert stream_kwargs["tools"] == [{"name": "say"}]
    assert stream_kwargs["max_tokens"] == 120
    morning_mock.assert_not_awaited()
    companion_mood_mock.assert_not_awaited()
    workspace_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_excluded_tools_are_removed_from_turn_surface():
    agent = _make_agent(with_tts=True, with_camera=True)
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="確認したで。"), "確認したで。")
    )

    with _patch_heavy():
        await agent.run("周りの様子を詳しく確認して", excluded_tools=frozenset({"look"}))

    tool_names = {tool["name"] for tool in agent.backend.stream_turn.await_args.kwargs["tools"]}
    assert "look" not in tool_names
    assert "see" in tool_names
    assert "say" in tool_names


@pytest.mark.parametrize(("emits_reasoning", "expected_max_tokens"), [(True, 800), (False, 120)])
@pytest.mark.asyncio
async def test_brief_reply_uses_backend_specific_token_cap(
    emits_reasoning: bool,
    expected_max_tokens: int,
):
    agent = _make_agent(with_tts=True)
    agent.backend.emits_reasoning = emits_reasoning
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="おはよう。"), "おはよう。")
    )

    with _patch_heavy():
        await agent.run("おはよう")

    assert agent.backend.stream_turn.await_args.kwargs["max_tokens"] == expected_max_tokens


@pytest.mark.parametrize(
    "user_input",
    [
        "MCP見える？",
        "使えるツールを教えて",
        "Which tools work?",
        "ウェブ検索できる？",
        "インターネット見れる？",
        "今日の天気を検索できる？",
        "Can you search the web?",
    ],
)
def test_tool_capability_question_is_not_treated_as_brief_reply(user_input: str):
    """Capability questions must retain the complete outer tool catalog."""
    agent = _make_agent()
    social_policy = SimpleNamespace(primary_act="clarification")

    assert not agent._should_use_brief_reply_mode(
        user_input=user_input,
        social_policy=social_policy,
        is_desire_turn=False,
    )


@pytest.mark.asyncio
async def test_run_increments_turn_count():
    """run() increments _turn_count on each invocation."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="Hi"), "Hi"))

    with _patch_heavy():
        assert agent._turn_count == 0
        await agent.run("test")
        assert agent._turn_count == 1
        await agent.run("test2")
        assert agent._turn_count == 2


@pytest.mark.asyncio
async def test_run_appends_user_message_to_history():
    """run() appends the user message to agent.messages."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="response"), "response")
    )

    with _patch_heavy():
        assert len(agent.messages) == 0
        await agent.run("hello from user")
        user_messages = [m for m in agent.messages if m.get("role") == "user"]
        assert user_messages
        assert "sent_at=" in user_messages[0]["content"]
        assert "weekday=" in user_messages[0]["content"]
        assert "hello from user" in user_messages[0]["content"]


@pytest.mark.asyncio
async def test_desire_turn_uses_history_but_does_not_persist_its_messages():
    agent = _make_agent()
    original_messages = [{"role": "user", "content": "earlier companion turn"}]
    agent.messages = original_messages
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="private autonomous reply"), "raw")
    )

    with _patch_heavy():
        result = await agent.run("", inner_voice="look around")

    assert result == "private autonomous reply"
    assert agent.messages is original_messages
    assert agent.messages == [{"role": "user", "content": "earlier companion turn"}]
    turn_messages = agent.backend.stream_turn.await_args.kwargs["messages"]
    assert turn_messages is not original_messages
    assert original_messages[0] in turn_messages
    assert any(message.get("content") == "private autonomous reply" for message in turn_messages)


@pytest.mark.asyncio
async def test_desire_turn_restores_history_when_prepare_fails():
    agent = _make_agent()
    original_messages = [{"role": "user", "content": "keep me"}]
    agent.messages = original_messages
    agent._hook.prepare_turn = AsyncMock(side_effect=RuntimeError("prepare failed"))

    with pytest.raises(RuntimeError, match="prepare failed"):
        await agent.run("", inner_voice="reflect")

    assert agent.messages is original_messages
    assert agent.messages == [{"role": "user", "content": "keep me"}]
    assert agent._turn_active is False


@pytest.mark.asyncio
async def test_repeated_tool_failure_raises_self_protect_without_irritable_tone(tmp_path):
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="落ち着いて進めよう。"), "落ち着いて進めよう。")
    )
    agent._tool_failure_streak = 3
    desires = DesireSystem(state_path=tmp_path / "desires.json", companion_name="Kota")

    with _patch_heavy():
        result = await agent.run("助けて", desires=desires)

    assert desires.level("self_protect") > 0.0
    assert "ugh" not in result.lower()
    assert "annoy" not in result.lower()


@pytest.mark.asyncio
async def test_existing_no_hardware_mode_still_works_with_mental_pipeline():
    agent = _make_agent(with_tts=False, with_camera=False, with_mcp=False)
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="hardwareなしでも動く"), "hardwareなしでも動く")
    )

    with _patch_heavy():
        result = await agent.run("こんにちは")

    assert result == "hardwareなしでも動く"


@pytest.mark.asyncio
async def test_run_accumulates_tokens():
    """run() adds input/output tokens to session totals."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(
        return_value=(
            _turn(
                "end_turn",
                text="ok",
            ),
            "ok",
        )
    )
    # Override to set token counts
    result_obj = TurnResult(stop_reason="end_turn", text="ok", input_tokens=200, output_tokens=80)
    agent.backend.stream_turn = AsyncMock(return_value=(result_obj, "ok"))

    with _patch_heavy():
        await agent.run("test")

    assert agent._session_input_tokens == 200
    assert agent._session_output_tokens == 80


# ---------------------------------------------------------------------------
# Tests: tool_use → end_turn sequence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_tool_use_then_end_turn():
    """run() executes a tool call then gets end_turn on the next iteration."""
    agent = _make_agent()

    tc = ToolCall(id="tc1", name="remember", input={"content": "test memory"})
    turn1 = TurnResult(stop_reason="tool_use", text="", tool_calls=[tc])
    turn2 = TurnResult(stop_reason="end_turn", text="Done!", tool_calls=[])

    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (turn1, None),
            (turn2, "Done!"),
        ]
    )

    with _patch_heavy():
        result = await agent.run("remember something")

    assert result == "Done!"
    assert agent._memory_tool.call.called


@pytest.mark.asyncio
async def test_run_tool_results_added_to_messages():
    """Tool results are added to message history after tool execution."""
    agent = _make_agent()

    tc = ToolCall(id="tc1", name="remember", input={"content": "hi"})
    turn1 = TurnResult(stop_reason="tool_use", text="", tool_calls=[tc])
    turn2 = TurnResult(stop_reason="end_turn", text="Saved.", tool_calls=[])

    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (turn1, None),
            (turn2, "Saved."),
        ]
    )

    with _patch_heavy():
        await agent.run("please remember")

    # make_tool_results was called with the tool call and its result
    assert agent.backend.make_tool_results.called


@pytest.mark.asyncio
async def test_run_tool_timeout_is_returned_as_tool_result():
    """A slow tool call is converted into a textual timeout result."""
    agent = _make_agent()
    tc = ToolCall(id="tc1", name="remember", input={"content": "slow"})
    turn1 = TurnResult(stop_reason="tool_use", text="", tool_calls=[tc])
    turn2 = TurnResult(stop_reason="end_turn", text="Done.", tool_calls=[])
    agent.backend.stream_turn = AsyncMock(side_effect=[(turn1, None), (turn2, "Done.")])

    async def _slow_execute(self, name, tool_input):  # noqa: ARG001
        await asyncio.sleep(0.2)
        return "late result", None

    patches = _heavy_patches()
    patches["familiar_agent.agent.EmbodiedAgent._execute_tool"] = _slow_execute
    patches["familiar_agent.agent.EmbodiedAgent._tool_timeout_seconds"] = MagicMock(
        return_value=0.01
    )

    with _patch_heavy(patches):
        result = await agent.run("remember slowly")

    assert result == "Done."
    collected = agent.backend.make_tool_results.call_args.args[1]
    assert collected[0][0].startswith("Tool timeout: remember exceeded")
    assert agent._last_tool_error == collected[0][0]
    assert agent._tool_failure_streak == 1


@pytest.mark.asyncio
async def test_run_passes_latest_pre_see_action_into_scene_update():
    """The last embodied action before see() conditions the scene update."""
    from familiar_agent.agent import EmbodiedAgent

    agent = _make_agent(with_camera=True)
    agent._scene = MagicMock()
    agent._scene.update = AsyncMock(return_value=[])
    agent._scene.context_for_prompt = MagicMock(return_value="")
    agent._scene_backend = MagicMock()
    agent._camera.call = AsyncMock(
        side_effect=[
            ("looked left", None),
            ("I see a room", "base64img"),
        ]
    )

    turn1 = TurnResult(
        stop_reason="tool_use",
        text="",
        tool_calls=[
            ToolCall(id="tc1", name="look", input={"direction": "left", "degrees": 45}),
            ToolCall(id="tc2", name="see", input={}),
        ],
    )
    turn2 = TurnResult(stop_reason="end_turn", text="There is a window.", tool_calls=[])
    agent.backend.stream_turn = AsyncMock(
        side_effect=[(turn1, None), (turn2, "There is a window.")]
    )

    patches = _heavy_patches()
    patches["familiar_agent.agent.EmbodiedAgent._run_post_response_pipeline"] = (
        EmbodiedAgent._run_post_response_pipeline
    )

    with _patch_heavy(patches):
        await agent.run("look and report")
        await agent._drain_background_tasks(timeout=0.5)

    agent._scene.update.assert_awaited_once()
    _, kwargs = agent._scene.update.call_args
    assert kwargs["action_name"] == "look"
    assert kwargs["action_input"] == {"direction": "left", "degrees": 45}


# ---------------------------------------------------------------------------
# Tests: auto-say
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_auto_say_fires_when_tts_available_and_no_say_call():
    """When TTS is present and model wrote text without calling say(), auto-say fires."""
    agent = _make_agent(with_tts=True)
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="Hello, I speak!"), "Hello, I speak!")
    )

    with _patch_heavy():
        await agent.run("speak to me")

    agent._tts.call.assert_awaited_once()
    call_args = agent._tts.call.call_args
    assert call_args[0][0] == "say"


@pytest.mark.asyncio
async def test_run_no_auto_say_when_tts_absent():
    """Without TTS, no auto-say even if model wrote text."""
    agent = _make_agent(with_tts=False)
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="Silent response"), "Silent response")
    )

    with _patch_heavy():
        await agent.run("respond")

    assert agent._tts is None


@pytest.mark.asyncio
async def test_run_no_auto_say_when_say_already_called():
    """If say() was called as a tool, auto-say should NOT fire again."""
    agent = _make_agent(with_tts=True)

    tc = ToolCall(id="tc1", name="say", input={"text": "I spoke"})
    turn1 = TurnResult(stop_reason="tool_use", text="", tool_calls=[tc])
    turn2 = TurnResult(stop_reason="end_turn", text="done", tool_calls=[])

    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (turn1, None),
            (turn2, "done"),
        ]
    )

    with _patch_heavy():
        await agent.run("speak via tool")

    # say() was called once via tool execution; auto-say must NOT add a second call
    assert agent._tts.call.call_count == 1


# ---------------------------------------------------------------------------
# Tests: morning reconstruction on first turn
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("initial_turn_count", "expected_calls"), [(0, 1), (5, 0)])
@pytest.mark.asyncio
async def test_run_only_reconstructs_morning_on_first_turn(
    initial_turn_count: int,
    expected_calls: int,
):
    agent = _make_agent()
    agent._turn_count = initial_turn_count
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="reply"), "reply"))

    morning_mock = AsyncMock(return_value="morning context")
    patches = _heavy_patches()
    patches["familiar_agent.agent.EmbodiedAgent._morning_reconstruction"] = morning_mock

    with _patch_heavy(patches):
        await agent.run("今日はどう？")

    assert morning_mock.await_count == expected_calls


# ---------------------------------------------------------------------------
# Tests: online temporal self + adaptive values
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_injects_online_temporal_context_into_user_message():
    agent = _make_agent()
    agent._turn_count = 1  # next run is a normal turn, not the first turn
    # Temporal context is now read from cache (populated by post-response pipeline)
    agent._cached_temporal_ctx = "[Temporal self]\n[Resurfaced memory]: 朝の空を探した"
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="reply"), "reply"))

    with _patch_heavy():
        await agent.run("今日はどう？")

    user_messages = [m["content"] for m in agent.messages if m.get("role") == "user"]
    assert any("[Temporal self]" in msg for msg in user_messages)


@pytest.mark.asyncio
async def test_maybe_update_self_narrative_uses_agency_error_trigger():
    agent = _make_agent()
    agent._utility_backend.complete = AsyncMock(return_value="ウチは少し揺れながら確かめ直した。")
    agent._self_narrative.write = MagicMock()
    agent._prediction.last_signal = MagicMock(
        return_value=SimpleNamespace(action_name="look", agency_error=0.72)
    )

    await agent._maybe_update_self_narrative(
        user_input="何が見えた？",
        final_text="まだ少しずれてる気がする",
        emotion="neutral",
        is_desire_turn=False,
    )

    agent._self_narrative.write.assert_called_once()
    assert agent._self_narrative.write.call_args.kwargs["trigger"] == "agency_error"


@pytest.mark.asyncio
async def test_maybe_adapt_values_updates_curiosity_and_support_policies():
    agent = _make_agent()
    agent._prediction.last_signal = MagicMock(
        return_value=SimpleNamespace(action_name="look", agency_error=0.68)
    )
    desires = MagicMock()
    desires.boost = MagicMock()

    await agent._maybe_adapt_values(
        user_input="大丈夫？",
        final_text="窓の向こうの空が気になったよ。",
        emotion="tender",
        camera_used=True,
        curiosity="窓の向こうの空",
        is_desire_turn=False,
        desires=desires,
    )

    calls = agent._memory.adjust_behavior_policy_confidence_async.await_args_list
    assert len(calls) >= 2
    assert any(call.args[:2] == ("curiosity:active", 0.08) for call in calls)
    assert any(call.args[:2] == ("curiosity:active", -0.05) for call in calls)
    assert any(call.args[:2] == ("conversation:supportive_style", 0.04) for call in calls)
    desires.boost.assert_called_once_with("share_memory", 0.08)


# ---------------------------------------------------------------------------
# Tests: empty / edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_empty_text_returns_no_response_placeholder():
    """When model returns no text, run() returns the placeholder string."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text=""), ""))

    with _patch_heavy():
        result = await agent.run("hi")

    assert result == "(no response)"


@pytest.mark.asyncio
async def test_run_schedules_post_response_pipeline_without_blocking_reply():
    """Post-response work should happen in the background after the reply is ready."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="Hello!"), "Hello!"))

    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_pipeline(self, **kwargs):  # noqa: ARG001
        started.set()
        await release.wait()

    patches = _heavy_patches()
    patches["familiar_agent.agent.EmbodiedAgent._run_post_response_pipeline"] = _slow_pipeline

    with _patch_heavy(patches):
        try:
            result = await agent.run("こんにちは")
            assert result == "Hello!"
            await asyncio.wait_for(started.wait(), timeout=0.5)
            assert any(not task.done() for task in agent._background_tasks)
        finally:
            release.set()
            await agent._drain_background_tasks(timeout=0.5)


@pytest.mark.asyncio
async def test_run_skips_tape_plan_when_no_separate_utility_backend():
    """No separate utility backend -> skip the extra TAPE planning round-trip."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="Hello!"), "Hello!"))

    plan_mock = AsyncMock(return_value="1. say")
    patches = _heavy_patches()
    patches["familiar_agent.agent.generate_plan"] = plan_mock

    with _patch_heavy(patches):
        await agent.run("こんにちは")

    plan_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests: deterministic ToM wiring (auto_tom_ctx -> mental_ctx -> system prompt)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flagged_turn_runs_auto_tom_off_critical_path():
    """A venting turn (should_use_tom=True) still runs ToM deterministically —
    but as a BACKGROUND task (roadmap PR7): it must not block the turn, and its
    value persists via the person model rather than this turn's prompt."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="うん"), "うん"))

    auto_tom = AsyncMock(return_value="TOM-SENTINEL-XYZ")
    patches = _heavy_patches()
    patches["familiar_agent.agent.EmbodiedAgent._run_auto_tom"] = auto_tom

    with _patch_heavy(patches):
        await agent.run("むかつくわ、ほんまに最悪な一日や")
        await agent._drain_background_tasks()

    auto_tom.assert_awaited_once()
    # Cooldown bookkeeping advances at trigger time.
    assert agent._last_auto_tom_turn == agent._turn_count
    # Off the critical path: nothing ToM-shaped is injected into THIS prompt.
    system = agent.backend.stream_turn.await_args.kwargs.get("system")
    if system is None:
        system = agent.backend.stream_turn.await_args.args[0]
    joined = "\n".join(system) if isinstance(system, tuple) else str(system)
    assert "TOM-SENTINEL-XYZ" not in joined


@pytest.mark.asyncio
async def test_brief_greeting_turn_skips_auto_tom():
    agent = _make_agent(with_tts=True)
    agent.backend.stream_turn = AsyncMock(
        return_value=(_turn("end_turn", text="おはよう。"), "おはよう。")
    )

    auto_tom = AsyncMock(return_value="TOM-SENTINEL-XYZ")
    patches = _heavy_patches()
    patches["familiar_agent.agent.EmbodiedAgent._run_auto_tom"] = auto_tom

    with _patch_heavy(patches):
        await agent.run("おはよう")

    auto_tom.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests: deferred-topic capture (user says "後で話す" -> unfinished business)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deferral_is_recorded_as_unfinished_business():
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="ええよ"), "ええよ"))
    agent._memory.list_unfinished_business_async = AsyncMock(return_value=[])
    open_mock = AsyncMock(return_value="biz-1")
    agent._memory.open_unfinished_business_async = open_mock

    with _patch_heavy():
        await agent.run("その話はあとで話すわ、ごめんな")

    open_mock.assert_awaited_once()
    summary = open_mock.await_args.args[0]
    assert summary.startswith("deferred topic: ")
    assert "あとで話す" in summary
    assert open_mock.await_args.kwargs.get("source") == "deferral"


@pytest.mark.asyncio
async def test_duplicate_deferral_not_recorded_twice():
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="ええよ"), "ええよ"))
    existing = {"id": "biz-1", "summary": "deferred topic: その話はあとで話すわ、ごめんな"}
    agent._memory.list_unfinished_business_async = AsyncMock(return_value=[existing])
    open_mock = AsyncMock(return_value="biz-2")
    agent._memory.open_unfinished_business_async = open_mock

    with _patch_heavy():
        await agent.run("その話はあとで話すわ、ごめんな")

    open_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_normal_input_records_no_deferral():
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="ん"), "ん"))
    agent._memory.list_unfinished_business_async = AsyncMock(return_value=[])
    open_mock = AsyncMock(return_value="biz-1")
    agent._memory.open_unfinished_business_async = open_mock

    with _patch_heavy():
        await agent.run("今日は新しいカメラの設定をいじっててんけど、なかなか難しいわ")

    open_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_deferral_dedup_sees_beyond_surfaced_top3():
    """A duplicate whose twin sits at position >=4 must still be deduped."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="ええよ"), "ええよ"))
    dup = {"id": "biz-4", "summary": "deferred topic: その話はあとで話すわ、ごめんな"}
    fresher = [{"id": f"biz-{i}", "summary": f"other {i}"} for i in range(3)]
    agent._memory.list_unfinished_business_async = AsyncMock(return_value=fresher + [dup])
    open_mock = AsyncMock(return_value="biz-5")
    agent._memory.open_unfinished_business_async = open_mock

    with _patch_heavy():
        await agent.run("その話はあとで話すわ、ごめんな")

    open_mock.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests: companion-thread surfacing ("presentation tomorrow" -> follow up later)
# ---------------------------------------------------------------------------


def _system_text(agent) -> str:
    system = agent.backend.stream_turn.await_args.kwargs.get("system")
    if system is None:
        system = agent.backend.stream_turn.await_args.args[0]
    return "\n".join(system) if isinstance(system, tuple) else str(system)


@pytest.mark.asyncio
async def test_companion_threads_render_in_their_own_block():
    """Threads get a follow-up instruction distinct from plain unfinished
    business — otherwise the model resolves "presentation tomorrow" right
    after wishing good luck, killing the next-day follow-up."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="ん"), "ん"))
    items = [
        {"id": "biz-1234567", "summary": "deferred topic: 旅行の話", "source": "deferral"},
        {"id": "thr-1234567", "summary": "presentation tomorrow", "source": "companion_thread"},
    ]
    agent._memory.list_unfinished_business_async = AsyncMock(return_value=items)
    agent._memory.open_unfinished_business_async = AsyncMock(return_value=None)

    with _patch_heavy():
        await agent.run("今日は新しいカメラの設定をいじっててんけど、なかなか難しいわ")

    joined = _system_text(agent)
    assert "[Companion's life threads" in joined
    assert "presentation tomorrow" in joined
    assert "[Open unfinished business" in joined
    assert "旅行の話" in joined
    # The thread must NOT sit under the resolve-once-addressed header
    business_block = joined.split("[Companion's life threads")[0]
    assert "presentation tomorrow" not in business_block.split("[Open unfinished business")[-1]


@pytest.mark.asyncio
async def test_all_three_stored_threads_render():
    """Render cap must match the storage cap (3): the model can only resolve
    what it sees, so a stored-but-hidden thread could only leave via expiry."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="ん"), "ん"))
    items = [
        {"id": f"thr-{i}", "summary": f"thread number {i}", "source": "companion_thread"}
        for i in range(3)
    ]
    agent._memory.list_unfinished_business_async = AsyncMock(return_value=items)
    agent._memory.open_unfinished_business_async = AsyncMock(return_value=None)

    with _patch_heavy():
        await agent.run("今日は新しいカメラの設定をいじっててんけど、なかなか難しいわ")

    joined = _system_text(agent)
    for i in range(3):
        assert f"thread number {i}" in joined


@pytest.mark.asyncio
async def test_no_thread_block_when_no_threads():
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="ん"), "ん"))
    items = [{"id": "biz-1", "summary": "deferred topic: 旅行の話", "source": "deferral"}]
    agent._memory.list_unfinished_business_async = AsyncMock(return_value=items)
    agent._memory.open_unfinished_business_async = AsyncMock(return_value=None)

    with _patch_heavy():
        await agent.run("今日は新しいカメラの設定をいじっててんけど、なかなか難しいわ")

    assert "[Companion's life threads" not in _system_text(agent)


# ---------------------------------------------------------------------------
# Tests: thin-wrap migration — the four inline loop behaviours now ride the
# substrate ReActLoop via EmbodiedAgentHook lifecycle methods.
# ---------------------------------------------------------------------------


def _user_texts(agent) -> list[str]:
    return [
        m["content"]
        for m in agent.messages
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)
    ]


@pytest.mark.asyncio
async def test_say_reminder_injected_after_two_silent_tools():
    """Two non-say tool iterations without say() must inject the reminder."""
    agent = _make_agent(with_camera=True)
    tc1 = ToolCall(id="t1", name="see", input={})
    tc2 = ToolCall(id="t2", name="look", input={"direction": "left"})
    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (_turn("tool_use", tool_calls=[tc1]), None),
            (_turn("tool_use", tool_calls=[tc2]), None),
            (_turn("end_turn", text="見えたで"), "見えたで"),
        ]
    )

    with _patch_heavy():
        result = await agent.run("外どうなってる？")

    assert result == "見えたで"
    reminders = [t for t in _user_texts(agent) if t.startswith("REMINDER: Writing text is silent")]
    assert reminders


@pytest.mark.asyncio
async def test_brief_reply_ends_turn_after_say():
    """Brief-reply turns must stop after say(); do not loop on the same phrase."""
    agent = _make_agent(with_tts=True)
    tc = ToolCall(id="t1", name="say", input={"text": "hello"})
    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (_turn("tool_use", tool_calls=[tc]), None),
            (_turn("end_turn", text=""), "done"),
        ]
    )

    with _patch_heavy():
        await agent.run("おはよう")

    assert any("You already spoke" in t for t in _user_texts(agent))


@pytest.mark.asyncio
async def test_normal_turn_stops_and_suppresses_second_say():
    """A normal conversational turn speaks once even if the model calls say twice."""
    agent = _make_agent(with_tts=True)
    # Long input (>80 chars) forces a normal, non-brief-reply turn so all tools
    # stay available and say() alone does not end the turn.
    long_input = (
        "ラズパイで使える7インチ以下のUSB-C一本で電源と映像の両方いけるディスプレイを"
        "いくつか調べて、できれば安いやつを教えてくれへんかな？予算は五千円くらいで考えてるんやけど、"
        "タッチ機能はなくてもええし、解像度もそこそこで十分やと思ってるわ。"
    )
    assert len(long_input) > 80
    say1 = ToolCall(id="t1", name="say", input={"text": "調べてみるわ"})
    say2 = ToolCall(id="t2", name="say", input={"text": "test"})
    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (_turn("tool_use", tool_calls=[say1]), None),
            (_turn("tool_use", tool_calls=[say2]), None),
            (_turn("end_turn", text=""), "done"),
        ]
    )

    with _patch_heavy():
        await agent.run(long_input)

    agent._tts.call.assert_awaited_once_with("say", {"text": "調べてみるわ"})
    assert any("You already spoke. End your turn now." in t for t in _user_texts(agent))


@pytest.mark.asyncio
async def test_exact_repeated_say_executes_only_once():
    """A model retry with identical speech must not replay audio or UI action."""
    agent = _make_agent(with_tts=True)
    long_input = "この文章を声に出したあと、処理結果も短く教えてほしい。" * 3
    say1 = ToolCall(id="t1", name="say", input={"text": "一度だけ話すで"})
    say2 = ToolCall(id="t2", name="say", input={"text": "一度だけ話すで"})
    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (_turn("tool_use", tool_calls=[say1]), None),
            (_turn("tool_use", tool_calls=[say2]), None),
            (_turn("end_turn", text="完了したで"), "完了したで"),
        ]
    )
    actions: list[tuple[str, dict]] = []

    with _patch_heavy():
        result = await agent.run(
            long_input,
            on_action=lambda name, tool_input: actions.append((name, tool_input)),
        )

    assert result == "完了したで"
    agent._tts.call.assert_awaited_once_with("say", {"text": "一度だけ話すで"})
    assert actions == [("say", {"text": "一度だけ話すで"})]


@pytest.mark.asyncio
async def test_interrupt_queue_drained_with_embodied_format():
    """Queued interrupts surface with the [User interrupted xN] say() directive."""
    agent = _make_agent()
    tc = ToolCall(id="t1", name="remember", input={"content": "x"})
    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (_turn("tool_use", tool_calls=[tc]), None),
            (_turn("end_turn", text="ん"), "ん"),
        ]
    )
    queue = asyncio.Queue()
    queue.put_nowait("なあ、聞いてる？")

    with _patch_heavy():
        await agent.run("覚えといて", interrupt_queue=queue)

    interrupted = [t for t in _user_texts(agent) if "[User interrupted x1]" in t]
    assert interrupted
    assert "weekday=" in interrupted[0]
    assert "なあ、聞いてる？" in interrupted[0]
    assert "say() now" in interrupted[0]


@pytest.mark.asyncio
async def test_interrupt_not_drained_before_first_model_call():
    """An input queued before the turn starts must not be folded into iteration 0."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(return_value=(_turn("end_turn", text="ん"), "ん"))
    queue = asyncio.Queue()
    queue.put_nowait("これは次のターンの入力")

    with _patch_heavy():
        await agent.run("おーい", interrupt_queue=queue)

    assert not [t for t in _user_texts(agent) if t.startswith("[User interrupted")]
    assert not queue.empty()  # left for the next turn


@pytest.mark.asyncio
async def test_tape_replan_spliced_into_tool_result():
    """A blocked plan splices [ADAPTIVE REPLAN] into the tool result text."""
    agent = _make_agent()
    tc = ToolCall(id="t1", name="remember", input={"content": "x"})
    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (_turn("tool_use", tool_calls=[tc]), None),
            (_turn("end_turn", text="ん"), "ん"),
        ]
    )

    patches = _heavy_patches()
    patches["familiar_agent.agent.check_plan_blocked"] = AsyncMock(return_value=True)
    patches["familiar_agent.agent.generate_replan"] = AsyncMock(return_value="try the camera")
    with _patch_heavy(patches):
        # prepare_turn computes plan/tape lazily; force the prep fields instead
        real_prepare = agent._hook.prepare_turn

        async def _prepare_with_plan(**kwargs):
            prep = await real_prepare(**kwargs)
            prep.tape_backend = object()
            prep.plan_ctx = "PLAN: inspect the room"
            return prep

        agent._hook.prepare_turn = _prepare_with_plan
        await agent.run("部屋見といて")

    collected = agent.backend.make_tool_results.call_args.args[1]
    assert "[ADAPTIVE REPLAN] try the camera" in collected[0][0]


@pytest.mark.asyncio
async def test_coherence_retry_reruns_loop_once(monkeypatch):
    """A coherence violation rejects the reply, injects SELF-CHECK, and retries."""
    agent = _make_agent()
    agent.backend.stream_turn = AsyncMock(
        side_effect=[
            (_turn("end_turn", text="矛盾した返事"), "矛盾した返事"),
            (_turn("end_turn", text="直した返事"), "直した返事"),
        ]
    )
    monkeypatch.setenv("FAMILIAR_COHERENCE_CHECK", "1")
    agent._check_response_coherence = AsyncMock(side_effect=["contradicts earlier turn", None])

    with _patch_heavy():
        result = await agent.run("どう思う？")

    assert result == "直した返事"
    self_checks = [t for t in _user_texts(agent) if t.startswith("[SELF-CHECK]")]
    assert self_checks
    assert "contradicts earlier turn" in self_checks[0]
    assert agent._coherence_retried is False  # reset after a clean finalisation
