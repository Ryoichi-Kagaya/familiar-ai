"""Non-critical persistence and adaptation after an agent response."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ._runtime_helpers import _noop_str
from .desires import DesireSystem, detect_worry_signal

logger = logging.getLogger(__name__)

GeneratePlan = Callable[[Any, str, list[str]], Awaitable[str]]


def flush_store(store: Any) -> None:
    """Best-effort flush a store that supports deferred persistence."""
    if store is None or not hasattr(store, "flush"):
        return
    try:
        store.flush()
    except Exception:  # noqa: BLE001
        pass


def _react_to_scene_events(events: list[dict], desires: DesireSystem | None) -> None:
    """Translate SceneTracker presence events into desire boosts."""
    if desires is None or not events:
        return
    for event in events:
        event_type = event.get("event_type", "")
        label = (event.get("entity_label") or "").lower()
        if "person" in label:
            if event_type == "appeared":
                desires.boost("greet_companion", 0.6)
            elif event_type == "disappeared":
                desires.boost("worry_companion", 0.2)


class PostResponsePipeline:
    """Persist a completed reply and prepare context for the next turn."""

    def __init__(self, agent: Any, *, generate_plan_fn: GeneratePlan) -> None:
        self._agent = agent
        self._generate_plan = generate_plan_fn

    async def _process_observation(
        self,
        *,
        final_text: str,
        camera_used: bool,
        observation_action_name: str | None,
        observation_action_input: dict | None,
        desires: DesireSystem | None,
    ) -> None:
        """Persist a camera observation and feed prediction/exploration state."""
        if not camera_used:
            return

        agent = self._agent
        recent_obs = await agent._memory.recall_async(final_text[:200], n=6, kind="observation")
        past_scores = [memory.get("score", 0.5) for memory in recent_obs[:3]]
        novelty = 1.0 - (sum(past_scores) / len(past_scores)) if past_scores else 0.8
        novelty = max(0.0, min(1.0, novelty))
        agent._exploration.record_novelty(novelty)
        if desires is not None:
            desires.boost("look_around", novelty * 0.3)

        if agent._scene is not None:
            scene_events = await agent._scene.update(
                final_text[:500],
                agent._scene_backend,
                prediction_engine=agent._prediction,
                action_name=observation_action_name,
                action_input=observation_action_input,
            )
            _react_to_scene_events(scene_events, desires)
            pred_signal = agent._prediction.last_signal()
            self_state = getattr(agent, "_self_state", None)
            if pred_signal is not None and self_state is not None:
                self_state.apply_prediction_feedback(
                    external_surprise=pred_signal.external_surprise,
                    agency_error=pred_signal.agency_error,
                    action_name=pred_signal.action_name,
                )
            pred_coalition = agent._prediction.as_coalition()
            if pred_coalition is not None:
                agent._workspace.apply_prediction_error(pred_coalition.novelty)

        await agent._memory.save_async(
            final_text[:500],
            direction="観察",
            kind="observation",
            dedupe_key=agent._memory_dedupe_key("observation", final_text[:500]),
            materialize_now=False,
        )

    async def _persist_conversation(
        self,
        *,
        user_input: str,
        final_text: str,
        is_desire_turn: bool,
    ) -> str:
        """Persist the exchange and update slow self-model state."""
        agent = self._agent
        emotion = await agent._infer_emotion(final_text)
        agent._update_mood(emotion)
        summary = await agent._summarize_exchange(user_input, final_text)
        await agent._memory.save_async(
            summary,
            direction="会話",
            kind="conversation",
            emotion=emotion,
            dedupe_key=agent._memory_dedupe_key("conversation", summary),
            materialize_now=False,
        )

        await agent._update_self_model(final_text, emotion)
        await agent._maybe_update_self_narrative(
            user_input=user_input,
            final_text=final_text,
            emotion=emotion,
            is_desire_turn=is_desire_turn,
        )
        return emotion

    def _update_relationship(
        self,
        *,
        user_input: str,
        is_desire_turn: bool,
        desires: DesireSystem | None,
    ) -> None:
        """Record a human exchange and react to explicit worry signals."""
        if is_desire_turn or not user_input:
            return

        agent = self._agent
        agent._relationship.record_conversation()
        if desires is None:
            return
        worry_boost = detect_worry_signal(user_input)
        if worry_boost > 0.0:
            desires.boost("worry_companion", worry_boost)
            logger.debug(
                "Worry signal detected (%.2f): boosting worry_companion",
                worry_boost,
            )

    async def _persist_curiosity(
        self,
        *,
        final_text: str,
        camera_used: bool,
        desires: DesireSystem | None,
    ) -> str | None:
        """Extract and persist visual curiosity when a desire system is active."""
        if desires is None or not camera_used:
            return None

        agent = self._agent
        curiosity = await agent.extract_curiosity(final_text)
        if not curiosity:
            return None
        desires.curiosity_target = curiosity
        desires.boost("look_around", 0.3)
        await agent._memory.save_async(
            curiosity,
            direction="好奇心",
            kind="curiosity",
            emotion="curious",
            dedupe_key=agent._memory_dedupe_key("curiosity", curiosity),
            materialize_now=False,
        )
        logger.info("Curiosity persisted: %s", curiosity)
        return curiosity

    def _update_continuity(
        self,
        *,
        emotion: str,
        companion_mood: str,
        curiosity: str | None,
    ) -> None:
        """Feed the turn outcome into concern and self-continuity state."""
        agent = self._agent
        pred_signal = agent._prediction.last_signal()
        concerns = getattr(agent, "_concerns", None)
        if concerns is not None:
            concerns.update_from_turn(
                turn_index=agent._turn_count,
                emotion=emotion,
                companion_mood=companion_mood,
                curiosity=curiosity,
                prediction_signal=pred_signal,
            )

        self_state = getattr(agent, "_self_state", None)
        if self_state is not None:
            self_state.apply_turn_context(
                emotion=emotion,
                companion_mood=companion_mood,
                curiosity=curiosity,
                prediction_signal=pred_signal,
            )

    async def _refresh_deferred_context(
        self,
        *,
        user_input: str,
        is_desire_turn: bool,
        desires: DesireSystem | None,
    ) -> None:
        """Compute and cache context consumed by the next turn."""
        agent = self._agent
        try:
            tape_backend = agent._tape_backend()
            tool_names = [tool["name"] for tool in agent._all_tool_defs] if tape_backend else []
            deferred_plan_task = (
                self._generate_plan(tape_backend, user_input, tool_names)
                if tape_backend and not is_desire_turn and user_input.strip()
                else _noop_str()
            )
            (
                deferred_plan,
                deferred_workspace,
                deferred_mood,
                deferred_temporal,
            ) = await asyncio.gather(
                deferred_plan_task,
                agent._gather_workspace_context(desires=desires),
                agent._infer_companion_mood(user_input),
                agent._online_temporal_context(desires=desires),
            )
            agent._cached_plan_ctx = deferred_plan
            agent._cached_workspace_ctx = deferred_workspace
            agent._cached_companion_mood = deferred_mood
            agent._cached_temporal_ctx = deferred_temporal
            if desires is not None and deferred_mood == "frustrated":
                desires.boost("worry_companion", 0.3)
                logger.debug("Companion mood frustrated: boosting worry_companion")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Deferred pre-response caching failed: %s", exc)

    async def run(
        self,
        *,
        user_input: str,
        final_text: str,
        camera_used: bool,
        observation_action_name: str | None,
        observation_action_input: dict | None,
        companion_mood: str,
        is_desire_turn: bool,
        desires: DesireSystem | None,
    ) -> None:
        """Persist and adapt after a reply without blocking that reply."""
        if not final_text or final_text == "(no response)":
            return

        agent = self._agent
        try:
            await self._process_observation(
                final_text=final_text,
                camera_used=camera_used,
                observation_action_name=observation_action_name,
                observation_action_input=observation_action_input,
                desires=desires,
            )
            emotion = await self._persist_conversation(
                user_input=user_input,
                final_text=final_text,
                is_desire_turn=is_desire_turn,
            )
            self._update_relationship(
                user_input=user_input,
                is_desire_turn=is_desire_turn,
                desires=desires,
            )
            curiosity = await self._persist_curiosity(
                final_text=final_text,
                camera_used=camera_used,
                desires=desires,
            )

            if user_input and not is_desire_turn:
                await agent._capture_companion_thread(user_input, desires)

            self._update_continuity(
                emotion=emotion,
                companion_mood=companion_mood,
                curiosity=curiosity,
            )
            await agent._maybe_adapt_values(
                user_input=user_input,
                final_text=final_text,
                emotion=emotion,
                camera_used=camera_used,
                curiosity=curiosity,
                is_desire_turn=is_desire_turn,
                desires=desires,
            )
            await agent._maybe_update_identity(
                user_input=user_input,
                final_text=final_text,
                is_desire_turn=is_desire_turn,
            )
            await self._refresh_deferred_context(
                user_input=user_input,
                is_desire_turn=is_desire_turn,
                desires=desires,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Post-response pipeline failed: %s", exc)
        finally:
            # Dense recurrence defers self-state writes; the turn's own affect
            # deltas must be durable once this background pipeline ends.
            flush_store(getattr(agent, "_self_state", None))
