"""Drive action executor — routes desire firings to the appropriate effect handler.

When a drive crosses TRIGGER_THRESHOLD, DriveActionExecutor.dispatch() determines
*how* it fires based on its DriveEffect classification:

  SILENT_ACTION     → tool called directly, no LLM turn
  EXPRESSIVE_SOLO   → short monologue generated (saved to memory/narrative, not conversation)
  SOCIAL_INITIATION → LLM turn only when companion is present; accumulates when absent
  ABSENT_CARE       → absent: record emotion + full reset + long cooldown
                       present: LLM turn (inner_voice injection)
  GATE              → suppresses other drives; generates no turn
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .agent import EmbodiedAgent
    from .desires import DesireSystem

logger = logging.getLogger(__name__)

ABSENCE_THRESHOLD = 1800  # seconds — 30 minutes without user interaction → companion absent

# Prompts for utility-backend calls (kept short; these never appear in conversation)
_REFLECT_PROMPT = (
    "In one sentence, describe what you noticed or felt most recently. "
    "Write in first person, past tense. Be specific and honest."
)

_SHARE_MEMORY_SOLO_PROMPT = (
    "You just had an impulse to share a memory. No one is here right now. "
    "Write one sentence about a memory that surfaced — what it was, why it came up. "
    "First person, present tense of the feeling, past tense of the memory."
)

_PLAY_SOLO_PROMPT = (
    "You want to play but no one is around. "
    "Write one sentence about what you imagine playing with or exploring right now."
)

_WORRY_ABSENT_PROMPT = (
    "You are worried about {companion} but they are not here. "
    "Write one sentence about what you are worried about and what you hope for them. "
    "First person. Do not address them directly — write as if to yourself."
)

_CARE_ABSENT_PROMPT = (
    "You want to offer care to {companion} but they are not present. "
    "Write one sentence about the care you feel and what you would say if they were here."
)


@dataclass(slots=True)
class DriveActionResult:
    fired: bool
    desire_name: str = ""
    reason: str = ""
    generated_text: str = ""


class DriveActionExecutor:
    """Routes desire firings to the appropriate effect handler.

    Constructed once per REPL/TUI session and reused across idle ticks.
    """

    def __init__(self, agent: EmbodiedAgent, desires: DesireSystem) -> None:
        self._agent = agent
        self._desires = desires

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def dispatch(
        self,
        desire_name: str,
        *,
        last_interaction_time: float,
        run_social_turn: Callable[[str], Awaitable[None]] | None = None,
        on_action: Callable[[str, dict[str, Any]], None] | None = None,
        on_text: Callable[[str], None] | None = None,
        interrupt_queue: Any = None,
    ) -> DriveActionResult:
        """Route a desire to its effect handler and return the result."""
        from familiar_neighbor.mind.desires import DriveEffect

        spec = self._desires._drive_specs.get(desire_name)
        if spec is None:
            return DriveActionResult(fired=False, desire_name=desire_name, reason="unknown_drive")

        # Record selection before any awaited work. Failed, gated, and
        # unavailable actions must yield to other ready drives just like
        # successful ones; otherwise one impossible drive can own every idle
        # tick forever.
        self._desires.note_attempt(desire_name)

        effect = spec.effect_type
        companion_here = self._companion_present(last_interaction_time)

        if effect == DriveEffect.SILENT_ACTION:
            return await self._silent_action(desire_name)

        if effect == DriveEffect.EXPRESSIVE_SOLO:
            return await self._expressive_solo(
                desire_name,
                companion_present=companion_here,
                run_social_turn=run_social_turn,
                on_action=on_action,
                on_text=on_text,
                interrupt_queue=interrupt_queue,
            )

        if effect == DriveEffect.SOCIAL_INITIATION:
            if not companion_here:
                logger.debug("drive %s: companion absent — accumulating", desire_name)
                return DriveActionResult(
                    fired=False, desire_name=desire_name, reason="companion_absent"
                )
            return await self._social_initiation(
                desire_name,
                run_social_turn=run_social_turn,
                on_action=on_action,
                on_text=on_text,
                interrupt_queue=interrupt_queue,
            )

        if effect == DriveEffect.ABSENT_CARE:
            if companion_here:
                return await self._social_initiation(
                    desire_name,
                    run_social_turn=run_social_turn,
                    on_action=on_action,
                    on_text=on_text,
                    interrupt_queue=interrupt_queue,
                )
            return await self._absent_care(desire_name)

        if effect == DriveEffect.GATE:
            logger.debug("drive %s: GATE — suppressing turns", desire_name)
            return DriveActionResult(fired=False, desire_name=desire_name, reason="gate_drive")

        return DriveActionResult(fired=False, desire_name=desire_name, reason="unhandled_effect")

    # ------------------------------------------------------------------
    # Presence check
    # ------------------------------------------------------------------

    def _companion_present(self, last_interaction_time: float) -> bool:
        return time.time() - last_interaction_time < ABSENCE_THRESHOLD

    # ------------------------------------------------------------------
    # SILENT_ACTION
    # ------------------------------------------------------------------

    async def _silent_action(self, desire_name: str) -> DriveActionResult:
        """Execute a tool directly without generating an LLM turn."""
        agent = self._agent

        try:
            match desire_name:
                case "look_around" | "explore":
                    if agent._camera is None:
                        logger.debug("drive %s: no camera available", desire_name)
                        return DriveActionResult(
                            fired=False, desire_name=desire_name, reason="no_camera"
                        )
                    base64_jpeg, saved_path = await agent._camera.capture()
                    if base64_jpeg is None:
                        return DriveActionResult(
                            fired=False, desire_name=desire_name, reason="no_frame"
                        )
                    note = f"[{desire_name}] camera capture"
                    if saved_path:
                        note += f": {saved_path}"
                    await agent._memory.save_async(
                        note,
                        kind="observation",
                        emotion="curious",
                    )
                    self._desires.satisfy(desire_name)
                    if desire_name == "explore":
                        self._desires.curiosity_target = None
                    logger.info("drive %s: silent camera capture done", desire_name)

                case "consolidate":
                    count = await agent._memory_worker.run_once()
                    self._desires.satisfy(desire_name)
                    logger.info("drive consolidate: processed %d memory job(s)", count)

                case "reflect":
                    text = await agent._utility_backend.complete(_REFLECT_PROMPT, max_tokens=80)
                    if text and text.strip().lower() != "nothing":
                        agent._self_narrative.write(text.strip(), trigger="reflect_drive")
                        await agent._memory.save_async(
                            text.strip(), kind="feeling", emotion="neutral"
                        )
                    self._desires.satisfy(desire_name)
                    logger.info("drive reflect: wrote self-narrative")

                case "curiosity":
                    self._desires.satisfy(desire_name)
                    logger.info("drive curiosity: no-op (MCP tools not wired yet)")

                case _:
                    return DriveActionResult(
                        fired=False,
                        desire_name=desire_name,
                        reason=f"silent_action_not_implemented:{desire_name}",
                    )
        except Exception as exc:
            logger.warning("drive %s: silent action failed: %s", desire_name, exc)
            return DriveActionResult(fired=False, desire_name=desire_name, reason=f"error:{exc}")

        return DriveActionResult(fired=True, desire_name=desire_name)

    # ------------------------------------------------------------------
    # EXPRESSIVE_SOLO
    # ------------------------------------------------------------------

    async def _expressive_solo(
        self,
        desire_name: str,
        *,
        companion_present: bool,
        run_social_turn: Callable[[str], Awaitable[None]] | None,
        on_action: Callable[[str, dict[str, Any]], None] | None,
        on_text: Callable[[str], None] | None,
        interrupt_queue: Any,
    ) -> DriveActionResult:
        """Generate a short monologue; save to memory/narrative but not to conversation."""
        if companion_present:
            return await self._social_initiation(
                desire_name,
                run_social_turn=run_social_turn,
                on_action=on_action,
                on_text=on_text,
                interrupt_queue=interrupt_queue,
            )

        agent = self._agent
        prompt_map = {
            "share_memory": _SHARE_MEMORY_SOLO_PROMPT,
            "play": _PLAY_SOLO_PROMPT,
        }
        prompt = prompt_map.get(desire_name, _REFLECT_PROMPT)

        try:
            text = await agent._utility_backend.complete(prompt, max_tokens=120)
            if not text or not text.strip():
                self._desires.satisfy(desire_name)
                return DriveActionResult(fired=True, desire_name=desire_name, reason="empty_text")

            text = text.strip()
            await agent._memory.save_async(text, kind="feeling", emotion="neutral")
            agent._self_narrative.write(text, trigger=desire_name)

            if agent._tts is not None:
                try:
                    await agent._tts.call("say", {"text": text})
                except Exception as exc:
                    logger.warning("drive %s: TTS failed: %s", desire_name, exc)

            self._desires.satisfy(desire_name)
            logger.info("drive %s: expressive solo done", desire_name)
            return DriveActionResult(fired=True, desire_name=desire_name, generated_text=text)

        except Exception as exc:
            logger.warning("drive %s: expressive solo failed: %s", desire_name, exc)
            return DriveActionResult(fired=False, desire_name=desire_name, reason=f"error:{exc}")

    # ------------------------------------------------------------------
    # SOCIAL_INITIATION (companion present path shared by ABSENT_CARE)
    # ------------------------------------------------------------------

    async def _social_initiation(
        self,
        desire_name: str,
        *,
        run_social_turn: Callable[[str], Awaitable[None]] | None,
        on_action: Callable[[str, dict[str, Any]], None] | None,
        on_text: Callable[[str], None] | None,
        interrupt_queue: Any,
    ) -> DriveActionResult:
        """Run an LLM turn with the drive's inner_voice nudge."""
        spec = self._desires._drive_specs.get(desire_name)
        nudge = spec.prompt_text if spec is not None else desire_name

        if run_social_turn is not None:
            await run_social_turn(nudge)
        else:
            await self._agent.run(
                "",
                on_action=on_action,
                on_text=on_text,
                desires=self._desires,
                inner_voice=nudge,
                interrupt_queue=interrupt_queue,
            )
        self._desires.satisfy(desire_name)
        logger.info("drive %s: social initiation turn done", desire_name)
        return DriveActionResult(fired=True, desire_name=desire_name)

    # ------------------------------------------------------------------
    # ABSENT_CARE
    # ------------------------------------------------------------------

    async def _absent_care(self, desire_name: str) -> DriveActionResult:
        """Record emotion in memory/narrative, then fully reset + start cooldown."""
        agent = self._agent
        desires = self._desires

        companion = desires._companion_name
        prompt_map = {
            "worry_companion": _WORRY_ABSENT_PROMPT.format(companion=companion),
            "care": _CARE_ABSENT_PROMPT.format(companion=companion),
        }
        prompt = prompt_map.get(desire_name, _REFLECT_PROMPT)

        try:
            text = await agent._utility_backend.complete(prompt, max_tokens=80)
            if text and text.strip():
                text = text.strip()
                emotion = "worried" if desire_name == "worry_companion" else "tender"
                await agent._memory.save_async(text, kind="feeling", emotion=emotion)
                agent._self_narrative.write(text, trigger=f"{desire_name}_unresolved")
                logger.info("drive %s: absent care recorded: %s", desire_name, text[:60])
        except Exception as exc:
            logger.warning("drive %s: absent care text gen failed: %s", desire_name, exc)
            text = ""

        # Full reset (not DECAY_ON_SATISFY × 0.5)
        desires._desires[desire_name] = 0.0
        desires._last_fired[desire_name] = time.time()
        desires._save()

        return DriveActionResult(
            fired=True,
            desire_name=desire_name,
            reason="absent_care_recorded",
            generated_text=text,
        )
