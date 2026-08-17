"""Final response repair, continuation handling, speech, and commit."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from familiar_neighbor.embodied_hook import PreparedTurn
from familiar_neighbor.mind.social_policy import SocialPolicyDecision
from familiar_runtime.models import ImageAttachment, UserTurn, coerce_user_turn

from .meta_monitor import MetaGateDecision
from .turn_coordinator import TurnRequest

logger = logging.getLogger(__name__)


def coerce_request_user_turn(request: TurnRequest) -> UserTurn:
    """Normalize current and legacy image inputs into one user turn."""
    user_turn = coerce_user_turn(request.user_input)
    if not request.user_images:
        return user_turn
    legacy_images = tuple(
        ImageAttachment.from_base64(image_data) for image_data in request.user_images
    )
    return UserTurn(
        text=user_turn.text,
        images=(*user_turn.images, *legacy_images),
        sent_at=user_turn.sent_at,
    )


def extract_continuation_status(final_text: str) -> tuple[str, str]:
    """Remove a trailing continuation marker and return its runtime status."""
    status_match = re.search(r"(?:^|\n)(DONE|CONTINUE:[^\n]+|DEFER:[^\n]+)\s*$", final_text)
    if not status_match:
        return final_text, "DONE"
    visible_text = final_text[: status_match.start(1)].rstrip() or "(no response)"
    return visible_text, status_match.group(1)


class TurnFinalizer:
    """Apply final safety checks and commit one completed turn."""

    def __init__(self, agent: Any) -> None:
        self._agent = agent

    def _check_identity_response(
        self,
        *,
        user_text: str,
        candidate_response: str,
    ) -> tuple[Any | None, list[Any]]:
        """Run the identity backstop without making it a turn-fatal dependency."""
        identity = getattr(self._agent, "_identity", None)
        if identity is None or not candidate_response or candidate_response == "(no response)":
            return identity, []
        try:
            violations = identity.check_response(
                user_text=user_text,
                candidate_response=candidate_response,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Identity response check failed: %s", exc)
            return identity, []
        return identity, violations

    def _apply_meta_gate(
        self,
        *,
        user_text: str,
        candidate_response: str,
        social_policy: SocialPolicyDecision,
        identity_violations: list[Any],
    ) -> str:
        """Apply deterministic final-response repair when the monitor requests it."""
        agent = self._agent
        gate_method = getattr(agent._meta_monitor, "gate_response", None)
        if not callable(gate_method):
            return candidate_response
        maybe_gate = gate_method(
            user_text=user_text,
            candidate_response=candidate_response,
            social_policy=social_policy,
            last_error=agent._last_tool_error,
            identity_violations=identity_violations or None,
        )
        if (
            isinstance(maybe_gate, MetaGateDecision)
            and maybe_gate.needs_repair
            and maybe_gate.repaired_response
        ):
            return maybe_gate.repaired_response
        return candidate_response

    def _record_identity_violation_effects(
        self,
        *,
        identity: Any | None,
        identity_violations: list[Any],
        desires: Any,
    ) -> None:
        """Persist dissonance and raise reflection pressure after a violation."""
        if identity is None or not identity_violations:
            return

        agent = self._agent
        top = identity_violations[0]
        try:
            identity.record_violation(top, turn_index=agent._turn_count)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Identity violation record failed: %s", exc)
        if desires is not None:
            desires.boost("identity_coherence", 0.3 + 0.4 * top.severity)
        concerns = getattr(agent, "_concerns", None)
        if concerns is not None:
            try:
                concerns.activate(
                    f"Something I hold was strained: {top.statement[:80]}",
                    category="identity",
                    intensity=0.4 + 0.5 * top.severity,
                    turn_index=agent._turn_count,
                )
            except Exception:  # noqa: BLE001
                pass

    async def _auto_say(
        self,
        *,
        prep: PreparedTurn,
        final_text: str,
        on_action: Callable[[str, dict], None] | None,
    ) -> None:
        """Speak a final reply when configured and no explicit say call occurred."""
        agent = self._agent
        if not (
            getattr(agent.config, "auto_say", False)
            and agent._tts
            and not prep.say_used
            and final_text
            and final_text != "(no response)"
        ):
            return
        if on_action:
            on_action("say", {"text": final_text})
        await agent._tts.call("say", {"text": final_text})

    async def commit_end_turn(
        self,
        *,
        prep: PreparedTurn,
        user_input: str,
        final_text: str,
        desires: Any,
        on_action: Callable[[str, dict], None] | None,
    ) -> str:
        """Repair, surface, and commit a successful end-turn response."""
        agent = self._agent
        identity, identity_violations = self._check_identity_response(
            user_text=user_input,
            candidate_response=final_text,
        )
        final_text = self._apply_meta_gate(
            user_text=user_input,
            candidate_response=final_text,
            social_policy=prep.social_policy,
            identity_violations=identity_violations,
        )
        self._record_identity_violation_effects(
            identity=identity,
            identity_violations=identity_violations,
            desires=desires,
        )

        final_text, continuation_status = extract_continuation_status(final_text)
        agent._heartbeat.apply_status(continuation_status)
        agent._coherence_retried = False

        await self._auto_say(
            prep=prep,
            final_text=final_text,
            on_action=on_action,
        )
        await agent._hook.commit_after_end_turn(
            prep=prep,
            user_input=user_input,
            final_text=final_text,
            is_desire_turn=prep.is_desire_turn,
            desires=desires,
        )
        return final_text

    async def force_final_response(
        self,
        *,
        prep: PreparedTurn,
        on_text: Callable[[str], None] | None,
    ) -> str:
        """Request a final tool-free answer after an unexpected loop stop."""
        agent = self._agent
        logger.warning(
            "Reached max iterations (%d). Forcing final response.",
            prep.turn_max_iterations,
        )
        agent.messages.append(
            agent.backend.make_user_message(
                "Please summarize what you found and provide your final answer now."
            )
        )
        result, _ = await agent._stream_with_retry(
            system=agent._system_prompt(
                morning_ctx=prep.morning_ctx,
                plan_ctx=prep.plan_ctx,
                continuity_ctx=prep.continuity_ctx,
                workspace_ctx=prep.workspace_ctx,
                mental_ctx=prep.mental_ctx,
            ),
            messages=agent.messages,
            tools=[],
            max_tokens=prep.turn_max_tokens,
            on_text=on_text,
        )
        return result.text or "(max iterations reached)"
