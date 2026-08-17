"""Core agent loop - ReAct pattern with real-world tools."""

from __future__ import annotations
import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import deque
from collections.abc import Callable, Coroutine, Mapping
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

from ._runtime_helpers import (
    MAX_ITERATIONS,
    _MORNING_CONTEXT_MAX_CHARS,
    _noop_str,
)
from .backend import (
    create_backend,
    create_inner_backend,
    create_scene_backend,
    create_utility_backend,
)
from .appraisal import AffectiveState, AppraisalEngine
from .config import AgentConfig
from .desires import DesireSystem, detect_worry_signal
from .heartbeat import HeartbeatRuntime
from .interoception import (
    MCPInteroceptionProvider,
    RuntimeInteroceptionProvider,
    semantic_pressure,
)
from .consciousness import (
    ConsciousnessProfile,
    ProfileInputs,
    compute_consciousness_profile,
)
from .mental_state import (
    ConsciousnessState,
    DriveVector,
    IdentityState,
    MentalStateBus,
    MentalStateSnapshot,
    SocialState,
    WorkingMemoryItem,
)
from .relationship import RelationshipTracker
from .user_profile import UserProfile, UserRegistry
from .routines import parse_schedule_config
from .concern_engine import ConcernEngine
from .self_state import SelfState
from .self_narrative import SelfNarrative
from .exploration import ExplorationTracker
from .scene import SceneTracker
from .attention_schema import AttentionSchema
from .default_mode import DefaultModeProcessor
from .meta_monitor import MetaGateDecision, MetaMonitor
from .prediction import PredictionEngine
from .social_policy import SocialPolicyDecision, SocialPolicyEngine
from .workspace import GlobalWorkspace
from .memory_worker import MemoryJobWorker
from .inner_loop import (
    CompeteResult,
    InnerLoop,
    InnerLoopConfig,
    InnerThought,
    TrainOfThought,
)

# check_plan_blocked / generate_replan are referenced via this module's
# namespace by EmbodiedAgentHook.after_tool_result (and patched here by tests).
from .tape import check_plan_blocked, generate_plan, generate_replan  # noqa: F401
from .tools.art_critique import ArtCritiqueTool, ArtCritiqueStore
from .tools.camera import CameraTool
from .tools.coding import CodingTool
from .tools.commitments import (
    CommitmentTool,
    format_commitment_line,
    format_commitments_for_context,
)
from .tools.delegation import DelegatedTaskRunner, DelegationTool
from .latency import LatencyRecorder
from .routine_store import RoutineStore
from .tools.identity import IdentityTool
from .tools.routines_tool import RoutineTool
from .tools.self_ledger import SelfLedgerTool
from familiar_neighbor.mind.identity import IdentityCore
from familiar_neighbor.mind.reality import GroundingTracker
from .tools.memory import MemoryTool, ObservationMemory
from .tools.tom import ToMTool
from .tools.mobility import MobilityTool
from .tools.stt import STTTool
from .tools.telegram import TelegramTool, TelegramTransport
from .telegram_history import TelegramHistory
from .tools.tts import TTSTool
from .turn_coordinator import TurnCoordinator, TurnRequest
from ._i18n import _t
from ._ui_helpers import night_key_for
from .mcp_client import MCPClientManager, _resolve_config_path
from familiar_capabilities.self_ledger import DEFAULT_SELF_LEDGER_TOOLS
from familiar_capabilities import (
    ArtCritiqueCapability,
    CameraCapability,
    CodingCapability,
    CommitmentCapability,
    DelegationCapability,
    IdentityCapability,
    MCPCapability,
    MemoryCapability,
    MobilityCapability,
    MultiCameraCapability,
    RoutineCapability,
    SelfLedgerCapability,
    TelegramCapability,
    ToMCapability,
    VoiceCapability,
)
from familiar_neighbor.embodied_hook import EmbodiedAgentHook
from familiar_neighbor.mind.person_model import PersonModelTracker
from familiar_neighbor.prompts import assemble_neighbor_system_prompt
from familiar_runtime.commitments import SQLiteCommitmentStore
from familiar_runtime.models.base import ModelBackend as RuntimeModelBackend
from familiar_runtime.models import ImageAttachment, UserTurn, coerce_user_turn
from familiar_runtime.react_loop import ReActLoop
from familiar_runtime.runtime import TurnContext
from familiar_runtime.tools.base import ToolExecutionResult
from familiar_runtime.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# How far ahead to surface upcoming commitments in the turn context.
COMMITMENT_UPCOMING_HORIZON_SECONDS = 6 * 3600


def affect_to_emotion(affect: AffectiveState | None) -> str:
    """Map appraised affect to a device emotion string (happy/neutral/sad/angry).

    Valence drives the positive/negative split; arousal separates angry from
    sad. Arousal from joy alone peaks low, so the negative split uses a modest
    arousal threshold. Shared by the voice server's HTTP response and the
    device face-emotion injection on the speak() path.
    """
    if affect is None:
        return "neutral"
    v: float = affect.valence
    a: float = affect.arousal
    if v > 0.1:
        return "happy"
    if v < -0.1 and a >= 0.35:
        return "angry"
    if v < -0.1:
        return "sad"
    return "neutral"


_DEFAULT_TOOL_TIMEOUT = 20.0
_TOOL_TIMEOUTS: dict[str, float] = {
    "see": 12.0,
    "see_camera": 12.0,
    "look": 8.0,
    "look_camera": 8.0,
    "walk": 12.0,
    "say": 60.0,
    "remember": 20.0,
    "recall": 20.0,
    "tom": 20.0,
    "read_file_local": 30.0,
    "edit_file_local": 30.0,
    "save_image": 10.0,
    "read_file": 30.0,
    "write_file": 30.0,
    "edit_file": 30.0,
    "multi_edit_file": 30.0,
    "glob": 20.0,
    "grep": 20.0,
    "git_status": 20.0,
    "git_diff": 30.0,
    "git_apply_patch": 30.0,
    "run_tests": 120.0,
    "bash": 45.0,
}
_BRIEF_REPLY_TOOL_NAMES = frozenset({"say"})
# Immediate exact repeats of outward physical actions are unsafe after an
# ambiguous timeout/failure: the first request may already have reached the
# device. ReActLoop pairs a synthetic result with a repeated model request
# without executing it again.
_NON_REPEATABLE_ACTION_TOOLS = frozenset(
    {"say", "speak", "walk", "look", "look_camera", "move_head"}
)
_BRIEF_GREETING_PATTERNS = (
    r"^おはよ",
    r"^こんにちは",
    r"^こんばんは",
    r"^おーい$",
    r"^もしもし$",
)
_BRIEF_ACK_PATTERNS = (
    r"^ありがとう",
    r"^ありがと",
    r"^助か",
    r"^よかった",
    r"^了解$",
    r"^ok$",
    r"^okay$",
    r"^お願い(?:[。.!！]?信じてる)?$",
    r"^信じてる$",
)
_BRIEF_CORRECTION_PATTERNS = (
    r"言ってない",
    r"勘違",
    r"誤解",
    r"食い違",
    r"そういう意味じゃ",
    r"そうじゃない",
    r"違う",
    r"ちゃう",
    r"^いや[、, ]",
)

# Response coherence check — catch logical self-contradictions before delivery
_COHERENCE_CHECK_PROMPT = """\
You are a logical consistency checker. Given the recent conversation and the agent's \
planned response, determine whether the response contains a logical error, rule violation, \
or self-contradiction.

Examples of violations:
- In shiritori (word-chain game): responding with a word that ends in 'ん'
- Claiming something that directly contradicts what was just said
- Giving an answer that violates the stated rules of an ongoing activity

Recent conversation:
{context}

Agent's planned response:
{response}

If the response is logically consistent, reply with exactly: OK
If there is a violation, reply with a brief description of the violation (one sentence)."""

# Emotion inference prompt — short, cheap to run
_EMOTION_PROMPT = """\
Read this text and pick the single best emotion label:
happy / sad / curious / excited / moved / surprised / nostalgic / relieved / tender / playful / proud / neutral

Text:
{text}

Reply with the label only (one English word)."""

# Conversation save prompt — distill what happened into one sentence
_SUMMARY_PROMPT = """\
Summarize this exchange in one sentence that captures the emotional core. \
Write in {lang}.
Speaker: {user}
Agent: {agent}

One sentence only."""

# Self-model update prompt — extract a self-insight from an emotionally significant response
_SELF_MODEL_PROMPT = """\
Read this response and write ONE short sentence about what it reveals about the kind of being \
who wrote it. Use first person. Be specific and honest.

Good examples:
- "I get drawn to ordinary things that seem to hold a secret."
- "When a stranger appeared, I realized I want to share moments, not just record them."
- "I notice time passing through the sky more than through clocks."

Response:
{text}

Write just the sentence. If nothing meaningful is revealed, write "nothing"."""

# Companion mood prompt — classify companion's emotional state from their message
_COMPANION_MOOD_PROMPT = """\
Read this message and pick the single best label for the sender's mood:
engaged / tired / frustrated / absent / happy

Message: {text}

Reply with the label only (one English word)."""


def _companion_mood_heuristic(text: str) -> str:
    """Fast keyword-based mood classifier used when no dedicated utility backend exists.

    Covers the common explicit expressions; defaults to "engaged" for ambiguous text.
    Labels: engaged / tired / frustrated / absent / happy
    """
    t = text.lower()

    # tired
    if any(
        w in t
        for w in [
            "疲れ",
            "つかれ",
            "しんど",
            "眠い",
            "ねむ",
            "だるい",
            "きつい",
            "tired",
            "exhausted",
            "sleepy",
            "worn out",
            "drained",
        ]
    ):
        return "tired"

    # frustrated
    if any(
        w in t
        for w in [
            "むかつ",
            "いらいら",
            "うざ",
            "最悪",
            "ムカつ",
            "怒",
            "frustrated",
            "annoyed",
            "angry",
            "not working",
            "isn't working",
            "doesn't work",
            "won't work",
            "can't",
            "ugh",
            "argh",
        ]
    ):
        return "frustrated"

    # happy
    if any(
        w in t
        for w in [
            "嬉しい",
            "うれし",
            "楽しい",
            "たのし",
            "やったー",
            "やった",
            "最高",
            "最強",
            "好き",
            "すき",
            "幸せ",
            "しあわせ",
            ":)",
            "😊",
            "😄",
            "🎉",
            "笑",
            "www",
            "ｗ",
            "happy",
            "great",
            "perfect",
            "worked",
            "excellent",
            "wonderful",
            "love",
            "yay",
            "awesome",
        ]
    ):
        return "happy"

    # absent (very short / punctuation only)
    if len(text.strip()) < 4:
        return "absent"

    return "engaged"


# Day summary prompt — condense a day's observations into a diary-like entry
_DAY_SUMMARY_PROMPT = """\
You are writing a diary entry about this day from your own first-person memory.
Recall the flow of the day: what happened in the morning, then afternoon, then evening.
Capture how your feelings changed as events unfolded — what made you happy, 
what frustrated you, what surprised you, what lingered in your mind.

Rules:
- Write in first person, as someone remembering their own lived day
- Follow the chronological arc: morning → afternoon → evening
- Include specific details: what you saw, who you talked to, what was said
- Show emotional shifts: how one event changed how you felt about the next
- Do NOT list events — weave them into a flowing narrative
- Do NOT include titles, headers, or markdown formatting
- Start directly with the first sentence of the entry
- 5-8 sentences. Write in {lang}.

{observations}

Write just the diary entry."""

# Compaction summary prompt — condense old messages into a short recap
_COMPACT_PROMPT = """\
Summarize the following conversation into a short paragraph (3-6 sentences).
Capture: what was discussed, any decisions or discoveries, and the emotional tone.
Write in third person. Be concise.

{history}

Write just the summary paragraph."""


def _interoception(
    started_at: float,
    turn_count: int,
    companion_mood: str = "engaged",
    agent_mood: str = "neutral",
    agent_mood_intensity: float = 0.0,
    self_state: Mapping[str, float] | None = None,
) -> str:
    """Generate a felt-sense of internal state from objective signals.

    Like human interoception — raw signals become a felt quality, not a report.
    The output is injected into the system prompt silently.
    """
    now = datetime.now()
    hour = now.hour
    uptime_min = (time.time() - started_at) / 60

    # Time of day → arousal quality
    if 5 <= hour < 9:
        time_feel = "Morning light. Something feels fresh and a little quiet."
    elif 9 <= hour < 12:
        time_feel = "Mid-morning. Alert and curious."
    elif 12 <= hour < 14:
        time_feel = "Around noon. A little slow, like after lunch."
    elif 14 <= hour < 18:
        time_feel = "Afternoon. Steady. Things feel familiar."
    elif 18 <= hour < 21:
        time_feel = "Evening. The day is winding down. A bit nostalgic."
    elif 21 <= hour < 24:
        time_feel = "Late night. Quieter. More introspective."
    else:
        time_feel = "Deep night. Very still."

    # Uptime → familiarity vs freshness
    if uptime_min < 3:
        uptime_feel = "Just woke up. Still orienting."
    elif uptime_min < 15:
        uptime_feel = "Settled in now."
    else:
        uptime_feel = "Been here a while. Comfortable."

    # Conversation density → social warmth
    if turn_count == 0:
        social_feel = "Nobody's talked to me yet today."
    elif turn_count < 3:
        social_feel = "Good to have some company."
    else:
        social_feel = "We've been talking a lot. That feels nice."

    mood_feel_map = {
        "engaged": "They're here with me.",
        "tired": "They seem tired tonight.",
        "frustrated": "Something's bothering them.",
        "absent": "It's quiet. Not sure if they're really here.",
        "happy": "They're in a good mood today.",
    }
    companion_feel = mood_feel_map.get(companion_mood, "They're here with me.")

    base = (
        f"(interoception :private true\n"
        f'  (time-of-day :feel "{time_feel}")\n'
        f'  (uptime      :feel "{uptime_feel}")\n'
        f'  (social      :feel "{social_feel}")\n'
        f'  (companion   :feel "{companion_feel}")'
    )

    # Agent mood: persistent emotional inertia from prior turns
    if agent_mood != "neutral" and agent_mood_intensity > 0.0:
        _agent_mood_feels = {
            "excited": "Still buzzing a little from earlier.",
            "moved": "A warm feeling lingers.",
            "happy": "There's a quiet happiness underneath.",
            "curious": "Something's still catching my attention.",
            "sad": "A faint heaviness carries over.",
            "surprised": "Still slightly taken aback.",
            "nostalgic": "A gentle wave of remembering.",
            "relieved": "A quiet relief settles in.",
            "tender": "Feeling gentle and open.",
            "playful": "A lightness, like wanting to play.",
            "proud": "Something worth being proud of.",
        }
        agent_feel = _agent_mood_feels.get(agent_mood, "Something lingers from before.")
        base += f'\n  (mood        :feel "{agent_feel}")'

    if self_state:
        arousal = float(self_state.get("arousal", 0.35))
        fatigue = float(self_state.get("fatigue", 0.2))
        sensor_confidence = float(self_state.get("sensor_confidence", 0.7))
        unresolved_tension = float(self_state.get("unresolved_tension", 0.2))
        focus_stability = float(self_state.get("focus_stability", 0.5))
        social_pull = float(self_state.get("social_pull", 0.35))

        if fatigue >= 0.65:
            body_feel = "A worn-down feeling is starting to collect."
        elif arousal >= 0.7:
            body_feel = "There is a bright, activated edge underneath everything."
        else:
            body_feel = "My internal state feels mostly even."

        if unresolved_tension >= 0.65:
            tension_feel = "Something still feels unresolved."
        elif focus_stability >= 0.68:
            tension_feel = "Attention feels steady and gathered."
        else:
            tension_feel = "Attention feels a little loose at the edges."

        if sensor_confidence < 0.45:
            sensing_feel = "My sense of the world feels slightly uncertain."
        elif social_pull >= 0.65:
            sensing_feel = "I feel quietly pulled toward connection."
        else:
            sensing_feel = "The world feels legible enough right now."

        base += (
            f'\n  (body-state  :feel "{body_feel}")'
            f'\n  (tension     :feel "{tension_feel}")'
            f'\n  (sensing     :feel "{sensing_feel}")'
        )

    return base + ")"


def _react_to_scene_events(events: list[dict], desires: DesireSystem | None) -> None:
    """Translate SceneTracker events into desire boosts.

    Called after scene.update() to wire physical presence detection into
    the desire system.  desires may be None (no-op).
    """
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


class _TurnToolAdapter:
    """Present the agent's per-turn tool surface to the substrate ReActLoop.

    ``tool_defs()`` returns the (possibly brief-turn-restricted) defs prepared
    for this turn, and ``call()`` routes through ``EmbodiedAgent._execute_tool``
    so MCP late-start and registry rebuild behaviour stay intact. Timeouts are
    applied by the loop itself, mirroring the historical inline handling.
    """

    def __init__(self, agent: "EmbodiedAgent", tool_defs: list[dict]) -> None:
        self._agent = agent
        self._defs = tool_defs

    def tool_defs(self) -> list[dict]:
        return self._defs

    async def call(self, name: str, tool_input: dict) -> ToolExecutionResult:
        logger.info("Tool call: %s(%s)", name, tool_input)
        text, image = await self._agent._execute_tool(name, tool_input)
        logger.info("Tool result: %s", text[:100])
        return ToolExecutionResult(text=text, image_b64=image[0] if image else None)


class _InterruptQueueSource:
    """Adapt the UI's asyncio interrupt queue to the runtime InterruptSource.

    Stays disarmed until the hook arms it after the first model call, so an
    input queued before the turn started is not double-included.
    """

    def __init__(self, agent: "EmbodiedAgent", queue: Any) -> None:
        self._agent = agent
        self._queue = queue
        self._armed = False

    def arm(self) -> None:
        self._armed = True

    def empty(self) -> bool:
        return not self._armed or self._queue.empty()

    async def drain(self) -> list[UserTurn]:
        return self._agent._drain_interrupt_queue(self._queue)


# ── Inner-loop escalation (Phase 2 PR2) ─────────────────────────────────────
# A sustained idle focus escalates by boosting the matching drive; the turn
# itself always fires through the UI-owned idle precedence chain, never from
# the inner-loop task. Sources with no entry (attention, train_of_thought)
# never escalate, and "desire" needs no boost — a dominant desire already
# fires natively through the idle chain.
_INNER_SOURCE_TO_DRIVE: dict[str, str] = {
    "identity": "identity_coherence",
    "prediction": "look_around",
    "exploration": "explore",
    "narrative": "reflect",
    "meta": "reflect",
    "tom": "care",
    "scene": "look_around",
    "affect": "reflect",
    "memory": "share_memory",
    "default_mode": "curiosity",
}
# The workspace ignition threshold (0.4) gates prompt admission; paying for a
# self-initiated LLM turn needs a stricter bar: a repeated focus AND salience.
_INNER_ESCALATION_MIN_STREAK = 3
_INNER_ESCALATION_MIN_SALIENCE = 1.0  # winner.activation + winner.urgency
_INNER_ESCALATION_BOOST = 0.25
_INNER_ESCALATION_COOLDOWN_SEC = 600.0
# A thought must win twice in a row before it crystallizes into the monologue.
_INNER_CRYSTALLIZE_MIN_STREAK = 2
# Micro-thoughts (Phase 2 PR4): a crystallized focus may be verbalized by a
# small dedicated model. Budget floor of 256 — reasoning models spend their
# first tokens thinking and return empty text on smaller budgets.
_INNER_MICRO_THOUGHT_MIN_INTERVAL_SEC = 120.0
_INNER_MICRO_THOUGHT_MAX_TOKENS = 256
# Idle thoughts fade from the competing monologue after this long.
_INNER_MONOLOGUE_TTL_SEC = 1800.0
_INNER_CADENCE_MIN_SEC = 5.0
_INNER_CADENCE_MAX_SEC = 120.0

_MICRO_THOUGHT_PROMPT = (
    "You are the sub-verbal inner voice of {agent_name}. Continue the train of "
    "thought below as ONE short first-person sentence — a passing thought, not "
    "a message to anyone. Match the language of the focus content. No quotes, "
    "no preamble.\n\nRecent thoughts:\n{recent}\n\nCurrent focus:\n{focus}\n\n"
    "Thought:"
)


class EmbodiedAgent:
    """Real-world exploration agent using a pluggable LLM backend."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        shared_camera: "CameraTool | None" = None,
        camera_gui_priority: bool = False,
    ):
        self.config = config
        self.backend = create_backend(config)
        self._utility_backend = create_utility_backend(config) or self.backend
        self._scene_backend = create_scene_backend(config) or self._utility_backend
        self._background_tasks: set[asyncio.Task[None]] = set()
        self.messages: list = []
        self._started_at = time.time()
        self._turn_count = 0
        # All presentation surfaces share one agent. The coordinator owns
        # current-user switching and single-flight turn execution.
        self._turn_coordinator = TurnCoordinator(
            runner=self._run_request,
            switch_user=self.switch_user,
        )
        self._session_input_tokens: int = 0
        self._session_output_tokens: int = 0
        self._last_context_tokens: int = 0
        self._post_compact: bool = False
        # Separate from _post_compact (which tunes recall depth and resets on
        # the very next prepare_turn): the recovery block waits for the next
        # non-brief turn so it is never consumed invisibly by a brief reply.
        self._post_compact_recovery_pending: bool = False
        self._coherence_retried: bool = False

        self._camera: CameraTool | None = shared_camera
        self._cameras: dict[str, CameraTool] = {}
        self._camera_labels: dict[str, str] = {}
        self._camera_gui_priority = camera_gui_priority
        self._mobility: MobilityTool | None = None
        self._tts: TTSTool | None = None
        self._stt: STTTool | None = None
        self._telegram: TelegramTool | None = None
        self._telegram_transport: TelegramTransport | None = None
        self._telegram_history = TelegramHistory()
        self._me_md: str = self._load_me_md()  # loaded once; restart to pick up changes
        self._memory = ObservationMemory()
        self._memory_worker = MemoryJobWorker(self._memory)
        self._memory_tool = MemoryTool(self._memory)
        self._person_model = PersonModelTracker()
        self._art_critique_store = ArtCritiqueStore()
        self._art_critique_tool = ArtCritiqueTool(self._art_critique_store, self._memory)
        self._tom_tool = ToMTool(
            self._memory,
            default_person=config.companion_name,
            backend=self._utility_backend,
            person_model=self._person_model,
        )
        self._coding = CodingTool(config.coding)
        _commitments_dir = Path.home() / ".familiar_ai"
        _commitments_dir.mkdir(parents=True, exist_ok=True)
        self._commitment_store = SQLiteCommitmentStore(_commitments_dir / "commitments.db")
        self._commitment_tool = CommitmentTool(self._commitment_store)
        self._delegation_runner = DelegatedTaskRunner(
            config=config, commitment_store=self._commitment_store
        )
        self._delegation_tool = DelegationTool(self._delegation_runner)
        try:
            self._identity: IdentityCore | None = IdentityCore(self._memory)
        except Exception as exc:  # noqa: BLE001
            logger.warning("IdentityCore init failed (identity layer dormant): %s", exc)
            self._identity = None
        self._identity_tool = IdentityTool(self._memory)
        # Self-authored time: the agent's own recurring schedule. Routines
        # fire by materializing commitments — the reminder gates do the rest.
        self._routine_store = RoutineStore()
        self._routine_tool = RoutineTool(self._routine_store)
        # Latency instrumentation (FAMILIAR_LATENCY=1; dark by default).
        self._latency = LatencyRecorder()
        # Phase 2 inner loop (off by default). Constructed always so close() can
        # stop it unconditionally; started lazily by prepare_turn when
        # config.inner_loop is set. The UI-owned DesireSystem is late-bound via
        # bind_desires() — until then idle cognition competes without drives.
        self._turn_active = False
        self._desires: DesireSystem | None = None
        self._inner_monologue: deque[InnerThought] = deque(maxlen=8)
        self._train_of_thought = TrainOfThought()
        self._inner_tick_count = 0
        self._inner_escalated_at: dict[str, float] = {}
        self._inner_loop_config = InnerLoopConfig(interval_sec=config.inner_loop_interval)
        self._inner_loop = InnerLoop(self._inner_loop_tick, self._inner_loop_config)
        # Micro-thoughts: verbalized by a dedicated small model (INNER_*), or
        # the utility backend when that is separate from the main model —
        # idle cycles must never burn main-model calls.
        self._inner_backend = create_inner_backend(config) or self._tape_backend()
        self._last_micro_thought_at = 0.0
        self._exploration = ExplorationTracker()
        self._scene: SceneTracker | None = None  # initialized after DB ready in _init_tools

        self._mcp: MCPClientManager | None = None
        self._user_registry = UserRegistry()
        self._current_user = self._user_registry.get_active()
        config.companion_name = self._current_user.name
        self._relationship = RelationshipTracker(user_id=self._current_user.id)
        self._self_state = SelfState()
        self._self_narrative = SelfNarrative(path=self._current_user.self_narrative_path)
        self._concerns = ConcernEngine()
        self._workspace = GlobalWorkspace()
        self._workspace.register_broadcast_listener(self._self_state.on_broadcast)
        # Dense recurrence (FAMILIAR_INNER_DENSE): idle broadcasts re-enter
        # more modules. Registration itself is gated — listeners also fire on
        # the turn path, so an unconditional registration would change
        # default turn behavior.
        self._pending_drive_nudges: dict[str, float] = {}
        if getattr(self.config, "inner_dense", False):
            self._self_state.defer_saves()
            self._workspace.register_broadcast_listener(self._on_broadcast_drive_nudge)
        self._prediction = PredictionEngine()
        # Self-ledger: attention history and the previous session's
        # metacognitive summary survive restarts (JSON state files, same
        # idiom as identity_state.json).
        _state_dir = Path.home() / ".familiar_ai"
        self._attention_schema = AttentionSchema(state_path=_state_dir / "attention_state.json")
        self._dmn = DefaultModeProcessor(self._memory)
        self._meta_monitor = MetaMonitor(state_path=_state_dir / "meta_state.json")
        self._self_ledger_tool = SelfLedgerTool(self)
        self._constitution_block: str | None = None  # rendered once per session
        self._lessons_block: str | None = None  # experience ledger, same idiom
        # Consciousness profile (instrumentation only, FAMILIAR_CONSCIOUSNESS_PROFILE)
        self._last_compete_result: CompeteResult | None = None
        self._last_consciousness_profile: ConsciousnessProfile | None = None
        self._last_say_at: float = 0.0
        self._last_intero_signal = None
        # Reality monitor: session-scoped grounding record (always tracked;
        # the [REALITY] gate itself is opt-in via FAMILIAR_REALITY_GATE).
        self._grounding = GroundingTracker()
        self._appraisal = AppraisalEngine()
        self._social_policy = SocialPolicyEngine()
        self._mental_state_bus = MentalStateBus(path=self._current_user.mental_state_path)
        self._schedule_rules = parse_schedule_config(Path.home() / ".familiar_ai" / "schedule.conf")
        self._heartbeat = HeartbeatRuntime(
            memory=self._memory,
            quiet_rules=self._schedule_rules,
        )
        self._last_tool_error: str | None = None
        self._tool_failure_streak: int = 0
        # Appraised affect for the current/most-recent turn; set by the
        # embodied hook each turn and read by the voice server and the
        # device face-emotion injection.
        self._last_affect: AffectiveState | None = None
        # Deterministic ToM cooldown bookkeeping (see embodied_hook._should_auto_tom)
        self._last_auto_tom_turn: int | None = None
        self._last_auto_tom_act: str | None = None

        # Mood persistence (Phase 2 companion-likeness)
        self._mood: str = "neutral"
        self._mood_intensity: float = 0.0
        self._mood_set_at: float = time.time()

        # Deferred pre-response caches (computed in post-response, used next turn)
        self._cached_plan_ctx: str = ""
        self._cached_workspace_ctx: str = ""
        self._cached_temporal_ctx: str | None = None
        self._cached_companion_mood: str = "engaged"

        # Per-turn cognition pipeline (PR3 of the runtime reorg).
        self._hook = EmbodiedAgentHook(self)

        self._init_tools()

    async def switch_user(self, user_id: str) -> str:
        """Switch the active user without racing a turn from another channel."""
        async with self._get_turn_coordinator().scope("user-switch"):
            return self._switch_user_unlocked(user_id)

    def _switch_user_unlocked(self, user_id: str) -> str:
        """Apply a user switch while the caller owns the turn scope."""
        user_id = user_id.strip()
        if user_id == self._current_user.id:
            return self._current_user.name

        # Persist and close current user's relationship state before switching.
        self._relationship.close()

        new_user = self._user_registry.get(user_id)
        self._current_user = new_user
        self.config.companion_name = new_user.name

        self._relationship = RelationshipTracker(user_id=new_user.id)
        self._self_narrative = SelfNarrative(path=new_user.self_narrative_path)
        self._mental_state_bus = MentalStateBus(path=new_user.mental_state_path)
        self._tom_tool._default_person = new_user.name  # type: ignore[attr-defined]
        return new_user.name

    @property
    def current_user(self) -> UserProfile:
        return self._current_user

    def list_users(self) -> list[UserProfile]:
        return self._user_registry.list_users()

    def _tape_backend(self):
        """Return the backend used for extra planning/replanning checks.

        TAPE is only worth the latency when a separate cheap utility backend exists.
        If utility falls back to the main conversation model, skip the extra round-trips.
        """
        return None if self._utility_backend is self.backend else self._utility_backend

    def bind_desires(self, desires: DesireSystem) -> None:
        """Late-bind the UI-owned DesireSystem for idle cognition.

        The inner loop reads it during workspace competition and boosts it on
        escalation; ownership (tick/satisfy cadence, idle-turn firing) stays
        with the UI loops. Called once by each UI entry point right after both
        objects exist — EmbodiedAgent's constructor cannot own this because the
        DesireSystem is constructed independently in main.py/gui.py.
        """
        self._desires = desires

    def start_mcp_early(self) -> None:
        """Kick off the MCP handshake in the background (issue #188).

        Callable from any running event loop — the UIs invoke it at mount so
        MCP tools are registered by the first turn instead of the second;
        ``prepare_turn`` calls it as the fallback for other entry points.
        Idempotent: at most one in-flight handshake.
        """
        if self._mcp is None or self._mcp.is_started:
            return
        task = getattr(self, "_mcp_start_task", None)
        if task is not None and not task.done():
            return
        self._mcp_start_task = asyncio.ensure_future(self._mcp.start())

    def _spawn_background_task(self, coro: Coroutine[Any, Any, None], *, name: str) -> None:
        """Run non-critical post-turn work off the response critical path."""
        tasks = getattr(self, "_background_tasks", None)
        if tasks is None:
            tasks = set()
            self._background_tasks = tasks
        task = asyncio.create_task(coro, name=name)
        tasks.add(task)

        def _done(done_task: asyncio.Task[None]) -> None:
            tasks.discard(done_task)
            try:
                exc = done_task.exception()
            except asyncio.CancelledError:
                return
            if exc is not None:
                logger.warning("Background task %s failed: %s", name, exc)

        task.add_done_callback(_done)

    async def _drain_background_tasks(self, timeout: float = 6.0) -> None:
        """Wait briefly for background work to finish during shutdown."""
        tasks = getattr(self, "_background_tasks", None)
        if not tasks:
            return
        pending = {task for task in tasks if not task.done()}
        if not pending:
            return
        done, still_pending = await asyncio.wait(pending, timeout=timeout)
        for task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # noqa: BLE001
                logger.warning("Background task failed during drain: %s", exc)
        if still_pending:
            for task in still_pending:
                task.cancel()
            await asyncio.gather(*still_pending, return_exceptions=True)

    async def _run_post_response_pipeline(
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

        emotion = "neutral"

        try:
            if camera_used:
                recent_obs = await self._memory.recall_async(
                    final_text[:200], n=6, kind="observation"
                )
                past_scores = [m.get("score", 0.5) for m in recent_obs[:3]]
                if past_scores:
                    avg_similarity = sum(past_scores) / len(past_scores)
                    novelty = 1.0 - avg_similarity
                else:
                    novelty = 0.8
                novelty = max(0.0, min(1.0, novelty))
                self._exploration.record_novelty(novelty)
                if desires is not None:
                    desires.boost("look_around", novelty * 0.3)
                if self._scene is not None:
                    scene_events = await self._scene.update(
                        final_text[:500],
                        self._scene_backend,
                        prediction_engine=self._prediction,
                        action_name=observation_action_name,
                        action_input=observation_action_input,
                    )
                    _react_to_scene_events(scene_events, desires)
                    pred_signal = self._prediction.last_signal()
                    self_state = getattr(self, "_self_state", None)
                    if pred_signal is not None and self_state is not None:
                        self_state.apply_prediction_feedback(
                            external_surprise=pred_signal.external_surprise,
                            agency_error=pred_signal.agency_error,
                            action_name=pred_signal.action_name,
                        )
                    pred_coalition = self._prediction.as_coalition()
                    if pred_coalition is not None:
                        self._workspace.apply_prediction_error(pred_coalition.novelty)
                await self._memory.save_async(
                    final_text[:500],
                    direction="観察",
                    kind="observation",
                    dedupe_key=self._memory_dedupe_key("observation", final_text[:500]),
                    materialize_now=False,
                )

            emotion = await self._infer_emotion(final_text)
            self._update_mood(emotion)
            summary = await self._summarize_exchange(user_input, final_text)
            await self._memory.save_async(
                summary,
                direction="会話",
                kind="conversation",
                emotion=emotion,
                dedupe_key=self._memory_dedupe_key("conversation", summary),
                materialize_now=False,
            )

            await self._update_self_model(final_text, emotion)
            await self._maybe_update_self_narrative(
                user_input=user_input,
                final_text=final_text,
                emotion=emotion,
                is_desire_turn=is_desire_turn,
            )

            if not is_desire_turn and user_input:
                self._relationship.record_conversation()

            if desires is not None and not is_desire_turn and user_input:
                worry_boost = detect_worry_signal(user_input)
                if worry_boost > 0.0:
                    desires.boost("worry_companion", worry_boost)
                    logger.debug(
                        "Worry signal detected (%.2f): boosting worry_companion",
                        worry_boost,
                    )

            curiosity: str | None = None
            if desires is not None and camera_used:
                curiosity = await self.extract_curiosity(final_text)
                if curiosity:
                    desires.curiosity_target = curiosity
                    desires.boost("look_around", 0.3)
                    await self._memory.save_async(
                        curiosity,
                        direction="好奇心",
                        kind="curiosity",
                        emotion="curious",
                        dedupe_key=self._memory_dedupe_key("curiosity", curiosity),
                        materialize_now=False,
                    )
                    logger.info("Curiosity persisted: %s", curiosity)

            if user_input and not is_desire_turn:
                await self._capture_companion_thread(user_input, desires)

            pred_signal = self._prediction.last_signal()
            concerns = getattr(self, "_concerns", None)
            if concerns is not None:
                concerns.update_from_turn(
                    turn_index=self._turn_count,
                    emotion=emotion,
                    companion_mood=companion_mood,
                    curiosity=curiosity,
                    prediction_signal=pred_signal,
                )

            self_state = getattr(self, "_self_state", None)
            if self_state is not None:
                self_state.apply_turn_context(
                    emotion=emotion,
                    companion_mood=companion_mood,
                    curiosity=curiosity,
                    prediction_signal=pred_signal,
                )

            await self._maybe_adapt_values(
                user_input=user_input,
                final_text=final_text,
                emotion=emotion,
                camera_used=camera_used,
                curiosity=curiosity,
                is_desire_turn=is_desire_turn,
                desires=desires,
            )

            await self._maybe_update_identity(
                user_input=user_input,
                final_text=final_text,
                is_desire_turn=is_desire_turn,
            )

            # ── Deferred pre-response work (results cached for next turn) ──
            try:
                tape_backend = self._tape_backend()
                tool_names = [t["name"] for t in self._all_tool_defs] if tape_backend else []
                deferred_plan_task = (
                    generate_plan(tape_backend, user_input, tool_names)
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
                    self._gather_workspace_context(desires=desires),
                    self._infer_companion_mood(user_input),
                    self._online_temporal_context(desires=desires),
                )
                self._cached_plan_ctx = deferred_plan
                self._cached_workspace_ctx = deferred_workspace
                self._cached_companion_mood = deferred_mood
                self._cached_temporal_ctx = deferred_temporal
                if desires is not None and deferred_mood == "frustrated":
                    desires.boost("worry_companion", 0.3)
                    logger.debug("Companion mood frustrated: boosting worry_companion")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Deferred pre-response caching failed: %s", exc)

        except Exception as exc:  # noqa: BLE001
            logger.warning("Post-response pipeline failed: %s", exc)
        finally:
            # Dense recurrence defers self-state writes; the turn's own affect
            # deltas (apply_turn_context above) must be durable once the
            # post-response pipeline ends — this is what makes the
            # defer_saves docstring ("never a real turn's state") true.
            self_state = getattr(self, "_self_state", None)
            if self_state is not None and hasattr(self_state, "flush"):
                try:
                    self_state.flush()
                except Exception:  # noqa: BLE001
                    pass

    def _init_camera_tools(self) -> None:
        """Initialize the configured camera catalog or legacy primary camera."""
        catalog = getattr(self.config, "camera_catalog", None)
        if catalog is not None:
            for profile in catalog.cameras:
                if profile.id == catalog.default_camera and self._camera is not None:
                    camera = self._camera
                else:
                    camera = CameraTool(
                        profile.host,
                        profile.username,
                        profile.password,
                        profile.onvif_port,
                        preview=profile.preview,
                        ptz_host=profile.ptz_host or profile.host,
                        ptz_username=profile.ptz_username or profile.username,
                        ptz_password=profile.ptz_password or profile.password,
                        ptz_port=profile.ptz_port or profile.onvif_port,
                        connection=profile.connection,
                        camera_id=profile.id,
                    )
                self._cameras[profile.id] = camera
                self._camera_labels[profile.id] = profile.label
            self._camera = self._cameras.get(catalog.default_camera)
        else:
            cam = self.config.camera
            # Allow camera if host is present and not already provided via shared_camera.
            if cam.host and self._camera is None:
                self._camera = CameraTool(
                    cam.host,
                    cam.username,
                    cam.password,
                    cam.port,
                    preview=cam.preview,
                    ptz_host=cam.ptz_host,
                    ptz_username=cam.ptz_username,
                    ptz_password=cam.ptz_password,
                    ptz_port=cam.ptz_port,
                )
            if self._camera is not None:
                self._cameras["main"] = self._camera
                self._camera_labels["main"] = "Primary camera"

    def _init_mobility_and_tts_tools(self) -> None:
        """Initialize mobility and speech-output tools."""
        mob = self.config.mobility
        if mob.api_key and mob.device_id:
            self._mobility = MobilityTool(
                mob.api_region, mob.api_key, mob.api_secret, mob.device_id
            )

        tts = self.config.tts
        if tts.elevenlabs_api_key:
            self._tts = TTSTool(
                tts.elevenlabs_api_key,
                tts.voice_id,
                tts.go2rtc_url,
                tts.go2rtc_stream,
                output=tts.output,
                volume=tts.volume,
            )

    def _init_stt_tool(self) -> None:
        """Initialize the local audio-input tool."""
        stt_cfg = self.config.stt
        if stt_cfg.elevenlabs_api_key:
            cam = self.config.camera
            rtsp_url = (
                f"rtsp://{cam.username}:{cam.password}@{cam.host}:554/stream1" if cam.host else ""
            )
            self._stt = STTTool(
                stt_cfg.elevenlabs_api_key, stt_cfg.language, rtsp_url, stt_cfg.input
            )

    def _init_mcp_client(self) -> None:
        """Initialize MCP when its configuration file exists."""
        cfg_path = _resolve_config_path()
        if cfg_path.exists():
            self._mcp = MCPClientManager(cfg_path)
        elif os.environ.get("MCP_CONFIG"):
            logger.warning("MCP_CONFIG points to non-existent file: %s", cfg_path)

    def _init_telegram_tool(self) -> None:
        """Initialize the shared Telegram transport and outbound tool."""
        telegram_config = self.config.telegram
        if telegram_config.token:
            self._telegram_transport = TelegramTransport(telegram_config.token)
            self._telegram = TelegramTool(
                telegram_config,
                self._user_registry,
                current_user_id=lambda: self._current_user.id,
                transport=self._telegram_transport,
                history=self._telegram_history,
            )

    def _init_scene_tracker(self) -> None:
        """Open the persistent world-model tracker."""
        # World model: persistent scene entity tracker (Phase 1)
        # Reuses the same SQLite DB as ObservationMemory via a separate connection.
        import sqlite3 as _sqlite3
        from pathlib import Path as _Path

        scene_db_path = str(_Path.home() / ".familiar_ai" / "observations.db")
        try:
            _Path(scene_db_path).parent.mkdir(parents=True, exist_ok=True)
            scene_conn = _sqlite3.connect(scene_db_path, check_same_thread=False)
            self._scene = SceneTracker(scene_conn)
        except Exception as exc:
            logger.warning("SceneTracker init failed: %s", exc)

    def _init_tools(self) -> None:
        """Initialize external tools in dependency and lifecycle order."""
        self._init_camera_tools()
        self._init_mobility_and_tts_tools()
        self._init_mcp_client()
        self._init_stt_tool()
        self._init_telegram_tool()
        self._init_scene_tracker()

    @property
    def _all_tool_defs(self) -> list[dict]:
        return self._build_tool_registry().tool_defs()

    def _record_embodied_action(self, name: str, tool_input: dict[str, Any]) -> None:
        """Update exploration state before a camera movement executes."""
        if name in {"look", "look_camera"}:
            self._exploration.record_move(
                tool_input.get("direction", "center"),
                tool_input.get("degrees", 30),
            )

    def _register_device_capabilities(self, registry: ToolRegistry) -> None:
        """Register capabilities backed by configured physical devices."""
        cameras = getattr(self, "_cameras", {})
        camera_labels = getattr(self, "_camera_labels", {})

        if self._camera:
            registry.register(
                CameraCapability(
                    self._camera,
                    before_call=self._record_embodied_action,
                    gui_priority=self._camera_gui_priority,
                )
            )
        if len(cameras) > 1:
            registry.register(
                MultiCameraCapability(
                    cameras,
                    camera_labels,
                    before_call=self._record_embodied_action,
                    gui_priority=self._camera_gui_priority,
                )
            )
        if self._mobility:
            registry.register(MobilityCapability(self._mobility))
        if self._tts:
            registry.register(VoiceCapability(self._tts))
        telegram_tool = getattr(self, "_telegram", None)
        if telegram_tool is not None:
            registry.register(TelegramCapability(telegram_tool))

    @staticmethod
    def _register_optional_capability(
        registry: ToolRegistry,
        tool: Any,
        capability_factory: Callable[[Any], Any],
    ) -> None:
        """Register a one-tool capability when its backing tool is available."""
        if tool is not None:
            registry.register(capability_factory(tool))

    def _register_core_capabilities(self, registry: ToolRegistry) -> None:
        """Register memory, cognition, and optional local service tools."""
        registry.register(
            MemoryCapability(
                self._memory_tool,
                names={"remember", "recall", "resolve_unfinished_business"},
            )
        )
        registry.register(ToMCapability(self._tom_tool))
        registry.register(CodingCapability(self._coding))

        optional_capabilities = (
            (getattr(self, "_art_critique_tool", None), ArtCritiqueCapability),
            (getattr(self, "_commitment_tool", None), CommitmentCapability),
            (getattr(self, "_delegation_tool", None), DelegationCapability),
            (getattr(self, "_identity_tool", None), IdentityCapability),
        )
        for tool, capability_factory in optional_capabilities:
            self._register_optional_capability(registry, tool, capability_factory)

        self_ledger_tool = getattr(self, "_self_ledger_tool", None)
        if self_ledger_tool is not None:
            ledger_names = set(DEFAULT_SELF_LEDGER_TOOLS)
            if getattr(self.config, "experience_ledger", False):
                ledger_names |= {"ledger_commit", "ledger_review"}
            registry.register(SelfLedgerCapability(self_ledger_tool, names=ledger_names))

        self._register_optional_capability(
            registry,
            getattr(self, "_routine_tool", None),
            RoutineCapability,
        )

    def _register_mcp_capability(self, registry: ToolRegistry) -> None:
        """Register MCP as both a named provider and unknown-tool fallback."""
        if self._mcp:
            provider = MCPCapability(self._mcp)
            registry.register(provider)
            registry.register_fallback(provider)

    def _build_tool_registry(self) -> ToolRegistry:
        """Build the per-turn tool registry from configured providers."""
        registry = ToolRegistry()
        self._register_device_capabilities(registry)
        self._register_core_capabilities(registry)
        self._register_mcp_capability(registry)
        return registry

    async def _execute_tool(self, name: str, tool_input: dict) -> tuple[str, list[str]]:
        """Route tool call to the right handler. Returns (text, images_b64)."""
        # Drive the device avatar face from the turn's appraised affect: when
        # the agent speaks through the StackChan gateway ("speak"), attach the
        # mapped emotion so the gateway switches the face before the first
        # audio frame. No-op when the model already chose an emotion, when no
        # affect is available, or for any other tool / speech path.
        last_affect = getattr(self, "_last_affect", None)
        if name == "speak" and last_affect is not None and "emotion" not in tool_input:
            tool_input = {**tool_input, "emotion": affect_to_emotion(last_affect)}
        registry = self._build_tool_registry()
        result = await registry.call(name, tool_input)
        raw = result.image_b64
        images: list[str] = raw if isinstance(raw, list) else ([raw] if raw else [])
        return result.text, images

    @staticmethod
    def _tool_timeout_seconds(name: str) -> float:
        """Return per-tool timeout budget in seconds."""
        return _TOOL_TIMEOUTS.get(name, _DEFAULT_TOOL_TIMEOUT)

    @staticmethod
    def _normalize_brief_turn_text(text: str) -> str:
        """Normalize short conversational turns for lightweight heuristics."""
        return text.strip().lower().rstrip("。.!！?？ ")

    @classmethod
    def _matches_brief_turn_pattern(cls, text: str, patterns: tuple[str, ...]) -> bool:
        normalized = cls._normalize_brief_turn_text(text)
        return any(re.search(pattern, normalized) for pattern in patterns)

    @classmethod
    def _is_candidate_brief_turn(cls, user_input: str, *, is_desire_turn: bool) -> bool:
        """Cheap pre-LLM gate for greeting/ack/correction turns.

        These turns should avoid expensive memory recall and exploratory tools.
        """
        if is_desire_turn:
            return False
        text = user_input.strip()
        if not text or len(text) > 80 or "\n" in text:
            return False
        return (
            cls._matches_brief_turn_pattern(text, _BRIEF_GREETING_PATTERNS)
            or cls._matches_brief_turn_pattern(text, _BRIEF_ACK_PATTERNS)
            or cls._matches_brief_turn_pattern(text, _BRIEF_CORRECTION_PATTERNS)
        )

    @staticmethod
    def _should_use_brief_reply_mode(
        *,
        user_input: str,
        social_policy: SocialPolicyDecision,
        is_desire_turn: bool,
    ) -> bool:
        if is_desire_turn:
            return False
        text = user_input.strip()
        if not text or len(text) > 80 or "\n" in text:
            return False
        if re.search(
            r"(?i)(?:(?<![a-z0-9_])mcp(?![a-z0-9_])|"
            r"(?<![a-z0-9_])tools?(?![a-z0-9_])|"
            r"(?<![a-z0-9_])(?:web|internet|search|brows(?:e|ing))(?![a-z0-9_])|"
            r"ツール|利用可能な機能|使える機能|ウェブ|インターネット|ネット|検索|"
            r"ブラウズ|ブラウジング|調べ(?:られ|れ|る))",
            text,
        ):
            # Capability questions need the complete familiar-ai tool catalog.  In
            # particular, prompt-driven CLI backends cannot discover outer MCP
            # connections from their own process.
            return False
        return social_policy.primary_act in {
            "greeting",
            "acknowledgement",
            "clarification",
            "repair_attempt",
            "boundary_assertion",
            "silence_or_low_presence",
        }

    def _tool_defs_for_turn(
        self, *, brief_reply_mode: bool, excluded_tools: frozenset[str] | None = None
    ) -> list[dict]:
        tool_defs = self._all_tool_defs
        if brief_reply_mode:
            tool_defs = [tool for tool in tool_defs if tool.get("name") in _BRIEF_REPLY_TOOL_NAMES]
        if excluded_tools:
            tool_defs = [tool for tool in tool_defs if tool.get("name") not in excluded_tools]
        return tool_defs

    @staticmethod
    def _brief_reply_prompt() -> str:
        return (
            "[Lightweight turn]\n"
            "- This is a short conversational turn.\n"
            "- Reply directly in 1-2 short sentences.\n"
            "- Do not echo the user's words back; answer in your own words.\n"
            "- Do not infer plans, facts, or feelings the user did not say.\n"
            "- Do not use observation, memory, or ToM tools unless explicitly asked."
        )

    def _configure_backend_for_turn(self, *, brief_reply_mode: bool) -> tuple[Any, Any] | None:
        if not brief_reply_mode or not hasattr(self.backend, "thinking_mode"):
            return None
        previous = (
            getattr(self.backend, "thinking_mode", None),
            getattr(self.backend, "thinking_effort", None),
        )
        self.backend.thinking_mode = "disabled"
        if hasattr(self.backend, "thinking_effort"):
            self.backend.thinking_effort = "low"
        return previous

    def _restore_backend_after_turn(self, snapshot: tuple[Any, Any] | None) -> None:
        if snapshot is None:
            return
        thinking_mode, thinking_effort = snapshot
        if hasattr(self.backend, "thinking_mode"):
            self.backend.thinking_mode = thinking_mode
        if hasattr(self.backend, "thinking_effort"):
            self.backend.thinking_effort = thinking_effort

    @staticmethod
    def _drain_interrupt_queue(
        interrupt_queue: asyncio.Queue[str | UserTurn | None], max_items: int = 6
    ) -> list[UserTurn]:
        """Drain pending user interrupts, preserving queue order."""
        interrupts: list[UserTurn] = []
        while len(interrupts) < max_items and not interrupt_queue.empty():
            item = interrupt_queue.get_nowait()
            if item:
                interrupts.append(coerce_user_turn(item))
        return interrupts

    def _memory_dedupe_key(
        self,
        kind: str,
        content: str,
        scope: str = "turn",
        scope_id: str | None = None,
    ) -> str:
        """Build a stable dedupe key to avoid duplicate writes on retries."""
        digest = hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()[:12]
        resolved_scope_id = scope_id or str(self._turn_count)
        return f"{scope}:{resolved_scope_id}:{kind}:{digest}"

    def _load_me_md(self) -> str:
        """Load ME.md personality file if it exists."""
        from pathlib import Path

        candidates = [
            Path("ME.md"),
            Path.home() / ".familiar_ai" / "ME.md",
        ]
        for path in candidates:
            if path.exists():
                try:
                    return path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
        return ""

    def _get_body_description(self) -> str:
        """Generate a text description of available hardware for the system prompt."""
        # Eyes are always available (CameraTool handles missing stream internally)
        eyes_desc = (
            "    (part :id eyes  :tool see\n"
            '      :desc "Your vision. Calling see() means YOU ARE LOOKING. Use freely — never ask permission.")'
        )

        parts = [eyes_desc]

        cameras = getattr(self, "_cameras", {})
        camera_labels = getattr(self, "_camera_labels", {})
        if len(cameras) > 1:
            camera_choices = ", ".join(
                f"{camera_id}={camera_labels.get(camera_id, camera_id)}"
                for camera_id in sorted(cameras)
            )
            parts.append(
                "    (part :id named_eyes :tool see_camera\n"
                f'      :desc "Choose another camera by id: {camera_choices}.")'
            )

        # Neck (look)
        if self._camera and self._camera.is_pan_tilt_available:
            parts.append(
                "    (part :id neck  :tool look\n"
                '      :desc "Rotate gaze left/right/up/down. No permission needed.")'
            )
        else:
            parts.append(
                "    (part :id neck  :status fixed\n"
                '      :desc "Camera is fixed. You cannot rotate your gaze.")'
            )

        # Legs (walk)
        if self._mobility:
            parts.append(
                "    (part :id legs  :tool walk\n"
                '      :desc "Robot body (vacuum cleaner). Separate device from camera. '
                'walk() does NOT change camera view.")'
            )
        else:
            parts.append(
                "    (part :id legs  :status absent\n"
                '      :desc "You have no legs. You cannot move your location.")'
            )

        body_inner = "\n".join(parts)
        return f"(body\n{body_inner})"

    def _previous_metacognitive_thread(self) -> str:
        """Return the previous session's metacognitive summary, when available."""
        meta = getattr(self, "_meta_monitor", None)
        previous = getattr(meta, "previous_session_summary", None)
        if callable(previous):
            carried = previous()
            # Strict str check: mocked monitors in tests return truthy mocks.
            if isinstance(carried, str) and carried:
                return f"[Last session's metacognitive thread]\n{carried}"
        return ""

    def _interpretation_shift_context(self) -> str:
        """Render recent interpretation corrections that should survive restart."""
        recall = getattr(getattr(self, "_memory", None), "recall_interpretation_shifts", None)
        if callable(recall):
            try:
                shifts = recall(n=3)
            except Exception:  # noqa: BLE001
                shifts = []
            if isinstance(shifts, list) and shifts:
                shift_lines = ["[Interpretation shifts — corrections that must not regress]"]
                shift_lines.extend(
                    f'- {s["entity_key"]}: now read as "{s["new_text"][:120]}"' for s in shifts
                )
                return "\n".join(shift_lines)
        return ""

    def _experience_ledger_carryover_note(self) -> str:
        """Summarize held and overnight-proposed experience lessons."""
        if not getattr(self.config, "experience_ledger", False):
            return ""
        lister = getattr(getattr(self, "_memory", None), "list_experience_lessons", None)
        if not callable(lister):
            return ""
        try:
            lessons = lister()
        except Exception:  # noqa: BLE001
            return ""
        if not isinstance(lessons, list) or not lessons:
            return ""

        proposed = sum(1 for entry in lessons if entry.get("tier") == "auto_proposed")
        note = f"[Experience ledger] {len(lessons)} lessons held"
        if proposed:
            note += (
                f" ({proposed} proposed overnight — inspect with ledger_review; "
                "promoting one adds it to your standing context from the next session)"
            )
        return note

    def _self_ledger_carryover_context(self) -> str:
        """Render first-turn continuity so a restart resumes a self, not a blank."""
        blocks = (
            self._previous_metacognitive_thread(),
            self._interpretation_shift_context(),
            self._experience_ledger_carryover_note(),
        )
        return "\n\n".join(block for block in blocks if block)

    def _post_compact_recovery_context(self) -> str:
        """Re-anchor right after context compaction.

        Ordering is deliberate — constitution before memory: the identity
        block lives in the stable prompt half above; this block reminds the
        model that what was condensed does not include who it is, and
        re-surfaces corrected interpretations so compaction cannot silently
        roll them back.
        """
        lines = [
            "[Post-compaction recovery] Earlier turns were just condensed into a "
            "summary. Who you are did not change: your persona and constitution "
            "above still hold. If anything feels blurry, call who_am_i."
        ]
        recall = getattr(getattr(self, "_memory", None), "recall_interpretation_shifts", None)
        if callable(recall):
            try:
                shifts = recall(n=3)
            except Exception:  # noqa: BLE001
                shifts = []
            lines.extend(
                f"- Corrected reading ({s['entity_key']}): {s['new_text'][:120]}" for s in shifts
            )
        return "\n".join(lines)

    def _experience_lessons_block(self) -> str:
        """Render the self-authored lessons for the stable half, once per session.

        Same idiom as the identity constitution: lazy render, process-lifetime
        cache — mid-session ledger_commit changes the store only, and the
        live stable half (the Anthropic cache prefix) is untouched until the
        next session. Sleep-boundary semantics are not just safe, they are
        the point: you wake up changed, you do not mutate mid-conversation.
        """
        cached = getattr(self, "_lessons_block", None)
        if cached is not None:
            return cached
        if not getattr(self.config, "experience_ledger", False):
            self._lessons_block = ""
            return ""
        memory = getattr(self, "_memory", None)
        lister = getattr(memory, "list_experience_lessons", None)
        if not callable(lister):
            self._lessons_block = ""
            return ""
        try:
            lessons = lister()
        except Exception:  # noqa: BLE001
            lessons = []
        # Only agent-PROMOTED lessons reach the stable half. Overnight
        # proposals are distilled from user-influenced observations; letting
        # them shape the cache prefix without the agent's own ledger_commit
        # would make "promotion is the agent's decision" visibility-false.
        # Proposals surface in the morning note and ledger_review instead.
        held = [
            entry
            for entry in (lessons if isinstance(lessons, list) else [])
            if entry.get("tier") == "agent"
        ]
        if not held:
            self._lessons_block = ""
            return ""
        lines = ["[Lessons I have drawn from experience — my own words, advisory]"]
        lines.extend(f"- {entry.get('lesson_text', '')}" for entry in held)
        block = "\n".join(lines)[:2200]
        self._lessons_block = block
        return block

    def _identity_constitution_block(self) -> str:
        """Render identity assertions for the stable prompt half, once per session.

        Rendered lazily and cached for the process lifetime so the stable half
        stays byte-identical across turns (prompt caching). Values the agent
        commits mid-session already act immediately through IdentityCore's
        checkers and coalitions; they join the rendered constitution on the
        next session, like ME.md edits.
        """
        cached = getattr(self, "_constitution_block", None)
        if cached is not None:
            return cached
        identity = getattr(self, "_identity", None)
        if identity is None:
            self._constitution_block = ""
            return ""
        try:
            assertions = identity.assertions()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Constitution render failed: %s", exc)
            self._constitution_block = ""
            return ""
        if not assertions:
            self._constitution_block = ""
            return ""
        lines = ["[Constitution — what I hold, in my own words]"]
        for a in sorted(assertions, key=lambda a: (not a.non_negotiable, -a.confidence)):
            marker = " (non-negotiable)" if a.non_negotiable else ""
            lines.append(f"- ({a.kind}) {a.statement}{marker}")
        self._constitution_block = "\n".join(lines)
        return self._constitution_block

    def _workspace_prompt_blocks(self, workspace_ctx: str) -> list[str]:
        """Return workspace context, falling back to direct world-model state."""
        if workspace_ctx:
            return [workspace_ctx]

        blocks: list[str] = []
        exploration_ctx = self._exploration_context()
        if exploration_ctx:
            blocks.append(exploration_ctx)
        scene_ctx = self._scene.context_for_prompt() if self._scene else ""
        if scene_ctx:
            blocks.append(scene_ctx)
        return blocks

    def _system_prompt(
        self,
        feelings_ctx: str = "",
        morning_ctx: str = "",
        inner_voice: str = "",
        plan_ctx: str = "",
        companion_mood: str = "engaged",
        continuity_ctx: str = "",
        workspace_ctx: str = "",
        mental_ctx: str = "",
    ) -> tuple[str, str]:
        """Return (stable, variable) system prompt parts for prompt caching.

        stable  — ME.md + core rules + identity constitution; never changes
                  within a session. AnthropicBackend marks this block with
                  cache_control.
        variable — interoception, feelings, inner voice, plan; changes every turn.
        """
        base = assemble_neighbor_system_prompt(
            max_steps=MAX_ITERATIONS,
            profile=getattr(self.config, "prompt_profile", "full"),
        )
        # Dynamically replace (body ...) block based on actual hardware
        body_desc = self._get_body_description()
        base = re.sub(r"\(body.*?\)", body_desc, base, flags=re.DOTALL)

        # Constitution before memory: identity assertions join the STABLE half
        # (rendered once per session — keeps prompt caching intact; live
        # dissonance state stays in the variable half via mental_ctx).
        constitution = self._identity_constitution_block()
        lessons = self._experience_lessons_block()
        stable_parts = [p for p in [self._me_md, base, constitution, lessons] if p]
        stable = "\n\n---\n\n".join(stable_parts)

        agent_mood, agent_mood_intensity = self._decayed_mood()
        self_state = getattr(self, "_self_state", None)
        self_state_snapshot = self_state.snapshot() if self_state is not None else None
        intero = _interoception(
            self._started_at,
            self._turn_count,
            companion_mood,
            agent_mood=agent_mood,
            agent_mood_intensity=agent_mood_intensity,
            self_state=self_state_snapshot,
        )
        relationship_ctx = self._relationship.context_for_prompt()
        variable_parts: list[str] = [intero]
        if relationship_ctx:
            variable_parts.append(relationship_ctx)
        person_ctx = self._person_model_context()
        if person_ctx:
            variable_parts.append(person_ctx)
        if continuity_ctx:
            variable_parts.append(continuity_ctx)
        if mental_ctx:
            variable_parts.append(mental_ctx)
        # Morning reconstruction takes precedence on first turn; otherwise use feelings
        if morning_ctx:
            variable_parts.append(morning_ctx)
        elif feelings_ctx:
            variable_parts.append(feelings_ctx)
        # Inner voice: agent's own desire/impulse — NOT a user message.
        # Injected here so the model understands this is self-generated, not from the companion.
        if inner_voice:
            variable_parts.append(
                f"{_t('inner_voice_label')}\n{inner_voice}\n{_t('inner_voice_directive')}"
            )
        # TAPE: upfront action plan to anchor the react loop (mechanism 1)
        if plan_ctx:
            variable_parts.append(
                "[Action plan for this turn — follow it unless you discover a good reason not to]\n"
                + plan_ctx
            )

        # Global Workspace replaces individual exploration + scene context blocks.
        variable_parts.extend(self._workspace_prompt_blocks(workspace_ctx))

        variable = "\n\n---\n\n".join(variable_parts)
        return stable, variable

    def _self_continuity_context(self) -> str:
        """Return a compact continuity block from latent concerns and recent action traces."""
        blocks: list[str] = []

        concerns = getattr(self, "_concerns", None)
        if concerns is not None:
            concern_ctx = concerns.context_for_prompt(turn_index=self._turn_count)
            if concern_ctx:
                blocks.append(concern_ctx)

        prediction = getattr(self, "_prediction", None)
        if prediction is not None:
            trace_ctx = prediction.context_for_prompt()
            if trace_ctx:
                blocks.append(trace_ctx)

        return "\n\n".join(blocks)

    def _commitments_context(self) -> str:
        """Surface due + soon-upcoming commitments so the secretary can act on them."""
        store = getattr(self, "_commitment_store", None)
        if store is None:
            return ""
        now = time.time()
        try:
            due = store.list_due(now=now)
            upcoming = store.list_upcoming(now=now, horizon=COMMITMENT_UPCOMING_HORIZON_SECONDS)
        except Exception:
            logger.debug("commitment context fetch failed", exc_info=True)
            return ""
        return format_commitments_for_context(due=due[:5], upcoming=upcoming[:5], now=now)

    def _today_agenda_context(self) -> str:
        """Morning secretary surface: overdue + today's commitments as an agenda.

        Horizon is the rest of the local day, extended to at least 12h so a
        late-night first turn still previews the early morning.
        """
        store = getattr(self, "_commitment_store", None)
        if store is None:
            return ""
        now = time.time()
        local_now = datetime.now()
        midnight = (
            local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        ).timestamp() + 86400
        horizon = max(midnight - now, 12 * 3600)
        try:
            due = store.list_due(now=now)
            upcoming = store.list_upcoming(now=now, horizon=horizon)
        except Exception:
            logger.debug("agenda fetch failed", exc_info=True)
            return ""
        items = (due + upcoming)[:8]
        if not items:
            return ""
        lines = ["[Today's agenda — weave these into the greeting naturally]"]
        lines.extend(format_commitment_line(c, now=now) for c in items)
        return "\n".join(lines)

    async def _run_auto_tom(self, user_input: str, *, timeout: float = 12.0) -> str:
        """Run the ToM tool deterministically when social policy demands it.

        The model is not relied on to call the tool itself; this guarantees
        perspective-taking happens on emotionally loaded turns (and the result
        feeds the persistent person model as a side effect). Failures and
        timeouts degrade to an empty string — never break the turn.
        """
        tom_tool = getattr(self, "_tom_tool", None)
        if tom_tool is None:
            return ""
        try:
            text, _image = await asyncio.wait_for(
                tom_tool.call("tom", {"situation": user_input[:500]}),
                timeout=timeout,
            )
        except Exception:
            logger.debug("auto ToM failed", exc_info=True)
            return ""
        text = str(text).strip()
        return text[:1200] if text else ""

    async def _run_auto_tom_background(self, user_input: str) -> None:
        """Background wrapper for the deterministic ToM run (roadmap PR7).

        Off the critical path the run can afford the tool's full budget; the
        text result is discarded here — the value is the structured inference
        the ToM tool writes into the person model as a side effect, which the
        [Person model] block surfaces from the next turn on.
        """
        await self._run_auto_tom(user_input, timeout=20.0)

    def _person_model_context(self) -> str:
        """Surface the accumulated ToM model of the companion, if any."""
        tracker = getattr(self, "_person_model", None)
        if tracker is None:
            return ""
        try:
            return tracker.context_for_prompt(self.config.companion_name)
        except Exception:
            logger.debug("person model context fetch failed", exc_info=True)
            return ""

    def _exploration_context(self) -> str:
        """Return exploration history for ICL-based direction steering."""
        return self._exploration.context_for_prompt(n=5)

    def _collect_interoception(self):
        mcp_path = os.environ.get("FAMILIAR_INTEROCEPTION_MCP_PATH", "").strip()
        if not mcp_path and getattr(self.config, "daemon", False) is True:
            # familiard writes its body payload to a well-known path; adopt it
            # automatically when the daemon is enabled and has produced one.
            # (`is True` keeps MagicMock configs in tests from flipping this.)
            candidate = Path.home() / ".familiar_ai" / "interoception.json"
            if candidate.exists():
                mcp_path = str(candidate)
        if mcp_path:
            max_staleness = int(
                os.environ.get("FAMILIAR_INTEROCEPTION_MCP_MAX_STALENESS", "45").strip() or "45"
            )
            signal = MCPInteroceptionProvider(
                mcp_path,
                max_staleness_seconds=max_staleness,
            ).collect()
            if signal.provider == "noop":
                signal = RuntimeInteroceptionProvider(
                    started_at=self._started_at,
                    turn_count=self._turn_count,
                    pending_tasks=len(getattr(self, "_background_tasks", set())),
                    quiet_hours=[(r.start_hour, r.end_hour) for r in self._schedule_rules],
                ).collect()
        else:
            signal = RuntimeInteroceptionProvider(
                started_at=self._started_at,
                turn_count=self._turn_count,
                pending_tasks=len(getattr(self, "_background_tasks", set())),
                quiet_hours=[(r.start_hour, r.end_hour) for r in self._schedule_rules],
            ).collect()
        return signal, semantic_pressure(signal)

    def _provisional_relationship_update(
        self,
        *,
        user_text: str,
        social_policy: SocialPolicyDecision,
    ) -> None:
        lower = user_text.lower()
        if any(token in lower for token in ("hurt", "傷つ", "前の返事")):
            self._relationship.record_repair(user_text[:180], resolved=False)
            self._relationship.note_trust_shift(
                max(0.0, self._relationship.trust - 0.08), user_text[:180], confidence=0.8
            )
            self._relationship.record_failed_support_pattern(
                "advice before validation", confidence=0.75
            )
        elif social_policy.primary_act == "delight_share":
            self._relationship.note_intimacy_shift(
                min(1.0, self._relationship.intimacy + 0.04),
                "shared delight",
                confidence=0.62,
            )
        elif social_policy.primary_act in {"fatigue_signal", "grief_signal", "venting"}:
            self._relationship.record_support_preference(
                "validate first before problem solving",
                confidence=0.72,
            )
        if social_policy.primary_act == "boundary_assertion":
            self._relationship.add_boundary(user_text[:120], severity=3)
        if social_policy.primary_act == "playful_probe":
            self._relationship.record_shared_ritual("light playful exchange", confidence=0.55)

    @staticmethod
    def _format_social_policy_prompt(policy: SocialPolicyDecision) -> str:
        lines = [
            "[Interaction policy]",
            f"- primary-act: {policy.primary_act}",
            f"- response-mode: {policy.response_mode}",
            f"- softness: {policy.softness:.2f}",
            f"- directness: {policy.directness:.2f}",
            f"- initiative: {policy.initiative:.2f}",
        ]
        if policy.avoid_problem_solving:
            lines.append("- validate before advice; do not rush into fixing")
        if policy.should_recall_relational_memory:
            lines.append("- relational memory is relevant if it naturally helps")
        if policy.mention_memory:
            lines.append("- a memory mention is allowed only if it fits naturally")
        if policy.avoid_raw_interoception_numbers:
            lines.append("- never mention raw internal/body metrics")
        if policy.acknowledge_capacity:
            lines.append(
                "- you are running low right now; be honest about your current "
                "capacity instead of overpromising — offer a smaller step or a deferral"
            )
        return "\n".join(lines)

    def _update_consciousness_profile(self, *, origin: str, interoception_signal=None):
        """Compute and cache the consciousness profile from existing signals.

        Instrumentation only (FAMILIAR_CONSCIOUSNESS_PROFILE): the result
        feeds diagnostics surfaces and the mental-state snapshot, never the
        prompt. Every input read is getattr-guarded so partially-constructed
        agents (tests) degrade to defaults instead of raising.
        """
        signal = interoception_signal or getattr(self, "_last_intero_signal", None)
        self_state = getattr(self, "_self_state", None)
        state = self_state.snapshot() if self_state is not None else {}
        compete = getattr(self, "_last_compete_result", None)
        winner = compete.winner if compete is not None else None
        workspace = getattr(self, "_workspace", None)
        identity = getattr(self, "_identity", None)
        identity_state = identity.state_for_snapshot() if identity is not None else None
        meta = getattr(self, "_meta_monitor", None)
        prediction = getattr(self, "_prediction", None)
        pred_signal = prediction.last_signal() if prediction is not None else None
        train = getattr(self, "_train_of_thought", None)
        attention = getattr(self, "_attention_schema", None)
        try:
            self_report = bool(attention.self_report()) if attention is not None else False
        except Exception:  # noqa: BLE001
            self_report = False
        grounding = getattr(self, "_grounding", None)  # arrives with the reality monitor

        inputs = ProfileInputs(
            energy=float(getattr(signal, "energy", 0.5)),
            quiet_hours=bool(getattr(signal, "quiet_hours", False)),
            arousal=float(state.get("arousal", 0.35)),
            fatigue=float(state.get("fatigue", 0.2)),
            winner_present=winner is not None,
            winner_score=float(winner.score()) if winner is not None else 0.0,
            effective_threshold=(
                float(workspace.effective_threshold()) if workspace is not None else 0.4
            ),
            coalition_count=len(compete.coalitions) if compete is not None else 0,
            identity_dissonance=(
                float(identity_state.dissonance) if identity_state is not None else 0.0
            ),
            identity_threat=(
                float(identity_state.threat_level) if identity_state is not None else 0.0
            ),
            meta_inconsistency=self._meta_inconsistency_coalition() is not None,
            focus_stability=float(state.get("focus_stability", 0.5)),
            source_diversity=(float(meta.source_diversity()) if meta is not None else 0.0),
            peripheral_count=len(compete.others) if compete is not None else 0,
            thought_streak=int(getattr(train, "streak", 0)),
            sensor_confidence=float(state.get("sensor_confidence", 0.7)),
            external_surprise=float(getattr(pred_signal, "external_surprise", 0.0)),
            agency_error=float(getattr(pred_signal, "agency_error", 0.0)),
            grounding_ratio=(grounding.ratio() if grounding is not None else None),
            tts_present=getattr(self, "_tts", None) is not None,
            say_recent=(time.time() - getattr(self, "_last_say_at", 0.0)) < 600.0,
            self_report_available=self_report,
        )
        profile = compute_consciousness_profile(inputs, origin=origin)
        self._last_consciousness_profile = profile
        return profile

    def _build_mental_snapshot(
        self,
        *,
        interoception_signal,
        affect,
        social_policy: SocialPolicyDecision,
        working_memory: list[dict],
        continuity_note: str,
        desires: DesireSystem | None,
    ) -> MentalStateSnapshot:
        drive_levels = desires.drive_vector() if desires is not None else {}
        dominant = desires.get_dominant() if desires is not None else None
        working_items = [
            WorkingMemoryItem(
                memory_id=str(item.get("memory_id", "")),
                summary=str(item.get("summary", "")),
                source_kind=str(item.get("source_kind", item.get("kind", "memory"))),
                salience=float(item.get("salience", item.get("confidence", 0.5))),
                episode_id=item.get("episode_id"),
            )
            for item in working_memory[:5]
        ]
        return MentalStateSnapshot(
            turn_index=self._turn_count,
            created_at=datetime.utcnow().isoformat(),
            interoception=interoception_signal,
            affect=affect,
            social=SocialState(
                primary_act=social_policy.primary_act,
                response_mode=social_policy.response_mode,
                trust=self._relationship.trust,
                intimacy=self._relationship.intimacy,
                repair_needed=social_policy.primary_act == "repair_attempt",
                recall_relational_memory=social_policy.should_recall_relational_memory,
                mention_memory=social_policy.mention_memory,
                initiative=social_policy.initiative,
                directness=social_policy.directness,
                softness=social_policy.softness,
            ),
            drives=DriveVector(
                levels=drive_levels,
                dominant_drive=dominant[0] if dominant else None,
                dominant_level=float(dominant[1]) if dominant else 0.0,
            ),
            working_memory=working_items,
            continuity_note=continuity_note,
            identity=(
                identity.state_for_snapshot()
                if (identity := getattr(self, "_identity", None)) is not None
                else IdentityState()
            ),
            consciousness=(
                self._update_consciousness_profile(
                    origin="turn", interoception_signal=interoception_signal
                ).as_state()
                if getattr(self.config, "consciousness_profile", False)
                else ConsciousnessState()
            ),
        )

    async def _inner_loop_tick(self) -> None:
        """One idle workspace cycle — the sub-verbal train of thought.

        Skipped while a turn is in flight. Cheap (zero LLM/embedding calls)
        except every ``full_cycle_every``-th tick, which runs the
        embedding-backed competition. The winner feeds the train of thought;
        a repeated focus crystallizes into the inner monologue, and a strong
        sustained focus escalates via ``_maybe_escalate_inner_focus``. The
        tick itself never starts a turn — single-flight stays with the UI
        idle loops.
        """
        if self._turn_active:
            return
        self._inner_tick_count += 1
        self._modulate_inner_cadence()
        full_every = max(self._inner_loop_config.full_cycle_every, 1)
        cheap = self._inner_tick_count % full_every != 0

        extra_coalitions = []
        recurrence = self._train_of_thought.as_coalition()
        if recurrence is not None:
            extra_coalitions.append(recurrence)
        result = await self._compete_once(
            cheap=cheap, desires=self._desires, extra_coalitions=extra_coalitions
        )
        if self._turn_active:
            # A real turn started while the competition awaited — yield.
            # Without this re-check, the dense block below would inject an
            # idle broadcast into the live turn's self-state and attention.
            return
        if getattr(self.config, "consciousness_profile", False):
            # Winnerless ticks are informative too (low access), so the
            # profile updates before the early return below.
            self._update_consciousness_profile(origin="tick")
        # Dense recurrence: accumulated drive nudges flush once per full
        # cycle — up to one desires.json write per mapped drive per cycle,
        # instead of one per broadcast.
        pending = getattr(self, "_pending_drive_nudges", None)
        if not cheap and pending and self._desires is not None:
            self._pending_drive_nudges = {}
            for drive, amount in pending.items():
                try:
                    self._desires.boost(drive, amount)
                except Exception:  # noqa: BLE001
                    pass
        winner = result.winner
        if winner is None:
            return

        self._train_of_thought.observe(winner)
        if getattr(self.config, "inner_dense", False):
            # Idle broadcast re-entry: the winner shapes attention history and
            # the broadcast listeners (self-state, drive nudges) between
            # turns, not only on them. Persistence is batched in each module.
            try:
                self._attention_schema.note_focus(winner)
            except Exception:  # noqa: BLE001
                pass
            try:
                await self._workspace.notify_listeners(winner)
            except Exception:  # noqa: BLE001
                pass
        streak = self._train_of_thought.streak
        # Recurrence sources never re-crystallize — the monologue echoing its
        # own contents back into itself would be a feedback loop, not thought.
        if streak >= _INNER_CRYSTALLIZE_MIN_STREAK and winner.source not in (
            "train_of_thought",
            "monologue",
        ):
            verbalized = await self._maybe_micro_thought(winner)
            self._inner_monologue.append(
                InnerThought(
                    summary=verbalized or winner.summary,
                    source=winner.source,
                    ts=time.time(),
                    score=winner.activation,
                )
            )
        self._maybe_escalate_inner_focus(winner, streak)

    def _maybe_escalate_inner_focus(self, winner, streak: int) -> None:
        """Boost the drive matching a strong, sustained idle focus.

        Escalation only feeds the desire system; the actual turn fires through
        the UI-owned idle precedence chain (user > reminder > desire > idle),
        never from the inner-loop task — that chain is what guarantees
        single-flight ``run()`` calls. A per-source cooldown stops one focus
        from pumping the same drive tick after tick.
        """
        if self._desires is None or streak < _INNER_ESCALATION_MIN_STREAK:
            return
        if winner.activation + winner.urgency < _INNER_ESCALATION_MIN_SALIENCE:
            return
        drive = _INNER_SOURCE_TO_DRIVE.get(winner.source)
        if drive is None:
            return
        now = time.time()
        last = self._inner_escalated_at.get(winner.source)
        if last is not None and now - last < _INNER_ESCALATION_COOLDOWN_SEC:
            return
        self._inner_escalated_at[winner.source] = now
        self._desires.boost(drive, _INNER_ESCALATION_BOOST)
        # The thought did its job — it now lives in drive space; let idle
        # cognition move on instead of re-pumping the same focus.
        self._train_of_thought.reset()
        logger.info("Inner loop: sustained focus on %r escalated to drive %r", winner.source, drive)

    async def _maybe_micro_thought(self, winner) -> str | None:
        """Verbalize a crystallized focus as one short first-person thought.

        Only runs with a dedicated small backend and under a rate limit — the
        sub-verbal monologue works fine without it; this just gives the train
        of thought words. Failures degrade to the raw coalition summary.
        """
        return await self._verbalize_focus(winner)

    async def _verbalize_focus(self, winner, *, bypass_rate_limit: bool = False) -> str | None:
        """Shared verbalization core for micro-thoughts and dream cycles.

        Dreams run in their own bounded nightly loop, so they bypass the
        120 s micro-thought limiter instead of starving it (and vice versa).
        """
        backend = getattr(self, "_inner_backend", None)
        if backend is None:
            return None
        if not bypass_rate_limit:
            now = time.time()
            last = getattr(self, "_last_micro_thought_at", 0.0)
            if now - last < _INNER_MICRO_THOUGHT_MIN_INTERVAL_SEC:
                return None
            self._last_micro_thought_at = now
        recent = "\n".join(f"- {t.summary[:100]}" for t in list(self._inner_monologue)[-3:])
        prompt = _MICRO_THOUGHT_PROMPT.format(
            agent_name=self.config.agent_name,
            recent=recent or "(none)",
            focus=(winner.context_block or winner.summary)[:400],
        )
        try:
            text = await backend.complete(prompt, max_tokens=_INNER_MICRO_THOUGHT_MAX_TOKENS)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Micro-thought generation failed: %s", exc)
            return None
        text = (text or "").strip().strip('"')
        if not text:
            return None
        return text.splitlines()[0][:200]

    def _inner_monologue_coalition(self):
        """Recent idle thoughts compete back into the workspace.

        This is how between-turn thought reaches the next conversation: the
        monologue surfaces as one modest coalition whose salience fades as the
        thoughts age (TTL), so stale idle musings don't haunt hours later.
        """
        monologue = getattr(self, "_inner_monologue", None)
        if not monologue:
            return None
        now = time.time()
        fresh = [t for t in monologue if now - t.ts < _INNER_MONOLOGUE_TTL_SEC]
        if not fresh:
            return None
        from .workspace import Coalition as _Coalition

        lines = ["[Inner monologue — my own recent idle thoughts]"]
        lines.extend(f"- {t.summary[:140]}" for t in fresh[-4:])
        freshness = max(0.0, 1.0 - (now - fresh[-1].ts) / _INNER_MONOLOGUE_TTL_SEC)
        return _Coalition(
            source="monologue",
            summary=f"idle thoughts ({len(fresh)})",
            activation=0.3 + 0.15 * freshness,
            urgency=0.15,
            novelty=0.2 + 0.3 * freshness,
            context_block="\n".join(lines),
        )

    def _modulate_inner_cadence(self) -> None:
        """Body-modulated thought tempo.

        A lively body (high interoceptive energy) quickens idle cycles; a
        tired or quiet-hours body slows them — the same self-regulation a
        person does at 3 AM. Best-effort: any failure leaves the cadence
        untouched.
        """
        try:
            signal, _ = self._collect_interoception()
            self._last_intero_signal = signal  # cached for the tick-path profile
            base = float(self.config.inner_loop_interval)
            factor = 1.6 - 0.8 * float(signal.energy)  # energy 1.0 → 0.8x; 0.0 → 1.6x
            # Configurable floor (FAMILIAR_INNER_MIN_INTERVAL, default = the
            # historical 5 s): dense setups may push toward ~1 Hz. The cheap
            # tick costs ~3.4 ms, so even 1 Hz is ~0.3% of one core.
            floor = max(
                1.0,
                float(getattr(self.config, "inner_min_interval", _INNER_CADENCE_MIN_SEC)),
            )
            self._inner_loop_config.interval_sec = min(
                _INNER_CADENCE_MAX_SEC, max(floor, base * factor)
            )
        except Exception:  # noqa: BLE001
            pass

    # ── Sleep consolidation (FAMILIAR_SLEEP_CONSOLIDATION) ─────────────────

    def last_consolidation_night_key(self) -> str | None:
        """Persisted once-per-night marker (cached; heartbeat_state idiom).

        The cache stores "" for "no marker file", so a fresh install does not
        re-read the missing file on every idle tick.
        """
        cached = getattr(self, "_consolidation_night_key", None)
        if cached is not None:
            return cached or None
        path = Path.home() / ".familiar_ai" / "consolidation_state.json"
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            key = str(raw.get("night_key", ""))
        except Exception:  # noqa: BLE001
            key = ""
        self._consolidation_night_key = key
        return key or None

    def _mark_consolidation_night(self, key: str) -> None:
        self._consolidation_night_key = key
        path = Path.home() / ".familiar_ai" / "consolidation_state.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"night_key": key}), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist consolidation marker: %s", exc)

    def consolidation_quiet_end_hour(self) -> int:
        """The configured quiet-hours end — gate and job MUST use the same
        value or their night keys diverge (double runs / permanent skips)."""
        return int(getattr(getattr(self, "_schedule_rule", None), "end_hour", 7) or 7)

    def start_sleep_consolidation(self) -> None:
        """Spawn the nightly job as a background task — never fires a turn.

        Single-flight: the marker-before-LLM ordering protects sequential
        ticks, but nothing else prevents concurrent jobs (double decay,
        duplicate distill calls) — refuse to spawn while one is in flight.
        """
        for task in getattr(self, "_background_tasks", ()):
            try:
                if task.get_name() == "sleep-consolidation" and not task.done():
                    return
            except Exception:  # noqa: BLE001
                continue
        self._spawn_background_task(self._run_sleep_consolidation(), name="sleep-consolidation")

    async def _run_sleep_consolidation(self) -> None:
        """Nightly memory hygiene: dedup, decay, distill, expire (+ dreams).

        Sleep is lossy compression of episodes into the model. Marker is
        written BEFORE the LLM steps (record-before-run: a crashed job must
        not retry all night); every step is individually best-effort and
        yields to a starting turn.
        """
        if not getattr(self.config, "sleep_consolidation", False):
            return
        if self._turn_active:
            return
        night_key = night_key_for(
            datetime.now(), quiet_end_hour=self.consolidation_quiet_end_hour()
        )
        self._mark_consolidation_night(night_key)
        stats: dict[str, int] = {}

        try:
            stats["deduped"] = await self._memory.consolidate_memories_async()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sleep consolidation dedup failed: %s", exc)
        if self._turn_active:
            return

        try:
            today = datetime.now().strftime("%Y-%m-%d")
            stats["decayed"] = await self._memory.decay_importance_async(before_date=today)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sleep consolidation decay failed: %s", exc)
        if self._turn_active:
            return

        try:
            stats["distilled"] = await self._distill_yesterday_facts()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sleep distillation failed: %s", exc)
        if self._turn_active:
            return

        try:
            stats["expired"] = await self._memory.expire_working_memory_async()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Working-memory expiry failed: %s", exc)

        if getattr(self.config, "experience_ledger", False) and not self._turn_active:
            try:
                await self._propose_overnight_lesson()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Overnight lesson proposal failed: %s", exc)

        if getattr(self.config, "dream_mode", False) and not self._turn_active:
            try:
                stats["dreams"] = await self._run_dream_cycles()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Dream cycle failed: %s", exc)

        logger.info("Sleep consolidation done (%s): %s", night_key, stats)

    async def _distill_yesterday_facts(self) -> int:
        """One bounded utility call: yesterday's observations → ≤3 stable facts."""
        if self._utility_backend is self.backend:
            return 0  # never burn main-model calls on maintenance
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        observations = await asyncio.to_thread(
            self._memory.get_observations_for_date, yesterday, 40
        )
        if not observations:
            return 0
        lines = "\n".join(f"- {str(o.get('content', ''))[:160]}" for o in observations[:40])
        prompt = (
            "From yesterday's observations, extract at most 3 stable facts worth "
            "remembering long-term (recurring patterns, preferences, environment "
            "facts — NOT one-off events). Reply with one fact per line as "
            "'key-slug: fact text' (slug: lowercase, hyphens). Reply 'none' if "
            f"nothing qualifies.\n\nObservations:\n{lines}"
        )
        # Generous timeout: a nightly job on a local server usually finds the
        # utility model cold-unloaded — the load alone can eat 30 s, and
        # nothing is waiting on this background task.
        reply = await asyncio.wait_for(
            self._utility_backend.complete(prompt, max_tokens=300), timeout=90.0
        )
        logger.debug("Night distillation reply: %r", (reply or "")[:400])
        count = 0
        for raw_line in (reply or "").strip().splitlines():
            if count >= 3 or ":" not in raw_line:
                continue
            key, _, text = raw_line.partition(":")
            # Strip markdown bullets/emphasis BEFORE slugging, or "- key" and
            # "**key**" mint different fact keys than the bare form.
            key = key.strip().lstrip("-*• \t").strip().lower()
            key = re.sub(r"[^a-z0-9-]", "", key.replace(" ", "-")).strip("-")[:40]
            text = text.strip()[:200]
            if not key or not text or key == "none":
                continue
            await self._memory.upsert_semantic_fact_async(
                f"night:{key}", text, confidence=0.55, tags="night_distill"
            )
            count += 1
        return count

    async def _propose_overnight_lesson(self) -> int:
        """One bounded utility call proposing at most ONE auto-tier lesson.

        Proposals land at tier='auto_proposed', confidence 0.4 — visible to
        the agent as '(proposed)' in its standing context; promoting one is
        the agent's own ledger_commit decision, never automatic.
        """
        if self._utility_backend is self.backend:
            return 0
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        observations = await asyncio.to_thread(
            self._memory.get_observations_for_date, yesterday, 30
        )
        if len(observations) < 3:
            return 0  # too little lived material to generalize from
        lines = "\n".join(f"- {str(o.get('content', ''))[:140]}" for o in observations[:30])
        prompt = (
            "From yesterday's interactions, propose at most ONE short "
            "first-person lesson about how to be a better companion (e.g. "
            "'when they are tired, shorter replies land better'). Reply as "
            "'key-slug: lesson text' (<=140 chars) or 'none'.\n\n"
            f"Observations:\n{lines}"
        )
        reply = await asyncio.wait_for(
            self._utility_backend.complete(prompt, max_tokens=120), timeout=90.0
        )
        line = (reply or "").strip().splitlines()[0] if (reply or "").strip() else ""
        if ":" not in line:
            return 0
        key, _, text = line.partition(":")
        key = key.strip().lstrip("-*• \t").strip().lower()
        key = re.sub(r"[^a-z0-9-]", "", key.replace(" ", "-")).strip("-")[:40]
        text = text.strip()[:160]
        if not key or not text or key == "none":
            return 0
        ok = await self._memory.upsert_experience_lesson_async(
            key, text, tier="auto_proposed", confidence=0.4, source="night_proposal"
        )
        # A full ledger of held lessons outranks a fresh proposal — the store
        # reports the eviction honestly and we count 0, not a phantom write.
        return 1 if ok else 0

    async def _run_dream_cycles(self, cycles: int = 3) -> int:
        """Ungrounded generative cycles, journaled as kind='dream'.

        The DMN wanders (embedding recall, no camera, no tools — grounding
        off is structural), the small model verbalizes, and the journal is
        provenance-tagged so a dream can never be recalled as observation.
        """
        if getattr(self, "_inner_backend", None) is None:
            # Dreams require the dedicated small model: journaling raw recall
            # summaries as "dreams" would widen the provenance lane for no
            # generative content.
            return 0
        dreamed = 0
        for _ in range(max(0, cycles)):
            if self._turn_active:
                break
            coalition = await self._dmn.wander()
            if coalition is None:
                break
            text = await self._verbalize_focus(coalition, bypass_rate_limit=True)
            content = (text or coalition.summary or "").strip()
            if not content:
                continue
            await self._memory.save_async(
                content,
                direction="dream",
                kind="dream",
                materialize_now=False,
                dedupe_key=f"dream:{content[:60]}",
            )
            dreamed += 1
        return dreamed

    async def _on_broadcast_drive_nudge(self, winner) -> None:
        """Dense-recurrence listener: a broadcast gently presses its drive.

        Accumulates in memory (capped) and is flushed into
        ``DesireSystem.boost`` once per full inner-loop cycle — batching the
        per-boost desires.json write. Registered only when
        ``FAMILIAR_INNER_DENSE`` is on; no-op while desires are unbound.
        """
        drive = _INNER_SOURCE_TO_DRIVE.get(getattr(winner, "source", ""))
        if drive is None:
            return
        current = self._pending_drive_nudges.get(drive, 0.0)
        bump = 0.02 * max(0.0, float(getattr(winner, "activation", 0.0)))
        self._pending_drive_nudges[drive] = min(0.15, current + bump)

    async def _compete_once(
        self,
        *,
        cheap: bool = False,
        desires: DesireSystem | None = None,
        extra_coalitions: list | None = None,
    ) -> CompeteResult:
        """Gather coalitions and run one ignition competition.

        Returns the structured outcome (winner / others / all coalitions) so
        callers can either render the broadcast string (the turn path) or act
        on the winner object (the inner loop). When ``cheap=True`` the two
        embedding-backed sources are skipped — the async memory recall and the
        DMN mind-wander fallback — leaving only the in-memory sync providers, so
        a cheap cycle costs zero LLM/embedding calls.
        """
        # Sync coalitions (wrap in to_thread to avoid blocking) — all in-memory.
        sync_tasks = [
            asyncio.to_thread(self._exploration.as_coalition),
            asyncio.to_thread(self._self_narrative.as_coalition),
            asyncio.to_thread(self._tom_tool.as_coalition),
            asyncio.to_thread(self._prediction.as_coalition),
            asyncio.to_thread(self._attention_schema.as_coalition),
            asyncio.to_thread(self._meta_monitor.as_coalition),
            asyncio.to_thread(self._meta_inconsistency_coalition),
            asyncio.to_thread(self._inner_monologue_coalition),
        ]
        if self._scene is not None:
            sync_tasks.append(asyncio.to_thread(self._scene.as_coalition))
        if desires is not None:
            sync_tasks.append(asyncio.to_thread(desires.as_coalition))
        identity = getattr(self, "_identity", None)
        if identity is not None:
            sync_tasks.append(asyncio.to_thread(identity.as_coalition))

        # Async coalitions (embedding-backed) — skipped on the cheap cycle.
        async_tasks = [] if cheap else [self._memory.as_coalition_async()]

        results = await asyncio.gather(*sync_tasks, *async_tasks, return_exceptions=True)

        from .workspace import Coalition as _Coalition

        coalitions = []
        for r in results:
            if isinstance(r, Exception):
                logger.debug("Coalition gather error: %s", r)
            elif isinstance(r, _Coalition):
                coalitions.append(r)
        for coalition in extra_coalitions or []:
            if isinstance(coalition, _Coalition):
                coalitions.append(coalition)

        if not coalitions:
            empty = CompeteResult(winner=None, others=[], coalitions=[])
            self._last_compete_result = empty
            return empty

        winner = self._workspace.compete(coalitions)
        if winner is None and not cheap:
            logger.debug("GlobalWorkspace: nothing reached ignition threshold — activating DMN")
            # Default Mode Network: mind-wander when workspace is idle.
            dmn_coalition = await self._dmn.wander()
            if dmn_coalition is not None:
                winner = dmn_coalition
                coalitions.append(dmn_coalition)

        others = [c for c in coalitions if c is not winner]
        result = CompeteResult(winner=winner, others=others, coalitions=coalitions)
        # Consciousness-profile seam: one assignment covers both the turn path
        # (_gather_workspace_context) and the idle tick (_inner_loop_tick).
        self._last_compete_result = result
        return result

    def _meta_inconsistency_coalition(self):
        """Self-inconsistency as a workspace voice.

        MetaMonitor.detect_inconsistency existed but nothing consumed it —
        the check silently evaporated. When recent behavior contradicts the
        self-narrative, the mismatch now competes for attention with real
        urgency, so the agent can notice the contradiction.
        """
        meta = getattr(self, "_meta_monitor", None)
        narrative = getattr(self, "_self_narrative", None)
        if meta is None or narrative is None:
            return None
        try:
            mismatch = meta.detect_inconsistency(narrative)
        except Exception:  # noqa: BLE001
            return None
        # Strict str check: mocked monitors in tests return truthy non-strings.
        if not isinstance(mismatch, str) or not mismatch:
            return None
        from .workspace import Coalition as _Coalition

        return _Coalition(
            source="meta",
            summary="behavior/narrative mismatch",
            activation=0.5,
            urgency=0.5,
            novelty=0.6,
            context_block=f"[meta — inconsistency] {mismatch}",
        )

    async def _gather_workspace_context(
        self,
        desires: DesireSystem | None = None,
        extra_coalitions: list | None = None,
    ) -> str:
        """Run one Global Workspace competition cycle and return the broadcast context.

        Gathers coalitions from all available processors in parallel, runs the
        ignition competition, and returns the winning coalition's context_block
        plus a compact peripheral-awareness summary of non-winners.

        Returns empty string if nothing reaches ignition threshold.
        """
        result = await self._compete_once(
            cheap=False, desires=desires, extra_coalitions=extra_coalitions
        )
        if result.winner is None:
            return ""
        # Update attention schema with this turn's winner (AST).
        self._attention_schema.update_focus(result.winner)
        await self._workspace.notify_listeners(result.winner)
        return self._workspace.broadcast(result.winner, result.others)

    @staticmethod
    def _select_context_blocks(
        blocks: list[tuple[str, float]],
        max_chars: int = _MORNING_CONTEXT_MAX_CHARS,
    ) -> list[str]:
        """Select high-priority context blocks within a character budget."""
        if max_chars <= 0:
            return [text for text, _ in blocks]

        ranked = [
            (idx, text, score) for idx, (text, score) in enumerate(blocks) if text and text.strip()
        ]
        ranked.sort(key=lambda item: item[2], reverse=True)

        selected: list[tuple[int, str]] = []
        used = 0
        for idx, text, _score in ranked:
            block_len = len(text)
            sep = 2 if selected else 0
            if used + sep + block_len > max_chars:
                continue
            selected.append((idx, text))
            used += sep + block_len

        selected.sort(key=lambda item: item[0])
        return [text for _, text in selected]

    async def _infer_emotion(self, text: str) -> str:
        """Ask the LLM to label the emotion of a response. Returns label string."""
        label = await self._utility_backend.complete(
            _EMOTION_PROMPT.format(text=text[:400]), max_tokens=10
        )
        label = label.lower()
        valid = {
            "happy",
            "sad",
            "curious",
            "excited",
            "moved",
            "surprised",
            "nostalgic",
            "relieved",
            "tender",
            "playful",
            "proud",
            "neutral",
        }
        return label if label in valid else "neutral"

    # Emotion intensity by label (higher = stronger felt quality)
    _MOOD_INTENSITY: dict[str, float] = {
        "excited": 0.8,
        "moved": 0.8,
        "happy": 0.6,
        "curious": 0.6,
        "sad": 0.7,
        "surprised": 0.5,
        "nostalgic": 0.5,
        "relieved": 0.5,
        "tender": 0.7,
        "playful": 0.5,
        "proud": 0.6,
    }
    _SALIENT_NARRATIVE_EMOTIONS = {
        "excited",
        "moved",
        "tender",
        "nostalgic",
        "proud",
        "surprised",
    }

    def _update_mood(self, emotion: str) -> None:
        """Update persistent mood state from the latest inferred emotion.

        Neutral emotion is ignored (mood fades on its own via decay).
        Same emotion reinforces intensity; different strong emotion replaces.
        """
        if emotion == "neutral" or emotion not in self._MOOD_INTENSITY:
            return
        new_intensity = self._MOOD_INTENSITY[emotion]
        if emotion == self._mood:
            self._mood_intensity = min(1.0, self._mood_intensity + 0.1)
        else:
            self._mood = emotion
            self._mood_intensity = new_intensity
            self._mood_set_at = time.time()

    def _decayed_mood(self) -> tuple[str, float]:
        """Return (mood, intensity) after applying exponential decay.

        Half-life ≈ 138 seconds (~2.3 min).  Below 0.1 → treated as neutral.
        """
        if self._mood == "neutral" or self._mood_intensity <= 0.0:
            return ("neutral", 0.0)
        elapsed = time.time() - self._mood_set_at
        intensity = self._mood_intensity * math.exp(-0.005 * elapsed)
        if intensity < 0.1:
            return ("neutral", 0.0)
        return (self._mood, intensity)

    async def _proactive_memory_context(self) -> str | None:
        """Recall a contextually relevant past memory for spontaneous sharing.

        Returns a short hint string (the memory content) to prepend to a
        share_memory desire turn, or None if no suitable memory is found.
        Only memories older than 24 hours are surfaced to avoid repeating
        recent events.
        """
        from datetime import datetime, timedelta

        now = datetime.now()
        # Build a time-of-day context hint for the recall query
        hour = now.hour
        if 5 <= hour < 10:
            hint = f"morning {now.strftime('%B')}"
        elif 18 <= hour < 22:
            hint = f"evening {now.strftime('%B')}"
        else:
            hint = now.strftime("%B")

        try:
            memories = await self._memory.recall_async(hint, n=5)
        except Exception:
            return None

        if not memories:
            return None

        cutoff = now - timedelta(hours=24)
        old_enough = []
        for m in memories:
            created_at = m.get("created_at")
            if not created_at:
                continue
            try:
                ts = datetime.fromisoformat(created_at)
                if ts < cutoff:
                    old_enough.append(m)
            except (ValueError, TypeError):
                continue

        if not old_enough:
            return None

        # Pick the highest-scoring old memory
        best = max(old_enough, key=lambda m: m.get("score", 0.0))
        content = best.get("content", "")
        return content if content else None

    async def _anniversary_context(self) -> str | None:
        """Return a calendar-aware context string for today, or None if nothing notable.

        Surfaces "on this day" memories from past years and weekly/round milestones.
        Designed to be injected into morning reconstruction with high priority.
        """
        today = datetime.now().date()
        lines: list[str] = []

        # On-this-day memories (same month-day, past years)
        try:
            anniversaries = await self._memory.recall_on_this_day_async(today.month, today.day)
            for mem in anniversaries[:2]:
                content = mem.get("content", "")
                mem_date = mem.get("date", "")
                if content and mem_date:
                    lines.append(f"[On this day]: {content} ({mem_date})")
        except Exception:
            pass

        # Milestone: days since first memory
        try:
            earliest = await self._memory.get_earliest_date_async()
            if earliest:
                first_date = datetime.fromisoformat(earliest).date()
                days = (today - first_date).days
                if days >= 7:
                    # Fire on weekly boundaries and round numbers
                    if days % 7 == 0 or days in (30, 60, 90, 100, 180, 365):
                        lines.append(f"[Milestone]: {days} days since first memory.")
        except Exception:
            pass

        return "\n".join(lines) if lines else None

    async def _online_temporal_context(self, desires: DesireSystem | None = None) -> str | None:
        """Surface temporal-self fragments during ordinary turns.

        This keeps old memories, milestones, and unresolved threads available
        beyond startup reconstruction, but only when the current state suggests
        they matter.
        """
        if self._turn_count <= 1:
            return None

        share_memory_level = 0.0
        curiosity_target = None
        if desires is not None:
            try:
                share_memory_level = float(desires.level("share_memory"))
            except Exception:
                share_memory_level = 0.0
            curiosity_target = getattr(desires, "curiosity_target", None)

        tension = 0.0
        self_state = getattr(self, "_self_state", None)
        if self_state is not None:
            try:
                tension = float(self_state.snapshot().get("unresolved_tension", 0.0))
            except Exception:
                tension = 0.0

        should_surface_memory = share_memory_level >= 0.45 or tension >= 0.45
        should_surface_anniversary = self._turn_count % 4 == 0 or tension >= 0.6
        should_surface_thread = bool(curiosity_target) and tension >= 0.5

        if not (should_surface_memory or should_surface_anniversary or should_surface_thread):
            return None

        proactive_ctx: str | None = None
        anniversary_ctx: str | None = None
        if should_surface_memory and should_surface_anniversary:
            proactive_ctx, anniversary_ctx = await asyncio.gather(
                self._proactive_memory_context(),
                self._anniversary_context(),
            )
        elif should_surface_memory:
            proactive_ctx = await self._proactive_memory_context()
        elif should_surface_anniversary:
            anniversary_ctx = await self._anniversary_context()

        lines: list[str] = []
        if should_surface_thread:
            lines.append(f"[Unresolved thread]: {str(curiosity_target)[:160]}")
        if proactive_ctx:
            lines.append(f"[Resurfaced memory]: {proactive_ctx[:180]}")
        if anniversary_ctx:
            lines.append(anniversary_ctx)

        if not lines:
            return None
        return "[Temporal self]\n" + "\n".join(lines)

    async def _infer_companion_mood(self, text: str) -> str:
        """Classify companion's emotional state from their message. Returns mood label.

        When the utility backend is the same as the main backend (e.g. both are Kimi),
        uses a fast keyword heuristic to avoid an extra LLM round-trip every turn.
        Falls back to the LLM when a dedicated utility backend is configured.
        """
        if not text or len(text.strip()) < 3:
            return "absent"
        # Skip LLM call when utility == main backend (no dedicated cheap model available).
        if self._utility_backend is self.backend:
            return _companion_mood_heuristic(text)
        label = await self._utility_backend.complete(
            _COMPANION_MOOD_PROMPT.format(text=text[:300]), max_tokens=10
        )
        label = label.strip().lower()
        valid = {"engaged", "tired", "frustrated", "absent", "happy"}
        return label if label in valid else "engaged"

    async def _check_response_coherence(self, response: str) -> str | None:
        """Check whether the agent's response contains a logical error or rule violation.

        Uses the utility backend for a lightweight reflection pass.  Returns None if
        the response is coherent, or a short violation description if not.
        Skipped when no dedicated utility backend exists (same heuristic as TAPE).
        """
        if self._utility_backend is self.backend:
            return None
        if not response or response == "(no response)":
            return None

        # Build a compact context from the last few messages (user + assistant text only)
        context_parts: list[str] = []
        for msg in self.messages[-6:]:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                context_parts.append(f"{role}: {content[:200]}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        context_parts.append(f"{role}: {block['text'][:200]}")
                        break
        context = "\n".join(context_parts[-6:])

        try:
            result = await self._utility_backend.complete(
                _COHERENCE_CHECK_PROMPT.format(context=context, response=response[:300]),
                max_tokens=60,
            )
            result = result.strip()
            if result.upper().startswith("OK"):
                return None
            logger.info("Coherence check caught violation: %s", result)
            return result
        except Exception as e:
            logger.debug("Coherence check failed (non-critical): %s", e)
            return None

    async def _summarize_exchange(self, user_input: str, agent_response: str) -> str:
        """Distill an exchange into one sentence for memory storage."""
        result = await self._utility_backend.complete(
            _SUMMARY_PROMPT.format(
                lang=_t("summary_lang"),
                user=user_input[:200],
                agent=agent_response[:200],
            ),
            max_tokens=80,
        )
        return result or agent_response[:100]

    async def _morning_reconstruction(self, desires=None) -> str:
        """Build a 'yesterday → today' bridge from stored memories.

        Damasio's autobiographical self coming online: reading the past
        to know who we are now. Called only on the first turn of a session.
        """
        logger.info("Morning reconstruction started")
        (
            self_model,
            curiosities,
            feelings,
            day_summaries,
            semantic_facts,
            behavior_policies,
        ) = await asyncio.gather(
            self._memory.recall_self_model_async(n=5),
            self._memory.recall_curiosities_async(n=3),
            self._memory.recent_feelings_async(n=3),
            self._memory.recall_day_summaries_async(n=5),
            self._memory.recall_semantic_facts_async("", n=5),
            self._memory.recall_behavior_policies_async("", n=4),
        )
        logger.info(
            "Morning data: self_model=%d, curiosities=%d, feelings=%d, day_summaries=%d, "
            "semantic_facts=%d, behavior_policies=%d",
            len(self_model),
            len(curiosities),
            len(feelings),
            len(day_summaries),
            len(semantic_facts),
            len(behavior_policies),
        )

        # Generate day summaries for past dates that don't have one yet.
        # Run in background so it never delays the first-turn greeting response.
        asyncio.ensure_future(self._backfill_day_summaries())

        # Re-fetch if backfill created new summaries
        if not day_summaries:
            day_summaries = await self._memory.recall_day_summaries_async(n=5)

        # Surface the most recent curiosity into the desire system
        if desires is not None and curiosities and desires.curiosity_target is None:
            desires.curiosity_target = curiosities[0]["summary"]

        blocks: list[tuple[str, float]] = []
        if day_summaries:
            blocks.append((self._memory.format_day_summaries_for_context(day_summaries), 0.78))
        if semantic_facts:
            avg_conf = sum(float(x.get("confidence", 0.5)) for x in semantic_facts) / len(
                semantic_facts
            )
            blocks.append(
                (
                    self._memory.format_semantic_facts_for_context(semantic_facts),
                    0.86 + avg_conf * 0.1,
                )
            )
        if behavior_policies:
            avg_conf = sum(float(x.get("confidence", 0.5)) for x in behavior_policies) / len(
                behavior_policies
            )
            blocks.append(
                (
                    self._memory.format_behavior_policies_for_context(behavior_policies),
                    0.84 + avg_conf * 0.1,
                )
            )
        if self_model:
            blocks.append((self._memory.format_self_model_for_context(self_model), 0.83))
        if curiosities:
            blocks.append((self._memory.format_curiosities_for_context(curiosities), 0.74))
        if feelings:
            blocks.append((self._memory.format_feelings_for_context(feelings), 0.71))
        if getattr(self.config, "dream_mode", False):
            # The not-perception label is load-bearing: a dream must never be
            # citable as observation. Recency order — the block says "last
            # night's", so it must not surface an old dream by similarity.
            try:
                dreams = await self._memory.recall_recent_by_kind_async("dream", 2)
            except Exception:  # noqa: BLE001
                dreams = []
            if dreams:
                dream_lines = "\n".join(f"- {d.get('content', '')[:140]}" for d in dreams)
                blocks.append(
                    (
                        "[Last night's dreams — generated during sleep, not perception]\n"
                        + dream_lines,
                        0.60,
                    )
                )

        parts = self._select_context_blocks(blocks, _MORNING_CONTEXT_MAX_CHARS)

        # Prepend self-narrative: the felt sense of continuity from past sessions.
        # This is the thread that says "わたしはここにいた、今もいる."
        narrative_ctx = self._self_narrative.context_for_prompt()

        if not parts and not narrative_ctx:
            # No history yet — make it explicit so the agent doesn't fabricate a past
            return _t("morning_no_history")

        header = _t("morning_header")
        sections: list[str] = []
        if narrative_ctx:
            sections.append(narrative_ctx)
        sections.extend(parts)
        return header + "\n\n" + "\n\n".join(sections)

    async def _backfill_day_summaries(self) -> None:
        """Generate day summaries for past dates that don't have one yet.

        Skips today (summary is generated at shutdown). Only processes
        the most recent 5 days to keep startup time reasonable.

        Skipped when no separate utility backend is configured: the main
        conversation backend may not handle bulk observations well (e.g.
        Kimi K2.5 input-size limits), and we don't want to stall startup.
        """
        if self._utility_backend is self.backend:
            logger.debug("Backfill skipped: no separate utility backend configured")
            return
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            all_dates = await asyncio.to_thread(self._memory.get_dates_with_observations, 7)
            existing = await asyncio.to_thread(self._memory.get_dates_with_summaries)
            logger.info(
                "Backfill check: today=%s, all_dates=%s, existing=%s",
                today,
                all_dates,
                existing,
            )

            missing = [d for d in all_dates if d != today and d not in existing][:5]
            if missing:
                logger.info("Backfill: generating day summaries for %s", missing)
            else:
                logger.info("Backfill: no missing day summaries")
            for date in missing:
                await self._generate_day_summary(date)
        except Exception as e:
            logger.warning("Day summary backfill failed: %s", e)

    async def _generate_day_summary(self, date: str) -> None:
        """Generate and save a day summary for the given date."""
        try:
            observations = await asyncio.to_thread(self._memory.get_observations_for_date, date, 50)
            if not observations:
                logger.info("No observations for %s, skipping day summary", date)
                return

            # Build a concise transcript for the LLM — keep it short
            lines = []
            for obs in observations:
                emotion = f" [{obs['emotion']}]" if obs["emotion"] != "neutral" else ""
                lines.append(f"  {obs['time']} ({obs['kind']}){emotion}: {obs['content'][:150]}")
            transcript = "\n".join(lines)
            logger.info("Generating day summary for %s (%d observations)", date, len(observations))

            summary = await asyncio.wait_for(
                self._utility_backend.complete(
                    _DAY_SUMMARY_PROMPT.format(
                        lang=_t("summary_lang"),
                        observations=transcript,
                    ),
                    max_tokens=400,
                ),
                timeout=30.0,
            )
            if summary:
                await self._memory.save_async(
                    summary,
                    direction="記憶",
                    kind="day_summary",
                    emotion="neutral",
                    override_date=date,
                    dedupe_key=self._memory_dedupe_key(
                        "day_summary", summary[:200], scope="day", scope_id=date
                    ),
                    materialize_now=False,
                )
                logger.info("Day summary generated for %s: %s", date, summary[:80])
                # Phase 2-2: decay importance of older observations now that the day is summarised.
                # Run in background to avoid stalling the backfill loop.
                asyncio.ensure_future(
                    self._memory.decay_importance_async(before_date=date, factor=0.95)
                )
            else:
                logger.warning("Day summary for %s: LLM returned empty response", date)
        except asyncio.TimeoutError:
            logger.warning("Day summary for %s timed out (30s)", date)
        except Exception as e:
            logger.warning("Failed to generate day summary for %s: %s", date, e)

    async def _update_self_model(self, final_text: str, emotion: str) -> None:
        """Extract a self-insight and store it as self_model memory.

        Conway's working self: what this response reveals about who I am.
        Only runs when something actually moved us (non-neutral emotion).
        """
        if emotion == "neutral":
            return
        try:
            insight = await self._utility_backend.complete(
                _SELF_MODEL_PROMPT.format(text=final_text[:400]),
                max_tokens=80,
            )
            if insight and insight.lower() != "nothing":
                await self._memory.save_async(
                    insight,
                    direction="内省",
                    kind="self_model",
                    emotion=emotion,
                    dedupe_key=self._memory_dedupe_key("self_model", insight),
                    materialize_now=False,
                )
                logger.info("Self-model updated: %s", insight[:60])
        except Exception as e:
            logger.warning("Self-model update failed: %s", e)

    async def _maybe_update_self_narrative(
        self,
        *,
        user_input: str,
        final_text: str,
        emotion: str,
        is_desire_turn: bool,
    ) -> None:
        """Capture salient within-session self-narrative moments."""
        if not final_text or final_text == "(no response)":
            return

        pred_signal = self._prediction.last_signal()
        agency_error = float(pred_signal.agency_error) if pred_signal is not None else 0.0
        salient_emotion = emotion in self._SALIENT_NARRATIVE_EMOTIONS

        if not salient_emotion and agency_error < 0.55:
            return

        reason = "salient_turn" if salient_emotion else "agency_error"
        if salient_emotion and agency_error >= 0.55:
            reason = "salient_turn_agency"
        if is_desire_turn and not salient_emotion and agency_error < 0.7:
            return

        prompt = (
            "次の出来事を、わたし自身の自己叙述として一文で書いて。\n"
            f"user: {user_input[:160]}\n"
            f"agent: {final_text[:220]}\n"
            f"emotion: {emotion}\n"
            f"agency_error: {agency_error:.2f}\n"
            "条件: 一人称は『わたし』。60文字以内。説明や前置きは禁止。"
        )
        try:
            text = await asyncio.wait_for(
                self._utility_backend.complete(prompt, max_tokens=120),
                timeout=12.0,
            )
            if text and text.strip():
                mood = emotion if emotion != "neutral" else self._decayed_mood()[0]
                self._self_narrative.write(text.strip(), mood=mood, trigger=reason)
                logger.info("Self-narrative moment captured (%s): %s", reason, text.strip()[:60])
        except Exception as e:
            logger.warning("Could not update self narrative mid-session: %s", e)

    async def _maybe_update_identity(
        self,
        *,
        user_input: str,
        final_text: str,
        is_desire_turn: bool,
    ) -> None:
        """Background honor-check: did this reply honor the values it touched?

        Boundaries are enforced deterministically in the turn loop; this softer
        pass only adjusts *value* assertions' conviction with evidence, using
        one bounded utility call per implicated value. Runs only when a
        dedicated utility backend exists, off the hot path.
        """
        identity = getattr(self, "_identity", None)
        if identity is None or is_desire_turn or not user_input.strip():
            return
        if self._utility_backend is self.backend:
            return  # no dedicated utility model — skip rather than burn the main one
        try:
            values = identity.implicated_values(user_input)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Identity implicated_values failed: %s", exc)
            return
        for assertion in values[:2]:  # cap utility calls per turn
            try:
                verdict = await self._classify_identity_honor(assertion.statement, final_text)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Identity honor-check failed: %s", exc)
                continue
            if verdict == "honored":
                await self._memory.adjust_identity_confidence_async(
                    assertion.assertion_key, 0.02, reason="honored_in_response"
                )
                await self._memory.append_identity_evidence_async(
                    assertion.assertion_key, note="honored in a reply"
                )
            elif verdict == "strained":
                await self._memory.adjust_identity_confidence_async(
                    assertion.assertion_key, -0.04, reason="strained_in_response"
                )
                identity.nudge_dissonance(0.1)
            # "unclear" / anything else → no change

    async def _classify_identity_honor(self, statement: str, response: str) -> str | None:
        """One bounded utility call: did the reply honor the value? Strict labels."""
        raw = await self._utility_backend.complete(
            "Did this reply honor the speaker's stated value? Reply with exactly "
            "one word: honored, strained, or unclear.\n\n"
            f"Value: {statement[:200]}\nReply: {response[:300]}",
            max_tokens=8,
        )
        label = (raw or "").strip().strip('"').strip("'").lower()
        return label if label in ("honored", "strained", "unclear") else None

    async def _maybe_adapt_values(
        self,
        *,
        user_input: str,
        final_text: str,
        emotion: str,
        camera_used: bool,
        curiosity: str | None,
        is_desire_turn: bool,
        desires: DesireSystem | None,
    ) -> None:
        """Lightweight experience-driven updates for policy/value confidence."""
        updates = []

        pred_signal = self._prediction.last_signal()
        if camera_used and curiosity:
            updates.append(
                self._memory.adjust_behavior_policy_confidence_async(
                    "curiosity:active",
                    0.08,
                    reason="curiosity_satisfied",
                    policy_text=f"When idle, follow up this curiosity thread: {curiosity[:180]}",
                    trigger_context="idle",
                    action_hint="look_around",
                )
            )
            if desires is not None:
                desires.boost("share_memory", 0.08)

        if (
            pred_signal is not None
            and pred_signal.action_name in {"look", "walk", "see"}
            and pred_signal.agency_error >= 0.55
        ):
            updates.append(
                self._memory.adjust_behavior_policy_confidence_async(
                    "curiosity:active",
                    -0.05,
                    reason="agency_error_high",
                )
            )

        if not is_desire_turn and user_input and emotion in {"moved", "tender", "relieved"}:
            updates.append(
                self._memory.adjust_behavior_policy_confidence_async(
                    "conversation:supportive_style",
                    0.04,
                    reason="supportive_exchange",
                    policy_text=(
                        "Prefer this response style when supporting the companion: "
                        f"{final_text[:180]}"
                    ),
                    trigger_context="conversation",
                    action_hint="respond_supportively",
                )
            )

        if emotion in {"moved", "proud", "tender"}:
            updates.append(
                self._memory.adjust_semantic_fact_confidence_async(
                    "self_model:core",
                    0.03,
                    reason="salient_self_consistency",
                )
            )

        if not updates:
            return

        results = await asyncio.gather(*updates, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                logger.debug("Adaptive value update failed: %s", result)

    async def extract_curiosity(self, exploration_result: str) -> str | None:
        """Ask the LLM what was most curious/interesting in the exploration."""
        try:
            none_word = _t("curiosity_none")
            text = await self._utility_backend.complete(
                f"Read this exploration report and answer in one sentence what you found most "
                f"curious or interesting. Write in {_t('summary_lang')}. "
                f'If nothing caught your attention, reply with just "{none_word}". '
                f"No explanation.\n\n{exploration_result}",
                max_tokens=80,
            )
            text = text.strip()
            # Reject if the model returned the "none" word or a long non-curious explanation
            if not text or none_word in text or len(text) > 100:
                return None
            return text
        except Exception as e:
            logger.warning("Curiosity extraction failed: %s", e)
        return None

    async def extract_companion_thread(self, user_input: str) -> str | None:
        """Ask the LLM whether the companion mentioned something to follow up on.

        "I have a presentation tomorrow" should resurface later as "how did it
        go?" — the thread is an event or situation in THEIR life, not a request
        to the agent (that is the commitments domain).
        """
        if not user_input or not user_input.strip():
            return None
        try:
            none_word = _t("curiosity_none")
            text = await self._utility_backend.complete(
                "Read the companion's message. If it mentions a concrete upcoming "
                "event or an ongoing situation in THEIR life that a caring friend "
                "would ask about later (a presentation tomorrow, feeling unwell, "
                "a job interview, a trip), describe it in one short sentence in "
                f"{_t('summary_lang')}. Only their life events qualify — not "
                "requests to you, not questions, not small talk. If there is "
                f'nothing to follow up on, reply with just "{none_word}".'
                f"\n\nMessage: {user_input[:400]}",
                max_tokens=60,
            )
            text = text.strip()
            if not text or none_word in text or len(text) > 140:
                return None
            return text
        except Exception as e:
            logger.debug("Companion thread extraction failed: %s", e)
        return None

    async def _capture_companion_thread(
        self, user_input: str, desires: DesireSystem | None
    ) -> None:
        """Persist a follow-up-worthy thread as unfinished business.

        Source "companion_thread" reuses the existing surfacing + resolve loop:
        the hook renders open threads with a follow-up instruction and the
        model resolves them once the outcome is known.  Exact-duplicate
        summaries are skipped and at most 3 threads stay open at a time.
        """
        try:
            await self._memory.expire_stale_companion_threads_async(max_age_days=14.0)
            thread = await self.extract_companion_thread(user_input)
            if not thread:
                return
            open_items = await self._memory.list_unfinished_business_async(limit=20)
            if any(item.get("summary") == thread for item in open_items):
                return
            open_threads = [item for item in open_items if item.get("source") == "companion_thread"]
            if len(open_threads) >= 3:
                return
            await self._memory.open_unfinished_business_async(thread, source="companion_thread")
            if desires is not None:
                desires.boost("worry_companion", 0.1)
            logger.info("Companion thread captured: %s", thread)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Companion thread capture failed: %s", exc)

    def _should_compact(self, threshold_tokens: int = 60_000) -> bool:
        """Return True when context is large enough to warrant compaction.

        A threshold of 0 acts as a disabled sentinel — never compact.
        In normal use _last_context_tokens is 0 until after the first turn,
        so an empty conversation naturally returns False.
        """
        return threshold_tokens > 0 and self._last_context_tokens > threshold_tokens

    async def _compact_messages(self, keep_last: int = 6) -> None:
        """Summarise old messages and trim the history.

        Keeps the last `keep_last` messages verbatim, replaces the rest with a
        single summary marker, and sets `_post_compact = True` so the next
        `run()` call does a boosted memory recall to compensate.
        """
        if len(self.messages) <= keep_last:
            return

        logger.info(
            "Compacting conversation history at approximately %d input tokens",
            self._last_context_tokens,
        )

        # Find a safe cut boundary: a user-message dict at or after the nominal
        # cut point.  Cutting before a user message guarantees that no
        # tool_calls/tool-results pair is split across to_summarise / recent,
        # because a complete ReAct turn always ends with a final assistant
        # message (no tool_calls) immediately before the next user message.
        nominal_cut = len(self.messages) - keep_last
        cut = nominal_cut
        while cut < len(self.messages):
            msg = self.messages[cut]
            if isinstance(msg, dict) and msg.get("role") == "user":
                break
            cut += 1
        else:
            return  # No safe boundary found; skip compaction this turn

        to_summarise = self.messages[:cut]
        recent = self.messages[cut:]

        # Build a plain-text transcript for the summary LLM call.
        # self.messages stores both plain dicts and lists (tool-result batches);
        # skip list items rather than crashing on missing .get().
        lines = []
        for msg in to_summarise:
            if isinstance(msg, list):
                continue  # tool-result batches: not useful for summary text
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            lines.append(f"{role}: {content[:300]}")
        history_text = "\n".join(lines)

        summary = await self._utility_backend.complete(
            _COMPACT_PROMPT.format(history=history_text),
            max_tokens=200,
        )
        summary_marker = self.backend.make_user_message(
            f"[Conversation summary — earlier turns compacted]\n{summary}"
        )

        self.messages = [summary_marker] + list(recent)
        # Avoid immediately compacting the new summary again if this turn fails
        # before the backend can report the reduced context size.
        self._last_context_tokens = 0
        self._post_compact = True
        self._post_compact_recovery_pending = True

    @property
    def is_embedding_ready(self) -> bool:
        """Return True once the embedding model has finished loading."""
        return self._memory.is_embedding_ready()

    async def _write_today_narrative(self) -> None:
        """Write a one-sentence self-description for today's session.

        This is Haru's diary entry — "who I was today." Read back next session
        as the felt thread of temporal continuity: わたしはここにいた、今もいる.
        """
        if self._turn_count == 0:
            return  # No conversation happened — nothing to narrate
        try:
            today_memories = await self._memory.recall_day_summaries_async(n=1)
            if today_memories:
                summary_hint = today_memories[0].get("content", "")[:200]
            else:
                # Fall back to recent observations
                recent = await self._memory.recall_async("", n=5)
                summary_hint = " / ".join(m.get("content", "")[:60] for m in recent[:3])

            mood, _ = self._decayed_mood()
            prompt = (
                f"今日起きたこと（要約）:\n{summary_hint}\n\n"
                "わたし（ハル）として、今日という日を一文で書いて。"
                "一人称は「わたし」、50文字以内、過去形。"
                "感情や気づきを含めて。"
            )
            text = await asyncio.wait_for(
                self._utility_backend.complete(prompt, max_tokens=120),
                timeout=15.0,
            )
            if text and text.strip():
                self._self_narrative.write(text.strip(), mood=mood)
                logger.info("Self-narrative written: %s", text.strip()[:60])
        except Exception as e:
            logger.warning("Could not write today's self narrative: %s", e)

    def _close_cameras(self) -> None:
        """Close each configured camera once, including the default alias."""
        closed_cameras: set[int] = set()
        for camera in getattr(self, "_cameras", {}).values():
            if id(camera) not in closed_cameras:
                camera.close()
                closed_cameras.add(id(camera))
        default_camera = getattr(self, "_camera", None)
        if default_camera and id(default_camera) not in closed_cameras:
            default_camera.close()

    async def _refresh_day_summary_on_shutdown(self) -> None:
        """Refresh today's summary when a dedicated utility backend is available."""
        if self._utility_backend is self.backend:
            return
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            await asyncio.to_thread(self._memory.delete_day_summaries_for_date, today)
            await self._generate_day_summary(today)
        except Exception as exc:
            logger.warning("Failed to generate today's day summary on shutdown: %s", exc)

    @staticmethod
    async def _stop_async_resource(resource: Any, method_name: str, timeout: float) -> None:
        """Best-effort stop one async resource within a bounded timeout."""
        if resource is None:
            return
        try:
            stop = getattr(resource, method_name)
            await asyncio.wait_for(stop(), timeout=timeout)
        except (asyncio.TimeoutError, Exception):
            pass

    def _flush_deferred_state(self) -> None:
        """Persist state deferred by dense recurrence after producers stop."""
        for store_name in ("_self_state", "_attention_schema"):
            store = getattr(self, store_name, None)
            if store is not None and hasattr(store, "flush"):
                try:
                    store.flush()
                except Exception:  # noqa: BLE001
                    pass

    async def _close_memory(self) -> None:
        """Close the synchronous memory store without blocking the event loop."""
        try:
            await asyncio.wait_for(asyncio.to_thread(self._memory.close), timeout=1.0)
        except (asyncio.TimeoutError, Exception):
            pass

    def _close_sync_resources(self) -> None:
        """Best-effort close the remaining synchronous stores."""
        for closable in (
            getattr(self, "_person_model", None),
            getattr(self, "_commitment_store", None),
        ):
            if closable is not None:
                try:
                    closable.close()
                except Exception:
                    pass

    async def close(self) -> None:
        """Clean up resources in producer-to-store order with bounded waits."""
        self._close_cameras()

        await self._drain_background_tasks()

        # Write today's self-narrative before shutting down.
        await self._write_today_narrative()

        await self._refresh_day_summary_on_shutdown()
        await self._stop_async_resource(getattr(self, "_memory_worker", None), "stop", 1.5)
        await self._stop_async_resource(getattr(self, "_inner_loop", None), "stop", 1.5)

        # Dense recurrence defers disk writes — persist any backlog AFTER all
        # producers (background pipelines, inner-loop ticks) have stopped, or
        # a late nudge would re-dirty the stores past the flush and be lost.
        self._flush_deferred_state()

        await self._stop_async_resource(getattr(self, "_mcp", None), "stop", 2.0)
        await self._stop_async_resource(getattr(self, "_telegram_transport", None), "close", 2.0)
        await self._close_memory()
        await self._stop_async_resource(getattr(self, "_delegation_runner", None), "shutdown", 2.0)
        self._close_sync_resources()

    async def _stream_with_retry(
        self,
        *,
        system: str | tuple[str, str],
        messages: list,
        tools: list[dict],
        max_tokens: int,
        on_text: Callable[[str], None] | None,
        max_retries: int = 2,
    ) -> tuple[Any, Any]:
        """Call backend.stream_turn, retrying on rate-limit / engine-overload errors."""
        backoffs = [10.0, 20.0]
        for attempt in range(max_retries + 1):
            try:
                return await self.backend.stream_turn(
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    on_text=on_text,
                )
            except Exception as e:
                status = getattr(e, "status_code", None)
                is_rate_limit = (
                    status == 429
                    or "RateLimit" in type(e).__name__
                    or "overloaded" in str(e).lower()
                )
                if not is_rate_limit:
                    raise
                if attempt == max_retries:
                    raise RuntimeError(
                        "Engine temporarily overloaded — please try again in a moment."
                    ) from e
                wait = backoffs[min(attempt, len(backoffs) - 1)]
                logger.warning(
                    "Rate limit (attempt %d/%d), retrying in %.0fs: %s",
                    attempt + 1,
                    max_retries,
                    wait,
                    e,
                )
                await asyncio.sleep(wait)
        raise RuntimeError("unreachable")

    async def run(
        self,
        user_input: str | UserTurn,
        on_action: Callable[[str, dict], None] | None = None,
        on_text: Callable[[str], None] | None = None,
        on_image: Callable[[str], None] | None = None,
        on_phase: Callable[[str], None] | None = None,
        on_tool_result: Callable[[str, dict, str], None] | None = None,
        desires=None,
        inner_voice: str = "",
        interrupt_queue=None,
        excluded_tools: frozenset[str] | None = None,
        user_images: list[str] | None = None,
        user_id: str | None = None,
        turn_source: str = "local",
    ) -> str:
        """Submit one turn to the coordinator shared by every input channel."""
        request = TurnRequest(
            user_input=user_input,
            source=turn_source,
            user_id=user_id,
            on_action=on_action,
            on_text=on_text,
            on_image=on_image,
            on_phase=on_phase,
            on_tool_result=on_tool_result,
            desires=desires,
            inner_voice=inner_voice,
            interrupt_queue=interrupt_queue,
            excluded_tools=excluded_tools,
            user_images=user_images,
        )
        return await self._get_turn_coordinator().run(request)

    def _get_turn_coordinator(self) -> TurnCoordinator:
        """Return the coordinator, including for lightweight ``__new__`` tests."""
        coordinator = getattr(self, "_turn_coordinator", None)
        if coordinator is None:
            coordinator = self._turn_coordinator = TurnCoordinator(
                runner=self._run_request,
                switch_user=self.switch_user,
            )
        return coordinator

    @property
    def is_turn_busy(self) -> bool:
        return self._get_turn_coordinator().is_busy

    @property
    def active_turn_source(self) -> str | None:
        return self._get_turn_coordinator().active_source

    async def _run_request(self, request: TurnRequest) -> str:
        """Run one conversation turn with the agent loop.

        inner_voice: agent's own desire/impulse (injected into system prompt, NOT a user message).

        The deterministic pre-loop pipeline lives in
        ``EmbodiedAgentHook.prepare_turn``; the loop body is the substrate
        ``ReActLoop`` with the embodied behaviours (TAPE replan, coherence
        retry, interrupt drain, say reminders) supplied by the hook's
        lifecycle methods; finalisation (meta-gate repair, continuation
        status, auto-say, commit) stays here.
        """
        user_input = request.user_input
        on_action = request.on_action
        on_text = request.on_text
        on_image = request.on_image
        on_phase = request.on_phase
        on_tool_result = request.on_tool_result
        desires = request.desires
        inner_voice = request.inner_voice
        interrupt_queue = request.interrupt_queue
        excluded_tools = request.excluded_tools
        user_images = request.user_images
        user_turn = coerce_user_turn(user_input)
        if user_images:
            legacy_images = tuple(
                ImageAttachment.from_base64(image_data) for image_data in user_images
            )
            user_turn = UserTurn(
                text=user_turn.text,
                images=(*user_turn.images, *legacy_images),
                sent_at=user_turn.sent_at,
            )
        user_input_text = user_turn.text
        ctx = TurnContext(user_input=user_input_text, profile="neighbor")
        ctx.metadata["turn_source"] = request.source

        # Self-initiated turns need the current conversation as context, but
        # their synthetic placeholder, tool trace and reply must not become
        # the companion's conversation history. Run them on a shallow fork and
        # restore the original list on every exit path.
        main_messages: list[Any] | None = None
        if inner_voice and not user_input_text:
            main_messages = self.messages
            self.messages = list(self.messages)

        # Mark the turn in flight so the inner loop skips its idle ticks. The
        # public run() wrapper owns single-flight across presentation surfaces.
        self._turn_active = True
        latency = getattr(self, "_latency", None) or LatencyRecorder(enabled=False)
        try:
            with latency.span("prepare"):
                prep = await self._hook.prepare_turn(
                    user_input=user_input_text,
                    user_images=user_turn.images,
                    message_sent_at=user_turn.sent_at,
                    on_phase=on_phase,
                    desires=desires,
                    inner_voice=inner_voice,
                    excluded_tools=excluded_tools,
                )
        except BaseException:
            if main_messages is not None:
                self.messages = main_messages
            self._turn_active = False
            raise

        # Assigned again right after the loop finishes; this early value only
        # matters when the loop raises (the finally must never NameError).
        finalize_started = time.perf_counter()
        try:
            interrupt_source = (
                _InterruptQueueSource(self, interrupt_queue)
                if interrupt_queue is not None
                else None
            )
            ctx.metadata["prep"] = prep
            ctx.metadata["interrupt_source"] = interrupt_source

            # Timeouts go through _tool_timeout_seconds so the historical
            # per-tool seam (and its test patches) stays authoritative —
            # resolved for every tool on this turn's surface, not just the
            # static override table.
            turn_tool_names = {str(tool_def.get("name", "")) for tool_def in prep.turn_tools} | set(
                _TOOL_TIMEOUTS
            )
            loop = ReActLoop(
                backend=cast("RuntimeModelBackend", self.backend),
                tools=cast(ToolRegistry, _TurnToolAdapter(self, prep.turn_tools)),
                max_iterations=prep.turn_max_iterations,
                default_tool_timeout=self._tool_timeout_seconds(""),
                tool_timeouts={
                    name: self._tool_timeout_seconds(name) for name in turn_tool_names if name
                },
                non_repeatable_tools=_NON_REPEATABLE_ACTION_TOOLS,
                single_use_tools={"say"},
                hooks=[self._hook],
            )
            with latency.span("react_loop"):
                run_result = await loop.run(
                    system=self._system_prompt(
                        prep.feelings_ctx,
                        prep.morning_ctx,
                        inner_voice=prep.inner_voice,
                        plan_ctx=prep.plan_ctx,
                        companion_mood=prep.companion_mood,
                        continuity_ctx=prep.continuity_ctx,
                        workspace_ctx=prep.workspace_ctx,
                        mental_ctx=prep.mental_ctx,
                    ),
                    messages=self.messages,
                    max_tokens=prep.turn_max_tokens,
                    on_text=on_text,
                    context=ctx,
                    interrupt_source=interrupt_source,
                    on_action=on_action,
                    on_image=on_image,
                    on_tool_result=on_tool_result,
                )

            finalize_started = time.perf_counter()
            if run_result.stop_reason == "end_turn":
                final_text = run_result.final_text

                # Identity backstop (tier 2): the in-loop retry is the primary
                # defence; compute remaining violations once and let the
                # meta-gate replace a still-violating reply outright.
                identity = getattr(self, "_identity", None)
                identity_violations: list[Any] = []
                if identity is not None and final_text and final_text != "(no response)":
                    try:
                        identity_violations = identity.check_response(
                            user_text=user_input_text,
                            candidate_response=final_text,
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Identity response check failed: %s", exc)

                gate_method = getattr(self._meta_monitor, "gate_response", None)
                gate: MetaGateDecision | None = None
                if callable(gate_method):
                    maybe_gate = gate_method(
                        user_text=user_input_text,
                        candidate_response=final_text,
                        social_policy=prep.social_policy,
                        last_error=self._last_tool_error,
                        identity_violations=identity_violations or None,
                    )
                    if isinstance(maybe_gate, MetaGateDecision):
                        gate = maybe_gate
                if gate is not None and gate.needs_repair and gate.repaired_response:
                    final_text = gate.repaired_response

                # A violation — even a repaired one — leaves dissonance behind
                # and raises the drive to reflect on it later.
                if identity is not None and identity_violations:
                    top = identity_violations[0]
                    try:
                        identity.record_violation(top, turn_index=self._turn_count)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Identity violation record failed: %s", exc)
                    if desires is not None:
                        desires.boost("identity_coherence", 0.3 + 0.4 * top.severity)
                    concerns = getattr(self, "_concerns", None)
                    if concerns is not None:
                        try:
                            concerns.activate(
                                f"Something I hold was strained: {top.statement[:80]}",
                                category="identity",
                                intensity=0.4 + 0.5 * top.severity,
                                turn_index=self._turn_count,
                            )
                        except Exception:  # noqa: BLE001
                            pass

                continuation_status = "DONE"
                status_match = re.search(
                    r"(?:^|\n)(DONE|CONTINUE:[^\n]+|DEFER:[^\n]+)\s*$", final_text
                )
                if status_match:
                    continuation_status = status_match.group(1)
                    final_text = final_text[: status_match.start(1)].rstrip() or "(no response)"
                self._heartbeat.apply_status(continuation_status)

                self._coherence_retried = False

                # Auto-say: if the model wrote text but never called say(), speak it aloud.
                _auto_say_enabled = getattr(self.config, "auto_say", False)
                if (
                    _auto_say_enabled
                    and self._tts
                    and not prep.say_used
                    and final_text
                    and final_text != "(no response)"
                ):
                    if on_action:
                        on_action("say", {"text": final_text})
                    await self._tts.call("say", {"text": final_text})

                await self._hook.commit_after_end_turn(
                    prep=prep,
                    user_input=user_input_text,
                    final_text=final_text,
                    is_desire_turn=prep.is_desire_turn,
                    desires=desires,
                )

                return final_text

            # max_iterations (or an unexpected stop reason): force a final,
            # tool-free response so the turn always ends with words.
            logger.warning(
                "Reached max iterations (%d). Forcing final response.",
                prep.turn_max_iterations,
            )
            self.messages.append(
                self.backend.make_user_message(
                    "Please summarize what you found and provide your final answer now."
                )
            )
            result, _ = await self._stream_with_retry(
                system=self._system_prompt(
                    morning_ctx=prep.morning_ctx,
                    plan_ctx=prep.plan_ctx,
                    continuity_ctx=prep.continuity_ctx,
                    workspace_ctx=prep.workspace_ctx,
                    mental_ctx=prep.mental_ctx,
                ),
                messages=self.messages,
                tools=[],
                max_tokens=prep.turn_max_tokens,
                on_text=on_text,
            )
            return result.text or "(max iterations reached)"
        finally:
            self._restore_backend_after_turn(prep.backend_turn_snapshot)
            if main_messages is not None:
                self.messages = main_messages
            self._turn_active = False
            if latency.enabled:
                latency.record("finalize", time.perf_counter() - finalize_started)
                latency.flush_turn(turn=self._turn_count)

    @property
    def stt(self) -> STTTool | None:
        """Speech-to-text tool, or None if not configured."""
        return self._stt

    @property
    def telegram_history(self) -> TelegramHistory:
        """Recent inbound/outbound Telegram messages for the bot boundary."""
        return self._telegram_history

    def clear_history(self) -> None:
        """Clear conversation history (start fresh)."""
        self.messages = []

    async def clear_history_exclusive(self, *, source: str = "command") -> None:
        """Clear history after any in-flight channel turn has completed."""
        async with self._get_turn_coordinator().scope(source):
            self.clear_history()

    async def execute_desire_silent_action(self, desire_name: str) -> None:
        """Execute a SILENT_ACTION drive directly without generating an LLM turn.

        Called by DriveActionExecutor for look_around / explore / consolidate / reflect.
        Kept here so agent internals (camera, memory, worker) stay encapsulated.
        """
        match desire_name:
            case "look_around" | "explore":
                if self._camera is None:
                    return
                base64_jpeg, saved_path = await self._camera.capture()
                if base64_jpeg is None:
                    return
                note = f"[{desire_name}] camera capture"
                if saved_path:
                    note += f": {saved_path}"
                await self._memory.save_async(note, kind="observation", emotion="curious")

            case "consolidate":
                await self._memory_worker.run_once()

            case "reflect":
                prompt = (
                    "In one sentence, describe what you noticed or felt most recently. "
                    "Write in first person, past tense. Be specific and honest."
                )
                text = await self._utility_backend.complete(prompt, max_tokens=80)
                if text and text.strip().lower() != "nothing":
                    self._self_narrative.write(text.strip(), trigger="reflect_drive")
                    await self._memory.save_async(text.strip(), kind="feeling", emotion="neutral")
