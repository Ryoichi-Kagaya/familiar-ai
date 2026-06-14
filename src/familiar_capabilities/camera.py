"""Camera capability adapter for the generic runtime."""

from __future__ import annotations

import inspect
from typing import Any

from familiar_agent.tools.camera import CameraTool
from familiar_runtime.tools.base import ToolExecutionResult
from familiar_runtime.tools.legacy import BeforeToolCall, LegacyToolProvider


class CameraCapability(LegacyToolProvider):
    """Expose the existing CameraTool through the ToolProvider protocol.

    The capability wraps an already-constructed CameraTool so the caller keeps
    control over hardware setup (Tapo/USB selection, RTSP URL, fallback wiring).
    An optional ``before_call`` hook lets neighbour mode record exploration
    state when ``look`` fires.  ``gui_priority=True`` makes PTZ moves acquire
    the shared _PriorityPTZLock with GUI priority so concurrent Telegram calls
    yield to the GUI on exact ties.
    """

    def __init__(
        self,
        tool: CameraTool,
        *,
        before_call: BeforeToolCall | None = None,
        gui_priority: bool = False,
    ) -> None:
        super().__init__(
            tool,
            names={"see", "look"},
            category="camera",
            tags={"neighbor", "perception"},
            before_call=before_call,
        )
        self._gui_priority = gui_priority

    async def call(self, name: str, tool_input: dict[str, Any]) -> ToolExecutionResult:
        if self._before_call is not None:
            maybe = self._before_call(name, tool_input)
            if inspect.isawaitable(maybe):
                await maybe
        text, image = await self._tool.call(name, tool_input, gui=self._gui_priority)
        return ToolExecutionResult(text=str(text), image_b64=image)
