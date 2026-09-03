"""Camera capability adapter for the generic runtime."""

from __future__ import annotations

import inspect
from typing import Any

from familiar_agent.tools.camera import CameraTool
from familiar_runtime.tools.base import ToolExecutionResult, ToolSpec
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


class MultiCameraCapability:
    """Expose stable selector-based tools for any number of named cameras."""

    def __init__(
        self,
        cameras: dict[str, CameraTool],
        labels: dict[str, str],
        *,
        before_call: BeforeToolCall | None = None,
        gui_priority: bool = False,
    ) -> None:
        self._cameras = cameras
        self._labels = labels
        self._before_call = before_call
        self._gui_priority = gui_priority

    def specs(self) -> list[ToolSpec]:
        camera_ids = sorted(self._cameras)
        choices = ", ".join(
            f"{camera_id} ({self._labels.get(camera_id, camera_id)})" for camera_id in camera_ids
        )
        camera_property = {"type": "string", "enum": camera_ids}
        return [
            ToolSpec(
                name="see_camera",
                description=f"See through a named camera. Available cameras: {choices}.",
                input_schema={
                    "type": "object",
                    "properties": {"camera": camera_property},
                    "required": ["camera"],
                },
                category="camera",
                tags={"neighbor", "perception"},
            ),
            ToolSpec(
                name="look_camera",
                description=f"Turn a named camera. Available cameras: {choices}.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "camera": camera_property,
                        "direction": {
                            "type": "string",
                            "enum": ["left", "right", "up", "down"],
                        },
                        "degrees": {"type": "integer", "default": 30},
                    },
                    "required": ["camera", "direction"],
                },
                category="camera",
                tags={"neighbor", "perception"},
            ),
        ]

    async def call(self, name: str, tool_input: dict[str, Any]) -> ToolExecutionResult:
        camera_id = str(tool_input.get("camera", ""))
        camera = self._cameras.get(camera_id)
        if camera is None:
            return ToolExecutionResult(
                text=f"Unknown camera: {camera_id}",
                success=False,
                error="camera_not_found",
            )
        if self._before_call is not None:
            maybe = self._before_call(name, tool_input)
            if inspect.isawaitable(maybe):
                await maybe
        internal_name = "see" if name == "see_camera" else "look"
        internal_input = {key: value for key, value in tool_input.items() if key != "camera"}
        text, image = await camera.call(
            internal_name,
            internal_input,
            gui=self._gui_priority,
        )
        label = self._labels.get(camera_id, camera_id)
        success = (
            bool(image) if internal_name == "see" else not str(text).lower().startswith("camera")
        )
        return ToolExecutionResult(
            text=f"[{label}] {text}",
            image_b64=image[0] if image else None,
            success=success,
            error=None if success else "camera_unavailable",
            metadata={"camera_id": camera_id, "camera_label": label},
        )
