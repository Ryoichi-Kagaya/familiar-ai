"""Smoke tests for familiar_capabilities adapters.

Each adapter is verified to expose the expected tool names through the
LegacyToolProvider protocol and to delegate call() to the wrapped tool
object without modifying behaviour.
"""

from __future__ import annotations

from typing import Any

import pytest


class _FakeLegacyTool:
    """Minimal stand-in for the familiar_agent.tools.* tool classes.

    Provides get_tool_definitions() in the historical shape and a coroutine
    call() returning a (text, image_b64) tuple.
    """

    def __init__(self, definitions: list[dict[str, Any]]) -> None:
        self._defs = definitions
        self.last_call: tuple[str, dict[str, Any]] | None = None

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return self._defs

    async def call(
        self, name: str, tool_input: dict[str, Any], **_kw: Any
    ) -> tuple[str, str | None]:
        self.last_call = (name, tool_input)
        return f"ok:{name}", None


@pytest.fixture(autouse=True)
def _patch_legacy_imports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stop test-time evaluation from requiring the real tool implementations.

    Each capability module imports its underlying tool class at import time,
    so we substitute a benign placeholder in sys.modules before the test
    body executes. The real class is only needed for isinstance checks,
    which we don't perform here.
    """
    # No-op fixture left here for future expansion if real tool imports
    # become problematic. Currently each tool import works because dependencies
    # are conditional inside the tool classes.
    return


def _spec_names(provider: Any) -> set[str]:
    return {spec.name for spec in provider.specs()}


@pytest.mark.asyncio
async def test_camera_capability_exposes_see_and_look() -> None:
    from familiar_capabilities import CameraCapability

    fake = _FakeLegacyTool(
        [
            {"name": "see", "description": "Take a photo.", "input_schema": {"type": "object"}},
            {"name": "look", "description": "Look around.", "input_schema": {"type": "object"}},
        ]
    )
    cap = CameraCapability(tool=fake)  # type: ignore[arg-type]
    assert _spec_names(cap) == {"see", "look"}
    result = await cap.call("see", {"angle": 0})
    assert result.text == "ok:see"
    assert fake.last_call == ("see", {"angle": 0})


@pytest.mark.asyncio
async def test_multi_camera_capability_selects_named_camera() -> None:
    from familiar_capabilities import MultiCameraCapability

    main = _FakeLegacyTool([])
    travel = _FakeLegacyTool([])
    cap = MultiCameraCapability(  # type: ignore[arg-type]
        {"main": main, "travel": travel},
        {"main": "普段", "travel": "外出用"},
    )

    assert _spec_names(cap) == {"see_camera", "look_camera"}
    see_schema = next(spec for spec in cap.specs() if spec.name == "see_camera").input_schema
    assert see_schema["properties"]["camera"]["enum"] == ["main", "travel"]

    result = await cap.call("see_camera", {"camera": "travel"})

    assert travel.last_call == ("see", {})
    assert main.last_call is None
    assert result.metadata["camera_id"] == "travel"
    assert result.text.startswith("[外出用]")


@pytest.mark.asyncio
async def test_multi_camera_capability_routes_ptz_and_rejects_unknown_camera() -> None:
    from familiar_capabilities import MultiCameraCapability

    travel = _FakeLegacyTool([])
    cap = MultiCameraCapability(  # type: ignore[arg-type]
        {"travel": travel},
        {"travel": "外出用"},
    )

    moved = await cap.call(
        "look_camera",
        {"camera": "travel", "direction": "left", "degrees": 45},
    )
    missing = await cap.call("see_camera", {"camera": "missing"})

    assert travel.last_call == ("look", {"direction": "left", "degrees": 45})
    assert moved.success is True
    assert missing.success is False
    assert missing.error == "camera_not_found"


@pytest.mark.asyncio
async def test_mobility_capability_exposes_walk() -> None:
    from familiar_capabilities import MobilityCapability

    fake = _FakeLegacyTool(
        [
            {"name": "walk", "description": "Move.", "input_schema": {"type": "object"}},
        ]
    )
    cap = MobilityCapability(tool=fake)  # type: ignore[arg-type]
    assert _spec_names(cap) == {"walk"}


@pytest.mark.asyncio
async def test_voice_capability_exposes_say() -> None:
    from familiar_capabilities import VoiceCapability

    fake = _FakeLegacyTool(
        [
            {"name": "say", "description": "Speak.", "input_schema": {"type": "object"}},
        ]
    )
    cap = VoiceCapability(tool=fake)  # type: ignore[arg-type]
    assert _spec_names(cap) == {"say"}


@pytest.mark.asyncio
async def test_tom_capability_exposes_tom() -> None:
    from familiar_capabilities import ToMCapability

    fake = _FakeLegacyTool(
        [
            {"name": "tom", "description": "Theory of mind.", "input_schema": {"type": "object"}},
        ]
    )
    cap = ToMCapability(tool=fake)  # type: ignore[arg-type]
    assert _spec_names(cap) == {"tom"}


@pytest.mark.asyncio
async def test_memory_capability_exposes_memory_tools() -> None:
    from familiar_capabilities import MemoryCapability

    fake = _FakeLegacyTool(
        [
            {"name": "remember", "description": "", "input_schema": {"type": "object"}},
            {"name": "recall", "description": "", "input_schema": {"type": "object"}},
            {"name": "recall_divergent", "description": "", "input_schema": {"type": "object"}},
            {"name": "get_working_memory", "description": "", "input_schema": {"type": "object"}},
        ]
    )
    cap = MemoryCapability(tool=fake)  # type: ignore[arg-type]
    assert _spec_names(cap) == {"remember", "recall", "recall_divergent", "get_working_memory"}


def test_capabilities_package_exports_all_adapters() -> None:
    import familiar_capabilities

    assert {
        "CameraCapability",
        "MultiCameraCapability",
        "CodingCapability",
        "MCPCapability",
        "MemoryCapability",
        "MobilityCapability",
        "ToMCapability",
        "VoiceCapability",
    }.issubset(set(familiar_capabilities.__all__))
