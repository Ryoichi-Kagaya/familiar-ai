"""Art critique capability adapter for the generic runtime."""

from __future__ import annotations

from familiar_agent.tools.art_critique import ArtCritiqueTool
from familiar_runtime.tools.legacy import LegacyToolProvider

ART_CRITIQUE_TOOL_NAMES = {"store_art_critique", "recall_art_critiques", "compare_artworks"}


class ArtCritiqueCapability(LegacyToolProvider):
    """Expose ArtCritiqueTool through the ToolProvider protocol."""

    def __init__(self, tool: ArtCritiqueTool) -> None:
        super().__init__(
            tool,
            names=set(ART_CRITIQUE_TOOL_NAMES),
            category="art_critique",
            tags={"art", "critique", "memory"},
        )
