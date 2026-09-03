"""Normalize anomalous model responses before display and persistence."""

from __future__ import annotations

import re

_EXACT_REPETITION_RE = re.compile(r"(.{20,}?)(?:\s*\1){1,3}", re.DOTALL)


def collapse_exact_repetition(text: str) -> str:
    """Collapse two to four exact response copies separated by whitespace."""
    stripped = text.strip()
    if not stripped:
        return stripped
    match = _EXACT_REPETITION_RE.fullmatch(stripped)
    return match.group(1).strip() if match else stripped
