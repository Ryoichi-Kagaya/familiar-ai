"""Provider-neutral user input content.

Front-ends should hand the runtime a :class:`UserTurn` instead of encoding
provider-specific image blocks themselves.  Model adapters remain responsible
for the final wire representation.
"""

from __future__ import annotations

import base64
import copy
from dataclasses import dataclass, field
from datetime import datetime


_WEEKDAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


@dataclass(frozen=True, slots=True)
class ImageAttachment:
    """Validated image bytes attached to a user turn."""

    data: bytes
    media_type: str
    filename: str | None = None

    @property
    def base64_data(self) -> str:
        """Return the image bytes encoded for JSON-based model transports."""

        return base64.b64encode(self.data).decode("ascii")

    @classmethod
    def from_base64(
        cls,
        data: str,
        *,
        media_type: str = "image/jpeg",
        filename: str | None = None,
    ) -> "ImageAttachment":
        """Build an attachment from a legacy base64 payload."""

        return cls(
            data=base64.b64decode(data, validate=True),
            media_type=media_type,
            filename=filename,
        )


@dataclass(frozen=True, slots=True)
class UserTurn:
    """One atomic user input, including any image attachments."""

    text: str
    images: tuple[ImageAttachment, ...] = ()
    sent_at: datetime = field(default_factory=lambda: datetime.now().astimezone())

    def with_text(self, text: str) -> "UserTurn":
        """Return the same attachments paired with context-enriched text."""

        return UserTurn(text=text, images=self.images, sent_at=self.sent_at)


def coerce_user_turn(value: str | UserTurn) -> UserTurn:
    """Normalize legacy text-only inputs at runtime boundaries."""

    return value if isinstance(value, UserTurn) else UserTurn(text=value)


def format_user_message_text(
    text: str,
    *,
    sent_at: datetime | float | None = None,
) -> str:
    """Attach trusted local send-time context to text passed to the model.

    ``sent_at`` should be captured when the runtime accepts the turn, before
    slower prompt preparation. A timezone-aware ISO timestamp and an explicit
    weekday keep relative-date phrases such as "tomorrow" grounded without
    asking the model to calculate the weekday itself.
    """
    if isinstance(sent_at, datetime):
        moment = sent_at if sent_at.tzinfo is not None else sent_at.astimezone()
    elif sent_at is not None:
        moment = datetime.fromtimestamp(sent_at).astimezone()
    else:
        moment = datetime.now().astimezone()
    timestamp = moment.isoformat(timespec="seconds")
    weekday = _WEEKDAY_NAMES[moment.weekday()]
    metadata = f"[familiar-ai message metadata: sent_at={timestamp}; weekday={weekday}]"
    return f"{metadata}\n{text}"


def compact_image_blocks(messages: list, keep_last: int = 3) -> list:
    """Clear old provider image blocks while preserving surrounding text.

    The traversal recognizes Anthropic/Claude-Code, OpenAI-compatible, and
    Gemini message shapes.  It operates on a deep copy only when an image
    actually needs clearing, so callers keep the historical no-op behavior.
    """

    positions: list[tuple[list, int, dict]] = []

    def visit(value: object) -> None:
        if isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    block_type = item.get("type")
                    source = item.get("source")
                    image_url = item.get("image_url")
                    inline_data = item.get("inline_data")
                    if block_type == "image" and isinstance(source, dict):
                        positions.append(
                            (value, index, {"type": "text", "text": "[image cleared]"})
                        )
                        continue
                    if block_type == "image_url" and isinstance(image_url, dict):
                        positions.append(
                            (value, index, {"type": "text", "text": "[image cleared]"})
                        )
                        continue
                    if isinstance(inline_data, dict) and str(
                        inline_data.get("mime_type", "")
                    ).startswith("image/"):
                        positions.append((value, index, {"text": "[image cleared]"}))
                        continue
                visit(item)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)

    visit(messages)
    clear_count = max(0, len(positions) - keep_last)
    if clear_count == 0:
        return messages

    compacted = copy.deepcopy(messages)
    positions = []
    visit(compacted)
    for parent, index, replacement in positions[:clear_count]:
        parent[index] = replacement
    return compacted
