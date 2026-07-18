"""Shared validation and normalization for user-supplied images."""

from __future__ import annotations

import io
import os
import shlex
from pathlib import Path

from familiar_runtime.models import ImageAttachment, UserTurn
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_DIMENSION = 2048
MAX_IMAGES_PER_TURN = 3

_FORMAT_TO_MEDIA_TYPE = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
_MEDIA_TYPE_TO_FORMAT = {value: key for key, value in _FORMAT_TO_MEDIA_TYPE.items()}
_MEDIA_TYPE_TO_SUFFIX = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}


class ImageInputError(ValueError):
    """Raised when a user-supplied image cannot be accepted safely."""


def image_suffix(media_type: str) -> str:
    """Return a safe file suffix for a supported image MIME type."""

    try:
        return _MEDIA_TYPE_TO_SUFFIX[media_type]
    except KeyError as exc:
        raise ImageInputError(f"Unsupported image type: {media_type}") from exc


def normalize_image_bytes(data: bytes, *, filename: str | None = None) -> ImageAttachment:
    """Validate, orient, resize, and re-encode image bytes.

    Re-encoding prevents extension spoofing and strips unrelated metadata before
    the image is sent to a model provider or written to a temporary transport.
    """

    if not data:
        raise ImageInputError("The selected image is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise ImageInputError("The selected image is larger than 20 MB.")

    try:
        with Image.open(io.BytesIO(data)) as opened:
            detected_format = (opened.format or "").upper()
            if detected_format not in _FORMAT_TO_MEDIA_TYPE:
                raise ImageInputError("Unsupported image format. Use JPEG, PNG, or WebP.")
            image = ImageOps.exif_transpose(opened)
            image.thumbnail(
                (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
                Image.Resampling.LANCZOS,
            )
            output = io.BytesIO()
            if detected_format == "JPEG":
                if image.mode not in {"RGB", "L"}:
                    image = image.convert("RGB")
                image.save(output, format="JPEG", quality=88, optimize=True)
            elif detected_format == "PNG":
                image.save(output, format="PNG", optimize=True)
            else:
                if image.mode not in {"RGB", "RGBA"}:
                    image = image.convert("RGBA" if "transparency" in image.info else "RGB")
                image.save(output, format="WEBP", quality=88, method=4)
    except ImageInputError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise ImageInputError("The selected file is not a valid image.") from exc

    normalized = output.getvalue()
    if len(normalized) > MAX_IMAGE_BYTES:
        raise ImageInputError("The normalized image is larger than 20 MB.")
    return ImageAttachment(
        data=normalized,
        media_type=_FORMAT_TO_MEDIA_TYPE[detected_format],
        filename=Path(filename).name if filename else None,
    )


def load_image_attachment(path: str | Path) -> ImageAttachment:
    """Load and normalize an image selected by a local front-end."""

    resolved = Path(path).expanduser()
    try:
        data = resolved.read_bytes()
    except OSError as exc:
        raise ImageInputError(f"Could not read image: {resolved}") from exc
    return normalize_image_bytes(data, filename=resolved.name)


def validate_image_count(count: int) -> None:
    """Reject turns that exceed the shared attachment-count limit."""

    if count > MAX_IMAGES_PER_TURN:
        raise ImageInputError(f"You can attach up to {MAX_IMAGES_PER_TURN} images at once.")


def parse_image_command(text: str) -> UserTurn | None:
    """Parse ``/image <path> [caption]`` for plain REPL and Textual UIs."""

    if text != "/image" and not text.startswith("/image "):
        return None
    try:
        parts = shlex.split(text, posix=os.name != "nt")
    except ValueError as exc:
        raise ImageInputError(f"Invalid /image command: {exc}") from exc
    if len(parts) < 2:
        raise ImageInputError('Usage: /image "<path>" [caption]')
    selected_path = parts[1].strip('"')
    caption = " ".join(parts[2:]).strip() or "この画像を見て。"
    return UserTurn(
        text=caption,
        images=(load_image_attachment(selected_path),),
    )
