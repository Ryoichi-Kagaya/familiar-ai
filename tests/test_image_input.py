from __future__ import annotations

import io

import pytest
from PIL import Image

from familiar_agent.image_input import (
    MAX_IMAGE_DIMENSION,
    ImageInputError,
    normalize_image_bytes,
    parse_image_command,
    validate_image_count,
)


def _image_bytes(format_name: str, *, size: tuple[int, int] = (32, 24)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (40, 90, 140)).save(output, format=format_name)
    return output.getvalue()


@pytest.mark.parametrize(
    ("format_name", "media_type"),
    [("JPEG", "image/jpeg"), ("PNG", "image/png"), ("WEBP", "image/webp")],
)
def test_normalize_image_bytes_detects_real_format(format_name: str, media_type: str) -> None:
    attachment = normalize_image_bytes(_image_bytes(format_name), filename="spoofed.txt")

    assert attachment.media_type == media_type
    assert attachment.filename == "spoofed.txt"
    assert attachment.data


def test_normalize_image_bytes_resizes_large_image() -> None:
    attachment = normalize_image_bytes(_image_bytes("JPEG", size=(MAX_IMAGE_DIMENSION + 200, 100)))

    with Image.open(io.BytesIO(attachment.data)) as normalized:
        assert max(normalized.size) == MAX_IMAGE_DIMENSION


def test_normalize_image_bytes_rejects_non_image() -> None:
    with pytest.raises(ImageInputError, match="not a valid image"):
        normalize_image_bytes(b"hello")


def test_validate_image_count_rejects_too_many() -> None:
    with pytest.raises(ImageInputError, match="up to 3"):
        validate_image_count(4)


def test_parse_image_command_builds_turn(tmp_path) -> None:
    path = tmp_path / "sample image.png"
    path.write_bytes(_image_bytes("PNG"))

    turn = parse_image_command(f'/image "{path}" describe this')

    assert turn is not None
    assert turn.text == "describe this"
    assert turn.images[0].media_type == "image/png"


def test_parse_image_command_uses_default_caption(tmp_path) -> None:
    path = tmp_path / "sample.png"
    path.write_bytes(_image_bytes("PNG"))

    turn = parse_image_command(f"/image {path}")

    assert turn is not None
    assert turn.text == "この画像を見て。"
