"""Synthetic pages drawn at test time. Nothing here is from a published comic."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw


def solid(size: tuple[int, int], color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, color)


def jpeg_bytes(color: tuple[int, int, int] = (20, 40, 60), size: tuple[int, int] = (48, 64)) -> bytes:
    buffer = io.BytesIO()
    solid(size, color).save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def grid_page(width: int = 900, height: int = 1200) -> Image.Image:
    """Four bordered panels, two rows, with a white gutter wide enough for the cv detector."""
    image = solid((width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    margin, gutter = 40, 48
    panel_w = (width - 2 * margin - gutter) // 2
    panel_h = (height - 2 * margin - gutter) // 2
    fills = ((210, 210, 210), (190, 190, 190), (170, 170, 170), (150, 150, 150))
    origins = (
        (margin, margin),
        (margin + panel_w + gutter, margin),
        (margin, margin + panel_h + gutter),
        (margin + panel_w + gutter, margin + panel_h + gutter),
    )
    for origin, fill in zip(origins, fills, strict=True):
        x, y = origin
        draw.rectangle((x, y, x + panel_w, y + panel_h), fill=fill, outline=(0, 0, 0), width=10)
    return image


def save_jpeg(path: Path, image: Image.Image) -> None:
    image.save(path, format="JPEG", quality=90)
