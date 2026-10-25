"""Draw panel boxes, text boxes, and reading-order numbers on a page."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from tate.boxes import Box
from tate.layout import PageLayout


def draw_debug(image: Image.Image, layout: PageLayout) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    line = max(4, min(canvas.size) // 350)
    font = ImageFont.load_default(size=max(28, min(canvas.size) // 32))
    for text in layout.texts:
        draw.rectangle(_corners(text), outline=(30, 90, 255), width=max(2, line // 2))
    color = (40, 160, 70) if not layout.fallback and not layout.spread else (220, 120, 20)
    for number, box in enumerate(layout.detected, start=1):
        draw.rectangle(_corners(box), outline=color, width=line)
        draw.text(
            (box.x + line, box.y + line),
            str(number),
            fill=color,
            font=font,
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
    if layout.spread:
        banner = "spread"
    elif layout.fallback:
        banner = "full page"
    else:
        banner = ""
    if banner:
        draw.text(
            (line, line),
            banner,
            fill=(190, 20, 20),
            font=font,
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
    return canvas


def _corners(box: Box) -> list[int]:
    return [box.x, box.y, max(box.x, box.x2 - 1), max(box.y, box.y2 - 1)]
