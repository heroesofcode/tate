"""Classical panel detector. No model, and no text boxes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from tate.boxes import Box, Detection, box_distance, union_box
from tate.detect.frames import background_color, frame_panels, fuse_panels
from tate.ink import recover_regions


class CvDetector:
    @property
    def name(self) -> str:
        return "cv"

    def detect(self, image: Image.Image) -> Detection:
        gutters = gutter_panels(image)
        frames = frame_panels(image)
        fused = fuse_panels(gutters, frames, image.width, image.height)
        return Detection(tuple(fused), ())


def gutter_panels(image: Image.Image) -> list[Box]:
    """Content regions separated by a gutter the colour of this page's background."""
    return _panels(image)


def _panels(image: Image.Image) -> list[Box]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if height < 2 or width < 2:
        return []
    content = _content_mask(rgb)
    gap = max(3, round(min(height, width) * 0.01))
    if gap % 2 == 0:
        gap += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (gap, gap))
    closed = cast(np.ndarray[Any, Any], cv2.morphologyEx(content, cv2.MORPH_CLOSE, kernel))
    found: Any = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours: Any = found[0]
    minimum = 0.012 * height * width
    boxes: list[Box] = []
    for contour in contours:
        rect: Any = cv2.boundingRect(contour)
        x, y, bw, bh = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
        if bw * bh < minimum:
            continue
        if bw < width * 0.05 or bh < height * 0.025:
            continue
        boxes.append(Box(x, y, bw, bh, 0.5))
    return _merge_fragments(boxes, width, height)


def _content_mask(rgb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Ink and paint, measured against the page background rather than pure white.

    A dark margin (a black webtoon gutter) is the background, so the art is
    whatever sits away from that colour. Yellowed paper works the same way:
    the gutter matches the sheet, and the panel does not.
    """
    background = background_color(rgb)
    diff = rgb.astype(np.float32) - background
    distance = np.sqrt(np.square(diff).sum(axis=2))
    luminance = float(0.299 * background[0] + 0.587 * background[1] + 0.114 * background[2])
    tau = 22.0 if luminance < 85.0 else 16.0
    mask = (distance > tau).astype(np.uint8) * 255
    return cast(np.ndarray[Any, Any], mask)


def _merge_fragments(boxes: list[Box], page_w: int, page_h: int) -> list[Box]:
    """Absorb specks into a nearby panel and drop the ones left on their own."""
    if not boxes:
        return []
    page_area = page_w * page_h
    tiny_area = int(0.015 * page_area)
    gap = max(8, int(0.015 * min(page_w, page_h)))
    kept = [box for box in boxes if box.area >= tiny_area]
    for speck in boxes:
        if speck.area >= tiny_area or not kept:
            continue
        nearest = min(range(len(kept)), key=lambda index: box_distance(kept[index], speck))
        if box_distance(kept[nearest], speck) <= gap:
            kept[nearest] = union_box(kept[nearest], speck).clamp(page_w, page_h)
    return kept


def recover_uncovered(image: Image.Image, panels: Sequence[Box]) -> list[Box]:
    """Bounding boxes of large ink regions that no panel covers.

    A tight halo around the panels already found is ignored, so this only
    brings back a panel the detector skipped. Paper tone and the page edge
    are not ink.
    """
    return recover_regions(image, panels)
