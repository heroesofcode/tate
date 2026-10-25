"""Classical panel detector. No model, and no text boxes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from tate.boxes import Box, Detection, box_distance, union_box


class CvDetector:
    @property
    def name(self) -> str:
        return "cv"

    def detect(self, image: Image.Image) -> Detection:
        return Detection(tuple(_panels(image)), ())


def _panels(image: Image.Image) -> list[Box]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cast(np.ndarray[Any, Any], cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    height, width = int(gray.shape[0]), int(gray.shape[1])
    if height < 2 or width < 2:
        return []
    _level, white_any = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY)
    white = cast(np.ndarray[Any, Any], white_any)
    content = cast(np.ndarray[Any, Any], cv2.bitwise_not(white))
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
        boxes.append(Box(x, y, bw, bh))
    return _merge_fragments(boxes, width, height)


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
    brings back a panel the detector skipped.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if height < 2 or width < 2 or not panels:
        return []
    step = max(1, int(max(height, width) / 600))
    small = rgb[::step, ::step]
    tone = small.astype(np.uint16).sum(axis=2) // 3
    ink = (tone < 235).astype(np.uint8) * 255
    mask = np.zeros(ink.shape, dtype=np.uint8)
    for panel in panels:
        x1 = max(0, min(int(panel.x / step), mask.shape[1]))
        x2 = max(0, min(int(np.ceil(panel.x2 / step)), mask.shape[1]))
        y1 = max(0, min(int(panel.y / step), mask.shape[0]))
        y2 = max(0, min(int(np.ceil(panel.y2 / step)), mask.shape[0]))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    radius = max(2, int(0.015 * min(mask.shape)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    covered = cast(np.ndarray[Any, Any], cv2.dilate(mask, kernel))
    uncovered = cast(np.ndarray[Any, Any], cv2.bitwise_and(ink, cv2.bitwise_not(covered)))
    close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    uncovered = cast(np.ndarray[Any, Any], cv2.morphologyEx(uncovered, cv2.MORPH_CLOSE, close))
    found: Any = cv2.findContours(uncovered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    extra: list[Box] = []
    minimum = 0.06 * height * width
    for contour in found[0]:
        rect: Any = cv2.boundingRect(contour)
        x, y, bw, bh = int(rect[0]) * step, int(rect[1]) * step, int(rect[2]) * step, int(rect[3]) * step
        if bw * bh < minimum or bw < width * 0.12 or bh < height * 0.08:
            continue
        extra.append(Box(x, y, bw, bh, 0.5))
    return extra
