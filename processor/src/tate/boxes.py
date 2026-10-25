"""Panel rectangles and the geometry used to clean them up."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Box:
    """Axis-aligned rectangle in source-page pixels. `x2` and `y2` are exclusive."""

    x: int
    y: int
    w: int
    h: int
    score: float = 1.0

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def clamp(self, width: int, height: int) -> Box:
        x1 = min(max(self.x, 0), width)
        y1 = min(max(self.y, 0), height)
        x2 = min(max(self.x2, 0), width)
        y2 = min(max(self.y2, 0), height)
        return Box(x1, y1, max(0, x2 - x1), max(0, y2 - y1), self.score)


@dataclass(frozen=True, slots=True)
class Detection:
    """Panels and speech-balloon or text boxes for one page."""

    panels: tuple[Box, ...]
    texts: tuple[Box, ...]


def intersection_area(a: Box, b: Box) -> int:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    if union <= 0:
        return 0.0
    return inter / union


def contained_ratio(inner: Box, outer: Box) -> float:
    """Share of `inner` that lies inside `outer`."""
    if inner.area <= 0:
        return 0.0
    return intersection_area(inner, outer) / inner.area


def box_distance(a: Box, b: Box) -> float:
    dx = max(a.x - b.x2, b.x - a.x2, 0)
    dy = max(a.y - b.y2, b.y - a.y2, 0)
    return math.hypot(dx, dy)


def union_box(a: Box, b: Box) -> Box:
    x1 = min(a.x, b.x)
    y1 = min(a.y, b.y)
    x2 = max(a.x2, b.x2)
    y2 = max(a.y2, b.y2)
    return Box(x1, y1, x2 - x1, y2 - y1, a.score)


def clean_panels(panels: Sequence[Box], page_w: int, page_h: int) -> list[Box]:
    """Drop tiny, duplicate, nested, and full-page wrapper boxes."""
    page_area = max(1, page_w * page_h)
    minimum = max(64, int(page_area * 0.002))
    candidates: list[Box] = []
    for panel in panels:
        box = panel.clamp(page_w, page_h)
        if box.w < 8 or box.h < 8 or box.area < minimum:
            continue
        candidates.append(box)

    ranked = sorted(candidates, key=lambda box: (box.score, box.area), reverse=True)
    kept: list[Box] = []
    for box in ranked:
        if any(iou(box, other) > 0.55 for other in kept):
            continue
        kept.append(box)

    wrappers: set[int] = set()
    for index, box in enumerate(kept):
        inner_count = 0
        for other_index, other in enumerate(kept):
            if other_index == index or other.area >= box.area:
                continue
            if contained_ratio(other, box) > 0.8:
                inner_count += 1
        if inner_count >= 2:
            wrappers.add(index)
    unwrapped = [box for index, box in enumerate(kept) if index not in wrappers]

    final: list[Box] = []
    for index, box in enumerate(unwrapped):
        nested = False
        for other_index, other in enumerate(unwrapped):
            if other_index == index:
                continue
            if box.area < other.area and contained_ratio(box, other) > 0.75:
                nested = True
                break
        if not nested:
            final.append(box)
    return final


def drop_slivers(boxes: Sequence[Box], page_w: int, page_h: int) -> list[Box]:
    """Drop thin strips that are just the edge of a larger panel."""
    kept: list[Box] = []
    for index, box in enumerate(boxes):
        if any(
            other_index != index and other.area > box.area * 1.4 and _is_sliver(box, other, page_w, page_h)
            for other_index, other in enumerate(boxes)
        ):
            continue
        kept.append(box)
    return kept


def separate_overlaps(boxes: Sequence[Box]) -> list[Box]:
    """Pull overlapping neighbours back to the middle of the shared band.

    A speech balloon is restored afterwards by expanding for text boxes, so this
    only stops the same strip of art being pasted twice.
    """
    current = list(boxes)
    for _ in range(4):
        changed = False
        for index in range(len(current)):
            for other in range(index + 1, len(current)):
                pair = _split_overlap(current[index], current[other])
                if pair is None:
                    continue
                current[index], current[other] = pair
                changed = True
        if not changed:
            break
    return [box for box in current if box.w >= 8 and box.h >= 8]


def _overlap_1d(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _is_sliver(small: Box, large: Box, page_w: int, page_h: int) -> bool:
    if small.w <= 0 or small.h <= 0:
        return False
    x_overlap = _overlap_1d(small.x, small.x2, large.x, large.x2) / small.w
    y_overlap = _overlap_1d(small.y, small.y2, large.y, large.y2) / small.h
    y_gap = 0 if y_overlap > 0 else min(abs(small.y - large.y2), abs(large.y - small.y2))
    x_gap = 0 if x_overlap > 0 else min(abs(small.x - large.x2), abs(large.x - small.x2))
    short = small.h < page_h * 0.055 and small.h < large.h * 0.4
    if short and x_overlap > 0.6 and y_gap < page_h * 0.025:
        return True
    narrow = small.w < page_w * 0.08 and small.w < large.w * 0.4
    return narrow and y_overlap > 0.6 and x_gap < page_w * 0.025


def _split_overlap(a: Box, b: Box) -> tuple[Box, Box] | None:
    inter_w = _overlap_1d(a.x, a.x2, b.x, b.x2)
    inter_h = _overlap_1d(a.y, a.y2, b.y, b.y2)
    if inter_w <= 8 or inter_h <= 8:
        return None
    if contained_ratio(a, b) > 0.7 or contained_ratio(b, a) > 0.7:
        return None
    if inter_h < 0.3 * min(a.h, b.h) and inter_w > 0.55 * min(a.w, b.w):
        mid = (max(a.y, b.y) + min(a.y2, b.y2)) // 2
        if a.y <= b.y:
            return (
                Box(a.x, a.y, a.w, max(1, mid - a.y), a.score),
                Box(b.x, mid, b.w, max(1, b.y2 - mid), b.score),
            )
        return (
            Box(a.x, mid, a.w, max(1, a.y2 - mid), a.score),
            Box(b.x, b.y, b.w, max(1, mid - b.y), b.score),
        )
    if inter_w < 0.3 * min(a.w, b.w) and inter_h > 0.55 * min(a.h, b.h):
        mid = (max(a.x, b.x) + min(a.x2, b.x2)) // 2
        if a.x <= b.x:
            return (
                Box(a.x, a.y, max(1, mid - a.x), a.h, a.score),
                Box(mid, b.y, max(1, b.x2 - mid), b.h, b.score),
            )
        return (
            Box(mid, a.y, max(1, a.x2 - mid), a.h, a.score),
            Box(b.x, b.y, max(1, mid - b.x), b.h, b.score),
        )
    return None
