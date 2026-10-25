"""Reading order for panel boxes: Japanese right to left, or Western left to right."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from tate.boxes import Box

ReadingDirection = Literal["rtl", "ltr"]


def reading_order(boxes: Sequence[Box], direction: ReadingDirection = "rtl") -> list[int]:
    """Return indices of `boxes` in reading order.

    Horizontal gutters are cut first (top band, then the band below). When a
    panel is tall enough to block that cut, the page is cut into columns.
    Side-by-side strips are the exception: if every column is a stack of four
    or more panels, each strip is finished before the next. ``rtl`` reads the
    right column first; ``ltr`` reads the left column first.
    ``ltr`` is ``rtl`` on boxes mirrored in x, so both directions share one
    algorithm. Small overlaps are ignored so a detector that bleeds across a
    gutter still produces a clean cut.
    """
    if direction == "ltr":
        boxes = [_mirror_x(box) for box in boxes]
    elif direction != "rtl":
        raise ValueError("direction must be rtl or ltr")
    indices = list(range(len(boxes)))
    if len(indices) <= 1:
        return indices
    return _order(indices, boxes, _tolerance(boxes))


def _mirror_x(box: Box) -> Box:
    """Reflect a box across x = 0. Width and the other fields stay put."""
    return Box(-box.x2, box.y, box.w, box.h, box.score)


def _tolerance(boxes: Sequence[Box]) -> int:
    heights = sorted(box.h for box in boxes)
    median = heights[len(heights) // 2]
    return max(2, int(median * 0.04))


def _order(indices: list[int], boxes: Sequence[Box], tolerance: int) -> list[int]:
    if len(indices) <= 1:
        return list(indices)
    columns = _column_stacks(indices, boxes, tolerance)
    if columns is not None:
        ordered: list[int] = []
        for group in reversed(columns):
            ordered.extend(_order(group, boxes, tolerance))
        return ordered
    horizontal = _groups_by_gaps(indices, boxes, "y", tolerance)
    if horizontal is not None:
        ordered = []
        for group in horizontal:
            ordered.extend(_order(group, boxes, tolerance))
        return ordered
    vertical = _groups_by_gaps(indices, boxes, "x", tolerance)
    if vertical is not None:
        ordered = []
        for group in reversed(vertical):
            ordered.extend(_order(group, boxes, tolerance))
        return ordered
    return _rows(indices, boxes)


def _span(box: Box, axis: str, tolerance: int) -> tuple[int, int]:
    if axis == "y":
        start, end, limit = box.y, box.y2, box.h
    else:
        start, end, limit = box.x, box.x2, box.w
    shrink = min(tolerance, max(0, limit // 6))
    if end - start <= 2 * shrink + 1:
        shrink = 0
    return start + shrink, end - shrink


def _center(box: Box, axis: str) -> float:
    if axis == "y":
        return box.y + box.h / 2
    return box.x + box.w / 2


def _groups_by_gaps(
    indices: list[int],
    boxes: Sequence[Box],
    axis: str,
    tolerance: int,
) -> list[list[int]] | None:
    spans = {index: _span(boxes[index], axis, tolerance) for index in indices}
    merged: list[list[int]] = []
    for index in sorted(indices, key=lambda item: spans[item][0]):
        start, end = spans[index]
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    gaps = [(merged[i][1], merged[i + 1][0]) for i in range(len(merged) - 1)]
    gaps = [gap for gap in gaps if gap[1] - gap[0] >= 1]
    if not gaps:
        return None
    cuts = [(start + end) // 2 for start, end in gaps]
    groups: list[list[int]] = [[] for _ in range(len(cuts) + 1)]
    for index in indices:
        center = _center(boxes[index], axis)
        band = 0
        for cut in cuts:
            if center >= cut:
                band += 1
            else:
                break
        groups[band].append(index)
    groups = [group for group in groups if group]
    if len(groups) <= 1:
        return None
    return groups


def _column_stacks(
    indices: list[int],
    boxes: Sequence[Box],
    tolerance: int,
) -> list[list[int]] | None:
    """Side-by-side strips: each column is its own stack of at least four panels.

    A normal row of two or three panels is left to the horizontal cut. A strip
    page is the case where every column separates into four or more single
    panels, so the reader finishes one strip before the next.
    """
    columns = _groups_by_gaps(indices, boxes, "x", tolerance)
    if columns is None or len(columns) < 2:
        return None
    if any(len(column) < 4 for column in columns):
        return None
    for column in columns:
        rows = _groups_by_gaps(column, boxes, "y", tolerance)
        if rows is None or len(rows) < 4 or any(len(row) != 1 for row in rows):
            return None
    return columns


def _vertical_overlap(a: Box, b: Box) -> float:
    overlap = min(a.y2, b.y2) - max(a.y, b.y)
    if overlap <= 0 or a.h <= 0 or b.h <= 0:
        return 0.0
    return overlap / min(a.h, b.h)


def _rows(indices: list[int], boxes: Sequence[Box]) -> list[int]:
    remaining = sorted(indices, key=lambda index: (boxes[index].y, -boxes[index].x2, boxes[index].x))
    rows: list[list[int]] = []
    for index in remaining:
        placed = False
        for row in rows:
            if any(_vertical_overlap(boxes[index], boxes[other]) >= 0.5 for other in row):
                row.append(index)
                placed = True
                break
        if not placed:
            rows.append([index])
    rows.sort(key=lambda row: min(boxes[index].y for index in row))
    ordered: list[int] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda index: (-boxes[index].x2, boxes[index].y, boxes[index].x)))
    return ordered
