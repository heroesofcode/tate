"""Reading order for panel boxes: Japanese right to left, or Western left to right."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Literal

from tate.boxes import Box

ReadingDirection = Literal["rtl", "ltr"]


def reading_order(boxes: Sequence[Box], direction: ReadingDirection = "rtl") -> list[int]:
    """Return indices of `boxes` in reading order.

    Horizontal gutters are cut first (top band, then the band below). When a
    panel is tall enough to block that cut, or a slanted gutter makes the
    boxes overlap, the page is cut into columns and the starting column is
    finished first. Side-by-side strips are the exception: if every column is
    a stack of four or more panels, each strip is finished before the next.
    ``rtl`` reads the right column first; ``ltr`` reads the left column first.
    ``ltr`` is ``rtl`` on boxes mirrored in x, so both directions share one
    algorithm. Small overlaps and corner bleeds are ignored so a detector that
    crosses a gutter still produces a clean cut. An inset is read with the
    row it sits in.
    """
    if direction == "ltr":
        boxes = [_mirror_x(box) for box in boxes]
    elif direction != "rtl":
        raise ValueError("direction must be rtl or ltr")
    if len(boxes) <= 1:
        return list(range(len(boxes)))
    # Work on the boxes in one canonical order so that ties (equal edges,
    # equal boxes) never depend on the order the detector listed them in.
    canonical = sorted(range(len(boxes)), key=lambda index: _box_key(boxes[index]))
    ordered = _order(list(range(len(canonical))), [boxes[index] for index in canonical], _tolerance(boxes))
    return [canonical[index] for index in ordered]


def _box_key(box: Box) -> tuple[int, int, int, int, float]:
    return (box.y, box.x, box.h, box.w, box.score)


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
    # A detector that bleeds across a row gutter closes the normal gap. A
    # wider shrink still finds the rows, and is preferred to cutting columns.
    loose = _loose_tolerance(indices, boxes)
    if loose > tolerance:
        bled = _groups_by_gaps(indices, boxes, "y", loose)
        if bled is not None and _bands_only_bleed(bled, boxes):
            ordered = []
            for group in bled:
                ordered.extend(_order(group, boxes, tolerance))
            return ordered
    vertical = _groups_by_gaps(indices, boxes, "x", tolerance)
    if vertical is not None:
        ordered = []
        for group in reversed(vertical):
            ordered.extend(_order(group, boxes, tolerance))
        return ordered
    blocked = _blocked_stack(indices, boxes)
    if blocked is not None:
        return blocked
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
    panels, so the reader finishes one strip before the next. The gutters may
    be closed by boxes that bleed a little; a wider shrink still sees them.
    """
    loose = max(tolerance, _loose_tolerance(indices, boxes))
    columns = _groups_by_gaps(indices, boxes, "x", loose)
    if columns is None or len(columns) < 2:
        return None
    if any(len(column) < 4 for column in columns):
        return None
    if not _columns_share_an_edge(columns, boxes):
        return None
    for column in columns:
        rows = _groups_by_gaps(column, boxes, "y", loose)
        if rows is None or len(rows) < 4 or any(len(row) != 1 for row in rows):
            return None
    return columns


def _columns_share_an_edge(columns: Sequence[Sequence[int]], boxes: Sequence[Box]) -> bool:
    """A strip's frames line up. A wide panel that only overlaps after shrinking does not."""
    for column in columns:
        starts = sorted(boxes[index].x for index in column)
        widths = sorted(boxes[index].w for index in column)
        median_x = starts[len(starts) // 2]
        median_w = max(1, widths[len(widths) // 2])
        limit = max(24, int(median_w * 0.06))
        if any(abs(start - median_x) > limit for start in starts):
            return False
    return True


def _bands_only_bleed(groups: Sequence[Sequence[int]], boxes: Sequence[Box]) -> bool:
    """A loose row cut is only a bleed when neighbouring bands barely overlap.

    A panel that reaches well into the band below is a column, not a torn edge,
    and the row cut would put it in the wrong place.
    """
    for upper, lower in itertools.pairwise(groups):
        for left in upper:
            for right in lower:
                overlap = _overlap_1d(boxes[left].y, boxes[left].y2, boxes[right].y, boxes[right].y2)
                shorter = min(boxes[left].h, boxes[right].h)
                if _ratio(overlap, shorter) > 0.28:
                    return False
    return True


def _loose_tolerance(indices: Sequence[int], boxes: Sequence[Box]) -> int:
    heights = sorted(boxes[index].h for index in indices)
    median = heights[len(heights) // 2]
    return max(2, int(median * 0.12))


def _overlap_1d(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _ratio(overlap: int, length: int) -> float:
    if length <= 0:
        return 0.0
    return overlap / length


def _blocked_stack(indices: list[int], boxes: Sequence[Box]) -> list[int] | None:
    """Order a column that a neighbouring panel overlaps, so the row cut is ambiguous.

    A stack is two panels one above the other in the same column. A panel beside
    that stack which reaches into both of them blocks a horizontal cut (a tall
    panel, or a slanted gutter whose boxes overlap). The stack is read from the
    starting side: the right stack before a left blocker, or a right blocker
    before the stack on its left. Blockers on both sides are left to row order.
    """
    best: tuple[tuple[int, int, int], list[int], list[int], list[int]] | None = None
    for upper in indices:
        for lower in indices:
            if upper == lower:
                continue
            top, bottom = boxes[upper], boxes[lower]
            if top.y > bottom.y:
                continue
            x_overlap = _overlap_1d(top.x, top.x2, bottom.x, bottom.x2)
            if _ratio(x_overlap, min(top.w, bottom.w)) < 0.55:
                continue
            y_overlap = _overlap_1d(top.y, top.y2, bottom.y, bottom.y2)
            if y_overlap > 0.35 * min(top.h, bottom.h) and bottom.y < top.y2:
                continue
            if bottom.y > top.y2 + max(top.h, bottom.h):
                continue
            cut = _stack_cut(top, bottom, y_overlap)
            above, below, blockers = _split_at_cut(indices, boxes, upper, lower, cut)
            if not above or not below:
                continue
            if not _blockers_on_one_side(boxes, upper, lower, blockers):
                continue
            # Panels sitting fully above or below this stack belong to another
            # band. A y-gap should have separated them; if they are still here
            # this pair is not the cut for the whole group.
            if any(not _overlaps_stack(boxes[index], boxes[upper], boxes[lower]) for index in indices):
                continue
            # Prefer a cut that actually has a blocker, then the one that
            # separates the most panels, then the highest cut on the page.
            rank = (1 if blockers else 0, len(above) + len(below), -cut)
            if best is None or rank > best[0]:
                best = (rank, above, below, blockers)
    if best is None:
        return None
    _, above, below, blockers = best
    if not blockers:
        return None
    ordered = _order_blocked(above, below, blockers, boxes)
    if len(ordered) != len(indices) or len(set(ordered)) != len(indices):
        return None
    return ordered


def _stack_cut(top: Box, bottom: Box, y_overlap: int) -> int:
    if y_overlap > 0:
        return (max(top.y, bottom.y) + min(top.y2, bottom.y2)) // 2
    return (top.y2 + bottom.y) // 2


def _split_at_cut(
    indices: Sequence[int],
    boxes: Sequence[Box],
    upper: int,
    lower: int,
    cut: int,
) -> tuple[list[int], list[int], list[int]]:
    above: list[int] = []
    below: list[int] = []
    blockers: list[int] = []
    for index in indices:
        if index == upper:
            above.append(index)
            continue
        if index == lower:
            below.append(index)
            continue
        box = boxes[index]
        center = box.y + box.h / 2
        into_lower = _reaches(box, boxes[lower])
        into_upper = _reaches(box, boxes[upper])
        reaches_across = (center < cut and into_lower) or (center >= cut and into_upper)
        if reaches_across:
            blockers.append(index)
        elif center < cut:
            above.append(index)
        else:
            below.append(index)
    return above, below, blockers


def _overlaps_stack(box: Box, upper: Box, lower: Box) -> bool:
    stack_top = min(upper.y, lower.y)
    stack_bottom = max(upper.y2, lower.y2)
    return _overlap_1d(box.y, box.y2, stack_top, stack_bottom) > 0


def _reaches(box: Box, other: Box) -> bool:
    """True when `box` overlaps `other` from the side, not as the same column."""
    y_overlap = _overlap_1d(box.y, box.y2, other.y, other.y2)
    x_overlap = _overlap_1d(box.x, box.x2, other.x, other.x2)
    return _ratio(y_overlap, min(box.h, other.h)) > 0.2 and _ratio(x_overlap, min(box.w, other.w)) < 0.45


def _blockers_on_one_side(
    boxes: Sequence[Box],
    upper: int,
    lower: int,
    blockers: Sequence[int],
) -> bool:
    if not blockers:
        return True
    stack_x = (min(boxes[upper].x, boxes[lower].x) + max(boxes[upper].x2, boxes[lower].x2)) / 2
    sides = [(boxes[index].x + boxes[index].w / 2) < stack_x for index in blockers]
    return all(sides) or not any(sides)


def _order_blocked(
    above: list[int],
    below: list[int],
    blockers: list[int],
    boxes: Sequence[Box],
) -> list[int]:
    stack = above + below
    stack_x = sum(boxes[index].x + boxes[index].w / 2 for index in stack) / len(stack)
    blocker_x = sum(boxes[index].x + boxes[index].w / 2 for index in blockers) / len(blockers)
    tolerance = _tolerance([boxes[index] for index in above + below + blockers])
    # Right-hand blockers are the starting column. Left-hand blockers wait
    # until the stack they overlap has been read.
    groups = (blockers, above, below) if blocker_x > stack_x else (above, below, blockers)
    ordered: list[int] = []
    for group in groups:
        ordered.extend(_order(group, boxes, tolerance))
    return ordered


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
