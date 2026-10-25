"""Fuse model boxes with the frames and gutters of the same page.

The frame-and-gutter merge runs first. Two corrections follow, and both need a
gutter between the pieces: a confident model box is put back when that merge
shattered it across the art, and a low-scoring model box is split when frames
sit on either side of a gutter. A gutter the merge missed is added when it
does not overlap a box already kept. Nothing here reads a ground-truth label.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from tate.boxes import Box, contained_ratio, intersection_area, iou
from tate.detect.frames import _prepare_guides, background_color, merge_panel_guides

# A model box this confident is a panel. Fragments inside it are the drawing.
_CONFIDENT = 0.8
# Below this, a model box is a blob that may have swallowed two panels.
_LOOSE = 0.5
_CHILD_AREA = 0.18
_COVER = 0.55
_CONTAINED = 0.6


def fuse_detections(
    image: Image.Image,
    model: Sequence[Box],
    frames: Sequence[Box],
    gutters: Sequence[Box],
) -> list[Box]:
    """Panels for one page, in source pixels."""
    page_w, page_h = image.size
    if page_w < 2 or page_h < 2:
        return [box.clamp(page_w, page_h) for box in model]
    merged = merge_panel_guides(model, frames, gutters, page_w, page_h)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    background = background_color(rgb)
    restored = _restore_shattered(merged, model, page_w, page_h)
    split = _split_loose(restored, frames, rgb, background, page_w, page_h)
    added = _add_gutters(split, gutters, page_w, page_h)
    return _drop_wrappers(added)


def _restore_shattered(
    merged: Sequence[Box],
    model: Sequence[Box],
    page_w: int,
    page_h: int,
) -> list[Box]:
    """Put back a confident box whose pieces overlap instead of meeting at a gutter."""
    kept = [box.clamp(page_w, page_h) for box in merged]
    for box in model:
        if box.score < _CONFIDENT:
            continue
        parent = box.clamp(page_w, page_h)
        inside = [
            index
            for index, piece in enumerate(kept)
            if piece.area < parent.area * 0.9 and contained_ratio(piece, parent) >= _CONTAINED
        ]
        if len(inside) < 2:
            continue
        pieces = [kept[index] for index in inside]
        if not _shattered(pieces, page_w, page_h):
            continue
        for index in sorted(inside, reverse=True):
            del kept[index]
        kept.append(parent)
    return kept


def _shattered(pieces: Sequence[Box], page_w: int, page_h: int) -> bool:
    """True when the pieces are one picture broken up, not a row of panels.

    Panels meet at a narrow gutter. Pieces that overlap, or that leave a wide
    hole between neighbors, were cut out of a single drawing. A wide span
    that only crosses another piece is that piece, not a hole.
    """
    for index, left in enumerate(pieces):
        for offset, right in enumerate(pieces[index + 1 :]):
            if _relation(left, right, page_w, page_h) == "overlap":
                return True
            y_overlap = _overlap(left.y, left.y2, right.y, right.y2) / max(1, min(left.h, right.h))
            x_overlap = _overlap(left.x, left.x2, right.x, right.x2) / max(1, min(left.w, right.w))
            x_gap = _opening(left.x, left.x2, right.x, right.x2)
            y_gap = _opening(left.y, left.y2, right.y, right.y2)
            if y_overlap > 0.45 and x_gap < -4:
                return True
            if x_overlap > 0.45 and y_gap < -4:
                return True
            if y_overlap > 0.45 and x_gap >= 0:
                horizontal = True
                gap, span = x_gap, page_w
            elif x_overlap > 0.45 and y_gap >= 0:
                horizontal = False
                gap, span = y_gap, page_h
            else:
                continue
            if _piece_between(pieces, index, index + 1 + offset, horizontal=horizontal):
                continue
            if gap > max(48, int(0.05 * span)):
                return True
    return False


def _piece_between(pieces: Sequence[Box], left_index: int, right_index: int, *, horizontal: bool) -> bool:
    """True when another piece lies in the opening on the axis the pair shares."""
    left, right = pieces[left_index], pieces[right_index]
    if horizontal:
        gap_start, gap_end = _span(left.x, left.x2, right.x, right.x2)
        band_start, band_end = max(left.y, right.y), min(left.y2, right.y2)
    else:
        gap_start, gap_end = _span(left.y, left.y2, right.y, right.y2)
        band_start, band_end = max(left.x, right.x), min(left.x2, right.x2)
    band = band_end - band_start
    if gap_end <= gap_start or band <= 0:
        return False
    for index, other in enumerate(pieces):
        if index in (left_index, right_index):
            continue
        if horizontal:
            start, end = other.x, other.x2
            cross_start, cross_end, cross_span = other.y, other.y2, other.h
        else:
            start, end = other.y, other.y2
            cross_start, cross_end, cross_span = other.x, other.x2, other.w
        if end <= gap_start or start >= gap_end:
            continue
        shared = _overlap(cross_start, cross_end, band_start, band_end)
        if shared / max(1, min(cross_span, band)) > 0.45:
            return True
    return False


def _split_loose(
    boxes: Sequence[Box],
    frames: Sequence[Box],
    rgb: np.ndarray[Any, Any],
    background: np.ndarray[Any, Any],
    page_w: int,
    page_h: int,
) -> list[Box]:
    """Replace a low-scoring blob with the frames on either side of a gutter."""
    guides = _prepare_guides(frames, page_w, page_h)
    if not guides:
        return list(boxes)
    used = [False] * len(guides)
    kept: list[Box] = []
    for box in boxes:
        if box.score >= _LOOSE:
            kept.append(box)
            _claim(box, guides, used)
            continue
        kids = [guides[index] for index in _child_indexes(box, guides, used)]
        if _real_split(rgb, background, box, kids, page_w, page_h):
            for guide in kids:
                used[guides.index(guide)] = True
            kept.extend(kids)
            continue
        side = None
        if not _has_peer(box, boxes):
            side = _side_split(rgb, background, box, kids, page_w, page_h)
        if side is not None:
            guide, remainder = side
            used[guides.index(guide)] = True
            kept.extend((guide, remainder))
            continue
        kept.append(box)
        _claim(box, guides, used)
    return kept


def _add_gutters(
    boxes: Sequence[Box],
    gutters: Sequence[Box],
    page_w: int,
    page_h: int,
) -> list[Box]:
    """Add a gutter panel the merge never covered.

    A gutter that already holds one of those panels is a second copy, so it
    stays out. Gutters are added in passes so a row can grow from a neighbour.
    """
    kept = list(boxes)
    guides = _prepare_guides(gutters, page_w, page_h)
    for _ in guides:
        added = False
        for guide in guides:
            if any(guide.x == other.x and guide.y == other.y and guide.w == other.w for other in kept):
                continue
            if _blocked(guide, kept):
                continue
            if any(contained_ratio(other, guide) >= 0.7 and other.area < guide.area * 0.9 for other in kept):
                continue
            if guide.area < 0.05 * page_w * page_h:
                continue
            if kept and not _beside(guide, kept, page_w, page_h):
                continue
            kept.append(guide)
            added = True
        if not added:
            break
    return kept


def _has_peer(box: Box, boxes: Sequence[Box]) -> bool:
    """True when another box already covers this same region."""
    return any(other is not box and iou(other, box) > 0.3 for other in boxes)


def _claim(box: Box, guides: Sequence[Box], used: list[bool]) -> None:
    for index, guide in enumerate(guides):
        if iou(guide, box) >= 0.5 or (
            guide.area < box.area * 0.92 and contained_ratio(guide, box) >= _CONTAINED
        ):
            used[index] = True


def _beside(guide: Box, kept: Sequence[Box], page_w: int, page_h: int) -> bool:
    return any(_relation(guide, other, page_w, page_h) == "neighbor" for other in kept)


def _child_indexes(box: Box, guides: Sequence[Box], used: Sequence[bool]) -> list[int]:
    ranked = sorted(
        (index for index, guide in enumerate(guides) if not used[index]),
        key=lambda index: guides[index].area,
        reverse=True,
    )
    kept: list[int] = []
    for index in ranked:
        guide = guides[index]
        if guide.area >= box.area * 0.92 or guide.area < box.area * _CHILD_AREA:
            continue
        if contained_ratio(guide, box) < _CONTAINED:
            continue
        if any(
            iou(guide, guides[other]) > 0.12 or contained_ratio(guide, guides[other]) > 0.45 for other in kept
        ):
            continue
        kept.append(index)
    return kept


def _real_split(
    rgb: np.ndarray[Any, Any],
    background: np.ndarray[Any, Any],
    box: Box,
    guides: Sequence[Box],
    page_w: int,
    page_h: int,
) -> bool:
    if len(guides) < 2 or box.area <= 0:
        return False
    covered = sum(intersection_area(guide, box) for guide in guides)
    if covered < _COVER * box.area:
        return False
    areas = [guide.area for guide in guides]
    if min(areas) < 0.22 * max(areas):
        return False
    return _separated(rgb, background, guides, page_w, page_h)


def _side_split(
    rgb: np.ndarray[Any, Any],
    background: np.ndarray[Any, Any],
    box: Box,
    guides: Sequence[Box],
    page_w: int,
    page_h: int,
) -> tuple[Box, Box] | None:
    """One frame on one side of a loose box, and the remainder past the gutter."""
    if len(guides) != 1 or box.area < 0.15 * page_w * page_h:
        return None
    guide = guides[0]
    remainder = _remainder(box, guide)
    if remainder is None or remainder.area < box.area * 0.2 or remainder.w < 8 or remainder.h < 8:
        return None
    if not _separated(rgb, background, (guide, remainder), page_w, page_h):
        return None
    return guide, remainder


def _remainder(box: Box, guide: Box) -> Box | None:
    tol_x = max(8, int(box.w * 0.15))
    tol_y = max(8, int(box.h * 0.15))
    rows = abs(guide.y - box.y) <= tol_y and abs(guide.y2 - box.y2) <= tol_y
    cols = abs(guide.x - box.x) <= tol_x and abs(guide.x2 - box.x2) <= tol_x
    if rows and guide.w < box.w * 0.82:
        if guide.x <= box.x + tol_x and box.x2 - guide.x2 >= box.w * 0.2:
            return Box(guide.x2, box.y, box.x2 - guide.x2, box.h, guide.score)
        if guide.x2 >= box.x2 - tol_x and guide.x - box.x >= box.w * 0.2:
            return Box(box.x, box.y, guide.x - box.x, box.h, guide.score)
    if cols and guide.h < box.h * 0.82:
        if guide.y <= box.y + tol_y and box.y2 - guide.y2 >= box.h * 0.2:
            return Box(box.x, guide.y2, box.w, box.y2 - guide.y2, guide.score)
        if guide.y2 >= box.y2 - tol_y and guide.y - box.y >= box.h * 0.2:
            return Box(box.x, box.y, box.w, guide.y - box.y, guide.score)
    return None


def _separated(
    rgb: np.ndarray[Any, Any],
    background: np.ndarray[Any, Any],
    boxes: Sequence[Box],
    page_w: int,
    page_h: int,
    *,
    strict: bool = False,
) -> bool:
    pairs = 0
    ordered = list(boxes)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            kind = _relation(left, right, page_w, page_h)
            if kind == "overlap":
                return False
            if kind == "apart":
                continue
            pairs += 1
            if not _gap_is_gutter(rgb, background, left, right, page_w, page_h, strict=strict):
                return False
    return pairs > 0


def _relation(left: Box, right: Box, page_w: int, page_h: int) -> str:
    if iou(left, right) > 0.12:
        return "overlap"
    y_overlap = _overlap(left.y, left.y2, right.y, right.y2) / max(1, min(left.h, right.h))
    x_overlap = _overlap(left.x, left.x2, right.x, right.x2) / max(1, min(left.w, right.w))
    x_gap = _opening(left.x, left.x2, right.x, right.x2)
    y_gap = _opening(left.y, left.y2, right.y, right.y2)
    # A few pixels of overlap is a shared rule. More than that is one region.
    horizontal = y_overlap > 0.45 and -4 <= x_gap <= 0.18 * page_w
    vertical = x_overlap > 0.45 and -4 <= y_gap <= 0.18 * page_h
    if horizontal and vertical:
        return "overlap"
    if horizontal or vertical:
        return "neighbor"
    return "apart"


def _gap_is_gutter(
    rgb: np.ndarray[Any, Any],
    background: np.ndarray[Any, Any],
    left: Box,
    right: Box,
    page_w: int,
    page_h: int,
    *,
    strict: bool = False,
) -> bool:
    """True when the boxes meet at a gutter rather than overlapping art.

    A gap of several pixels is the rule or the paper the two guides were cut
    on. Boxes that only touch need that band to actually be paper or a rule,
    so a line inside a drawing does not count.
    """
    y_overlap = _overlap(left.y, left.y2, right.y, right.y2) / max(1, min(left.h, right.h))
    x_overlap = _overlap(left.x, left.x2, right.x, right.x2) / max(1, min(left.w, right.w))
    if y_overlap >= x_overlap:
        gap = _opening(left.x, left.x2, right.x, right.x2)
        start, end = _span(left.x, left.x2, right.x, right.x2)
        cross_0 = max(left.y, right.y)
        cross_1 = min(left.y2, right.y2)
        axis, span = 1, page_w
    else:
        gap = _opening(left.y, left.y2, right.y, right.y2)
        start, end = _span(left.y, left.y2, right.y, right.y2)
        cross_0 = max(left.x, right.x)
        cross_1 = min(left.x2, right.x2)
        axis, span = 0, page_h
    if gap < -4 or gap > 0.18 * span:
        return False
    if gap >= 4 and not strict:
        return True
    return _strip_is_gutter(rgb, background, axis, start, end, cross_0, cross_1, span)


def _strip_is_gutter(
    rgb: np.ndarray[Any, Any],
    background: np.ndarray[Any, Any],
    axis: int,
    start: int,
    end: int,
    cross_0: int,
    cross_1: int,
    page_span: int,
) -> bool:
    """True when the band between two touching boxes is paper or a thin rule."""
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if cross_1 - cross_0 < 4:
        return False
    # A real gap is measured on its own. Padding is only for boxes that touch,
    # so the art beside a wide gutter is not counted as the gutter.
    if end - start >= 2:
        band_0, band_1 = start, end
    else:
        pad = max(3, round(page_span * 0.006))
        band_0 = max(0, start - pad)
        band_1 = min(page_span, max(start, end) + pad)
    if band_1 - band_0 < 2:
        return False
    if axis == 1:
        strip = rgb[max(0, cross_0) : min(height, cross_1), max(0, band_0) : min(width, band_1)]
    else:
        strip = rgb[max(0, band_0) : min(height, band_1), max(0, cross_0) : min(width, cross_1)]
    if strip.size == 0:
        return False
    distance = np.sqrt(np.square(strip.astype(np.float32) - background).sum(axis=2))
    if float((distance < 28.0).mean()) >= 0.55:
        return True
    luminance = strip.astype(np.float32).mean(axis=2)
    paper = float(0.299 * background[0] + 0.587 * background[1] + 0.114 * background[2])
    dark = luminance < paper - 30.0
    return float(dark.mean()) >= 0.35 and band_1 - band_0 <= max(8, round(page_span * 0.03))


def _blocked(guide: Box, kept: Sequence[Box]) -> bool:
    """True when `guide` is already represented by a box of similar size."""
    center_x = guide.x + guide.w / 2
    center_y = guide.y + guide.h / 2
    for other in kept:
        if contained_ratio(guide, other) >= 0.55 or iou(guide, other) >= 0.5:
            return True
        # A speck inside a panel must not hide that panel. Only a box about
        # as large as the guide can claim its centre.
        if other.area < guide.area * 0.8:
            continue
        if other.x <= center_x <= other.x2 and other.y <= center_y <= other.y2:
            return True
    return False


def _contains_many(box: Box, others: Sequence[Box]) -> bool:
    inside = [other for other in others if other.area < box.area and contained_ratio(other, box) >= 0.7]
    if len(inside) < 2:
        return False
    # Children on one side of a wide box are not a partition of it.
    return _cover_fraction(box, inside) >= 0.7


def _cover_fraction(parent: Box, children: Sequence[Box]) -> float:
    cols, rows = 8, 8
    hits = 0
    for row in range(rows):
        cy = parent.y + (row + 0.5) * parent.h / rows
        for col in range(cols):
            cx = parent.x + (col + 0.5) * parent.w / cols
            if any(child.x <= cx < child.x2 and child.y <= cy < child.y2 for child in children):
                hits += 1
    return hits / (cols * rows)


def _drop_wrappers(boxes: Sequence[Box]) -> list[Box]:
    drop = {index for index, box in enumerate(boxes) if _contains_many(box, boxes)}
    return [box for index, box in enumerate(boxes) if index not in drop]


def _opening(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    if end_a <= start_b:
        return start_b - end_a
    if end_b <= start_a:
        return start_a - end_b
    return -_overlap(start_a, end_a, start_b, end_b)


def _span(start_a: int, end_a: int, start_b: int, end_b: int) -> tuple[int, int]:
    if end_a <= start_b:
        return end_a, start_b
    if end_b <= start_a:
        return end_b, start_a
    return max(start_a, start_b), min(end_a, end_b)


def _overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))
