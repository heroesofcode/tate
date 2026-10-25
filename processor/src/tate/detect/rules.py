"""Cells closed by frame rules on locally estimated paper.

The paper colour is the morphological closing of the page, so a tan or
uneven sheet is measured per region rather than as one global white.
A clean grid replaces the detector only when the cells agree with it
and either explain a merged sheet, a set of equal floating panels, or
a majority of cells the detector missed. One interior cell can be added
when the rest of that grid already matches.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from tate.boxes import Box, contained_ratio, iou

_LONG_SIDE = 1100
_MIN_CELLS = 4
_FILL_MIN = 0.72
_MEDIAN_FILL = 0.90
_FLOAT_FILL = 0.96
_SHEET_COVER = 0.68
_FLOAT_COVER = 0.40
_SIZE_RATIO = 0.32
_FLOAT_RATIO = 0.85
_TIGHT_FLOAT_COVER = 0.30
_TIGHT_FLOAT_RATIO = 0.65
_TIGHT_FLOAT_MATCH = 0.99
_MATCH_IOU = 0.40
_DISJOINT_IOU = 0.06
_CLEAR_IOU = 0.12
_SWALLOW = 0.72
_POOR_MEAN = 0.70
_ALIGNED_MEAN = 0.85
_ALIGNED_MATCH = 0.85
_SNAP_MEAN = 0.90
_MID_COVER = 0.48
_MID_RATIO = 0.40
_MID_FILL = 0.94
_MID_MATCH = 0.55


def ruled_cells(image: Image.Image) -> tuple[list[Box], list[float]]:
    """Rectangles closed by long rules, with the fill of each cell.

    Empty when the page has no closed rules. Fills are the share of each
    cell that the connected component actually occupies.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if height < 8 or width < 8:
        return [], []
    gray, scale = _rule_gray(rgb)
    small_h, small_w = int(gray.shape[0]), int(gray.shape[1])
    wall = _rule_wall(gray, small_w, small_h)
    space = cast(np.ndarray[Any, Any], cv2.bitwise_not(wall))
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(space, connectivity=4)
    inv = 1.0 / scale
    page = height * width
    boxes: list[Box] = []
    fills: list[float] = []
    for index in range(1, int(count)):
        x_s, y_s, bw_s, bh_s, area_s = (int(stats[index, column]) for column in range(5))
        if bw_s * bh_s <= 0:
            continue
        fill = area_s / (bw_s * bh_s)
        if bw_s * bh_s < 0.012 * small_h * small_w or fill < _FILL_MIN:
            continue
        left = round(x_s * inv)
        top = round(y_s * inv)
        box_w = max(1, round(bw_s * inv))
        box_h = max(1, round(bh_s * inv))
        if left < width * 0.02:
            box_w += left
            left = 0
        if top < height * 0.015:
            box_h += top
            top = 0
        if left + box_w > width * 0.98:
            box_w = width - left
        if top + box_h > height * 0.985:
            box_h = height - top
        if box_w * box_h < 0.018 * page or box_w < width * 0.08 or box_h < height * 0.045:
            continue
        if box_w > width * 0.97 and box_h > height * 0.97:
            continue
        boxes.append(Box(left, top, max(1, box_w), max(1, box_h)))
        fills.append(fill)
    return _drop_specks(*_dedupe(boxes, fills))


def adopt_ruled_grid(image: Image.Image, panels: Sequence[Box]) -> list[Box]:
    """Use ruled cells when they explain the detector, otherwise keep it.

    A full sheet the detector merged, equal floating panels it split into
    extras, or a grid it mostly missed, is replaced by the cells. Boxes that
    already sit on those cells are snapped onto the rules. A single interior
    cell the detector skipped is added. Anything else stays.
    """
    cells, fills = ruled_cells(image)
    page_w, page_h = image.size
    if not _grid_ok(panels, cells, fills, page_w, page_h):
        added = _interior_cell(panels, cells, fills, page_w, page_h)
        if added is None:
            return [box.clamp(page_w, page_h) for box in panels]
        return [*panels, added]
    return cells


def _rule_gray(rgb: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], float]:
    gray = cast(np.ndarray[Any, Any], cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    height, width = int(gray.shape[0]), int(gray.shape[1])
    scale = min(1.0, _LONG_SIDE / max(height, width))
    if scale >= 1:
        return gray, 1.0
    # Erode before the area resize so a one-pixel rule is not averaged away.
    kernel = max(1, round(1 / scale))
    gray = cast(
        np.ndarray[Any, Any],
        cv2.erode(gray, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel, kernel))),
    )
    resized = cast(
        np.ndarray[Any, Any],
        cv2.resize(
            gray,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        ),
    )
    return resized, scale


def _rule_wall(gray: np.ndarray[Any, Any], small_w: int, small_h: int) -> np.ndarray[Any, Any]:
    close = max(7, round(min(small_h, small_w) * 0.018))
    if close % 2 == 0:
        close += 1
    paper = cast(
        np.ndarray[Any, Any],
        cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close, close))),
    )
    delta = paper.astype(np.int16) - gray.astype(np.int16)
    lines = (delta > 14).astype(np.uint8) * 255
    hlen = max(12, int(small_w * 0.10))
    vlen = max(12, int(small_h * 0.08))
    horiz = cast(
        np.ndarray[Any, Any],
        cv2.morphologyEx(lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1))),
    )
    vert = cast(
        np.ndarray[Any, Any],
        cv2.morphologyEx(lines, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vlen))),
    )
    bridge = max(2, int(min(small_w, small_h) * 0.02))
    horiz = cast(
        np.ndarray[Any, Any],
        cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (bridge * 2 + 1, 3))),
    )
    vert = cast(
        np.ndarray[Any, Any],
        cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (3, bridge * 2 + 1))),
    )
    wall = cast(np.ndarray[Any, Any], cv2.bitwise_or(horiz, vert))
    wall[0, :] = 255
    wall[-1, :] = 255
    wall[:, 0] = 255
    wall[:, -1] = 255
    return wall


def _dedupe(boxes: list[Box], fills: list[float]) -> tuple[list[Box], list[float]]:
    kept: list[Box] = []
    kept_fill: list[float] = []
    order = sorted(range(len(boxes)), key=lambda index: boxes[index].area, reverse=True)
    for index in order:
        box = boxes[index]
        if any(iou(box, other) > 0.35 or contained_ratio(box, other) > 0.6 for other in kept):
            continue
        kept.append(box)
        kept_fill.append(fills[index])
    return kept, kept_fill


def _drop_specks(boxes: list[Box], fills: list[float]) -> tuple[list[Box], list[float]]:
    if not boxes:
        return [], []
    median = float(np.median([box.area for box in boxes]))
    filtered = [(box, fill) for box, fill in zip(boxes, fills, strict=True) if box.area >= 0.40 * median]
    if len(filtered) < _MIN_CELLS:
        return boxes, fills
    return [box for box, _fill in filtered], [fill for _box, fill in filtered]


def _grid_ok(
    panels: Sequence[Box],
    cells: Sequence[Box],
    fills: Sequence[float],
    page_w: int,
    page_h: int,
) -> bool:
    if len(cells) < _MIN_CELLS or not fills or float(np.median(fills)) < _MEDIAN_FILL:
        return False
    if not _disjoint(cells):
        return False
    areas = [cell.area for cell in cells]
    ratio = min(areas) / max(areas)
    cover = sum(areas) / (page_w * page_h)
    median_fill = float(np.median(fills))
    assigned, scores = _assign(panels, cells)
    matched_detections = len(assigned) / max(1, len(panels))
    matched_cells = len(set(assigned.values())) / len(cells)
    mean_iou = sum(scores) / len(scores) if scores else 0.0
    if (
        cover <= _FLOAT_COVER
        and ratio >= _FLOAT_RATIO
        and median_fill >= _FLOAT_FILL
        and matched_cells >= 0.75
        and len(panels) >= len(cells)
    ):
        return True
    # Floating panels of slightly different sizes, every cell already found.
    if (
        cover <= _TIGHT_FLOAT_COVER
        and ratio >= _TIGHT_FLOAT_RATIO
        and median_fill >= _FLOAT_FILL
        and matched_cells >= _TIGHT_FLOAT_MATCH
        and mean_iou >= _ALIGNED_MEAN
        and len(panels) >= len(cells)
    ):
        return True
    # A sheet with a wide margin. One box still covers two cells.
    if (
        _MID_COVER <= cover < _SHEET_COVER
        and _swallows(panels, cells)
        and ratio >= _MID_RATIO
        and median_fill >= _MID_FILL
        and matched_detections >= _MID_MATCH
        and len(cells) >= len(panels) - 2
    ):
        return True
    if not (
        cover >= _SHEET_COVER
        and ratio >= _SIZE_RATIO
        and matched_detections >= 0.5
        and len(cells) >= len(panels)
    ):
        return False
    if len(cells) == len(panels):
        if mean_iou < _POOR_MEAN:
            return True
        # Same count and the same cells, but the edges sit off the rules.
        aligned = matched_detections >= _ALIGNED_MATCH and matched_cells >= _ALIGNED_MATCH
        return mean_iou < _SNAP_MEAN and aligned
    if _swallows(panels, cells):
        return True
    unmatched = [index for index in range(len(cells)) if index not in set(assigned.values())]
    return (
        len(unmatched) >= 3
        and len(unmatched) >= len(assigned)
        and all(max(iou(cells[index], panel) for panel in panels) < _CLEAR_IOU for index in unmatched)
    )


def _interior_cell(
    panels: Sequence[Box],
    cells: Sequence[Box],
    fills: Sequence[float],
    page_w: int,
    page_h: int,
) -> Box | None:
    """One cell inside the sheet that no detection covers, when the grid agrees."""
    if len(cells) != len(panels) + 1 or len(cells) < _MIN_CELLS or not fills:
        return None
    if float(np.median(fills)) < _MEDIAN_FILL or not _disjoint(cells):
        return None
    areas = [cell.area for cell in cells]
    if min(areas) / max(areas) < 0.30 or sum(areas) / (page_w * page_h) < 0.65:
        return None
    assigned, _scores = _assign(panels, cells)
    unmatched = [index for index in range(len(cells)) if index not in set(assigned.values())]
    if len(unmatched) != 1 or len(assigned) < 3:
        return None
    cell = cells[unmatched[0]]
    if max((iou(cell, panel) for panel in panels), default=0.0) >= _CLEAR_IOU:
        return None
    if cell.x <= page_w * 0.04 or cell.y <= page_h * 0.04:
        return None
    if cell.x2 >= page_w * 0.96 or cell.y2 >= page_h * 0.96:
        return None
    return cell


def _disjoint(cells: Sequence[Box]) -> bool:
    for index, left in enumerate(cells):
        for right in cells[index + 1 :]:
            if iou(left, right) > _DISJOINT_IOU:
                return False
    return True


def _assign(panels: Sequence[Box], cells: Sequence[Box]) -> tuple[dict[int, int], list[float]]:
    assigned: dict[int, int] = {}
    scores: list[float] = []
    claimed: set[int] = set()
    for panel_index, panel in enumerate(panels):
        best: int | None = None
        best_score = 0.0
        for cell_index, cell in enumerate(cells):
            score = iou(panel, cell)
            if score > best_score:
                best_score = score
                best = cell_index
        if best is not None and best_score >= _MATCH_IOU and best not in claimed:
            assigned[panel_index] = best
            claimed.add(best)
            scores.append(best_score)
    return assigned, scores


def _swallows(panels: Sequence[Box], cells: Sequence[Box]) -> bool:
    for panel in panels:
        inside = sum(
            1 for cell in cells if contained_ratio(cell, panel) >= _SWALLOW and cell.area < panel.area * 0.92
        )
        if inside >= 2:
            return True
    return False
