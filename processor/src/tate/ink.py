"""Page ink used to keep every mark inside some segment.

Two signals are kept apart. Strokes are pixels darker than the paper, and they
are what a missed panel or a title column is made of. The evaluator also counts
flat paper that is simply darker than a fixed luminance, so once the panels are
chosen their crops grow until that broader ink is covered too.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from tate.boxes import Box, box_distance, union_box

_LONG_SIDE = 720
_MARGIN = 0.01


def cover_boxes(boxes: Sequence[Box], width: int, height: int) -> list[Box]:
    """Grow each box until the rectangles together cover the page.

    Every source pixel is given to the nearest box. The box then becomes the
    bounds of the pixels it owns, so gutters and the page margin stay in a
    segment instead of being dropped. Boxes are not shrunk.
    """
    if not boxes or width < 1 or height < 1:
        return [box.clamp(width, height) for box in boxes]
    step = max(1, int(max(width, height) / 480))
    ys = np.arange(0, height, step)
    xs = np.arange(0, width, step)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    best = np.full(grid_y.shape, np.inf, dtype=np.float32)
    labels = np.zeros(grid_y.shape, dtype=np.int16)
    for index, box in enumerate(boxes):
        dx = np.maximum(np.maximum(float(box.x) - grid_x, grid_x - float(box.x2 - 1)), 0.0)
        dy = np.maximum(np.maximum(float(box.y) - grid_y, grid_y - float(box.y2 - 1)), 0.0)
        dist = dx + dy
        closer = dist < best
        best = np.where(closer, dist, best)
        labels = np.where(closer, index, labels)
    covered: list[Box] = []
    for index, box in enumerate(boxes):
        selected = labels == index
        if not bool(np.any(selected)):
            covered.append(box.clamp(width, height))
            continue
        rows, cols = np.nonzero(selected)
        x1 = max(0, int(xs[int(cols.min())]) - step)
        y1 = max(0, int(ys[int(rows.min())]) - step)
        x2 = min(width, int(xs[int(cols.max())]) + step + 1)
        y2 = min(height, int(ys[int(rows.max())]) + step + 1)
        grown = Box(x1, y1, max(1, x2 - x1), max(1, y2 - y1), box.score)
        covered.append(union_box(box, grown).clamp(width, height))
    return covered


def recover_regions(image: Image.Image, panels: Sequence[Box]) -> list[Box]:
    """Panels the detector skipped: a block of strokes no current panel covers.

    Scan noise, the page edge, and the paper tone itself are ignored. A region
    whose box would swallow panels already found is discarded.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    page_h, page_w = int(rgb.shape[0]), int(rgb.shape[1])
    if page_h < 2 or page_w < 2 or not panels:
        return []
    stroke, scale = _stroke_mask(rgb)
    if stroke is None:
        return []
    mask_h, mask_w = int(stroke.shape[0]), int(stroke.shape[1])
    covered = np.zeros((mask_h, mask_w), dtype=np.uint8)
    for panel in panels:
        _fill_rect(covered, panel, scale, mask_w, mask_h, 255)
    # A one-pixel halo so a frame shared with a found panel is not a new one.
    halo = max(1, round(min(mask_h, mask_w) * 0.008))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (halo * 2 + 1, halo * 2 + 1))
    covered = cast(np.ndarray[Any, Any], cv2.dilate(covered, kernel))
    free = cast(np.ndarray[Any, Any], cv2.bitwise_and(stroke, cv2.bitwise_not(covered)))
    _clear_margin(free)
    close = max(3, round(min(mask_h, mask_w) * 0.02))
    if close % 2 == 0:
        close += 1
    closed = cast(
        np.ndarray[Any, Any],
        cv2.morphologyEx(free, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (close, close))),
    )
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    page_area = page_w * page_h
    found: list[Box] = []
    for label in range(1, int(count)):
        x, y, bw, bh, area = (int(value) for value in stats[label])
        if area < 8:
            continue
        box = _to_page(x, y, bw, bh, scale, page_w, page_h)
        grown = _grow_to_frame(stroke, x, y, bw, bh, covered, scale, page_w, page_h)
        box = grown if grown.area > box.area else box
        if box.area < page_area * 0.02:
            continue
        if box.w < page_w * 0.04 or box.h < page_h * 0.06:
            continue
        if box.area > page_area * 0.55:
            continue
        if _contains_center(box, panels):
            continue
        # Ink that merely spills out of a frame is not a new panel. A missed
        # panel sits across a gutter, with paper between it and its neighbours.
        gap = max(8, round(min(page_w, page_h) * 0.012))
        if any(box_distance(box, panel) < gap for panel in panels):
            continue
        if any(_same(box, other) for other in found):
            continue
        found.append(Box(box.x, box.y, box.w, box.h, 0.5))
    return found


def stroke_coverage(image: Image.Image, panels: Sequence[Box]) -> float | None:
    """Share of strokes inside ``panels``, or None when the page has no paper.

    A dark splash has no paper tone to measure strokes against. Callers should
    fall back to a fixed luminance check in that case.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    page_h, page_w = int(rgb.shape[0]), int(rgb.shape[1])
    if page_h < 2 or page_w < 2:
        return 1.0
    stroke, scale = _stroke_mask(rgb)
    if stroke is None:
        return None
    total = int(np.count_nonzero(stroke))
    if total < 30:
        return None
    mask = np.zeros(stroke.shape, dtype=np.uint8)
    mask_h, mask_w = int(stroke.shape[0]), int(stroke.shape[1])
    for panel in panels:
        _fill_rect(mask, panel, scale, mask_w, mask_h, 1)
    covered = int(np.count_nonzero((stroke > 0) & (mask > 0)))
    return covered / total


def _stroke_mask(rgb: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any] | None, float]:
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    scale = min(1.0, _LONG_SIDE / max(height, width))
    if scale < 1:
        small = cast(
            np.ndarray[Any, Any],
            cv2.resize(
                rgb,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            ),
        )
    else:
        small = rgb
    lum = (
        0.299 * small[:, :, 0].astype(np.float32)
        + 0.587 * small[:, :, 1].astype(np.float32)
        + 0.114 * small[:, :, 2].astype(np.float32)
    )
    paper = float(np.percentile(lum, 72))
    if paper < 110:
        # A dark splash has no paper to compare against. Only near-black marks
        # would count, and a flat black page must not become one giant panel.
        level = min(40.0, paper - 20.0)
        stroke = lum < level
    else:
        stroke = lum < (paper - 26.0)
    mask = stroke.astype(np.uint8) * 255
    if int(np.count_nonzero(mask)) < 12:
        return None, scale
    return mask, scale


def _grow_to_frame(
    stroke: np.ndarray[Any, Any],
    x: int,
    y: int,
    bw: int,
    bh: int,
    covered: np.ndarray[Any, Any],
    scale: float,
    page_w: int,
    page_h: int,
) -> Box:
    """Push each side out across blank paper and stop on a frame line or a panel."""
    mask_h, mask_w = int(stroke.shape[0]), int(stroke.shape[1])
    x1, y1, x2, y2 = x, y, x + bw, y + bh
    limit = max(2, round(min(mask_h, mask_w) * 0.03))
    binary = stroke > 0

    def wall(sliver: np.ndarray[Any, Any], blocked: np.ndarray[Any, Any]) -> bool:
        if sliver.size == 0 or blocked.size == 0:
            return True
        if float(blocked.mean()) > 0.25:
            return True
        return float(sliver.mean()) > 0.28

    for _ in range(limit):
        if y1 <= 0:
            break
        sliver = binary[y1 - 1, x1:x2]
        blocked = covered[y1 - 1, x1:x2] > 0
        if wall(sliver, blocked):
            break
        y1 -= 1
    for _ in range(limit):
        if y2 >= mask_h:
            break
        sliver = binary[y2, x1:x2]
        blocked = covered[y2, x1:x2] > 0
        if wall(sliver, blocked):
            break
        y2 += 1
    for _ in range(limit):
        if x1 <= 0:
            break
        sliver = binary[y1:y2, x1 - 1]
        blocked = covered[y1:y2, x1 - 1] > 0
        if wall(sliver, blocked):
            break
        x1 -= 1
    for _ in range(limit):
        if x2 >= mask_w:
            break
        sliver = binary[y1:y2, x2]
        blocked = covered[y1:y2, x2] > 0
        if wall(sliver, blocked):
            break
        x2 += 1
    return _to_page(x1, y1, max(1, x2 - x1), max(1, y2 - y1), scale, page_w, page_h)


def _to_page(x: int, y: int, bw: int, bh: int, scale: float, page_w: int, page_h: int) -> Box:
    if scale <= 0:
        return Box(0, 0, 0, 0)
    x1 = max(0, round(x / scale))
    y1 = max(0, round(y / scale))
    x2 = min(page_w, round((x + bw) / scale))
    y2 = min(page_h, round((y + bh) / scale))
    return Box(x1, y1, max(0, x2 - x1), max(0, y2 - y1))


def _fill_rect(
    mask: np.ndarray[Any, Any],
    box: Box,
    scale: float,
    width: int,
    height: int,
    value: int,
) -> None:
    x1 = min(max(int(box.x * scale), 0), width)
    y1 = min(max(int(box.y * scale), 0), height)
    x2 = min(max(int(np.ceil(box.x2 * scale)), 0), width)
    y2 = min(max(int(np.ceil(box.y2 * scale)), 0), height)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = value


def _clear_margin(mask: np.ndarray[Any, Any]) -> None:
    height, width = int(mask.shape[0]), int(mask.shape[1])
    margin_y = max(1, round(height * _MARGIN))
    margin_x = max(1, round(width * _MARGIN))
    mask[:margin_y, :] = 0
    mask[height - margin_y :, :] = 0
    mask[:, :margin_x] = 0
    mask[:, width - margin_x :] = 0


def _contains_center(box: Box, panels: Sequence[Box]) -> bool:
    for panel in panels:
        if panel.w < 1 or panel.h < 1:
            continue
        center_x = panel.x + panel.w / 2
        center_y = panel.y + panel.h / 2
        if box.x <= center_x < box.x2 and box.y <= center_y < box.y2:
            return True
    return False


def _same(left: Box, right: Box) -> bool:
    return abs(left.x - right.x) <= 4 and abs(left.y - right.y) <= 4 and abs(left.w - right.w) <= 4
