"""Rectangles enclosed by frame lines, and the merge with other panel boxes.

A page edge counts as a wall, so a panel that bleeds off the scan still closes.
Guides are general: frame rectangles or gutter boxes. Nothing here is tied to a
particular comic.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from tate.boxes import Box, contained_ratio, intersection_area, iou

_GUIDE_SCORE = 0.5
_TARGET = 1100
# A guide this large is the page itself, not a panel.
_PAGE_GUIDE = 0.72
# Two boxes past this overlap are the same region counted twice.
_OVERLAP_IOU = 0.45
# Guides this far inside every side of a huge box are drawings on that picture,
# not the panels that structure the page.
_INSET_MARGIN = 0.08


def background_color(rgb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Median colour of the outer border. Scans pick up the page, not the art."""
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    border_y = max(1, round(height * 0.04))
    border_x = max(1, round(width * 0.04))
    strips = (
        rgb[:border_y, :, :],
        rgb[height - border_y :, :, :],
        rgb[:, :border_x, :],
        rgb[:, width - border_x :, :],
    )
    pixels = np.concatenate([strip.reshape(-1, 3) for strip in strips], axis=0)
    median = np.median(pixels.astype(np.float32), axis=0)
    return cast(np.ndarray[Any, Any], median)


def frame_panels(image: Image.Image) -> list[Box]:
    """Panels closed by long dark rules. Empty when the page has no such rules."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if height < 8 or width < 8:
        return []
    gray, scale = _line_gray(rgb)
    small_h, small_w = int(gray.shape[0]), int(gray.shape[1])
    paper = float(np.percentile(gray, 70))
    floor = float(np.percentile(gray, 8))
    drop = min(58.0, max(16.0, 0.48 * (paper - floor)))
    dark = (gray < paper - drop).astype(np.uint8) * 255
    if int(np.count_nonzero(dark)) < 40:
        return []
    min_frac = 0.08
    bridge = 2
    hlen = max(15, int(small_w * min_frac))
    vlen = max(15, int(small_h * min_frac))
    horiz = cast(
        np.ndarray[Any, Any],
        cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hlen, 1))),
    )
    vert = cast(
        np.ndarray[Any, Any],
        cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vlen))),
    )
    horiz = cast(
        np.ndarray[Any, Any],
        cv2.dilate(horiz, cv2.getStructuringElement(cv2.MORPH_RECT, (bridge * 2 + 1, bridge))),
    )
    vert = cast(
        np.ndarray[Any, Any],
        cv2.dilate(vert, cv2.getStructuringElement(cv2.MORPH_RECT, (bridge, bridge * 2 + 1))),
    )
    lines = cast(np.ndarray[Any, Any], cv2.bitwise_or(horiz, vert))
    # The scan edge closes a panel that bleeds off the page.
    lines[0, :] = 255
    lines[-1, :] = 255
    lines[:, 0] = 255
    lines[:, -1] = 255
    open_space = cast(np.ndarray[Any, Any], cv2.bitwise_not(lines))
    # External contours stop at the page border and miss the panels inside it.
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(open_space, connectivity=4)
    inv = 1.0 / scale
    page_area = height * width
    boxes: list[Box] = []
    for index in range(1, int(count)):
        x_s, y_s, bw_s, bh_s, area_s = (int(stats[index, column]) for column in range(5))
        if bw_s * bh_s < 0.012 * small_w * small_h or area_s < 0.55 * bw_s * bh_s:
            continue
        x = round(x_s * inv)
        y = round(y_s * inv)
        bw = max(1, round(bw_s * inv))
        bh = max(1, round(bh_s * inv))
        snap = max(2, round(3 * inv))
        if x <= snap:
            bw += x
            x = 0
        if y <= snap:
            bh += y
            y = 0
        if x + bw >= width - snap:
            bw = width - x
        if y + bh >= height - snap:
            bh = height - y
        box = Box(x, y, bw, bh, _GUIDE_SCORE).clamp(width, height)
        if box.area < 0.015 * page_area or box.w < width * 0.07 or box.h < height * 0.04:
            continue
        if box.w > width * 0.98 and box.h > height * 0.98:
            continue
        boxes.append(box)
    return _dedupe(boxes)


def fuse_panels(
    primary: Sequence[Box],
    guides: Sequence[Box],
    page_w: int,
    page_h: int,
) -> list[Box]:
    """Split a primary box that covers several guides, and add guides it missed.

    One guide overlapping one primary box does not replace that box. A split
    happens only when the guides tile that box. A detail drawn inside a panel,
    or a second copy of a region already split, stays out.
    """
    prepared = _prepare_guides(guides, page_w, page_h)
    if not prepared:
        return [box.clamp(page_w, page_h) for box in primary]
    if not primary:
        return prepared
    page_area = max(1, page_w * page_h)
    # A box around most of the page is not itself a panel. A partition or a
    # tiling replaces it. So does a set of guides that reaches the frame.
    # Guides that float inside the box are insets on a splash, and the splash
    # stays one panel.
    if len(primary) == 1 and primary[0].area > 0.65 * page_area:
        covers = len(prepared) >= 3 and sum(guide.area for guide in prepared) > 0.35 * page_area
        partitioned = _is_partition(primary[0], prepared) or _is_tiling(prepared, page_w, page_h)
        if partitioned or (covers and not _floats_inside(primary[0], prepared)):
            return prepared
    used = [False] * len(prepared)
    fused: list[Box] = []
    for box in primary:
        inside = [
            index
            for index, guide in enumerate(prepared)
            if not used[index] and guide.area < box.area * 0.92 and contained_ratio(guide, box) >= 0.72
        ]
        if _is_partition(box, [prepared[index] for index in inside]):
            for index in inside:
                used[index] = True
                fused.append(prepared[index])
            continue
        covered = [
            index
            for index, guide in enumerate(prepared)
            if used[index] and contained_ratio(guide, box) >= 0.72 and guide.area < box.area * 0.92
        ]
        covered_area = sum(intersection_area(prepared[index], box) for index in covered)
        if len(covered) >= 2 and covered_area >= box.area * 0.5:
            continue
        fused.append(box.clamp(page_w, page_h))
        for index, guide in enumerate(prepared):
            if iou(guide, box) >= 0.35 or contained_ratio(guide, box) >= 0.5:
                used[index] = True
    for index, guide in enumerate(prepared):
        if used[index]:
            continue
        if any(iou(guide, other) >= 0.2 or contained_ratio(guide, other) >= 0.4 for other in fused):
            continue
        center_x = guide.x + guide.w / 2
        center_y = guide.y + guide.h / 2
        if any(other.x <= center_x <= other.x2 and other.y <= center_y <= other.y2 for other in fused):
            continue
        fused.append(guide)
    return fused


def merge_panel_guides(
    primary: Sequence[Box],
    frames: Sequence[Box],
    gutters: Sequence[Box],
    page_w: int,
    page_h: int,
) -> list[Box]:
    """Frames split or fill gaps. A gutter tiling wins only when boxes overlap.

    When the model boxes are one region, a huge region, or a heavy overlap, an
    aligned set of frame rectangles replaces them.
    """
    override = _aligned_override(primary, frames, page_w, page_h)
    if override is not None:
        return override
    guides = list(frames) if primary else [*frames, *gutters]
    fused = fuse_panels(primary, guides, page_w, page_h)
    if not primary or not _pairs_overlap(fused, _OVERLAP_IOU):
        return fused
    tiling = tiling_panels(gutters, page_w, page_h)
    if tiling and len(tiling) + 1 >= len(fused):
        return tiling
    return fused


def tiling_panels(guides: Sequence[Box], page_w: int, page_h: int) -> list[Box]:
    """Guides that sit side by side and cover the page. Empty when they do not."""
    prepared = _prepare_guides(guides, page_w, page_h)
    if not _is_tiling(prepared, page_w, page_h):
        return []
    return prepared


def _is_partition(box: Box, guides: Sequence[Box]) -> bool:
    """True when the guides are siblings that fill the box, not details inside it."""
    if len(guides) < 2 or box.area <= 0:
        return False
    areas: list[int] = []
    for guide in guides:
        inter = intersection_area(guide, box)
        if guide.area <= 0 or inter < guide.area * 0.72:
            return False
        areas.append(inter)
    if min(areas) < 0.25 * max(areas):
        return False
    # The box is often a little larger than the frames it swallowed. The
    # leftover is that margin, not a third panel, so the bar stays under 1.
    if sum(areas) < 0.70 * box.area:
        return False
    for index, left in enumerate(guides):
        for right in guides[index + 1 :]:
            if iou(left, right) > 0.12:
                return False
            if contained_ratio(left, right) > 0.55 or contained_ratio(right, left) > 0.55:
                return False
    return True


def _floats_inside(box: Box, guides: Sequence[Box]) -> bool:
    """True when every guide sits in the interior, clear of the box edges."""
    if box.w <= 0 or box.h <= 0 or not guides:
        return False
    margin_x = box.w * _INSET_MARGIN
    margin_y = box.h * _INSET_MARGIN
    inner_x1 = box.x + margin_x
    inner_y1 = box.y + margin_y
    inner_x2 = box.x2 - margin_x
    inner_y2 = box.y2 - margin_y
    return all(
        guide.x >= inner_x1 and guide.y >= inner_y1 and guide.x2 <= inner_x2 and guide.y2 <= inner_y2
        for guide in guides
    )


def _is_tiling(boxes: Sequence[Box], page_w: int, page_h: int) -> bool:
    if len(boxes) < 2:
        return False
    page_area = max(1, page_w * page_h)
    if any(box.area > 0.45 * page_area for box in boxes):
        return False
    if sum(box.area for box in boxes) < 0.5 * page_area:
        return False
    for index, left in enumerate(boxes):
        for right in boxes[index + 1 :]:
            if iou(left, right) > 0.08:
                return False
            if contained_ratio(left, right) > 0.5 or contained_ratio(right, left) > 0.5:
                return False
    return True


def _aligned_override(
    primary: Sequence[Box],
    frames: Sequence[Box],
    page_w: int,
    page_h: int,
) -> list[Box] | None:
    tiling = tiling_panels(frames, page_w, page_h)
    if len(tiling) < 3 or len(tiling) < len(primary):
        return None
    if not _shares_edges(tiling, page_w, page_h):
        return None
    if not _detection_is_broken(primary, page_w, page_h):
        return None
    return tiling


def _shares_edges(boxes: Sequence[Box], page_w: int, page_h: int) -> bool:
    """True when the boxes line up in rows or columns, not as scattered insets."""
    if len(boxes) < 3:
        return False
    tol_x = max(8, int(page_w * 0.02))
    tol_y = max(8, int(page_h * 0.02))
    shares = 0
    for index, left in enumerate(boxes):
        for right in boxes[index + 1 :]:
            same_x = abs(left.x - right.x) <= tol_x or abs(left.x2 - right.x2) <= tol_x
            same_y = abs(left.y - right.y) <= tol_y or abs(left.y2 - right.y2) <= tol_y
            if same_x or same_y:
                shares += 1
    return shares >= len(boxes) - 1


def _detection_is_broken(panels: Sequence[Box], page_w: int, page_h: int) -> bool:
    if len(panels) <= 2:
        return True
    page_area = max(1, page_w * page_h)
    if any(panel.area > 0.40 * page_area for panel in panels):
        return True
    return _pairs_overlap(panels, _OVERLAP_IOU)


def _pairs_overlap(boxes: Sequence[Box], threshold: float) -> bool:
    for index, left in enumerate(boxes):
        for right in boxes[index + 1 :]:
            if iou(left, right) > threshold:
                return True
    return False


def _line_gray(rgb: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], float]:
    """Grayscale shrunk so a one-pixel rule survives the resize."""
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    gray = cast(np.ndarray[Any, Any], cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    scale = min(1.0, _TARGET / max(height, width))
    if scale >= 1:
        return gray, 1.0
    kernel = max(1, round(1 / scale))
    if kernel > 1:
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


def _prepare_guides(guides: Sequence[Box], page_w: int, page_h: int) -> list[Box]:
    page_area = max(1, page_w * page_h)
    kept: list[Box] = []
    for guide in guides:
        box = guide.clamp(page_w, page_h)
        if box.w < page_w * 0.07 or box.h < page_h * 0.04 or box.area < page_area * 0.015:
            continue
        if box.area > page_area * _PAGE_GUIDE:
            continue
        if box.w > page_w * 0.98 and box.h > page_h * 0.98:
            continue
        short = min(box.w, box.h)
        long = max(box.w, box.h)
        if short <= 0 or long / short > 6:
            continue
        kept.append(Box(box.x, box.y, box.w, box.h, _GUIDE_SCORE))
    return _absorb_insets(_dedupe(kept), page_w, page_h)


def _dedupe(boxes: Sequence[Box]) -> list[Box]:
    ranked = sorted(boxes, key=lambda box: box.area, reverse=True)
    kept: list[Box] = []
    for box in ranked:
        if any(iou(box, other) > 0.5 for other in kept):
            continue
        kept.append(box)
    return kept


def _absorb_insets(boxes: Sequence[Box], page_w: int, page_h: int) -> list[Box]:
    """Fold a caption box that shares an edge into the panel it is attached to."""
    current = list(boxes)
    changed = True
    while changed and len(current) > 1:
        changed = False
        current.sort(key=lambda box: box.area, reverse=True)
        for big_index, big in enumerate(current):
            for small_index in range(big_index + 1, len(current)):
                small = current[small_index]
                if small.area > big.area * 0.18:
                    continue
                thin = small.h < big.h * 0.42 or small.w < big.w * 0.42
                if not thin or not _shares_edge(small, big, page_w, page_h):
                    continue
                current[big_index] = _union(big, small)
                del current[small_index]
                changed = True
                break
            if changed:
                break
    return current


def _shares_edge(small: Box, big: Box, page_w: int, page_h: int) -> bool:
    tol_x = max(6, int(page_w * 0.012))
    tol_y = max(6, int(page_h * 0.012))
    y_overlap = _overlap(small.y, small.y2, big.y, big.y2) / max(1, small.h)
    x_overlap = _overlap(small.x, small.x2, big.x, big.x2) / max(1, small.w)
    touch_x = abs(small.x2 - big.x) <= tol_x or abs(big.x2 - small.x) <= tol_x
    touch_y = abs(small.y2 - big.y) <= tol_y or abs(big.y2 - small.y) <= tol_y
    return (touch_x and y_overlap > 0.65) or (touch_y and x_overlap > 0.65)


def _overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _union(a: Box, b: Box) -> Box:
    x1 = min(a.x, b.x)
    y1 = min(a.y, b.y)
    x2 = max(a.x2, b.x2)
    y2 = max(a.y2, b.y2)
    return Box(x1, y1, x2 - x1, y2 - y1, a.score)
