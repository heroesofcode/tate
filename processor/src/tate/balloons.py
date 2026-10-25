"""Speech balloons and caption lines that a crop must not slice.

The panel detector's text boxes cover the letters, not the outline or the tail.
A balloon is a light region closed by a dark outline, with text-like strokes
inside. A caption is a short line of those strokes sitting in a gutter. Either
one is given to a single panel so the crop holds the whole mark.
"""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from tate.boxes import Box

_LONG_SIDE = 720


def find_balloons(image: Image.Image) -> list[Box]:
    """Outlined balloons and gutter caption lines, in page pixels."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    page_h, page_w = int(rgb.shape[0]), int(rgb.shape[1])
    if page_h < 16 or page_w < 16:
        return []
    small, scale = _resize(rgb)
    lum = small.astype(np.float32).mean(axis=2)
    paper = float(np.percentile(lum, 72))
    if paper < 90:
        return []
    balloons = _outlined(lum, paper, scale, page_w, page_h)
    captions = _caption_lines(lum, paper, scale, page_w, page_h)
    return _merge(balloons + captions, page_w, page_h)


def _resize(rgb: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], float]:
    page_h, page_w = int(rgb.shape[0]), int(rgb.shape[1])
    scale = min(1.0, _LONG_SIDE / max(page_h, page_w))
    if scale >= 1:
        return rgb, 1.0
    size = (max(2, round(page_w * scale)), max(2, round(page_h * scale)))
    resized = cast(np.ndarray[Any, Any], cv2.resize(rgb, size, interpolation=cv2.INTER_AREA))
    return resized, scale


def _outlined(
    lum: np.ndarray[Any, Any],
    paper: float,
    scale: float,
    page_w: int,
    page_h: int,
) -> list[Box]:
    """Interiors sealed by an outline, grown out to that outline and its tail."""
    ink = (lum < paper - 20.0).astype(np.uint8) * 255
    kernel = _odd(max(3, round(min(int(lum.shape[0]), int(lum.shape[1])) * 0.007)))
    shape = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    closed = cast(np.ndarray[Any, Any], cv2.morphologyEx(ink, cv2.MORPH_CLOSE, shape))
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    links = hierarchy[0]
    page_area = page_w * page_h
    found: list[Box] = []
    for index, contour in enumerate(contours):
        if int(links[index][3]) < 0:
            continue
        box = _hole_box(lum, contour, paper, scale, page_w, page_h, page_area)
        if box is not None:
            found.append(box)
    return found


def _hole_box(
    lum: np.ndarray[Any, Any],
    contour: np.ndarray[Any, Any],
    paper: float,
    scale: float,
    page_w: int,
    page_h: int,
    page_area: int,
) -> Box | None:
    _x, _y, bw, bh = (int(value) for value in cv2.boundingRect(contour))
    if bw < 8 or bh < 8:
        return None
    area = float(cv2.contourArea(contour))
    if area < bw * bh * 0.40:
        return None
    rough = _to_page(0, 0, bw, bh, scale, page_w, page_h)
    if rough.area < page_area * 0.003 or rough.area > page_area * 0.14:
        return None
    if rough.w > page_w * 0.75 or rough.h > page_h * 0.50:
        return None
    seed = np.zeros(lum.shape, dtype=np.uint8)
    cv2.drawContours(seed, [contour], -1, 1, thickness=-1)
    inside = lum < paper - 20.0
    if float(inside[seed > 0].mean()) < 0.02:
        return None
    margin = round(min(int(lum.shape[0]), int(lum.shape[1])) * 0.07) + 2
    grown = _grow_near(seed, lum, paper, margin)
    box = _mask_box(grown, scale, page_w, page_h)
    if box is None or box.area > page_area * 0.16:
        return None
    if box.w > page_w * 0.80 or box.h > page_h * 0.55:
        return None
    return _pad(box, page_w, page_h, 0.012)


def _grow_near(
    seed: np.ndarray[Any, Any],
    lum: np.ndarray[Any, Any],
    paper: float,
    margin: int,
) -> np.ndarray[Any, Any]:
    """Grow one balloon on a window, not the whole page."""
    ys, xs = np.nonzero(seed)
    if int(xs.size) == 0:
        return seed
    height, width = int(lum.shape[0]), int(lum.shape[1])
    y1 = max(0, int(ys.min()) - margin)
    y2 = min(height, int(ys.max()) + margin + 1)
    x1 = max(0, int(xs.min()) - margin)
    x2 = min(width, int(xs.max()) + margin + 1)
    window = lum[y1:y2, x1:x2]
    local = seed[y1:y2, x1:x2]
    first = max(4, margin // 2)
    local = _grow(local, window < paper - 34.0, first)
    local = _grow(local, window < paper - 48.0, max(4, margin - first))
    grown = np.zeros(lum.shape, dtype=np.uint8)
    grown[y1:y2, x1:x2] = local
    return grown


def _grow(seed: np.ndarray[Any, Any], barrier: np.ndarray[Any, Any], steps: int) -> np.ndarray[Any, Any]:
    """Spread through paper, stopping at dark outline pixels."""
    if steps <= 0:
        return seed
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    grown = seed
    for _ in range(steps):
        nxt = cv2.dilate(grown, kernel)
        nxt = np.where(barrier, grown, nxt)
        if int(nxt.sum()) == int(grown.sum()):
            break
        grown = nxt
    return grown


def _mask_box(mask: np.ndarray[Any, Any], scale: float, page_w: int, page_h: int) -> Box | None:
    ys, xs = np.nonzero(mask)
    if int(xs.size) == 0:
        return None
    return _to_page(
        int(xs.min()),
        int(ys.min()),
        int(xs.max() - xs.min() + 1),
        int(ys.max() - ys.min() + 1),
        scale,
        page_w,
        page_h,
    )


def _caption_lines(
    lum: np.ndarray[Any, Any],
    paper: float,
    scale: float,
    page_w: int,
    page_h: int,
) -> list[Box]:
    """A short run of text strokes with paper immediately above and below."""
    ink = lum < paper - 28.0
    frac = ink.mean(axis=1)
    height, width = int(ink.shape[0]), int(ink.shape[1])
    limit = max(16, round(height * 0.09))
    found: list[Box] = []
    row = 0
    while row < height:
        if not (0.05 <= float(frac[row]) <= 0.72):
            row += 1
            continue
        end = row
        while end < height and 0.05 <= float(frac[end]) <= 0.72:
            end += 1
        above = float(frac[row - 1]) if row else 1.0
        below = float(frac[end]) if end < height else 1.0
        tall = end - row
        if above < 0.05 and below < 0.05 and 3 <= tall <= limit:
            band = ink[row:end]
            cols = np.flatnonzero(band.any(axis=0))
            if int(cols.size) and int(cols[-1] - cols[0]) > width * 0.08:
                box = _to_page(int(cols[0]), row, int(cols[-1] - cols[0] + 1), tall, scale, page_w, page_h)
                found.append(_pad(box, page_w, page_h, 0.01))
        row = end
    return found


def _to_page(x: int, y: int, bw: int, bh: int, scale: float, page_w: int, page_h: int) -> Box:
    if scale <= 0:
        return Box(0, 0, 0, 0)
    left = max(0, round(x / scale))
    top = max(0, round(y / scale))
    right = min(page_w, round((x + bw) / scale))
    bottom = min(page_h, round((y + bh) / scale))
    return Box(left, top, max(0, right - left), max(0, bottom - top))


def _pad(box: Box, page_w: int, page_h: int, frac: float) -> Box:
    pad = max(8, round(min(page_w, page_h) * frac))
    return Box(box.x - pad, box.y - pad, box.w + 2 * pad, box.h + 2 * pad).clamp(page_w, page_h)


def _merge(boxes: list[Box], page_w: int, page_h: int) -> list[Box]:
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda item: item.area, reverse=True):
        if box.w < 8 or box.h < 6:
            continue
        if any(_same(box, other) for other in kept):
            continue
        kept.append(box.clamp(page_w, page_h))
    return kept


def _same(a: Box, b: Box) -> bool:
    overlap_w = min(a.x2, b.x2) - max(a.x, b.x)
    overlap_h = min(a.y2, b.y2) - max(a.y, b.y)
    if overlap_w <= 0 or overlap_h <= 0:
        return False
    inter = overlap_w * overlap_h
    smaller = min(a.area, b.area)
    return smaller > 0 and inter / smaller > 0.6


def _odd(value: int) -> int:
    return value if value % 2 else value + 1
