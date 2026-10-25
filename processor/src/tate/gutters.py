"""Split a box that covers several panels along the gutters between them.

A gutter is either a band of paper with drawings on both sides, or a thin dark
rule whose shoulders are paper. A hairline white gap is not a gutter. A cut
splits a box only when that gutter crosses the box, edge to edge apart from a
short tolerance, so a rule through one column does not slice the panel beside
it. Cuts are searched inside one row at a time, because a vertical rule usually
stops at the horizontal gutter. Nothing here is tuned to a particular page: a
cut is kept only when both pieces are large and inky.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
from PIL import Image

from tate.boxes import Box

_LONG_SIDE = 720


def split_on_gutters(image: Image.Image, panels: Sequence[Box]) -> list[Box]:
    """Replace any panel that spans a gutter with one box per cell."""
    page_w, page_h = image.size
    if page_w < 8 or page_h < 8 or not panels:
        return [panel.clamp(page_w, page_h) for panel in panels]
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lum, scale = _resize(rgb)
    paper = float(np.percentile(lum, 72))
    dark = _dark(lum, paper)
    y_cuts = _cuts(dark, lum, paper, scale, page_h, axis=0)
    pieces: list[Box] = []
    for panel in panels:
        box = panel.clamp(page_w, page_h)
        if box.w < 2 or box.h < 2:
            continue
        for band in _apply(box, y_cuts, "y", page_w, page_h, dark, lum, paper, scale):
            x_cuts = _cuts(dark, lum, paper, scale, page_w, axis=1, limit=band)
            pieces.extend(_apply(band, x_cuts, "x", page_w, page_h, dark, lum, paper, scale))
    return pieces


def _resize(rgb: np.ndarray[Any, Any]) -> tuple[np.ndarray[Any, Any], float]:
    page_h, page_w = int(rgb.shape[0]), int(rgb.shape[1])
    scale = _LONG_SIDE / max(page_h, page_w)
    if scale > 1:
        scale = 1.0
    size = (max(2, round(page_w * scale)), max(2, round(page_h * scale)))
    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    lum = resized.astype(np.float32).mean(axis=2)
    return lum, scale


def _dark(lum: np.ndarray[Any, Any], paper: float) -> np.ndarray[Any, Any]:
    level = min(40.0, max(1.0, paper - 20.0)) if paper < 110 else paper - 24.0
    return np.asarray(lum < level)


def _cuts(
    dark: np.ndarray[Any, Any],
    lum: np.ndarray[Any, Any],
    paper: float,
    scale: float,
    page_span: int,
    *,
    axis: int,
    limit: Box | None = None,
) -> list[int]:
    """Gutter positions along ``axis`` (0 = y, 1 = x), in page pixels."""
    view_dark, view_lum, origin = _window(dark, lum, scale, axis, limit)
    if view_dark.size == 0 or int(view_dark.shape[axis]) < 8:
        return []
    frac = view_dark.mean(axis=1 - axis)
    valleys = _valley_mask(frac)
    mask_h = int(view_dark.shape[0])
    mask_w = int(view_dark.shape[1])
    span = mask_h if axis == 0 else mask_w
    found = _clusters(valleys, frac, span, scale, page_span, kind="valley")
    found.extend(_shoulder_rules(view_lum, view_dark, paper, axis, scale, page_span))
    found.extend(_frame_rules(view_lum, paper, axis))
    cuts = sorted({origin + round(index / scale) for index in found})
    return _dedupe(cuts, max(8, round(page_span * 0.03)))


def _window(
    dark: np.ndarray[Any, Any],
    lum: np.ndarray[Any, Any],
    scale: float,
    axis: int,
    limit: Box | None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], int]:
    if limit is None or axis == 0:
        return dark, lum, 0
    y1 = _index(limit.y, scale, int(dark.shape[0]))
    y2 = _index(limit.y2, scale, int(dark.shape[0]))
    if y2 - y1 < 4:
        return dark, lum, 0
    return dark[y1:y2], lum[y1:y2], 0


def _index(pixel: int, scale: float, limit: int) -> int:
    return min(limit, max(0, round(pixel * scale)))


def _valley_mask(frac: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.asarray(frac < 0.07)


def _rule_mask(lum: np.ndarray[Any, Any], paper: float, axis: int) -> np.ndarray[Any, Any]:
    mask_h, mask_w = int(lum.shape[0]), int(lum.shape[1])
    k = max(2, round(min(mask_h, mask_w) * 0.008))
    if axis == 0:
        before = np.pad(lum, ((k, 0), (0, 0)), mode="edge")[:-k]
        after = np.pad(lum, ((0, k), (0, 0)), mode="edge")[k:]
        both = (before - lum > 8) & (after - lum > 8) & (lum < paper - 12)
        return np.asarray(both.mean(axis=1) > 0.7)
    before = np.pad(lum, ((0, 0), (k, 0)), mode="edge")[:, :-k]
    after = np.pad(lum, ((0, 0), (0, k)), mode="edge")[:, k:]
    both = (before - lum > 8) & (after - lum > 8) & (lum < paper - 12)
    return np.asarray(both.mean(axis=0) > 0.7)


def _shoulder_rules(
    lum: np.ndarray[Any, Any],
    dark: np.ndarray[Any, Any],
    paper: float,
    axis: int,
    scale: float,
    page_span: int,
) -> list[int]:
    """A rule counts only when the pixels beside it are paper, not more drawing."""
    response = _rule_mask(lum, paper, axis)
    frac = dark.mean(axis=1 - axis)
    span = int(response.shape[0])
    gap = max(2, round(span * 0.012))
    kept: list[int] = []
    for index in _clusters(response, frac, span, scale, page_span, kind="rule"):
        start = max(0, index - gap)
        end = min(span, index + gap + 1)
        if axis == 0:
            above = lum[start:index]
            below = lum[index + 1 : end]
            shoulders = (above.size and float((above > paper - 14).mean()) >= 0.62) and (
                below.size and float((below > paper - 14).mean()) >= 0.62
            )
        else:
            left = lum[:, start:index]
            right = lum[:, index + 1 : end]
            shoulders = (left.size and float((left > paper - 14).mean()) >= 0.62) and (
                right.size and float((right > paper - 14).mean()) >= 0.62
            )
        if shoulders:
            kept.append(index)
    return kept


def _frame_rules(lum: np.ndarray[Any, Any], paper: float, axis: int) -> list[int]:
    """A thin dark rule with paper on both sides, even when the rule is thick.

    The ridge test looks only a few pixels away, so a rule of several pixels
    reads as flat black and is missed. This measures the dark column itself
    and the paper beside it.
    """
    dark = lum < paper - 30.0
    if axis == 0:
        frac = dark.mean(axis=1)
        span = int(frac.shape[0])
    else:
        frac = dark.mean(axis=0)
        span = int(frac.shape[0])
    gap = max(3, round(span * 0.02))
    found: list[int] = []
    index = 0
    while index < span:
        if float(frac[index]) <= 0.52:
            index += 1
            continue
        end = index
        while end < span and float(frac[end]) > 0.52:
            end += 1
        thick = end - index
        interior = 1 <= thick <= 6 and index > gap and end < span - gap
        if interior and _paper_shoulders(lum, paper, axis, index, end, gap):
            found.append((index + end) // 2)
        index = end
    return found


def _paper_shoulders(
    lum: np.ndarray[Any, Any],
    paper: float,
    axis: int,
    start: int,
    end: int,
    gap: int,
) -> bool:
    level = paper - 18.0
    if axis == 0:
        above = lum[max(0, start - gap) : start]
        below = lum[end : end + gap]
        return _mostly_paper(above, level) and _mostly_paper(below, level)
    left = lum[:, max(0, start - gap) : start]
    right = lum[:, end : end + gap]
    return _mostly_paper(left, level) and _mostly_paper(right, level)


def _mostly_paper(patch: np.ndarray[Any, Any], level: float) -> bool:
    if patch.size == 0:
        return False
    return float((patch > level).mean()) >= 0.60


def _clusters(
    mask: np.ndarray[Any, Any],
    frac: np.ndarray[Any, Any],
    span: int,
    scale: float,
    page_span: int,
    *,
    kind: str,
) -> list[int]:
    minimum = 1
    maximum = max(3, round(page_span * (0.07 if kind == "valley" else 0.025) * scale))
    margin = max(2, round(span * 0.04))
    side = max(4, round(span * 0.025))
    found: list[int] = []
    index = 0
    while index < span:
        if not bool(mask[index]):
            index += 1
            continue
        end = index
        while end < span and bool(mask[end]):
            end += 1
        thick = end - index
        if minimum <= thick <= maximum and index > margin and end < span - margin:
            center = (index + end) // 2
            above = float(frac[max(0, index - side) : index].mean()) if index else 0.0
            below = float(frac[end : min(span, end + side)].mean()) if end < span else 0.0
            if _valley_shoulders(kind, thick, above, below):
                found.append(center)
        index = end
    return found


def _valley_shoulders(kind: str, thick: int, above: float, below: float) -> bool:
    """A white gap needs drawings on both sides.

    A gap only a few pixels wide also needs one solid drawing beside it, so a
    scratch inside a picture is not a gutter. A wider band of paper can sit
    next to sparser line art.
    """
    if kind != "valley":
        return above >= 0.06 and below >= 0.06
    if thick <= 3:
        return min(above, below) >= 0.15 and max(above, below) >= 0.55
    return above >= 0.15 and below >= 0.15


def _dedupe(cuts: list[int], gap: int) -> list[int]:
    kept: list[int] = []
    for cut in cuts:
        if not kept or cut - kept[-1] >= gap:
            kept.append(cut)
        else:
            kept[-1] = (kept[-1] + cut) // 2
    return kept


def _crosses_box(
    lum: np.ndarray[Any, Any],
    dark: np.ndarray[Any, Any],
    paper: float,
    box: Box,
    axis: str,
    cut: int,
    scale: float,
) -> bool:
    """True when a frame rule or a paper gap at ``cut`` crosses ``box``.

    The mark is measured inside the box, not on the rest of the page. It may
    stop a little short of each edge. ``axis`` ``"y"`` is a horizontal line.
    """
    height, width = int(lum.shape[0]), int(lum.shape[1])
    y1 = _index(box.y, scale, height)
    y2 = _index(box.y2, scale, height)
    x1 = _index(box.x, scale, width)
    x2 = _index(box.x2, scale, width)
    if y2 - y1 < 8 or x2 - x1 < 8:
        return False
    if axis == "y":
        local = _index(cut, scale, height) - y1
        view = lum[y1:y2, x1:x2]
        ink = dark[y1:y2, x1:x2]
        page_span = height
    else:
        local = _index(cut, scale, width) - x1
        view = np.transpose(lum[y1:y2, x1:x2])
        ink = np.transpose(dark[y1:y2, x1:x2])
        page_span = width
    if view.size == 0 or not (-2 <= local <= int(view.shape[0]) + 1):
        return False
    local = min(max(local, 0), int(view.shape[0]) - 1)
    return _paper_gap_crosses(view, ink, local, page_span) or _rule_crosses(view, paper, local)


def _spans(flags: np.ndarray[Any, Any], *, cover: float = 0.72) -> bool:
    """True when a mark reaches both ends of a line, apart from a short edge margin.

    The margin is an eighth of the line. A rule under a title, or a gutter
    inside a box that includes a blank border, still counts. A mark that stops
    in the middle of the line does not.
    """
    span = int(flags.shape[0])
    if span < 8:
        return False
    tolerance = max(2, round(span * 0.12))
    hits = np.flatnonzero(flags)
    if hits.size == 0:
        return False
    if int(hits[0]) > tolerance or int(hits[-1]) < span - 1 - tolerance:
        return False
    core = flags[tolerance:-tolerance] if span > 2 * tolerance + 4 else flags
    if int(core.shape[0]) < 4:
        return False
    return float(core.mean()) >= cover


def _paper_gap_crosses(
    view: np.ndarray[Any, Any],
    ink: np.ndarray[Any, Any],
    local: int,
    page_span: int,
) -> bool:
    """A band of paper across the view, with drawings on both sides."""
    span = int(view.shape[0])
    open_row = ~ink
    frac = ink.mean(axis=1)
    # A gutter row is mostly paper and reaches both ends. Inside one box the
    # line can be a little darker than the page-wide valley test.
    candidates = [
        row
        for row in range(max(0, local - 6), min(span, local + 7))
        if float(frac[row]) < 0.17 and _spans(open_row[row])
    ]
    candidates.sort(key=lambda row: (abs(row - local), row))
    limit = max(3, round(page_span * 0.07))
    side = max(4, round(span * 0.025))
    for origin in candidates:
        start = origin
        while start > 0 and float(frac[start - 1]) < 0.17 and _spans(open_row[start - 1]):
            start -= 1
        end = origin + 1
        while end < span and float(frac[end]) < 0.17 and _spans(open_row[end]):
            end += 1
        # The page-level cut can sit on the rule beside the gap.
        if not start - 4 <= local < end + 4:
            continue
        thick = end - start
        if thick < 1 or thick > limit:
            continue
        above = ink[max(0, start - side) : start]
        below = ink[end : min(span, end + side)]
        if above.size == 0 or below.size == 0:
            continue
        # The cut was already a gutter on a wider view. Drawings only have to
        # sit on both sides inside this box, so a light panel still counts.
        if float(above.mean()) >= 0.12 and float(below.mean()) >= 0.12:
            return True
    return False


def _rule_crosses(view: np.ndarray[Any, Any], paper: float, local: int) -> bool:
    """A thin dark rule, or a ridge, with paper on both sides, across the view."""
    span = int(view.shape[0])
    band = 14
    lo = max(0, local - band)
    hi = min(span, local + band + 1)
    ruled = [_row_is_rule(view, paper, row) for row in range(lo, hi)]
    origin = local - lo
    near: int | None = None
    for delta in range(5):
        for candidate in (origin - delta, origin + delta):
            if 0 <= candidate < len(ruled) and ruled[candidate]:
                near = candidate
                break
        if near is not None:
            break
    if near is None:
        return False
    start = near
    while start > 0 and ruled[start - 1]:
        start -= 1
    end = near + 1
    while end < len(ruled) and ruled[end]:
        end += 1
    # A run that reaches the edge of this window continues into a dark field.
    if (start == 0 and lo > 0) or (end == len(ruled) and hi < span):
        return False
    thick = end - start
    if not 1 <= thick <= 8:
        return False
    gap = max(3, round(span * 0.02))
    return _paper_shoulders(view, paper, 0, lo + start, lo + end, gap)


def _row_is_rule(view: np.ndarray[Any, Any], paper: float, row: int) -> bool:
    line = view[row]
    # A frame rule is thinner than a paper gutter, so a broken line still counts.
    if _spans(line < paper - 30.0, cover=0.55):
        return True
    height, width = int(view.shape[0]), int(view.shape[1])
    k = max(2, round(min(height, width) * 0.008))
    if row - k < 0 or row + k >= height:
        return False
    ridge = (view[row - k] - line > 8) & (view[row + k] - line > 8) & (line < paper - 12)
    return _spans(ridge, cover=0.55)


def _apply(
    box: Box,
    cuts: Sequence[int],
    axis: str,
    page_w: int,
    page_h: int,
    dark: np.ndarray[Any, Any],
    lum: np.ndarray[Any, Any],
    paper: float,
    scale: float,
) -> list[Box]:
    page_span = page_h if axis == "y" else page_w
    span = box.h if axis == "y" else box.w
    start = box.y if axis == "y" else box.x
    end = box.y2 if axis == "y" else box.x2
    # A single wide panel is not a row of panels. Split only a box that is
    # large on both axes, so a frame rule inside one panel stays put.
    cross = box.w / page_w if axis == "y" else box.h / page_h
    if span < page_span * 0.34 or cross < 0.32:
        return [box]
    inset = max(round(span * 0.14), round(page_span * 0.05))
    # The candidate was measured on a wider view. Keep it only when the same
    # kind of gutter runs across this box, so a column rule cannot cut its neighbour.
    interior = [
        cut
        for cut in cuts
        if start + inset < cut < end - inset and _crosses_box(lum, dark, paper, box, axis, cut, scale)
    ]
    if not interior:
        return [box]
    edges = [start]
    minimum = round(page_span * 0.13)
    for cut in interior:
        if cut - edges[-1] >= minimum and end - cut >= minimum:
            edges.append(cut)
    edges.append(end)
    if len(edges) < 3:
        return [box]
    parts: list[Box] = []
    for left, right in itertools.pairwise(edges):
        if axis == "y":
            part = Box(box.x, left, box.w, right - left, box.score)
        else:
            part = Box(left, box.y, right - left, box.h, box.score)
        if _ink(dark, part, scale) < 0.04:
            return [box]
        parts.append(part)
    return parts


def _ink(dark: np.ndarray[Any, Any], box: Box, scale: float) -> float:
    y1 = _index(box.y, scale, int(dark.shape[0]))
    y2 = _index(box.y2, scale, int(dark.shape[0]))
    x1 = _index(box.x, scale, int(dark.shape[1]))
    x2 = _index(box.x2, scale, int(dark.shape[1]))
    if y2 <= y1 or x2 <= x1:
        return 0.0
    return float(dark[y1:y2, x1:x2].mean())
