"""Split a box that covers several panels along the gutters between them.

A gutter is a band of paper with drawings on both sides, a bright strip of this
page's own sheet when that colour is not the border sample, or a thin dark rule
whose shoulders are paper. The line has to cross the whole box, edge to edge
apart from a short tolerance, so a rule through one column does not slice the
panel beside it. A stroke that stops inside the drawing, a hairline white gap,
a grey strip between strokes, and a cut that would leave a thin sliver are not
gutters. Cuts are searched inside one row at a time, because a vertical rule
usually stops at the horizontal gutter. Nothing here is tuned to a particular
page: a cut is kept only when both pieces are large and inky.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from tate.boxes import Box, union_box

_LONG_SIDE = 720


def page_view(image: Image.Image) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float]:
    """RGB, paper mask, and sheet light for ``image``, reused on the same page."""
    found = image.__dict__.get("_tate_view")
    if isinstance(found, tuple) and len(found) == 3:
        return cast(tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], float], found)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    view = (rgb, paper_mask(rgb), _sheet_light(rgb))
    image.__dict__["_tate_view"] = view
    return view


def split_on_gutters(image: Image.Image, panels: Sequence[Box]) -> list[Box]:
    """Replace any panel that spans a gutter with one box per cell."""
    page_w, page_h = image.size
    if page_w < 8 or page_h < 8 or not panels:
        return [panel.clamp(page_w, page_h) for panel in panels]
    rgb, sheet, light = page_view(image)
    lum, scale = _resize(rgb)
    paper = float(np.percentile(lum, 72))
    dark = _dark(lum, paper)
    y_cuts = _cuts(dark, lum, paper, scale, page_h, axis=0)
    pieces: list[Box] = []
    for panel in panels:
        box = panel.clamp(page_w, page_h)
        if box.w < 2 or box.h < 2:
            continue
        for band in _apply(box, y_cuts, "y", page_w, page_h, dark, lum, paper, scale, rgb, sheet, light):
            x_cuts = _cuts(dark, lum, paper, scale, page_w, axis=1, limit=band)
            pieces.extend(
                _apply(band, x_cuts, "x", page_w, page_h, dark, lum, paper, scale, rgb, sheet, light)
            )
    return pieces


def paper_mask(rgb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Pixels that are this page's paper, including a light sheet inside a dark margin."""
    from tate.detect.frames import background_color

    background = background_color(rgb)
    distance = np.sqrt(np.square(rgb.astype(np.float32) - background).sum(axis=2))
    near_margin = distance < 30.0
    luminance = _luminance(background)
    if luminance >= 105.0:
        return np.asarray(near_margin)
    # The outer scan is dark, so the sheet is the light colour in the gutters.
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    y0, y1 = height // 8, max(height // 8 + 1, height - height // 8)
    x0, x1 = width // 8, max(width // 8 + 1, width - width // 8)
    inner = rgb[y0:y1, x0:x1].astype(np.float32).mean(axis=2)
    bright = inner[inner > 175.0]
    if bright.size < inner.size * 0.015:
        return np.asarray(near_margin)
    sheet = float(np.median(bright))
    lum = rgb.astype(np.float32).mean(axis=2)
    return np.asarray(near_margin | (lum >= sheet - 22.0))


def full_span_divider(
    rgb: np.ndarray[Any, Any],
    sheet: np.ndarray[Any, Any],
    box: Box,
    axis: str,
    cut: int,
    *,
    allow_ink_rule: bool = False,
    page_light: float | None = None,
    search: int = 48,
) -> bool:
    """True when a frame rule or a paper gap at ``cut`` crosses all of ``box``.

    ``axis`` ``"x"`` is a vertical line. ``"y"`` is a horizontal line. The line
    may stop a little short of the box edge. A paper gap, a bright sheet that is
    not the border sample, or a thin dark rule with paper on both sides, counts.
    ``allow_ink_rule`` also accepts a thin dark band that reaches both edges,
    which is how a coloured panel is framed when the paint goes up to the rule.
    ``search`` is how far, in page pixels, that mark may sit from ``cut``.
    """
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    x1, y1 = max(0, box.x), max(0, box.y)
    x2, y2 = min(width, box.x2), min(height, box.y2)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return False
    light = _sheet_light(rgb) if page_light is None else page_light
    band = max(1, search)
    walk = 56
    if axis == "x":
        lo = max(0, cut - band - walk)
        hi = min(width, cut + band + walk + 1)
        if hi - lo < 8 or not (lo <= cut < hi):
            return False
        view_sheet = sheet[y1:y2, lo:hi]
        view_lum = rgb[y1:y2, lo:hi].astype(np.float32).mean(axis=2)
        return _divider_view(view_sheet, view_lum, cut - lo, band, allow_ink_rule, light)
    lo = max(0, cut - band - walk)
    hi = min(height, cut + band + walk + 1)
    if hi - lo < 8 or not (lo <= cut < hi):
        return False
    view_sheet = np.transpose(sheet[lo:hi, x1:x2])
    view_lum = np.transpose(rgb[lo:hi, x1:x2].astype(np.float32).mean(axis=2))
    return _divider_view(view_sheet, view_lum, cut - lo, band, allow_ink_rule, light)


def absorb_slivers(image: Image.Image, panels: Sequence[Box]) -> list[Box]:
    """Fold a thin fragment into the neighbouring panel when no gutter divides them.

    A split that leaves a piece much thinner than a panel is rejected. A narrow
    cell that really is its own panel stays, because a frame or a paper gap
    crosses the whole edge between it and its neighbour. Boxes that only touch,
    or that overlap, are held apart the same way: the mark is sought in a short
    band around that shared edge.
    """
    page_w, page_h = image.size
    current = [panel.clamp(page_w, page_h) for panel in panels if panel.w >= 2 and panel.h >= 2]
    if page_w < 8 or page_h < 8 or len(current) < 2:
        return current
    rgb, sheet, light = page_view(image)
    lum, scale = _resize(rgb)
    paper = float(np.percentile(lum, 72))
    dark = _dark(lum, paper)
    changed = True
    while changed and len(current) > 1:
        changed = False
        order = sorted(range(len(current)), key=lambda index: current[index].area)
        for index in order:
            host = _sliver_host(current, index, page_w, page_h)
            if host is None:
                continue
            small, big = current[index], current[host]
            axis = _thin_axis(small, big, page_w, page_h)
            if axis is None:
                continue
            parent = union_box(big, small)
            if _gutter_between(
                rgb, sheet, light, lum, dark, paper, scale, parent, small, big, axis, page_w, page_h
            ):
                continue
            current[host] = parent
            del current[index]
            changed = True
            break
    return current


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


def _luminance(pixel: np.ndarray[Any, Any]) -> float:
    return float(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2])


def _sheet_light(rgb: np.ndarray[Any, Any]) -> float:
    """The page's light sheet. A tan border does not hide a brighter interior.

    The level is the bright end of the page, not a mid tone. A grey strip
    between two strokes stays below it and is not a gutter. When the border is
    already that light colour, the border itself is the sheet.
    """
    from tate.detect.frames import background_color

    sample = rgb[::8, ::8]
    if sample.size == 0:
        return 180.0
    bright = float(np.percentile(sample.astype(np.float32).mean(axis=2), 90))
    border = _luminance(background_color(rgb))
    if bright >= border + 18.0:
        return bright
    return max(bright, border)


def _divider_view(
    sheet: np.ndarray[Any, Any],
    lum: np.ndarray[Any, Any],
    local_cut: int,
    search: int,
    allow_ink_rule: bool,
    page_light: float,
) -> bool:
    """``sheet`` and ``lum`` are shaped (span, along). ``local_cut`` is an index on along."""
    span = int(sheet.shape[0])
    along = int(sheet.shape[1])
    if span < 8 or along < 3:
        return False
    start = max(0, local_cut - search)
    end = min(along, local_cut + search + 1)
    window = sheet[:, start:end]
    if window.size == 0:
        return False
    spanning = _span_columns(window)
    run = _longest_run([bool(flag) for flag in spanning])
    if run is not None and run[1] - run[0] >= 2 and _ink_both_sides(sheet, start + run[0], start + run[1]):
        return True
    if _light_gap_view(lum, start, end, page_light):
        return True
    if _frame_rule_view(sheet, lum, start, end):
        return True
    return allow_ink_rule and _ink_rule_view(lum, start, end)


def _span_columns(flags: np.ndarray[Any, Any], coverage: float = 0.72) -> np.ndarray[Any, Any]:
    """Which columns of ``flags`` (span, along) reach both ends."""
    span = int(flags.shape[0])
    along = int(flags.shape[1]) if flags.ndim == 2 else 1
    if flags.ndim == 1:
        flags = flags[:, None]
    if span < 8 or along == 0:
        return np.zeros(along, dtype=bool)
    tolerance = max(2, round(span * 0.055))
    core = flags[tolerance:-tolerance] if span > 2 * tolerance + 4 else flags
    length = int(core.shape[0])
    if length < 4:
        return np.zeros(along, dtype=bool)
    edge = max(1, round(length * 0.03))
    covered = core.mean(axis=0) >= coverage
    ends = np.any(core[:edge], axis=0) & np.any(core[-edge:], axis=0)
    return np.asarray(covered & ends)


def _longest_run(flags: Sequence[bool]) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    index = 0
    count = len(flags)
    while index < count:
        if not flags[index]:
            index += 1
            continue
        end = index
        while end < count and flags[end]:
            end += 1
        if best is None or end - index > best[1] - best[0]:
            best = (index, end)
        index = end
    return best


def _ink_both_sides(sheet: np.ndarray[Any, Any], run_start: int, run_end: int) -> bool:
    """Drawings sit on both sides of a paper run. A blank margin does not count."""
    along = int(sheet.shape[1])

    def inky(origin: int, direction: int) -> bool:
        for delta in range(1, 57):
            column = origin + direction * delta
            if column < 0 or column >= along:
                return False
            if float((~sheet[:, column]).mean()) >= 0.10:
                return True
        return False

    return inky(run_start - 1, -1) and inky(run_end, 1)


def _light_gap_view(lum: np.ndarray[Any, Any], start: int, end: int, page_light: float) -> bool:
    """A bright band of the sheet across the whole span, with drawing on both sides.

    The band has to be as light as the page's sheet, so a grey strip between two
    strokes is not a gutter. It still counts when that light colour is not the
    border sample. A light panel fills the search window and is not a gutter;
    the band has to be a strip with an edge on both sides. The strip stays
    narrow: a wide light field is the panel, not the gutter, and treating it as
    one makes a classical page fall back to a single crop.
    """
    if end - start < 5:
        return False
    thresh = page_light - 12.0
    flags = _span_columns(lum[:, start:end] > thresh, 0.90)
    index = 0
    while index < len(flags):
        if not bool(flags[index]):
            index += 1
            continue
        end_run = index
        while end_run < len(flags) and bool(flags[end_run]):
            end_run += 1
        width = end_run - index
        bounded = index > 0 and end_run < len(flags)
        if (
            2 <= width <= 48
            and bounded
            and _dark_both_sides(lum, start + index, start + end_run, thresh - 28.0)
        ):
            return True
        index = end_run
    return False


def _dark_both_sides(lum: np.ndarray[Any, Any], run_start: int, run_end: int, level: float) -> bool:
    """Drawing, not more blank paper, sits within a short walk of both sides."""
    along = int(lum.shape[1])

    def inky(origin: int, direction: int) -> bool:
        for delta in range(1, 57):
            column = origin + direction * delta
            if column < 0 or column >= along:
                return False
            if float((lum[:, column] < level).mean()) >= 0.10:
                return True
        return False

    return inky(run_start - 1, -1) and inky(run_end, 1)


def _frame_rule_view(
    sheet: np.ndarray[Any, Any],
    lum: np.ndarray[Any, Any],
    start: int,
    end: int,
) -> bool:
    """A thin dark rule with paper on both sides, running the length of the view."""
    if end - start < 3:
        return False
    paper_l = _paper_level(lum, sheet)
    dark = lum < paper_l - 30.0
    along = int(lum.shape[1])
    spans = _span_columns(dark)
    for index in range(start, end):
        if not bool(spans[index]):
            continue
        left_0, left_1 = max(0, index - 12), max(0, index - 1)
        right_0, right_1 = min(along, index + 2), min(along, index + 13)
        if left_1 <= left_0 or right_1 <= right_0:
            continue
        left_paper = np.any(sheet[:, left_0:left_1], axis=1)
        right_paper = np.any(sheet[:, right_0:right_1], axis=1)
        both = dark[:, index] & left_paper & right_paper
        band_0, band_1 = max(0, index - 8), min(along, index + 9)
        thick = int(np.count_nonzero(dark[:, band_0:band_1].mean(axis=0) > 0.55))
        if thick <= 8 and bool(_span_columns(both)[0]):
            return True
    return False


def _ink_rule_view(lum: np.ndarray[Any, Any], start: int, end: int) -> bool:
    """A thin dark band that reaches both edges. A solid black field does not.

    Coloured panels are often painted up to the rule, so the shoulders are not
    paper. The band still has to be thin: a filled shape is wider. A wide dark
    area beside a real rule does not hide the rule.
    """
    flags = _span_columns(lum[:, start:end] < 115.0)
    index = 0
    while index < len(flags):
        if not bool(flags[index]):
            index += 1
            continue
        end_run = index
        while end_run < len(flags) and bool(flags[end_run]):
            end_run += 1
        if 1 <= end_run - index <= 64:
            return True
        index = end_run
    return False


def _paper_level(lum: np.ndarray[Any, Any], sheet: np.ndarray[Any, Any]) -> float:
    bright = lum[sheet] if sheet.shape == lum.shape else lum
    if bright.size == 0:
        return float(np.percentile(lum, 70))
    light = bright[bright > 100.0]
    sample = light if light.size > 20 else bright
    return float(np.median(sample))


def _sliver_host(boxes: Sequence[Box], index: int, page_w: int, page_h: int) -> int | None:
    small = boxes[index]
    best: int | None = None
    best_key: tuple[int, int] | None = None
    for other, big in enumerate(boxes):
        if other == index or big.area < small.area * 2:
            continue
        axis = _thin_axis(small, big, page_w, page_h)
        if axis is None or not _shares_band(small, big, axis):
            continue
        gap = _axis_gap(small, big, axis)
        limit = max(8, round((page_w if axis == "x" else page_h) * 0.06))
        if gap > limit:
            continue
        key = (gap, -big.area)
        if best_key is None or key < best_key:
            best = other
            best_key = key
    return best


def _thin_axis(small: Box, big: Box, page_w: int, page_h: int) -> str | None:
    """``"x"`` when ``small`` is a vertical sliver. ``"y"`` when it is a flat one."""
    if small.w < page_w * 0.11 and small.w < big.w * 0.16 and small.h > small.w:
        return "x"
    if small.h < page_h * 0.08 and small.h < big.h * 0.16 and small.w > small.h:
        return "y"
    return None


def _shares_band(small: Box, big: Box, axis: str) -> bool:
    if axis == "x":
        overlap = _overlap_span(small.y, small.y2, big.y, big.y2)
        return overlap > 0.6 * max(1, small.h)
    overlap = _overlap_span(small.x, small.x2, big.x, big.x2)
    return overlap > 0.6 * max(1, small.w)


def _gutter_between(
    rgb: np.ndarray[Any, Any],
    sheet: np.ndarray[Any, Any],
    light: float,
    lum: np.ndarray[Any, Any],
    dark: np.ndarray[Any, Any],
    paper: float,
    scale: float,
    parent: Box,
    small: Box,
    big: Box,
    axis: str,
    page_w: int,
    page_h: int,
) -> bool:
    """True when a frame rule or a paper gap divides ``small`` from ``big``.

    A positive gap is the gutter already measured between the boxes. Boxes that
    touch or overlap have no such gap, so the mark has to sit on the shared
    edge. Only a short band around that edge is searched: a rule further inside
    either drawing is not what divides them.
    """
    gap = _axis_gap(small, big, axis)
    if gap > 0:
        cut = _shared_cut(small, big, axis)
        return full_span_divider(rgb, sheet, parent, axis, cut, page_light=light) or _crosses_box(
            lum, dark, paper, parent, axis, cut, scale
        )
    page_span = page_w if axis == "x" else page_h
    band = max(8, min(24, round(page_span * 0.015)))
    for cut in _contact_cuts(small, big, axis):
        if full_span_divider(rgb, sheet, parent, axis, cut, page_light=light, search=band):
            return True
    return False


def _contact_cuts(a: Box, b: Box, axis: str) -> list[int]:
    """The shared edge, or both lips of the overlap when the boxes cross."""
    if axis == "x":
        start_a, end_a, start_b, end_b = a.x, a.x2, b.x, b.x2
    else:
        start_a, end_a, start_b, end_b = a.y, a.y2, b.y, b.y2
    if end_a <= start_b:
        return [(end_a + start_b) // 2]
    if end_b <= start_a:
        return [(end_b + start_a) // 2]
    left = max(start_a, start_b)
    right = min(end_a, end_b)
    if right <= left:
        return [left]
    return [left, right]


def _axis_gap(a: Box, b: Box, axis: str) -> int:
    if axis == "x":
        if a.x2 < b.x:
            return b.x - a.x2
        if b.x2 < a.x:
            return a.x - b.x2
        return 0
    if a.y2 < b.y:
        return b.y - a.y2
    if b.y2 < a.y:
        return a.y - b.y2
    return 0


def _shared_cut(a: Box, b: Box, axis: str) -> int:
    if axis == "x":
        if a.x2 <= b.x:
            return (a.x2 + b.x) // 2
        if b.x2 <= a.x:
            return (b.x2 + a.x) // 2
        return (max(a.x, b.x) + min(a.x2, b.x2)) // 2
    if a.y2 <= b.y:
        return (a.y2 + b.y) // 2
    if b.y2 <= a.y:
        return (b.y2 + a.y) // 2
    return (max(a.y, b.y) + min(a.y2, b.y2)) // 2


def _overlap_span(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


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
    rgb: np.ndarray[Any, Any],
    sheet: np.ndarray[Any, Any],
    page_light: float,
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
    # A full-resolution sheet gap or rule counts too, including a light band of
    # the page's own sheet that the resized valley test missed.
    interior = [
        cut
        for cut in cuts
        if start + inset < cut < end - inset
        and (
            _crosses_box(lum, dark, paper, box, axis, cut, scale)
            or full_span_divider(rgb, sheet, box, axis, cut, allow_ink_rule=True, page_light=page_light)
        )
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
