"""Turn detected boxes into strip segments. Content is kept; balloons are not cut."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from PIL import Image

from tate.boxes import (
    Box,
    Detection,
    box_distance,
    clean_panels,
    drop_slivers,
    intersection_area,
    iou,
    separate_overlaps,
    union_box,
)
from tate.detect.cv import recover_uncovered
from tate.gutters import split_on_gutters
from tate.ink import cover_boxes, stroke_coverage
from tate.order import ReadingDirection, reading_order

MAX_UPSCALE = 1.6
MAX_SEGMENT_HEIGHT = 4096
SPREAD_RATIO = 1.2
PAD_FRAC = 0.004
TEXT_OVERLAP = 0.30
OVERLAP_IOU = 0.60
INK_COVERAGE_MIN = 0.68
STROKE_COVERAGE_MIN = 0.40
INK_LEVEL = 240
BACKGROUND_RGB = (255, 255, 255)
BACKGROUND_HEX = "#ffffff"


@dataclass
class SegmentDraft:
    """One strip segment before gaps and encoding.

    ``box`` is the crop in source-page pixels, after text expansion and padding.
    ``panel`` is the panel that crop came from, before expansion and padding.
    ``continuation`` is true for every piece of a tall crop after the first.
    """

    image: Image.Image
    page: int
    box: Box
    continuation: bool
    panel: Box


@dataclass
class PageLayout:
    """Segments for one page, plus the boxes drawn in a debug image.

    ``detected`` is the list drawn on a debug image. Panels in the order they
    are emitted are ``emitted_panels``: continuation pieces collapse back onto
    the panel they were cut from.
    """

    segments: list[SegmentDraft]
    detected: list[Box]
    texts: list[Box]
    fallback: bool
    spread: bool

    def emitted_panels(self) -> list[Box]:
        """Panel boxes in emission order, with tall-panel pieces collapsed."""
        panels: list[Box] = []
        for segment in self.segments:
            if not segment.continuation:
                panels.append(segment.panel)
        return panels


def gap_sizes(width: int) -> tuple[int, int]:
    """Panel gap and page gap, in strip pixels."""
    panel = max(12, round(width * 0.018))
    page = max(40, round(width * 0.06))
    return panel, page


def assign_gaps(segments: Sequence[SegmentDraft], width: int) -> list[int]:
    """Vertical space the reader draws above each segment. The first is 0."""
    panel_gap, page_gap = gap_sizes(width)
    gaps: list[int] = []
    previous: int | None = None
    for index, segment in enumerate(segments):
        if index == 0 or segment.continuation:
            gaps.append(0)
        elif previous is not None and segment.page != previous:
            gaps.append(page_gap)
        else:
            gaps.append(panel_gap)
        previous = segment.page
    return gaps


def prepare_texts(texts: Sequence[Box], page_w: int, page_h: int) -> list[Box]:
    page_area = max(1, page_w * page_h)
    kept: list[Box] = []
    ranked = sorted(texts, key=lambda box: (box.score, box.area), reverse=True)
    for text in ranked:
        box = text.clamp(page_w, page_h)
        if box.w < 8 or box.h < 8 or box.area > page_area * 0.5:
            continue
        if any(iou(box, other) > 0.6 for other in kept):
            continue
        kept.append(box)
    return kept


def expand_for_text(
    panels: Sequence[Box],
    texts: Sequence[Box],
    page_w: int,
    page_h: int,
) -> list[Box]:
    """Grow every panel that substantially overlaps a text box, so the balloon survives.

    A text box that sits in no panel is attached to the nearest one.
    """
    original = [panel.clamp(page_w, page_h) for panel in panels]
    expanded = list(original)
    if not expanded:
        return []
    for text in texts:
        text_box = text.clamp(page_w, page_h)
        if text_box.area <= 0:
            continue
        owners = [
            index
            for index, panel in enumerate(original)
            if intersection_area(panel, text_box) / text_box.area >= TEXT_OVERLAP
        ]
        if not owners:
            owners = [min(range(len(original)), key=lambda index: box_distance(original[index], text_box))]
        for index in owners:
            expanded[index] = union_box(expanded[index], text_box).clamp(page_w, page_h)
    return expanded


def pad_box(box: Box, page_w: int, page_h: int) -> Box:
    px = max(2, round(page_w * PAD_FRAC))
    py = max(2, round(page_h * PAD_FRAC))
    padded = Box(box.x - px, box.y - py, box.w + 2 * px, box.h + 2 * py, box.score)
    return padded.clamp(page_w, page_h)


def ink_coverage(image: Image.Image, panels: Sequence[Box]) -> float:
    """Share of non-paper pixels that fall inside some panel. Sampled for speed."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    step = max(1, int(max(height, width) / 480))
    sample = rgb[::step, ::step]
    if sample.size == 0:
        return 1.0
    tone = sample.astype(np.uint16).sum(axis=2) // 3
    ink = tone < INK_LEVEL
    total = int(np.count_nonzero(ink))
    if total < 40:
        return 1.0
    mask = np.zeros(ink.shape, dtype=np.bool_)
    for panel in panels:
        x1 = max(0, min(int(panel.x / step), mask.shape[1]))
        x2 = max(0, min(math.ceil(panel.x2 / step), mask.shape[1]))
        y1 = max(0, min(int(panel.y / step), mask.shape[0]))
        y2 = max(0, min(math.ceil(panel.y2 / step), mask.shape[0]))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    covered = int(np.count_nonzero(np.logical_and(ink, mask)))
    return covered / total


def layout_page(
    image: Image.Image,
    detection: Detection,
    *,
    page_index: int,
    width: int,
    direction: ReadingDirection = "rtl",
) -> PageLayout:
    """Lay one page out. Spreads and unreliable detections stay as a single segment.

    ``direction`` is the reading order of the panels. ``rtl`` is the default.
    A spread is judged on the panels found before uncovered ink is recovered.
    A title or a bleed brought back afterwards stays inside the spread.
    """
    page_w, page_h = image.size
    texts = prepare_texts(detection.texts, page_w, page_h)
    base = _panels_before_recovery(image, detection.panels)
    spread = _is_spread(page_w, page_h, base)
    cleaned = base if spread else _recover_panels(image, base)
    ordered = [cleaned[index] for index in reading_order(cleaned, direction)]
    if spread or _cut_splash(image, cleaned) or _unreliable(cleaned, image):
        whole = Box(0, 0, page_w, page_h)
        return PageLayout(
            segments=_segments_for_box(image, whole, page_index, width, whole),
            detected=ordered,
            texts=texts,
            fallback=not spread,
            spread=spread,
        )
    expanded = expand_for_text(cleaned, texts, page_w, page_h)
    # Crops, not the panel boxes, grow to meet so ink outside every frame
    # (a margin, a caption, a balloon that crosses a border) is still emitted.
    expanded = cover_boxes(expanded, page_w, page_h)
    expanded = expand_for_text(expanded, texts, page_w, page_h)
    order = reading_order(cleaned, direction)
    segments: list[SegmentDraft] = []
    used: list[Box] = []
    for index in order:
        box = pad_box(expanded[index], page_w, page_h)
        if box.w < 2 or box.h < 2:
            continue
        pieces = _segments_for_box(image, box, page_index, width, cleaned[index])
        if not pieces:
            continue
        used.append(box)
        segments.extend(pieces)
    if not segments:
        whole = Box(0, 0, page_w, page_h)
        return PageLayout(
            segments=_segments_for_box(image, whole, page_index, width, whole),
            detected=ordered,
            texts=texts,
            fallback=True,
            spread=False,
        )
    return PageLayout(segments=segments, detected=used, texts=texts, fallback=False, spread=False)


def _panels_before_recovery(image: Image.Image, panels: Sequence[Box]) -> list[Box]:
    page_w, page_h = image.size
    cleaned = clean_panels(panels, page_w, page_h)
    cleaned = drop_slivers(cleaned, page_w, page_h)
    cleaned = split_on_gutters(image, cleaned)
    cleaned = clean_panels(cleaned, page_w, page_h)
    cleaned = drop_slivers(cleaned, page_w, page_h)
    return separate_overlaps(cleaned)


def _recover_panels(image: Image.Image, panels: Sequence[Box]) -> list[Box]:
    page_w, page_h = image.size
    recovered = recover_uncovered(image, panels)
    if not recovered:
        return list(panels)
    # A recovered column is a missed region, not a thin edge of a panel
    # already found, so sliver cleanup does not see it.
    cleaned = clean_panels([*panels, *recovered], page_w, page_h)
    return separate_overlaps(cleaned)


def _is_spread(page_w: int, page_h: int, panels: Sequence[Box]) -> bool:
    """A spread is two facing pages, not a landscape page of panels.

    A wide page with several panels, or with more than one row, is laid out
    panel by panel. A wide page with no panel split, or with exactly two
    boxes that cover the left and right halves, stays one piece.
    """
    if page_w <= page_h * SPREAD_RATIO:
        return False
    if len(panels) >= 3 or _row_count(panels) >= 2:
        return False
    if len(panels) <= 1:
        return True
    return _facing_pages(panels, page_w, page_h)


def _row_count(panels: Sequence[Box]) -> int:
    if not panels:
        return 0
    ordered = sorted(panels, key=lambda box: box.y)
    rows = 1
    bottom = ordered[0].y2
    for box in ordered[1:]:
        if box.y >= bottom - max(4, round(box.h * 0.2)):
            rows += 1
            bottom = box.y2
        else:
            bottom = max(bottom, box.y2)
    return rows


def _facing_pages(panels: Sequence[Box], page_w: int, page_h: int) -> bool:
    left, right = sorted(panels, key=lambda box: box.x)
    if left.x2 > page_w * 0.58 or right.x < page_w * 0.42:
        return False
    tall = left.h >= page_h * 0.7 and right.h >= page_h * 0.7
    wide = left.w >= page_w * 0.3 and right.w >= page_w * 0.3
    return tall and wide


def _cut_splash(image: Image.Image, panels: Sequence[Box]) -> bool:
    """Two wide bands with no paper gutter are one picture, not two panels."""
    if len(panels) != 2:
        return False
    first, second = panels
    width, height = image.size
    x_overlap = _span_overlap(first.x, first.x2, second.x, second.x2)
    y_overlap = _span_overlap(first.y, first.y2, second.y, second.y2)
    wide = first.w > width * 0.65 and second.w > width * 0.65 and x_overlap > 0.7 * min(first.w, second.w)
    tall = first.h > height * 0.65 and second.h > height * 0.65 and y_overlap > 0.7 * min(first.h, second.h)
    if wide and not _paper_between(image, first, second, "y"):
        return True
    return tall and not _paper_between(image, first, second, "x")


def _span_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _paper_between(image: Image.Image, first: Box, second: Box, axis: str) -> bool:
    rgb = np.asarray(image.convert("RGB"))
    if axis == "y":
        if first.y > second.y:
            first, second = second, first
        if first.y2 <= second.y:
            y1, y2 = first.y2, second.y
        else:
            y1, y2 = second.y, first.y2
        if y2 - y1 < 4:
            y1 = max(0, min(first.y2, second.y) - 8)
            y2 = min(rgb.shape[0], max(first.y2, second.y) + 8)
        x1, x2 = max(first.x, second.x), min(first.x2, second.x2)
        if y2 <= y1 or x2 - x1 < 8:
            return False
        band = rgb[max(0, y1) : min(rgb.shape[0], y2), max(0, x1) : min(rgb.shape[1], x2)]
    else:
        if first.x > second.x:
            first, second = second, first
        if first.x2 <= second.x:
            x1, x2 = first.x2, second.x
        else:
            x1, x2 = second.x, first.x2
        if x2 - x1 < 4:
            x1 = max(0, min(first.x2, second.x) - 8)
            x2 = min(rgb.shape[1], max(first.x2, second.x) + 8)
        y1, y2 = max(first.y, second.y), min(first.y2, second.y2)
        if x2 <= x1 or y2 - y1 < 8:
            return False
        band = rgb[max(0, y1) : min(rgb.shape[0], y2), max(0, x1) : min(rgb.shape[1], x2)]
    if band.size == 0:
        return False
    tone = band.astype(np.uint16).sum(axis=2) // 3
    # A gutter matches the paper tone of this band. Fixed white misses a tan scan.
    paper = float(np.percentile(tone, 80))
    if paper < 110:
        return False
    return float(np.mean(tone >= paper - 18)) > 0.6


def _unreliable(panels: Sequence[Box], image: Image.Image) -> bool:
    if len(panels) <= 1:
        return True
    for index, left in enumerate(panels):
        for right in panels[index + 1 :]:
            if iou(left, right) > OVERLAP_IOU:
                return True
    strokes = stroke_coverage(image, panels)
    if strokes is None:
        return ink_coverage(image, panels) < INK_COVERAGE_MIN
    # Frame rules and colour that bleeds past a correct box are still strokes.
    # Only a detection that leaves most of the drawing outside is unreliable;
    # the crops grow afterwards, so the rest of the ink is not dropped.
    return strokes < STROKE_COVERAGE_MIN


def _segments_for_box(
    image: Image.Image,
    box: Box,
    page_index: int,
    width: int,
    panel: Box,
) -> list[SegmentDraft]:
    clamped = box.clamp(image.width, image.height)
    if clamped.w < 1 or clamped.h < 1:
        return []
    crop = image.crop((clamped.x, clamped.y, clamped.x2, clamped.y2))
    rendered = _scale_to_strip(crop, width)
    if rendered.height <= MAX_SEGMENT_HEIGHT:
        return [SegmentDraft(rendered, page_index, clamped, False, panel)]
    parts = _split_height(rendered, clamped)
    rendered.close()
    return [
        SegmentDraft(part, page_index, part_box, index > 0, panel)
        for index, (part, part_box) in enumerate(parts)
    ]


def _scale_to_strip(crop: Image.Image, width: int) -> Image.Image:
    raw_scale = width / crop.width
    scale = min(raw_scale, MAX_UPSCALE)
    new_w = width if raw_scale <= MAX_UPSCALE else max(1, round(crop.width * MAX_UPSCALE))
    new_h = max(1, round(crop.height * scale))
    resized = crop.resize((new_w, new_h), Image.Resampling.LANCZOS)
    if new_w == width:
        return resized
    canvas = Image.new("RGB", (width, new_h), BACKGROUND_RGB)
    canvas.paste(resized, ((width - new_w) // 2, 0))
    resized.close()
    return canvas


def _split_height(image: Image.Image, box: Box) -> list[tuple[Image.Image, Box]]:
    parts: list[tuple[Image.Image, Box]] = []
    top = 0
    total = image.height
    while top < total:
        height = min(MAX_SEGMENT_HEIGHT, total - top)
        part = image.crop((0, top, image.width, top + height)).copy()
        y1 = box.y + round(box.h * (top / total))
        y2 = box.y2 if top + height >= total else box.y + round(box.h * ((top + height) / total))
        if y2 <= y1:
            y2 = min(box.y2, y1 + 1)
        parts.append((part, Box(box.x, y1, box.w, y2 - y1, box.score)))
        top += height
    return parts
