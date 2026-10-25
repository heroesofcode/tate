"""Scores for one page and the aggregates the report is built from.

Boxes are half-open. IoU, coverage, and ink use that same convention.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from tate.boxes import Box, contained_ratio, iou
from tate.layout import PageLayout

INK_LUMINANCE = 200
PAPER_DISTANCE = 30.0
DARK_BACKGROUND_SHARE = 0.8
BORDER_FRAC = 0.02
MARGIN_FRAC = 0.01
BALLOON_COVER = 0.98
WHOLE_COVER = 0.95
LOSS_PAGE = 0.005
STRIP_WIDTH = 1080


@dataclass(frozen=True, slots=True)
class Counts:
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 1.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 1.0
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self) -> float:
        if self.tp + self.fp + self.fn == 0:
            return 1.0
        precision = self.precision
        recall = self.recall
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def add(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)


@dataclass(frozen=True, slots=True)
class Crop:
    box: Box
    continuation: bool
    order: int


@dataclass(frozen=True, slots=True)
class Match:
    pred: int
    gt: int
    score: float


@dataclass(frozen=True, slots=True)
class PageScore:
    page_id: str
    work: str
    styles: tuple[str, ...]
    whole: bool
    fallback: bool
    spread: bool
    unexpected_fallback: bool
    wrong_split: bool
    wrong_order: bool
    elapsed: float
    gt_panels: tuple[Box, ...]
    pred_panels: tuple[Box, ...]
    crops: tuple[Crop, ...]
    balloons_total: int
    balloons_cut: int
    cut_balloons: tuple[Box, ...]
    ignore: tuple[Box, ...]
    content_loss: float
    counts_50: Counts
    counts_75: Counts
    order_exact: bool | None
    kendall_tau: float | None

    @property
    def f1_50(self) -> float:
        return self.counts_50.f1

    @property
    def f1_75(self) -> float:
        return self.counts_75.f1

    def reasons(self) -> tuple[str, ...]:
        found: list[str] = []
        if self.counts_50.fp or self.counts_50.fn:
            found.append("panels")
        if self.wrong_order:
            found.append("order")
        if self.balloons_cut:
            found.append("balloon")
        if self.content_loss > LOSS_PAGE:
            found.append("content")
        if self.unexpected_fallback:
            found.append("fallback")
        if self.wrong_split:
            found.append("split")
        return tuple(found)


@dataclass(frozen=True, slots=True)
class MetricBlock:
    pages: int
    panel_50: Counts
    panel_75: Counts
    mean_page_f1_50: float
    mean_page_f1_75: float
    order_exact: float | None
    order_pages: int
    mean_kendall_tau: float | None
    tau_pages: int
    balloons: int
    balloons_cut: int
    content_loss_mean: float
    content_loss_max: float
    content_loss_pages: int
    fallback_pages: int
    fallback_rate: float
    whole_pages: int
    wrong_split_pages: int
    wrong_split_rate: float | None
    time_mean_s: float
    time_p95_s: float

    @property
    def panel_f1_50(self) -> float:
        return self.panel_50.f1

    @property
    def panel_f1_75(self) -> float:
        return self.panel_75.f1

    @property
    def balloon_cut_rate(self) -> float | None:
        if self.balloons == 0:
            return None
        return self.balloons_cut / self.balloons


def predicted_panels(layout: PageLayout, width: int, height: int) -> list[Box]:
    """Panels in emission order. A fallback or a spread is the whole page."""
    if layout.fallback or layout.spread:
        return [Box(0, 0, width, height)]
    return list(layout.emitted_panels())


def match_boxes(pred: Sequence[Box], gt: Sequence[Box], threshold: float) -> list[Match]:
    """Greedy one-to-one matches, highest IoU first."""
    pairs: list[tuple[float, int, int]] = []
    for pred_index, pred_box in enumerate(pred):
        for gt_index, gt_box in enumerate(gt):
            score = iou(pred_box, gt_box)
            if score >= threshold:
                pairs.append((score, pred_index, gt_index))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[Match] = []
    for score, pred_index, gt_index in pairs:
        if pred_index in used_pred or gt_index in used_gt:
            continue
        used_pred.add(pred_index)
        used_gt.add(gt_index)
        matches.append(Match(pred_index, gt_index, score))
    return matches


def kendall_tau(pred_ranks: Sequence[int]) -> float:
    """Kendall tau-a of predicted ranks listed in ground-truth order. No ties."""
    count = len(pred_ranks)
    if count < 2:
        raise ValueError("kendall tau needs two ranks")
    agree = 0
    disagree = 0
    for left in range(count):
        for right in range(left + 1, count):
            if pred_ranks[left] < pred_ranks[right]:
                agree += 1
            elif pred_ranks[left] > pred_ranks[right]:
                disagree += 1
    total = count * (count - 1) / 2
    return (agree - disagree) / total


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear percentile. Rank ``(n - 1) * fraction`` is interpolated."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * fraction
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def score_page(
    image: Image.Image,
    *,
    page_id: str,
    work: str,
    styles: Sequence[str],
    whole: bool,
    gt_panels: Sequence[Box],
    balloons: Sequence[Box],
    ignore: Sequence[Box],
    layout: PageLayout,
    elapsed: float,
) -> PageScore:
    width, height = image.size
    emitted = layout.emitted_panels()
    predicted = predicted_panels(layout, width, height)
    counts_50, counts_75 = _detection_counts(predicted, gt_panels, whole, width, height)
    order_exact, tau, wrong_order = _order_stats(predicted, gt_panels, whole)
    crops = _crops(layout)
    cut = tuple(
        balloon
        for balloon in balloons
        if not any(contained_ratio(balloon, crop.box) >= BALLOON_COVER for crop in crops)
    )
    loss = content_loss(
        image, color="color" in styles, ignore=ignore, crops=tuple(crop.box for crop in crops)
    )
    unexpected = (not whole) and layout.fallback
    wrong_split = whole and (not layout.fallback) and (not layout.spread) and len(emitted) > 1
    return PageScore(
        page_id=page_id,
        work=work,
        styles=tuple(styles),
        whole=whole,
        fallback=layout.fallback,
        spread=layout.spread,
        unexpected_fallback=unexpected,
        wrong_split=wrong_split,
        wrong_order=wrong_order,
        elapsed=elapsed,
        gt_panels=tuple(gt_panels),
        pred_panels=tuple(predicted),
        crops=crops,
        balloons_total=len(balloons),
        balloons_cut=len(cut),
        cut_balloons=cut,
        ignore=tuple(ignore),
        content_loss=loss,
        counts_50=counts_50,
        counts_75=counts_75,
        order_exact=order_exact,
        kendall_tau=tau,
    )


def aggregate(pages: Sequence[PageScore]) -> MetricBlock:
    if not pages:
        empty = Counts(0, 0, 0)
        return MetricBlock(
            pages=0,
            panel_50=empty,
            panel_75=empty,
            mean_page_f1_50=1.0,
            mean_page_f1_75=1.0,
            order_exact=None,
            order_pages=0,
            mean_kendall_tau=None,
            tau_pages=0,
            balloons=0,
            balloons_cut=0,
            content_loss_mean=0.0,
            content_loss_max=0.0,
            content_loss_pages=0,
            fallback_pages=0,
            fallback_rate=0.0,
            whole_pages=0,
            wrong_split_pages=0,
            wrong_split_rate=None,
            time_mean_s=0.0,
            time_p95_s=0.0,
        )
    panel_50 = Counts(0, 0, 0)
    panel_75 = Counts(0, 0, 0)
    for page in pages:
        panel_50 = panel_50.add(page.counts_50)
        panel_75 = panel_75.add(page.counts_75)
    order_pages = [page for page in pages if page.order_exact is not None]
    tau_pages = [page for page in pages if page.kendall_tau is not None]
    losses = [page.content_loss for page in pages]
    times = [page.elapsed for page in pages]
    whole_pages = sum(1 for page in pages if page.whole)
    wrong_split_pages = sum(1 for page in pages if page.wrong_split)
    fallback_pages = sum(1 for page in pages if page.unexpected_fallback)
    exact_rate = None
    if order_pages:
        exact_rate = sum(1 for page in order_pages if page.order_exact) / len(order_pages)
    tau_mean = None
    if tau_pages:
        tau_mean = sum(page.kendall_tau or 0.0 for page in tau_pages) / len(tau_pages)
    split_rate = None if whole_pages == 0 else wrong_split_pages / whole_pages
    return MetricBlock(
        pages=len(pages),
        panel_50=panel_50,
        panel_75=panel_75,
        mean_page_f1_50=sum(page.f1_50 for page in pages) / len(pages),
        mean_page_f1_75=sum(page.f1_75 for page in pages) / len(pages),
        order_exact=exact_rate,
        order_pages=len(order_pages),
        mean_kendall_tau=tau_mean,
        tau_pages=len(tau_pages),
        balloons=sum(page.balloons_total for page in pages),
        balloons_cut=sum(page.balloons_cut for page in pages),
        content_loss_mean=sum(losses) / len(losses),
        content_loss_max=max(losses),
        content_loss_pages=sum(1 for loss in losses if loss > LOSS_PAGE),
        fallback_pages=fallback_pages,
        fallback_rate=fallback_pages / len(pages),
        whole_pages=whole_pages,
        wrong_split_pages=wrong_split_pages,
        wrong_split_rate=split_rate,
        time_mean_s=sum(times) / len(times),
        time_p95_s=percentile(times, 0.95),
    )


def aggregate_styles(pages: Sequence[PageScore]) -> dict[str, MetricBlock]:
    grouped: dict[str, list[PageScore]] = {}
    for page in pages:
        for style in page.styles:
            grouped.setdefault(style, []).append(page)
    return {style: aggregate(grouped[style]) for style in sorted(grouped)}


def aggregate_works(pages: Sequence[PageScore]) -> dict[str, MetricBlock]:
    grouped: dict[str, list[PageScore]] = {}
    for page in pages:
        grouped.setdefault(page.work, []).append(page)
    return {work: aggregate(grouped[work]) for work in sorted(grouped)}


def content_loss(
    image: Image.Image,
    *,
    color: bool,
    ignore: Sequence[Box],
    crops: Sequence[Box],
) -> float:
    ink, _covered, lost = ink_masks(image, color=color, ignore=ignore, crops=crops)
    total = int(np.count_nonzero(ink))
    if total == 0:
        return 0.0
    return int(np.count_nonzero(lost)) / total


def ink_masks(
    image: Image.Image,
    *,
    color: bool,
    ignore: Sequence[Box],
    crops: Sequence[Box],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Return ``(ink, covered, lost)`` boolean masks in source pixels."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    luminance = _luminance(rgb)
    ink = luminance < INK_LUMINANCE
    paper = _paper_color(rgb)
    paper_dark = _dark_background(luminance, ignore)
    if color or paper_dark:
        delta = rgb.astype(np.float64) - paper
        distance = np.sqrt(np.square(delta).sum(axis=2))
        away = distance > PAPER_DISTANCE
        # On a dark page background the gutters are dark too; only pixels that
        # differ from that background are content.
        ink = np.logical_and(ink, away) if paper_dark else ink
        if color:
            ink = np.logical_or(ink, away)
    _clear_margin(ink, width, height)
    for box in ignore:
        _fill(ink, box, False)
    covered = np.zeros((height, width), dtype=np.bool_)
    for box in crops:
        _fill(covered, box, True)
    lost = np.logical_and(ink, np.logical_not(covered))
    return ink, covered, lost


def _detection_counts(
    predicted: Sequence[Box],
    gt_panels: Sequence[Box],
    whole: bool,
    width: int,
    height: int,
) -> tuple[Counts, Counts]:
    if whole:
        target = _whole_target(gt_panels, width, height)
        return _whole_counts(predicted, target), _whole_counts(predicted, target)
    return (
        _pair_counts(predicted, gt_panels, 0.5),
        _pair_counts(predicted, gt_panels, 0.75),
    )


def _whole_target(gt_panels: Sequence[Box], width: int, height: int) -> Box:
    if not gt_panels:
        return Box(0, 0, width, height)
    box = gt_panels[0]
    for extra in gt_panels[1:]:
        x1 = min(box.x, extra.x)
        y1 = min(box.y, extra.y)
        x2 = max(box.x2, extra.x2)
        y2 = max(box.y2, extra.y2)
        box = Box(x1, y1, x2 - x1, y2 - y1)
    return box


def _whole_counts(predicted: Sequence[Box], target: Box) -> Counts:
    if len(predicted) == 1 and contained_ratio(target, predicted[0]) >= WHOLE_COVER:
        return Counts(1, 0, 0)
    return Counts(0, len(predicted), 1)


def _pair_counts(predicted: Sequence[Box], gt_panels: Sequence[Box], threshold: float) -> Counts:
    matches = match_boxes(predicted, gt_panels, threshold)
    true_positive = len(matches)
    return Counts(true_positive, len(predicted) - true_positive, len(gt_panels) - true_positive)


def _order_stats(
    predicted: Sequence[Box],
    gt_panels: Sequence[Box],
    whole: bool,
) -> tuple[bool | None, float | None, bool]:
    if whole or len(gt_panels) < 2:
        return None, None, False
    matches = match_boxes(predicted, gt_panels, 0.5)
    by_gt = sorted(matches, key=lambda match: match.gt)
    tau = kendall_tau([match.pred for match in by_gt]) if len(matches) >= 2 else None
    identical = len(matches) == len(gt_panels) and len(matches) == len(predicted)
    if identical:
        identical = [match.pred for match in by_gt] == list(range(len(gt_panels)))
    wrong_order = tau is not None and tau < 1.0
    return identical, tau, wrong_order


def _crops(layout: PageLayout) -> tuple[Crop, ...]:
    crops: list[Crop] = []
    order = 0
    for segment in layout.segments:
        if not segment.continuation:
            order += 1
        crops.append(Crop(segment.box, segment.continuation, order))
    return tuple(crops)


def _luminance(rgb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    channels = rgb.astype(np.float64)
    return 0.299 * channels[:, :, 0] + 0.587 * channels[:, :, 1] + 0.114 * channels[:, :, 2]


def _paper_color(rgb: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    border_y = _edge(height, BORDER_FRAC)
    border_x = _edge(width, BORDER_FRAC)
    mask = np.zeros((height, width), dtype=np.bool_)
    if border_y:
        mask[:border_y, :] = True
        mask[height - border_y :, :] = True
    if border_x:
        mask[:, :border_x] = True
        mask[:, width - border_x :] = True
    if not bool(mask.any()):
        mask[:, :] = True
    pixels = rgb[mask].astype(np.float64)
    median = np.median(pixels, axis=0)
    return np.asarray(median, dtype=np.float64)


def _dark_background(luminance: np.ndarray[Any, Any], ignore: Sequence[Box]) -> bool:
    """True when the ring just inside the margin is dark nearly all the way round.

    A dark scan edge or a frame on one side covers only part of the ring, so a
    white page with dark edges still reads as white paper.
    """
    height, width = int(luminance.shape[0]), int(luminance.shape[1])
    ring = np.zeros((height, width), dtype=np.bool_)
    outer_y, outer_x = _edge(height, MARGIN_FRAC), _edge(width, MARGIN_FRAC)
    inner_y = min(height // 2, outer_y + _edge(height, BORDER_FRAC))
    inner_x = min(width // 2, outer_x + _edge(width, BORDER_FRAC))
    ring[outer_y:inner_y, outer_x : width - outer_x] = True
    ring[height - inner_y : height - outer_y, outer_x : width - outer_x] = True
    ring[outer_y : height - outer_y, outer_x:inner_x] = True
    ring[outer_y : height - outer_y, width - inner_x : width - outer_x] = True
    for box in ignore:
        _fill(ring, box, False)
    count = int(np.count_nonzero(ring))
    if count == 0:
        return False
    dark = int(np.count_nonzero(np.logical_and(ring, luminance < INK_LUMINANCE)))
    return dark >= DARK_BACKGROUND_SHARE * count


def _clear_margin(ink: np.ndarray[Any, Any], width: int, height: int) -> None:
    margin_y = _edge(height, MARGIN_FRAC)
    margin_x = _edge(width, MARGIN_FRAC)
    if margin_y:
        ink[:margin_y, :] = False
        ink[height - margin_y :, :] = False
    if margin_x:
        ink[:, :margin_x] = False
        ink[:, width - margin_x :] = False


def _edge(length: int, fraction: float) -> int:
    if length <= 1:
        return 0
    return max(1, round(length * fraction))


def _fill(mask: np.ndarray[Any, Any], box: Box, value: bool) -> None:
    height, width = int(mask.shape[0]), int(mask.shape[1])
    x1 = min(max(box.x, 0), width)
    x2 = min(max(box.x2, 0), width)
    y1 = min(max(box.y, 0), height)
    y2 = min(max(box.y2, 0), height)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = value


def xyxy(box: Box) -> list[int]:
    return [box.x, box.y, box.x2, box.y2]
