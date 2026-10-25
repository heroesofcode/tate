from PIL import Image

from tate.boxes import Box
from tate.eval.metrics import (
    Counts,
    PageScore,
    aggregate,
    aggregate_styles,
    content_loss,
    kendall_tau,
    match_boxes,
    percentile,
    score_page,
)
from tate.layout import PageLayout, SegmentDraft


def _segment(box: Box, panel: Box | None = None, continuation: bool = False) -> SegmentDraft:
    return SegmentDraft(Image.new("RGB", (2, 2), "white"), 0, box, continuation, panel or box)


def _layout(
    segments: list[SegmentDraft],
    *,
    fallback: bool = False,
    spread: bool = False,
) -> PageLayout:
    return PageLayout(segments, [], [], fallback, spread)


def _score(
    image: Image.Image,
    layout: PageLayout,
    panels: tuple[Box, ...],
    *,
    whole: bool = False,
    balloons: tuple[Box, ...] = (),
    ignore: tuple[Box, ...] = (),
    styles: tuple[str, ...] = ("standard",),
    page_id: str = "p",
    work: str = "demo",
) -> PageScore:
    return score_page(
        image,
        page_id=page_id,
        work=work,
        styles=styles,
        whole=whole,
        gt_panels=panels,
        balloons=balloons,
        ignore=ignore,
        layout=layout,
        elapsed=1.0,
    )


def test_matching_f1_and_order() -> None:
    gt = (Box(0, 0, 100, 100), Box(0, 100, 100, 100), Box(200, 0, 100, 100))
    pred = (Box(0, 0, 100, 100), Box(200, 0, 100, 100), Box(0, 100, 50, 100))
    assert len(match_boxes(pred, gt, 0.5)) == 3
    assert len(match_boxes(pred, gt, 0.75)) == 2
    assert kendall_tau([0, 2, 1]) == (2 - 1) / 3
    assert kendall_tau([0, 1, 2]) == 1
    assert kendall_tau([2, 1, 0]) == -1

    image = Image.new("RGB", (300, 200), "white")
    page = _score(
        image,
        _layout([_segment(box, box) for box in pred]),
        gt,
        balloons=(Box(0, 0, 10, 10),),
    )
    assert page.counts_50 == Counts(3, 0, 0)
    assert page.counts_75 == Counts(2, 1, 1)
    assert page.order_exact is False
    assert page.kendall_tau == (2 - 1) / 3
    assert page.wrong_order is True
    assert page.f1_50 == 1

    exact = _score(image, _layout([_segment(box, box) for box in gt]), gt)
    assert exact.order_exact is True
    assert exact.kendall_tau == 1
    assert exact.wrong_order is False

    both = aggregate([page, exact])
    assert both.panel_50.tp == 6
    assert both.panel_75.fn == 1
    assert both.order_exact == 0.5
    assert both.mean_kendall_tau == (1 + (1 / 3)) / 2


def test_whole_page_fallback_and_wrong_split() -> None:
    image = Image.new("RGB", (100, 80), "white")
    whole = Box(0, 0, 100, 80)
    kept = _score(image, _layout([_segment(whole)], fallback=True), (whole,), whole=True)
    assert kept.counts_50 == Counts(1, 0, 0)
    assert kept.unexpected_fallback is False
    assert kept.wrong_split is False
    assert kept.order_exact is None

    split = _score(
        image,
        _layout([_segment(Box(0, 0, 100, 40)), _segment(Box(0, 40, 100, 40))]),
        (whole,),
        whole=True,
    )
    assert split.counts_50 == Counts(0, 2, 1)
    assert split.wrong_split is True

    partial = _score(image, _layout([_segment(Box(0, 0, 40, 80))]), (whole,), whole=True)
    assert partial.counts_50 == Counts(0, 1, 1)
    assert partial.wrong_split is False

    fell_back = _score(
        image,
        _layout([_segment(whole)], fallback=True),
        (Box(10, 10, 30, 30), Box(50, 10, 30, 30)),
        whole=False,
    )
    assert fell_back.unexpected_fallback is True
    assert fell_back.pred_panels == (whole,)
    assert aggregate([fell_back]).fallback_rate == 1


def test_balloon_cut_counts_each_crop_separately() -> None:
    image = Image.new("RGB", (100, 40), "white")
    balloon = Box(0, 10, 100, 20)
    crops = [
        _segment(Box(0, 0, 100, 20), Box(0, 0, 100, 40), continuation=False),
        _segment(Box(0, 20, 100, 20), Box(0, 0, 100, 40), continuation=True),
    ]
    cut = _score(image, _layout(crops), (Box(0, 0, 100, 40),), balloons=(balloon,))
    assert cut.balloons_cut == 1
    intact = _score(
        image,
        _layout([_segment(Box(0, 0, 100, 40))]),
        (Box(0, 0, 100, 40),),
        balloons=(balloon,),
    )
    assert intact.balloons_cut == 0


def test_content_loss_ignores_margin_ignore_boxes_and_color_paper() -> None:
    image = Image.new("RGB", (100, 100), "white")
    pixels = image.load()
    assert pixels is not None
    for y in range(40, 60):
        for x in range(40, 60):
            pixels[x, y] = (0, 0, 0)
    pixels[0, 50] = (0, 0, 0)
    covered = content_loss(
        image,
        color=False,
        ignore=(Box(40, 40, 10, 20),),
        crops=(Box(50, 40, 10, 20),),
    )
    assert covered == 0
    uncovered = content_loss(image, color=False, ignore=(Box(40, 40, 10, 20),), crops=())
    assert uncovered == 1

    color = Image.new("RGB", (100, 100), (250, 240, 200))
    paint = color.load()
    assert paint is not None
    for y in range(30, 50):
        for x in range(30, 50):
            paint[x, y] = (180, 220, 250)
    half = (Box(30, 30, 10, 20),)
    assert content_loss(color, color=False, ignore=(), crops=half) == 0
    assert content_loss(color, color=True, ignore=(), crops=half) == 0.5


def test_style_breakdown_and_percentile() -> None:
    image = Image.new("RGB", (40, 40), "white")
    box = Box(4, 4, 20, 20)
    standard = _score(image, _layout([_segment(box)]), (box,), styles=("standard", "dense"), page_id="a")
    other = _score(image, _layout([]), (box,), styles=("dense",), page_id="b", work="other")
    styles = aggregate_styles([standard, other])
    assert styles["standard"].pages == 1
    assert styles["dense"].pages == 2
    assert styles["dense"].panel_f1_50 < 1
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([10], 0.95) == 10


def test_f1_is_zero_on_a_complete_miss_and_one_when_empty() -> None:
    assert Counts(0, 2, 3).f1 == 0.0
    assert Counts(0, 0, 4).f1 == 0.0
    assert Counts(0, 0, 0).f1 == 1.0
    assert Counts(2, 0, 0).f1 == 1.0
