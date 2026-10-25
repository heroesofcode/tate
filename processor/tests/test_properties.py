"""Properties of the pure layout helpers, plus deterministic packages."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from tate.boxes import Box, Detection, clean_panels, drop_slivers, separate_overlaps
from tate.detect.page import PageDetector
from tate.errors import TateError
from tate.io.inputs import parse_page_range
from tate.layout import MAX_SEGMENT_HEIGHT, expand_for_text, layout_page
from tate.order import reading_order
from tate.pipeline import process_chapter

_SCORE = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


def _loose_box() -> st.SearchStrategy[Box]:
    return st.builds(
        Box,
        st.integers(-40, 360),
        st.integers(-40, 360),
        st.integers(-8, 220),
        st.integers(-8, 220),
        _SCORE,
    )


@st.composite
def _inside_boxes(draw: st.DrawFn) -> tuple[int, int, list[Box]]:
    page_w = draw(st.integers(1, 240))
    page_h = draw(st.integers(1, 240))
    count = draw(st.integers(0, 6))
    boxes: list[Box] = []
    for _ in range(count):
        width = draw(st.integers(1, page_w))
        height = draw(st.integers(1, page_h))
        boxes.append(
            Box(
                draw(st.integers(0, page_w - width)),
                draw(st.integers(0, page_h - height)),
                width,
                height,
                draw(_SCORE),
            )
        )
    return page_w, page_h, boxes


@st.composite
def _any_boxes(draw: st.DrawFn) -> tuple[int, int, list[Box]]:
    page_w = draw(st.integers(1, 400))
    page_h = draw(st.integers(1, 400))
    boxes = draw(st.lists(_loose_box(), max_size=8))
    return page_w, page_h, boxes


def _assert_inside(box: Box, page_w: int, page_h: int, *, min_side: int) -> None:
    assert box.w >= min_side
    assert box.h >= min_side
    assert box.x >= 0 and box.y >= 0
    assert box.x2 <= page_w and box.y2 <= page_h


def _mirror_x(box: Box) -> Box:
    return Box(-box.x2, box.y, box.w, box.h, box.score)


@given(st.lists(_loose_box(), max_size=8))
def test_reading_order_is_a_permutation(boxes: list[Box]) -> None:
    order = reading_order(boxes)
    assert sorted(order) == list(range(len(boxes)))
    assert sorted(reading_order(boxes, "ltr")) == list(range(len(boxes)))


@given(st.lists(_loose_box(), max_size=8))
def test_ltr_order_matches_rtl_of_mirrored_boxes(boxes: list[Box]) -> None:
    mirrored = [_mirror_x(box) for box in boxes]
    assert reading_order(boxes, "ltr") == reading_order(mirrored, "rtl")


@given(
    rows=st.integers(1, 5),
    cols=st.integers(1, 5),
    cell_w=st.integers(12, 48),
    cell_h=st.integers(12, 48),
    gap_x=st.integers(1, 16),
    gap_y=st.integers(1, 16),
)
def test_strict_grid_is_right_to_left_then_top_to_bottom(
    rows: int,
    cols: int,
    cell_w: int,
    cell_h: int,
    gap_x: int,
    gap_y: int,
) -> None:
    boxes = [
        Box(col * (cell_w + gap_x), row * (cell_h + gap_y), cell_w, cell_h)
        for row in range(rows)
        for col in range(cols)
    ]
    strips = rows >= 4 and cols >= 2
    if strips:
        expected = [row * cols + col for col in range(cols - 1, -1, -1) for row in range(rows)]
        expected_ltr = [row * cols + col for col in range(cols) for row in range(rows)]
    else:
        expected = [row * cols + col for row in range(rows) for col in range(cols - 1, -1, -1)]
        expected_ltr = list(range(len(boxes)))
    assert reading_order(boxes) == expected
    assert reading_order(boxes, "rtl") == expected
    assert reading_order(boxes, "ltr") == expected_ltr


@given(_any_boxes())
def test_clean_panels_stay_inside_and_non_degenerate(page: tuple[int, int, list[Box]]) -> None:
    page_w, page_h, boxes = page
    for box in clean_panels(boxes, page_w, page_h):
        _assert_inside(box, page_w, page_h, min_side=8)


@given(_inside_boxes())
def test_separate_overlaps_stay_inside_and_non_degenerate(page: tuple[int, int, list[Box]]) -> None:
    page_w, page_h, boxes = page
    for box in separate_overlaps(boxes):
        _assert_inside(box, page_w, page_h, min_side=8)


@given(_inside_boxes())
def test_drop_slivers_stay_inside_and_non_degenerate(page: tuple[int, int, list[Box]]) -> None:
    page_w, page_h, boxes = page
    for box in drop_slivers(boxes, page_w, page_h):
        _assert_inside(box, page_w, page_h, min_side=1)


@st.composite
def _panels_and_text(draw: st.DrawFn) -> tuple[int, int, list[Box], list[Box]]:
    page_w = draw(st.integers(1, 300))
    page_h = draw(st.integers(1, 300))
    panels = draw(st.lists(_loose_box(), max_size=5))
    texts = draw(st.lists(_loose_box(), max_size=5))
    return page_w, page_h, panels, texts


@given(_panels_and_text())
def test_expand_for_text_never_shrinks_a_panel(sample: tuple[int, int, list[Box], list[Box]]) -> None:
    page_w, page_h, panels, texts = sample
    original = [panel.clamp(page_w, page_h) for panel in panels]
    expanded = expand_for_text(panels, texts, page_w, page_h)
    assert len(expanded) == len(original)
    for before, after in zip(original, expanded, strict=True):
        assert after.x <= before.x
        assert after.y <= before.y
        assert after.x2 >= before.x2
        assert after.y2 >= before.y2
        _assert_inside(after, page_w, page_h, min_side=0)


@st.composite
def _random_layout(draw: st.DrawFn) -> tuple[int, int, int, int, tuple[Box, ...], tuple[Box, ...]]:
    page_w = draw(st.integers(1, 110))
    page_h = draw(st.integers(1, 140))
    width = draw(st.integers(64, 160))
    page_index = draw(st.integers(0, 5))

    def box_in_range(limit_w: int, limit_h: int) -> Box:
        return Box(
            draw(st.integers(-limit_w, limit_w)),
            draw(st.integers(-limit_h, limit_h)),
            draw(st.integers(1, max(1, limit_w))),
            draw(st.integers(1, max(1, limit_h))),
            draw(_SCORE),
        )

    panels = tuple(box_in_range(page_w, page_h) for _ in range(draw(st.integers(0, 4))))
    texts = tuple(box_in_range(page_w, page_h) for _ in range(draw(st.integers(0, 3))))
    return page_w, page_h, width, page_index, panels, texts


@given(_random_layout())
def test_layout_page_covers_detected_panels(
    sample: tuple[int, int, int, int, tuple[Box, ...], tuple[Box, ...]],
) -> None:
    page_w, page_h, width, page_index, panels, texts = sample
    image = Image.new("RGB", (page_w, page_h), (255, 255, 255))
    laid = None
    try:
        laid = layout_page(image, Detection(panels, texts), page_index=page_index, width=width)
        assert laid.segments
        mask = np.zeros((page_h, page_w), dtype=np.bool_)
        for segment in laid.segments:
            assert segment.image.width == width
            assert 1 <= segment.image.height <= MAX_SEGMENT_HEIGHT
            assert segment.page == page_index
            box = segment.box
            _assert_inside(box, page_w, page_h, min_side=1)
            mask[box.y : box.y2, box.x : box.x2] = True
        for box in laid.detected:
            _assert_inside(box, page_w, page_h, min_side=1)
            assert bool(mask[box.y : box.y2, box.x : box.x2].all())
    finally:
        image.close()
        if laid is not None:
            for segment in laid.segments:
                segment.image.close()


@given(
    count=st.integers(0, 40),
    start=st.integers(1, 80),
    end=st.integers(1, 80),
)
def test_parse_page_range_accepts_only_real_pages(count: int, start: int, end: int) -> None:
    spec = f" {start}-{end} "
    if count < 1 or start > count or end < start or end > count:
        with pytest.raises(TateError):
            parse_page_range(spec, count)
        return
    assert parse_page_range(spec, count) == list(range(start - 1, end))
    assert parse_page_range(f"{start}", count) == [start - 1]


@given(spec=st.text(max_size=16), count=st.integers(-1, 30))
def test_parse_page_range_handles_arbitrary_specs(spec: str, count: int) -> None:
    try:
        indexes = parse_page_range(spec, count)
    except TateError:
        return
    assert indexes
    assert indexes == list(range(indexes[0], indexes[-1] + 1))
    assert indexes[0] >= 0
    assert indexes[-1] < count


@given(
    width=st.integers(64, 140),
    pages=st.lists(
        st.tuples(
            st.integers(16, 64),
            st.integers(16, 80),
            st.integers(0, 255),
            st.integers(0, 255),
            st.integers(0, 255),
        ),
        min_size=1,
        max_size=3,
    ),
)
@settings(max_examples=4, deadline=None)
def test_same_input_and_epoch_produce_the_same_bytes(
    width: int,
    pages: Sequence[tuple[int, int, int, int, int]],
) -> None:
    previous = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "1700000000"
    try:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            folder = root / "chapter"
            folder.mkdir()
            for index, (page_w, page_h, red, green, blue) in enumerate(pages):
                image = Image.new("RGB", (page_w, page_h), (red, green, blue))
                image.save(folder / f"{index}.png", format="PNG")
                image.close()
            first = process_chapter(folder, root / "a.tate", width=width, detector=PageDetector())
            second = process_chapter(folder, root / "b.tate", width=width, detector=PageDetector())
            assert first.output.read_bytes() == second.output.read_bytes()
    finally:
        if previous is None:
            os.environ.pop("SOURCE_DATE_EPOCH", None)
        else:
            os.environ["SOURCE_DATE_EPOCH"] = previous
