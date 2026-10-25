from tate.boxes import Box
from tate.order import reading_order


def test_grid_is_right_to_left_then_down() -> None:
    boxes = [
        Box(0, 0, 100, 100),
        Box(120, 0, 100, 100),
        Box(0, 120, 100, 100),
        Box(120, 120, 100, 100),
    ]
    assert reading_order(boxes) == [1, 0, 3, 2]
    assert reading_order(boxes, "rtl") == [1, 0, 3, 2]


def test_grid_is_left_to_right_then_down() -> None:
    boxes = [
        Box(0, 0, 100, 100),
        Box(120, 0, 100, 100),
        Box(0, 120, 100, 100),
        Box(120, 120, 100, 100),
    ]
    assert reading_order(boxes, "ltr") == [0, 1, 2, 3]


def test_tall_panel_on_the_right_is_read_first() -> None:
    boxes = [
        Box(0, 0, 100, 90),
        Box(0, 110, 100, 90),
        Box(120, 0, 100, 200),
    ]
    assert reading_order(boxes) == [2, 0, 1]


def test_tall_left_column_is_read_first_left_to_right() -> None:
    boxes = [
        Box(120, 0, 100, 90),
        Box(120, 110, 100, 90),
        Box(0, 0, 100, 200),
    ]
    assert reading_order(boxes, "ltr") == [2, 0, 1]


def test_tall_right_column_is_read_last_left_to_right() -> None:
    boxes = [
        Box(0, 0, 100, 90),
        Box(0, 110, 100, 90),
        Box(120, 0, 100, 200),
    ]
    assert reading_order(boxes, "ltr") == [0, 1, 2]


def test_tall_panel_on_the_left_is_read_last() -> None:
    boxes = [
        Box(120, 0, 100, 90),
        Box(120, 110, 100, 90),
        Box(0, 0, 100, 200),
    ]
    assert reading_order(boxes) == [0, 1, 2]


def test_full_width_banner_then_the_row_below() -> None:
    boxes = [
        Box(0, 0, 220, 80),
        Box(0, 100, 100, 100),
        Box(120, 100, 100, 100),
    ]
    assert reading_order(boxes) == [0, 2, 1]


def test_columns_in_one_row() -> None:
    boxes = [Box(0, 0, 50, 100), Box(60, 0, 50, 100), Box(120, 0, 50, 100)]
    assert reading_order(boxes) == [2, 1, 0]


def test_slight_vertical_overlap_still_splits_rows() -> None:
    boxes = [
        Box(0, 0, 100, 100),
        Box(120, 0, 100, 100),
        Box(0, 96, 100, 100),
        Box(120, 96, 100, 100),
    ]
    assert reading_order(boxes) == [1, 0, 3, 2]


def test_staircase_falls_back_to_rows() -> None:
    # Right panel blocks a horizontal cut; the bottom banner blocks a vertical one.
    boxes = [
        Box(0, 0, 100, 80),
        Box(110, 0, 100, 150),
        Box(0, 160, 210, 80),
    ]
    assert reading_order(boxes) == [1, 0, 2]
