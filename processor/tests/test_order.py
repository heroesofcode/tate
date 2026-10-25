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


def test_side_by_side_strips_are_read_a_column_at_a_time() -> None:
    boxes = [
        Box(0, 0, 100, 100),
        Box(0, 120, 100, 100),
        Box(0, 240, 100, 100),
        Box(0, 360, 100, 100),
        Box(160, 0, 100, 100),
        Box(160, 120, 100, 100),
        Box(160, 240, 100, 100),
        Box(160, 360, 100, 100),
    ]
    assert reading_order(boxes, "ltr") == [0, 1, 2, 3, 4, 5, 6, 7]
    assert reading_order(boxes) == [4, 5, 6, 7, 0, 1, 2, 3]


def test_slanted_gutter_reads_the_right_column_before_the_left_panel() -> None:
    # The left panel's box overlaps the right column because the gutter slants.
    boxes = [
        Box(620, 0, 610, 880),
        Box(570, 850, 660, 260),
        Box(0, 0, 675, 970),
        Box(900, 1140, 330, 850),
        Box(0, 1140, 380, 850),
    ]
    assert reading_order(boxes) == [0, 1, 2, 3, 4]
    assert reading_order(boxes, "ltr") == [2, 0, 1, 4, 3]


def test_staggered_frames_are_read_top_to_bottom() -> None:
    boxes = [
        Box(80, 150, 500, 400),
        Box(400, 570, 520, 400),
        Box(80, 980, 500, 400),
        Box(400, 1400, 520, 400),
    ]
    assert reading_order(boxes) == [0, 1, 2, 3]
    assert reading_order(boxes, "ltr") == [0, 1, 2, 3]


def test_inset_is_read_in_the_row_it_sits_in() -> None:
    boxes = [
        Box(0, 0, 200, 200),
        Box(220, 0, 200, 200),
        Box(280, 40, 60, 50),
        Box(0, 220, 420, 160),
    ]
    assert reading_order(boxes) == [1, 2, 0, 3]
    assert reading_order(boxes, "ltr") == [0, 1, 2, 3]


def test_row_order_survives_boxes_that_bleed_across_the_gutter() -> None:
    boxes = [
        Box(0, 0, 110, 110),
        Box(100, 0, 110, 110),
        Box(0, 100, 110, 110),
        Box(100, 100, 110, 110),
    ]
    assert reading_order(boxes) == [1, 0, 3, 2]
    assert reading_order(boxes, "ltr") == [0, 1, 2, 3]


def test_side_by_side_strips_survive_a_small_overlap() -> None:
    boxes = [Box(col * 150 - 6, row * 110 - 6, 112, 100) for col in range(2) for row in range(4)]
    assert reading_order(boxes, "ltr") == [0, 1, 2, 3, 4, 5, 6, 7]
    assert reading_order(boxes) == [4, 5, 6, 7, 0, 1, 2, 3]


def test_staircase_falls_back_to_rows() -> None:
    # Right panel blocks a horizontal cut; the bottom banner blocks a vertical one.
    boxes = [
        Box(0, 0, 100, 80),
        Box(110, 0, 100, 150),
        Box(0, 160, 210, 80),
    ]
    assert reading_order(boxes) == [1, 0, 2]


def test_equal_boxes_keep_one_order_whatever_the_input_order() -> None:
    a = Box(0, 0, 10, 10, 0.2)
    b = Box(0, 0, 10, 10, 0.9)
    first = [[a, b][index] for index in reading_order([a, b])]
    second = [[b, a][index] for index in reading_order([b, a])]
    assert first == second
