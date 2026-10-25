"""Boxes that look like slivers but are real panels."""

from tate.boxes import Box, drop_slivers


def test_a_narrow_panel_that_spans_the_row_is_kept() -> None:
    wide = Box(0, 0, 400, 500)
    narrow = Box(420, 0, 40, 500)
    kept = drop_slivers([wide, narrow], 600, 520)
    assert kept == [wide, narrow]
