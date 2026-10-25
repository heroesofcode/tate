from PIL import Image, ImageDraw, ImageFont

from tate.boxes import Box, Detection, clean_panels, drop_slivers, separate_overlaps
from tate.debug import draw_debug
from tate.detect.page import PageDetector
from tate.gutters import _valley_shoulders, absorb_slivers, split_on_gutters
from tate.ink import cover_boxes
from tate.layout import (
    MAX_SEGMENT_HEIGHT,
    assign_gaps,
    expand_for_text,
    gap_sizes,
    layout_page,
)


def _layout(image: Image.Image, panels: tuple[Box, ...], texts: tuple[Box, ...] = ()) -> list:
    return layout_page(image, Detection(panels, texts), page_index=0, width=1080).segments


def test_balloon_that_crosses_a_border_is_kept_whole() -> None:
    image = Image.new("RGB", (500, 700), "white")
    panels = (Box(20, 20, 200, 300), Box(280, 20, 200, 300))
    # Overlaps the left panel by 40% of its area and the right panel by less.
    text = Box(180, 80, 100, 60)
    expanded = expand_for_text(panels, (text,), 500, 400)
    assert expanded[0].x <= 180
    assert expanded[0].y <= 80
    assert expanded[0].x2 >= 280
    assert expanded[1].x == 280

    segments = _layout(image, panels, (text,))
    assert len(segments) == 2
    left = min(segments, key=lambda segment: segment.box.x)
    assert left.box.x <= 180
    assert left.box.x2 >= 280
    # The crop grows to keep the balloon; the panel box stays the detector box.
    assert (left.panel.x, left.panel.y, left.panel.w, left.panel.h) == (20, 20, 200, 300)
    assert left.panel.x2 < left.box.x2


def test_orphan_text_attaches_to_the_nearest_panel() -> None:
    panels = (Box(20, 20, 180, 200), Box(280, 20, 180, 200))
    text = Box(10, 360, 30, 30)
    expanded = expand_for_text(panels, (text,), 500, 500)
    assert expanded[0].y2 >= 390
    assert expanded[1].y2 == 220


def test_one_panel_falls_back_to_the_whole_page() -> None:
    image = Image.new("RGB", (300, 400), "white")
    segments = _layout(image, (Box(10, 10, 80, 80),))
    assert len(segments) == 1
    assert (segments[0].box.x, segments[0].box.y, segments[0].box.w, segments[0].box.h) == (
        0,
        0,
        300,
        400,
    )


def test_tan_paper_around_real_panels_is_not_a_fallback() -> None:
    image = Image.new("RGB", (400, 640), (176, 164, 140))
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 384, 290), outline=(10, 10, 10), width=6)
    draw.rectangle((16, 330, 384, 610), outline=(10, 10, 10), width=6)
    draw.line((40, 40, 300, 250), fill=(10, 10, 10), width=4)
    draw.line((40, 360, 300, 560), fill=(10, 10, 10), width=4)
    laid = layout_page(
        image,
        Detection((Box(16, 16, 368, 274), Box(16, 330, 368, 280)), ()),
        page_index=0,
        width=400,
    )
    assert laid.fallback is False
    assert laid.spread is False
    assert len(laid.segments) == 2


def _ink_along_the_inset(draw: ImageDraw.ImageDraw, panel: Box) -> None:
    """Dark pixels just inside the box, staggered so they are not a straight rule."""
    for y in range(panel.y, panel.y2):
        column = (y // 36) % 3
        draw.point((panel.x + column, y), fill=(0, 0, 0))
        draw.point((panel.x2 - 1 - column, y), fill=(0, 0, 0))
    for x in range(panel.x, panel.x2):
        row = (x // 36) % 3
        draw.point((x, panel.y + row), fill=(0, 0, 0))
        draw.point((x, panel.y2 - 1 - row), fill=(0, 0, 0))


def test_dark_art_inside_the_border_does_not_hide_a_missed_panel() -> None:
    image = Image.new("RGB", (900, 1000), "white")
    draw = ImageDraw.Draw(image)
    panels = (Box(40, 40, 240, 860), Box(320, 40, 240, 860))
    for panel in panels:
        _ink_along_the_inset(draw, panel)
        draw.rectangle((panel.x + 36, panel.y + 36, panel.x + 140, panel.y + 200), fill=(10, 10, 10))
    draw.rectangle((640, 80, 860, 900), fill=(30, 30, 30))
    laid = layout_page(image, Detection(panels, ()), page_index=0, width=1080)
    assert laid.fallback is False
    assert len(laid.segments) == 3
    assert any(segment.panel.x > 500 for segment in laid.segments)


def test_dark_art_inside_the_border_still_falls_back_when_ink_is_missed() -> None:
    image = Image.new("RGB", (500, 700), "white")
    draw = ImageDraw.Draw(image)
    # Larger than a panel, so recovery cannot adopt it, and outside both boxes.
    draw.rectangle((20, 20, 480, 490), fill=(0, 0, 0))
    panels = (Box(30, 530, 180, 150), Box(280, 530, 180, 150))
    for panel in panels:
        _ink_along_the_inset(draw, panel)
    laid = layout_page(image, Detection(panels, ()), page_index=0, width=1080)
    assert laid.fallback is True
    assert len(laid.segments) == 1
    assert (laid.segments[0].box.w, laid.segments[0].box.h) == (500, 700)


def test_low_ink_coverage_falls_back_to_the_whole_page() -> None:
    image = Image.new("RGB", (400, 400), "white")
    ImageDraw.Draw(image).rectangle((30, 30, 369, 369), fill=(0, 0, 0))
    segments = _layout(image, (Box(40, 40, 40, 40), Box(300, 300, 40, 40)))
    assert len(segments) == 1
    assert segments[0].box.w == 400
    assert segments[0].box.h == 400


def test_yellow_paper_outside_the_panels_is_not_missed_ink() -> None:
    paper = (212, 184, 140)
    image = Image.new("RGB", (400, 800), paper)
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 360, 300), fill=(20, 20, 20))
    draw.rectangle((40, 480, 360, 740), fill=(20, 20, 20))
    panels = (Box(40, 40, 320, 260), Box(40, 480, 320, 260))
    laid = layout_page(image, Detection(panels, ()), page_index=0, width=1080)
    assert laid.fallback is False
    assert len(laid.segments) == 2


def test_a_rule_just_outside_the_box_keeps_the_page() -> None:
    image = Image.new("RGB", (500, 800), (230, 210, 170))
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 30, 240, 760), outline=(0, 0, 0), width=4)
    draw.rectangle((260, 30, 480, 760), outline=(0, 0, 0), width=4)
    draw.rectangle((246, 40, 256, 750), fill=(0, 0, 0))
    draw.rectangle((40, 50, 55, 65), fill=(0, 0, 0))
    draw.rectangle((280, 50, 295, 65), fill=(0, 0, 0))
    panels = (Box(28, 40, 200, 710), Box(270, 40, 200, 710))
    laid = layout_page(image, Detection(panels, ()), page_index=0, width=1080)
    assert laid.fallback is False
    assert len(laid.segments) == 2


def test_spread_stays_whole_and_scales_to_the_strip() -> None:
    image = Image.new("RGB", (1600, 1000), (8, 8, 8))
    laid = layout_page(
        image,
        Detection((Box(0, 0, 800, 1000), Box(800, 0, 800, 1000)), ()),
        page_index=2,
        width=1080,
    )
    assert laid.spread
    assert len(laid.segments) == 1
    segment = laid.segments[0]
    assert (segment.box.x, segment.box.y, segment.box.w, segment.box.h) == (0, 0, 1600, 1000)
    assert segment.image.size == (1080, round(1000 * 1080 / 1600))
    assert segment.page == 2


def test_narrow_page_is_upscaled_at_most_1_6_and_centered() -> None:
    image = Image.new("RGB", (100, 3000), (1, 2, 3))
    segments = _layout(image, ())
    assert len(segments) == 2
    assert segments[0].continuation is False
    assert segments[1].continuation is True
    assert segments[0].image.size == (1080, MAX_SEGMENT_HEIGHT)
    assert segments[1].image.height == round(3000 * 1.6) - MAX_SEGMENT_HEIGHT
    assert segments[0].image.getpixel((0, 0)) == (255, 255, 255)
    assert segments[0].image.getpixel((540, 200)) == (1, 2, 3)
    assert segments[0].box.y == 0
    assert segments[1].box.y == segments[0].box.y2
    assert segments[-1].box.y2 == 3000
    assert segments[0].panel == segments[1].panel == Box(0, 0, 100, 3000)
    laid = layout_page(image, Detection((), ()), page_index=0, width=1080)
    assert laid.emitted_panels() == [Box(0, 0, 100, 3000)]


def test_nested_duplicate_is_removed() -> None:
    panels = [Box(10, 10, 200, 300), Box(20, 20, 170, 200)]
    cleaned = clean_panels(panels, 400, 500)
    assert len(cleaned) == 1
    assert cleaned[0].w == 200


def test_bottom_sliver_is_dropped() -> None:
    page = Box(40, 40, 300, 400)
    sliver = Box(50, 425, 280, 30)
    kept = drop_slivers([page, sliver], 500, 700)
    assert kept == [page]


def test_stacked_overlap_is_split_instead_of_repeated() -> None:
    top = Box(0, 0, 200, 120)
    bottom = Box(0, 100, 200, 120)
    separated = separate_overlaps([top, bottom])
    assert separated[0].y2 <= separated[1].y
    assert separated[0].h + separated[1].h == 220


def test_title_column_across_a_gutter_is_its_own_segment() -> None:
    image = Image.new("RGB", (800, 1100), "white")
    draw = ImageDraw.Draw(image)
    panel = Box(40, 80, 340, 860)
    # Wide enough to stay a column, close enough that growing across the
    # blank paper would meet the panel and hide the gutter.
    column = Box(panel.x2 + 26, 120, 70, 700)
    draw.rectangle((panel.x, panel.y, panel.x2, panel.y2), fill=(30, 30, 30))
    draw.rectangle((column.x, column.y, column.x2, column.y2), fill=(20, 20, 20))
    laid = layout_page(image, Detection((panel,), ()), page_index=0, width=1080)
    assert laid.fallback is False
    assert len(laid.segments) == 2
    assert any(segment.panel.x >= panel.x2 and segment.panel.x2 >= column.x2 - 4 for segment in laid.segments)
    image.close()


def test_narrow_recovered_column_survives_sliver_cleanup() -> None:
    image = Image.new("RGB", (800, 1100), "white")
    draw = ImageDraw.Draw(image)
    column = Box(772, 100, 28, 760)
    panel = Box(30, 80, column.x - 36 - 30, 780)
    draw.rectangle((panel.x, panel.y, panel.x2, panel.y2), fill=(30, 30, 30))
    draw.rectangle((column.x, column.y, column.x2, column.y2), fill=(20, 20, 20))
    laid = layout_page(image, Detection((panel,), ()), page_index=0, width=1080)
    assert laid.fallback is False
    assert len(laid.segments) == 2
    assert any(segment.panel.x > panel.x2 and segment.panel.w < 80 for segment in laid.segments)
    image.close()


def test_a_missed_panel_is_recovered_from_uncovered_ink() -> None:
    image = Image.new("RGB", (800, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 340, 940), fill=(40, 40, 40))
    draw.rectangle((460, 40, 760, 940), fill=(40, 40, 40))
    laid = layout_page(
        image,
        Detection((Box(40, 40, 300, 900),), ()),
        page_index=0,
        width=1080,
    )
    assert laid.fallback is False
    assert len(laid.segments) == 2
    # The crop may take in the gutter; the panel itself stays on the right.
    assert any(segment.panel.x > 350 and segment.box.x2 > 700 for segment in laid.segments)


def test_two_bands_through_one_drawing_stay_one_page() -> None:
    image = Image.new("RGB", (400, 800), "white")
    ImageDraw.Draw(image).rectangle((20, 20, 380, 780), fill=(30, 30, 30))
    segments = _layout(image, (Box(20, 20, 360, 420), Box(20, 380, 360, 400)))
    assert len(segments) == 1
    assert segments[0].box.h == 800


def test_captions_outside_a_frame_do_not_drop_the_page() -> None:
    image = Image.new("RGB", (600, 800), (220, 190, 150))
    draw = ImageDraw.Draw(image)
    panels = (Box(30, 40, 220, 720), Box(360, 40, 210, 720))
    for panel in panels:
        draw.rectangle((panel.x, panel.y, panel.x2 - 1, panel.y2 - 1), outline=(0, 0, 0), width=5)
        draw.rectangle((panel.x + 16, panel.y + 16, panel.x + 48, panel.y + 48), fill=(0, 0, 0))
    draw.rectangle((270, 80, 340, 740), fill=(0, 0, 0))
    laid = layout_page(image, Detection(panels, ()), page_index=0, width=1080)
    assert laid.fallback is False
    assert len(laid.segments) == 2


def test_overlapping_boxes_with_a_white_gutter_stay_split() -> None:
    image = Image.new("RGB", (400, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 380, 340), fill=(30, 30, 30))
    draw.rectangle((20, 460, 380, 760), fill=(30, 30, 30))
    segments = _layout(image, (Box(20, 20, 360, 400), Box(20, 380, 360, 400)))
    assert len(segments) == 2


def test_two_panels_with_a_white_gutter_stay_split() -> None:
    image = Image.new("RGB", (400, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 380, 340), fill=(30, 30, 30))
    draw.rectangle((20, 420, 380, 760), fill=(30, 30, 30))
    segments = _layout(image, (Box(20, 20, 360, 320), Box(20, 420, 360, 340)))
    assert len(segments) == 2


def test_gaps_are_larger_between_pages() -> None:
    panel, page = gap_sizes(1080)
    assert page > panel
    first = layout_page(Image.new("RGB", (80, 80), "white"), Detection((), ()), page_index=0, width=1080)
    second = layout_page(Image.new("RGB", (80, 80), "white"), Detection((), ()), page_index=1, width=1080)
    gaps = assign_gaps([*first.segments, *second.segments], 1080)
    assert gaps == [0, page]


def _glyph_pixels(digit: str, font_size: int) -> set[tuple[int, int]]:
    font = ImageFont.load_default(size=font_size)
    canvas = Image.new("RGB", (font_size * 2, font_size * 2), (255, 255, 255))
    ImageDraw.Draw(canvas).text(
        (0, 0),
        digit,
        fill=(40, 160, 70),
        font=font,
        stroke_width=2,
        stroke_fill=(255, 255, 255),
    )
    found: set[tuple[int, int]] = set()
    for y in range(canvas.height):
        for x in range(canvas.width):
            if canvas.getpixel((x, y)) == (40, 160, 70):
                found.add((x, y))
    canvas.close()
    return found


def _shows_digit(overlay: Image.Image, box: Box, digit: str) -> bool:
    line = max(4, min(overlay.size) // 350)
    font_size = max(28, min(overlay.size) // 32)
    glyph = _glyph_pixels(digit, font_size)
    assert glyph
    origin_x = box.x + line
    origin_y = box.y + line
    for x, y in glyph:
        px = origin_x + x
        py = origin_y + y
        if px < 0 or py < 0 or px >= overlay.width or py >= overlay.height:
            return False
        if overlay.getpixel((px, py)) != (40, 160, 70):
            return False
    return True


def test_direction_orders_panels_and_debug_numbers() -> None:
    image = Image.new("RGB", (420, 560), "white")
    panels = (Box(16, 20, 174, 240), Box(220, 20, 180, 240))
    detection = Detection(panels, ())
    rtl = layout_page(image, detection, page_index=0, width=200)
    explicit = layout_page(image, detection, page_index=0, width=200, direction="rtl")
    ltr = layout_page(image, detection, page_index=0, width=200, direction="ltr")
    assert [segment.panel.x for segment in rtl.segments] == [segment.panel.x for segment in explicit.segments]
    assert [segment.panel.x for segment in rtl.segments] == [220, 16]
    assert [segment.panel.x for segment in ltr.segments] == [16, 220]
    assert [box.x for box in ltr.detected] == [box.x for box in (segment.box for segment in ltr.segments)]
    overlay = draw_debug(image, ltr)
    assert _shows_digit(overlay, ltr.detected[0], "1")
    assert _shows_digit(overlay, ltr.detected[1], "2")
    overlay.close()
    image.close()


def test_same_page_panels_use_the_smaller_gap() -> None:
    image = Image.new("RGB", (420, 560), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 20, 190, 520), fill=(180, 180, 180))
    draw.rectangle((220, 20, 400, 520), fill=(160, 160, 160))
    panels = (Box(16, 20, 174, 500), Box(220, 20, 180, 500))
    segments = _layout(image, panels)
    assert len(segments) == 2
    gaps = assign_gaps(segments, 1080)
    assert gaps == [0, gap_sizes(1080)[0]]


def test_landscape_page_of_panels_is_not_a_spread() -> None:
    image = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(image)
    panels: list[Box] = []
    for row in range(2):
        for col in range(3):
            x = 24 + col * 294
            y = 24 + row * 286
            draw.rectangle((x, y, x + 250, y + 250), fill=(30, 30, 30))
            panels.append(Box(x, y, 250, 250))
    laid = layout_page(image, Detection(tuple(panels), ()), page_index=0, width=400)
    assert laid.spread is False
    assert laid.fallback is False
    assert len(laid.segments) == 6


def _wide_strip() -> Image.Image:
    """Landscape page. The gutters cut a page-sized box into six panels."""
    image = Image.new("RGB", (1200, 920), "white")
    draw = ImageDraw.Draw(image)
    origin_x = (1200 - 640) // 2
    for row in range(2):
        for col in range(3):
            x = origin_x + 20 + col * 206
            y = 24 + row * 450
            draw.rectangle((x, y, x + 176, y + 400), fill=(20, 20, 20))
    return image


def _spread_with_title() -> tuple[Image.Image, Box, Box, Box]:
    image = Image.new("RGB", (1600, 1000), "white")
    draw = ImageDraw.Draw(image)
    left = Box(40, 180, 700, 780)
    right = Box(860, 180, 700, 780)
    title = Box(500, 30, 600, 110)
    draw.rectangle((left.x, left.y, left.x2, left.y2), fill=(30, 30, 30))
    draw.rectangle((right.x, right.y, right.x2, right.y2), fill=(30, 30, 30))
    draw.rectangle((title.x, title.y, title.x2, title.y2), fill=(10, 10, 10))
    return image, left, right, title


def test_a_full_page_box_is_split_along_the_gutters() -> None:
    image = _wide_strip()
    page_w, page_h = image.size
    boxes = (
        Box(0, 0, page_w, page_h),
        Box(-8, -6, page_w + 16, page_h + 12),
    )
    try:
        for box in boxes:
            laid = layout_page(image, Detection((box,), ()), page_index=0, width=400)
            assert laid.spread is False
            assert laid.fallback is False
            assert len(laid.segments) == 6
    finally:
        image.close()


def test_page_mode_keeps_a_guttered_page_whole() -> None:
    image = _wide_strip()
    try:
        detection = PageDetector().detect(image)
        assert detection.panels == ()
        laid = layout_page(image, detection, page_index=0, width=400)
        assert laid.spread is True
        assert laid.fallback is False
        assert len(laid.segments) == 1
        segment = laid.segments[0]
        assert (segment.box.x, segment.box.y, segment.box.w, segment.box.h) == (0, 0, *image.size)
    finally:
        image.close()


def test_recovered_title_on_a_spread_stays_one_piece() -> None:
    image, left, right, _title = _spread_with_title()
    try:
        laid = layout_page(image, Detection((left, right), ()), page_index=0, width=800)
        assert laid.spread is True
        assert laid.fallback is False
        assert len(laid.segments) == 1
        segment = laid.segments[0]
        assert (segment.box.x, segment.box.y, segment.box.w, segment.box.h) == (0, 0, 1600, 1000)
    finally:
        image.close()


def test_a_detected_title_is_not_part_of_the_spread() -> None:
    image, left, right, title = _spread_with_title()
    try:
        laid = layout_page(image, Detection((left, right, title), ()), page_index=0, width=800)
        assert laid.spread is False
        assert laid.fallback is False
        assert len(laid.segments) == 3
    finally:
        image.close()


def test_a_merged_strip_is_split_along_the_gutters() -> None:
    image = Image.new("RGB", (640, 920), "white")
    draw = ImageDraw.Draw(image)
    for row in range(2):
        for col in range(3):
            x = 20 + col * 206
            y = 24 + row * 450
            draw.rectangle((x, y, x + 176, y + 400), fill=(20, 20, 20))
    laid = layout_page(
        image,
        Detection((Box(16, 16, 608, 880),), ()),
        page_index=0,
        width=400,
    )
    assert laid.spread is False
    assert laid.fallback is False
    assert len(laid.segments) == 6


def test_a_frame_rule_between_two_cells_is_a_gutter() -> None:
    image = Image.new("RGB", (600, 400), (210, 190, 160))
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 280, 360), outline=(0, 0, 0), width=4)
    draw.rectangle((300, 20, 560, 360), outline=(0, 0, 0), width=4)
    draw.rectangle((50, 80, 140, 180), fill=(0, 0, 0))
    draw.rectangle((360, 80, 460, 180), fill=(0, 0, 0))
    parts = split_on_gutters(image, [Box(20, 20, 540, 340)])
    assert len(parts) == 2
    assert parts[0].x2 <= parts[1].x


def test_a_tall_side_panel_is_not_split_by_the_neighbouring_column_rule() -> None:
    """A frame rule across one column must not cut the tall panel beside it."""
    image = Image.new("RGB", (1000, 900), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 590, 430), outline=(0, 0, 0), width=6)
    draw.rectangle((30, 450, 590, 860), outline=(0, 0, 0), width=6)
    draw.rectangle((70, 80, 280, 250), fill=(0, 0, 0))
    draw.rectangle((70, 520, 280, 720), fill=(0, 0, 0))
    draw.rectangle((640, 30, 970, 860), outline=(0, 0, 0), width=6)
    draw.rectangle((680, 80, 930, 820), fill=(30, 30, 30))
    column = Box(30, 30, 560, 830)
    tall = Box(640, 30, 330, 830)
    parts = split_on_gutters(image, [column, tall])
    column_parts = [part for part in parts if part.x2 <= 600]
    tall_parts = [part for part in parts if part.x >= 600]
    assert len(column_parts) == 2
    assert column_parts[0].y2 <= column_parts[1].y
    assert len(tall_parts) == 1
    assert tall_parts[0].y <= 40
    assert tall_parts[0].y2 >= 850


def test_a_bright_gap_counts_when_it_is_not_the_border_colour() -> None:
    image = Image.new("RGB", (1000, 800), (190, 140, 90))
    draw = ImageDraw.Draw(image)
    draw.rectangle((50, 50, 950, 750), fill=(250, 250, 245))
    draw.rectangle((80, 80, 460, 700), fill=(30, 40, 70))
    draw.rectangle((520, 80, 920, 700), fill=(70, 30, 40))
    parts = split_on_gutters(image, [Box(80, 80, 840, 620)])
    assert len(parts) == 2
    assert parts[0].x2 <= parts[1].x or parts[1].x2 <= parts[0].x
    image.close()


def test_a_grey_strip_between_strokes_is_not_a_gutter() -> None:
    image = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 860, 660), fill=(200, 200, 200))
    draw.rectangle((80, 80, 400, 600), fill=(10, 10, 10))
    draw.rectangle((480, 80, 820, 600), fill=(10, 10, 10))
    parts = split_on_gutters(image, [Box(80, 80, 740, 520)])
    assert len(parts) == 1
    image.close()


def test_a_line_that_stops_short_of_the_panel_is_not_a_gutter() -> None:
    image = Image.new("RGB", (800, 520), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 770, 490), fill=(210, 210, 210))
    draw.line((400, 110, 400, 390), fill=(0, 0, 0), width=5)
    draw.ellipse((140, 160, 280, 340), outline=(0, 0, 0), width=4)
    parts = split_on_gutters(image, [Box(30, 30, 740, 460)])
    assert len(parts) == 1
    image.close()


def test_a_thin_fragment_without_a_gutter_rejoins_its_panel() -> None:
    image = Image.new("RGB", (900, 900), (232, 220, 186))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 24, 876, 300), fill=(72, 84, 64))
    for origin in (690, 760, 830):
        draw.rectangle((origin, 36, origin + 30, 288), fill=(18, 18, 18))
    draw.rectangle((24, 480, 400, 860), fill=(40, 40, 40))
    draw.rectangle((480, 480, 876, 860), fill=(50, 50, 50))
    wide = Box(24, 24, 640, 276)
    slivers = (Box(680, 24, 52, 276), Box(748, 24, 44, 276), Box(818, 24, 52, 276))
    kept = absorb_slivers(image, [wide, *slivers])
    assert len(kept) == 1
    assert kept[0].x <= 24 and kept[0].x2 >= 860
    right = Box(480, 480, 396, 380)
    left = Box(24, 480, 376, 380)
    laid = layout_page(
        image,
        Detection((wide, *slivers, right, left), ()),
        page_index=0,
        width=400,
        direction="rtl",
    )
    assert laid.fallback is False
    assert len(laid.segments) == 3
    top = laid.segments[0].panel
    assert top.w > 700
    assert laid.segments[1].panel.x > laid.segments[2].panel.x
    image.close()


def test_a_narrow_framed_panel_sharing_an_edge_stays_separate() -> None:
    """A real narrow panel stays when a rule or a paper gap sits on the shared edge.

    The neighbour may meet that edge exactly, or the two boxes may overlap there.
    """
    wide = Box(40, 36, 560, 624)
    for shift in (0, 4):
        narrow = Box(600 - shift, 36, 70, 624)
        ruled = Image.new("RGB", (900, 700), "white")
        draw = ImageDraw.Draw(ruled)
        draw.rectangle((wide.x, wide.y, wide.x2, wide.y2), outline=(0, 0, 0), width=3)
        draw.rectangle((narrow.x, narrow.y, narrow.x2, narrow.y2), outline=(0, 0, 0), width=3)
        draw.rectangle((wide.x + 14, wide.y + 14, wide.x2 - 14, wide.y2 - 14), fill=(30, 30, 30))
        draw.rectangle((narrow.x + 14, narrow.y + 14, narrow.x2 - 14, narrow.y2 - 14), fill=(24, 24, 24))
        assert len(absorb_slivers(ruled, [wide, narrow])) == 2
        ruled.close()

        gapped = Image.new("RGB", (900, 700), "white")
        draw = ImageDraw.Draw(gapped)
        draw.rectangle((wide.x, wide.y, wide.x2 - 12, wide.y2), fill=(30, 30, 30))
        draw.rectangle((narrow.x + 12, narrow.y, narrow.x2, narrow.y2), fill=(24, 24, 24))
        assert len(absorb_slivers(gapped, [wide, narrow])) == 2
        gapped.close()

    wide_row = Box(30, 30, 840, 500)
    flat = Box(30, 530, 840, 44)
    ruled = Image.new("RGB", (900, 800), "white")
    draw = ImageDraw.Draw(ruled)
    draw.rectangle((wide_row.x, wide_row.y, wide_row.x2, wide_row.y2), outline=(0, 0, 0), width=3)
    draw.rectangle((flat.x, flat.y, flat.x2, flat.y2), outline=(0, 0, 0), width=3)
    draw.rectangle((wide_row.x + 14, wide_row.y + 14, wide_row.x2 - 14, wide_row.y2 - 14), fill=(30, 30, 30))
    draw.rectangle((flat.x + 14, flat.y + 14, flat.x2 - 14, flat.y2 - 14), fill=(24, 24, 24))
    assert len(absorb_slivers(ruled, [wide_row, flat])) == 2
    ruled.close()


def test_a_touching_sliver_without_a_rule_on_the_edge_still_folds() -> None:
    """A fragment that merely touches its panel still folds when the edge is open.

    A frame further inside the panel is not the edge between them.
    """
    wide = Box(40, 36, 560, 624)
    for shift in (0, 4):
        narrow = Box(600 - shift, 36, 70, 624)
        image = Image.new("RGB", (900, 700), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((wide.x, wide.y, 552, wide.y2), fill=(30, 30, 30))
        draw.rectangle((576, wide.y, narrow.x2, wide.y2), fill=(30, 30, 30))
        draw.line((564, wide.y, 564, wide.y2), fill=(0, 0, 0), width=3)
        kept = absorb_slivers(image, [wide, narrow])
        assert len(kept) == 1
        assert kept[0].x <= wide.x and kept[0].x2 >= narrow.x2
        image.close()

    wide_row = Box(30, 30, 840, 500)
    flat = Box(30, 530, 840, 44)
    image = Image.new("RGB", (900, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((wide_row.x, wide_row.y, wide_row.x2, 486), fill=(30, 30, 30))
    draw.rectangle((flat.x, 510, flat.x2, flat.y2), fill=(30, 30, 30))
    draw.line((wide_row.x, 498, wide_row.x2, 498), fill=(0, 0, 0), width=3)
    kept = absorb_slivers(image, [wide_row, flat])
    assert len(kept) == 1
    assert kept[0].y <= wide_row.y and kept[0].y2 >= flat.y2
    image.close()


def test_a_narrow_panel_across_a_paper_gap_is_not_a_sliver() -> None:
    image = Image.new("RGB", (800, 600), "white")
    draw = ImageDraw.Draw(image)
    wide = Box(30, 40, 520, 500)
    narrow = Box(580, 40, 48, 500)
    draw.rectangle((wide.x, wide.y, wide.x2, wide.y2), fill=(30, 30, 30))
    draw.rectangle((narrow.x, narrow.y, narrow.x2, narrow.y2), fill=(20, 20, 20))
    kept = absorb_slivers(image, [wide, narrow])
    assert len(kept) == 2
    image.close()


def test_a_hairline_gap_is_a_gutter_only_beside_a_solid_drawing() -> None:
    assert _valley_shoulders("valley", 1, 0.20, 0.20) is False
    assert _valley_shoulders("valley", 1, 0.90, 0.20) is True
    assert _valley_shoulders("valley", 8, 0.20, 0.20) is True


def test_a_stroke_through_a_drawing_is_not_a_gutter() -> None:
    image = Image.new("RGB", (800, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((180, 180, 620, 620), fill=(70, 70, 70))
    draw.line((400, 180, 400, 620), fill=(0, 0, 0), width=6)
    parts = split_on_gutters(image, [Box(180, 180, 440, 440)])
    assert len(parts) == 1


def test_crops_tile_the_page_so_margin_ink_is_kept() -> None:
    covered = cover_boxes([Box(40, 30, 80, 90), Box(200, 40, 70, 100)], 320, 240)
    for x, y in ((0, 0), (319, 0), (0, 239), (319, 239), (160, 120)):
        assert any(box.x <= x < box.x2 and box.y <= y < box.y2 for box in covered)

    image = Image.new("RGB", (500, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 80, 460, 360), fill=(40, 40, 40))
    draw.rectangle((40, 440, 460, 740), fill=(55, 55, 55))
    draw.rectangle((70, 8, 160, 40), fill=(0, 0, 0))
    laid = layout_page(
        image,
        Detection((Box(40, 80, 420, 280), Box(40, 440, 420, 300)), ()),
        page_index=0,
        width=400,
    )
    assert laid.fallback is False
    assert len(laid.segments) == 2
    assert any(
        segment.box.x <= 70 and segment.box.y <= 8 and segment.box.x2 >= 160 and segment.box.y2 >= 40
        for segment in laid.segments
    )
