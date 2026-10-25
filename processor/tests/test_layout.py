from PIL import Image, ImageDraw, ImageFont

from tate.boxes import Box, Detection, clean_panels, drop_slivers, separate_overlaps
from tate.debug import draw_debug
from tate.detect.page import PageDetector
from tate.gutters import split_on_gutters
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


def test_low_ink_coverage_falls_back_to_the_whole_page() -> None:
    image = Image.new("RGB", (400, 400), "black")
    segments = _layout(image, (Box(10, 10, 40, 40), Box(300, 300, 40, 40)))
    assert len(segments) == 1
    assert segments[0].box.w == 400
    assert segments[0].box.h == 400


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
