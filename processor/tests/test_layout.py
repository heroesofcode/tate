from PIL import Image, ImageDraw

from tate.boxes import Box, Detection, clean_panels, drop_slivers, separate_overlaps
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
    assert any(segment.box.x > 350 for segment in laid.segments)


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
