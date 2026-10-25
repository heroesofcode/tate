"""Balloons and captions that cross a gutter stay inside one crop."""

from PIL import Image, ImageDraw

from tate.balloons import find_balloons
from tate.boxes import Box, Detection, contained_ratio
from tate.layout import layout_page


def test_an_outlined_balloon_crossing_the_gutter_stays_in_one_crop() -> None:
    image = Image.new("RGB", (400, 560), (245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((24, 30, 176, 520), fill=(35, 35, 35))
    draw.rectangle((224, 30, 376, 520), fill=(35, 35, 35))
    draw.ellipse((150, 160, 270, 280), fill=(255, 255, 255), outline=(0, 0, 0), width=3)
    draw.line((175, 200, 245, 200), fill=(0, 0, 0), width=2)
    draw.line((175, 224, 230, 224), fill=(0, 0, 0), width=2)
    balloon = Box(150, 160, 120, 120)
    found = find_balloons(image)
    assert any(contained_ratio(balloon, box) >= 0.98 for box in found)

    panels = (Box(16, 16, 176, 528), Box(208, 16, 176, 528))
    laid = layout_page(image, Detection(panels, ()), page_index=0, width=400)
    assert laid.fallback is False
    assert sum(contained_ratio(balloon, segment.box) >= 0.98 for segment in laid.segments) == 1


def test_a_caption_line_in_the_gutter_stays_in_one_crop() -> None:
    image = Image.new("RGB", (320, 520), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 300, 200), fill=(30, 30, 30))
    draw.rectangle((16, 300, 300, 500), fill=(30, 30, 30))
    for y in range(236, 250):
        for x in range(40, 250, 4):
            draw.line((x, y, x + 2, y), fill=(0, 0, 0))
    caption = Box(40, 234, 210, 18)
    found = find_balloons(image)
    assert any(contained_ratio(caption, box) >= 0.98 for box in found)

    panels = (Box(16, 16, 284, 184), Box(16, 300, 284, 200))
    laid = layout_page(image, Detection(panels, ()), page_index=0, width=400)
    assert laid.fallback is False
    assert sum(contained_ratio(caption, segment.box) >= 0.98 for segment in laid.segments) == 1
