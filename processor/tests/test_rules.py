import numpy as np
from PIL import Image, ImageDraw

from tate.boxes import Box, Detection, iou
from tate.detect.rules import adopt_ruled_grid, ruled_cells
from tate.layout import layout_page


def _tan_page(width: int, height: int) -> Image.Image:
    """Tan paper, lighter toward the bottom and warmer toward the right."""
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    tone = 150.0 + 48.0 * y + 28.0 * x
    rgb = np.stack(
        [
            np.clip(tone + 22.0, 0, 255),
            np.clip(tone, 0, 255),
            np.clip(tone - 30.0, 0, 255),
        ],
        axis=2,
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _ruled_sheet() -> tuple[Image.Image, list[Box]]:
    """Two by two cells on uneven paper, closed by thin rules."""
    image = _tan_page(800, 1000)
    draw = ImageDraw.Draw(image)
    cells = [
        Box(36, 36, 350, 440),
        Box(410, 36, 354, 440),
        Box(36, 510, 350, 454),
        Box(410, 510, 354, 454),
    ]
    for cell in cells:
        draw.rectangle((cell.x, cell.y, cell.x2 - 1, cell.y2 - 1), outline=(18, 16, 12), width=3)
    return image, cells


def test_a_merged_sheet_on_tan_paper_is_replaced_by_its_cells() -> None:
    image, cells = _ruled_sheet()
    # One box sits on a cell. The other covers the two cells in the bottom row.
    detections = (cells[0], Box(36, 510, 728, 454))
    adopted = adopt_ruled_grid(image, detections)
    assert len(adopted) == 4
    for cell in cells:
        assert max(iou(cell, box) for box in adopted) >= 0.7


def test_equal_floating_panels_drop_the_extra_boxes() -> None:
    image = _tan_page(1000, 1400)
    draw = ImageDraw.Draw(image)
    cells = [
        Box(80, 80, 300, 340),
        Box(620, 80, 300, 340),
        Box(80, 980, 300, 340),
        Box(620, 980, 300, 340),
    ]
    for cell in cells:
        draw.rectangle((cell.x, cell.y, cell.x2 - 1, cell.y2 - 1), outline=(12, 10, 8), width=4)
    extras = tuple(Box(cell.x + 12, cell.y + 18, cell.w - 20, cell.h - 30) for cell in cells[:2])
    adopted = adopt_ruled_grid(image, (*cells, *extras))
    assert len(adopted) == 4
    for cell in cells:
        assert max(iou(cell, box) for box in adopted) >= 0.7


def test_a_grid_the_detector_already_matched_is_kept() -> None:
    image, cells = _ruled_sheet()
    adopted = adopt_ruled_grid(image, cells)
    assert len(adopted) == len(cells)
    for original, adopted_box in zip(cells, adopted, strict=True):
        assert (adopted_box.x, adopted_box.y, adopted_box.w, adopted_box.h) == (
            original.x,
            original.y,
            original.w,
            original.h,
        )


def test_slightly_unequal_floating_panels_drop_the_extra_boxes() -> None:
    image = _tan_page(1000, 1400)
    draw = ImageDraw.Draw(image)
    cells = [
        Box(70, 70, 300, 340),
        Box(630, 70, 250, 330),
        Box(70, 980, 300, 340),
        Box(630, 980, 300, 340),
    ]
    for cell in cells:
        draw.rectangle((cell.x, cell.y, cell.x2 - 1, cell.y2 - 1), outline=(12, 10, 8), width=4)
    extras = tuple(Box(cell.x + 10, cell.y + 16, cell.w - 24, cell.h - 28) for cell in cells[:2])
    adopted = adopt_ruled_grid(image, (*cells, *extras))
    assert len(adopted) == 4
    for cell in cells:
        assert max(iou(cell, box) for box in adopted) >= 0.7


def test_insets_on_a_matched_grid_snap_to_the_rules() -> None:
    image, cells = _ruled_sheet()
    insets = tuple(Box(cell.x + 18, cell.y + 22, cell.w - 40, cell.h - 48) for cell in cells)
    adopted = adopt_ruled_grid(image, insets)
    assert len(adopted) == 4
    for cell in cells:
        assert max(iou(cell, box) for box in adopted) >= 0.85


def test_a_wide_margin_still_splits_a_merged_row() -> None:
    image = _tan_page(800, 900)
    draw = ImageDraw.Draw(image)
    cells = [
        Box(20, 180, 360, 280),
        Box(420, 180, 360, 280),
        Box(20, 500, 360, 280),
        Box(420, 500, 360, 280),
    ]
    for cell in cells:
        draw.rectangle((cell.x, cell.y, cell.x2 - 1, cell.y2 - 1), outline=(16, 14, 10), width=3)
    detections = (cells[0], cells[1], Box(20, 500, 760, 280))
    adopted = adopt_ruled_grid(image, detections)
    assert len(adopted) == 4
    for cell in cells:
        assert max(iou(cell, box) for box in adopted) >= 0.7


def test_a_stroke_through_a_drawing_is_not_a_ruled_grid() -> None:
    image = Image.new("RGB", (800, 800), (186, 176, 150))
    draw = ImageDraw.Draw(image)
    draw.rectangle((160, 160, 640, 640), fill=(72, 72, 72))
    draw.line((400, 160, 400, 640), fill=(0, 0, 0), width=6)
    panel = Box(160, 160, 480, 480)
    cells, _fills = ruled_cells(image)
    assert len(cells) < 4
    adopted = adopt_ruled_grid(image, (panel,))
    assert len(adopted) == 1
    assert iou(adopted[0], panel) > 0.95


def test_layout_emits_the_cells_of_a_merged_tan_sheet() -> None:
    image, cells = _ruled_sheet()
    laid = layout_page(
        image,
        Detection((cells[0], Box(36, 510, 728, 454)), ()),
        page_index=0,
        width=400,
    )
    assert laid.fallback is False
    assert laid.spread is False
    assert len(laid.emitted_panels()) == 4
