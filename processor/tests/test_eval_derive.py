from PIL import Image

from tate.boxes import Box
from tate.eval.derive import apply_geometry, render_ops, rotation_mapper
from tate.eval.model import (
    Contrast,
    Gamma,
    Geometry,
    Jpeg,
    Noise,
    PanelAnn,
    Rotate,
    Scale,
    Yellow,
)


def _geometry() -> Geometry:
    return Geometry(
        size=(100, 80),
        panels=(PanelAnn(Box(10, 20, 30, 30)),),
        balloons=(Box(12, 22, 8, 6),),
        ignore=(Box(0, 70, 10, 10),),
        whole=False,
    )


def test_right_angles_and_scale_move_box_corners() -> None:
    turned = apply_geometry(_geometry(), (Rotate(90),))
    assert turned.size == (80, 100)
    assert (
        turned.panels[0].box.x,
        turned.panels[0].box.y,
        turned.panels[0].box.w,
        turned.panels[0].box.h,
    ) == (
        20,
        60,
        30,
        30,
    )

    half = apply_geometry(_geometry(), (Rotate(180),))
    assert half.size == (100, 80)
    assert (half.panels[0].box.x, half.panels[0].box.y) == (60, 30)

    scaled = apply_geometry(_geometry(), (Scale(2),))
    assert scaled.size == (200, 160)
    assert (
        scaled.panels[0].box.x,
        scaled.panels[0].box.y,
        scaled.panels[0].box.w,
        scaled.panels[0].box.h,
    ) == (
        20,
        40,
        60,
        60,
    )


def test_polygon_bounds_follow_the_transformed_points() -> None:
    panel = PanelAnn(Box(10, 20, 30, 30), ((10, 20), (40, 35), (10, 50)))
    geometry = Geometry((100, 80), (panel,), (), (), False)
    turned = apply_geometry(geometry, (Rotate(90),))
    assert turned.panels[0].poly is not None
    assert (
        turned.panels[0].box.x,
        turned.panels[0].box.y,
        turned.panels[0].box.x2,
        turned.panels[0].box.y2,
    ) == (
        20,
        60,
        50,
        90,
    )


def test_photometric_ops_keep_the_geometry() -> None:
    ops = (Contrast(0.5), Gamma(1.2), Noise(3, 1), Jpeg(80), Yellow(0.2))
    assert apply_geometry(_geometry(), ops) == _geometry()


def test_rotation_canvas_matches_pillow_and_a_pixel_stays_inside_its_box() -> None:
    image = Image.new("RGB", (100, 80), "white")
    image.putpixel((10, 20), (255, 0, 0))
    turned = render_ops(image, (Rotate(90),))
    _mapper, size = rotation_mapper(100, 80, 90)
    assert turned.size == size
    moved = apply_geometry(
        Geometry((100, 80), (PanelAnn(Box(10, 20, 1, 1)),), (), (), False),
        (Rotate(90),),
    )
    box = moved.panels[0].box
    assert any(turned.getpixel((x, y))[0] > 200 for y in range(box.y, box.y2) for x in range(box.x, box.x2))

    wide = Image.new("RGB", (30, 20), "white")
    rendered = render_ops(wide, (Rotate(15),))
    assert rendered.size == wide.rotate(15, expand=True, fillcolor=(255, 255, 255)).size


def test_photometric_pixels_are_deterministic() -> None:
    image = Image.new("RGB", (6, 4), (100, 150, 200))
    assert render_ops(image, (Contrast(0),)).getpixel((0, 0)) == (128, 128, 128)
    assert render_ops(image, (Gamma(1),)).getpixel((1, 1)) == (100, 150, 200)
    assert render_ops(image, (Yellow(1),)).getpixel((0, 0)) == (242, 228, 196)
    first = render_ops(image, (Noise(12, 7),)).tobytes()
    assert first == render_ops(image, (Noise(12, 7),)).tobytes()
    assert first != render_ops(image, (Noise(12, 8),)).tobytes()
    assert render_ops(image, (Jpeg(40),)).mode == "RGB"
