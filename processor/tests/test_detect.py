import hashlib
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image, ImageDraw

from tate.boxes import Box, contained_ratio
from tate.detect.cv import CvDetector
from tate.detect.frames import frame_panels, fuse_panels, merge_panel_guides
from tate.detect.fuse import fuse_detections
from tate.detect.model import MODEL_SHA256, MODEL_URL, write_verified
from tate.detect.yolo import PANEL_CONF, YoloDetector
from tate.errors import TateError
from tests.pages import grid_page


def test_model_pin_is_the_published_export() -> None:
    assert MODEL_SHA256 == "e66667bc6d5f00013ff27efc15d21e521825369d44dfd5d7f6e43cda2ca512b7"
    assert "f6b7f3cb2bada1ec7d8b16501f9c984890811445" in MODEL_URL


def test_write_verified_refuses_a_mismatch_and_leaves_no_partial(tmp_path: Any) -> None:
    dest = tmp_path / "model.onnx"
    with pytest.raises(TateError, match="sha256"):
        write_verified(b"not a model", dest, "0" * 64)
    assert not dest.exists()
    assert list(tmp_path.iterdir()) == []


def test_write_verified_replaces_atomically(tmp_path: Any) -> None:
    dest = tmp_path / "cache" / "model.onnx"
    data = b"weights"
    write_verified(data, dest, hashlib.sha256(data).hexdigest())
    assert dest.read_bytes() == data
    assert not list(dest.parent.glob(".*.part"))


def test_yolo_maps_letterboxed_boxes() -> None:
    image = Image.new("RGB", (200, 400), "white")

    class FakeSession:
        def get_inputs(self) -> list[SimpleNamespace]:
            return [SimpleNamespace(name="images")]

        def run(self, _names: object, feed: dict[str, np.ndarray[Any, Any]]) -> list[np.ndarray[Any, Any]]:
            tensor = feed["images"]
            assert tensor.shape == (1, 3, 1024, 1024)
            assert tensor.dtype == np.float32
            assert float(tensor.min()) >= 0.0
            assert float(tensor.max()) <= 1.0
            output = np.zeros((1, 300, 6), dtype=np.float32)
            output[0, 0] = (256, 0, 768, 1024, 0.9, 0)
            output[0, 1] = (400, 100, 500, 200, 0.8, 1)
            output[0, 2] = (10, 10, 40, 40, PANEL_CONF - 0.05, 0)
            return [output]

    detection = YoloDetector(FakeSession()).detect(image)
    assert len(detection.panels) == 1
    panel = detection.panels[0]
    assert panel.x == 0
    assert panel.y == 0
    assert panel.w == 200
    assert panel.h == 400
    assert len(detection.texts) == 1
    text = detection.texts[0]
    assert text.x == pytest.approx(56, abs=1)
    assert text.y == pytest.approx(39, abs=1)


def test_cv_finds_four_panels_and_a_bleed_panel() -> None:
    detection = CvDetector().detect(grid_page())
    assert len(detection.panels) == 4
    assert detection.texts == ()

    image = Image.new("RGB", (800, 1000), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 40, 340, 960), fill=(200, 200, 200), outline=(0, 0, 0), width=8)
    draw.rectangle((430, 0, 799, 999), fill=(170, 170, 170))
    found = CvDetector().detect(image)
    bleed = next(box for box in found.panels if box.x2 > 700)
    assert contained_ratio(type(bleed)(430, 0, 360, 900), bleed) > 0.8 or bleed.w > 300


def test_cv_finds_frames_on_yellowed_paper() -> None:
    paper = (212, 184, 140)
    image = Image.new("RGB", (900, 1200), paper)
    draw = ImageDraw.Draw(image)
    # Fill matches the sheet, so only the rules separate the panels.
    panels = ((40, 40, 420, 560), (480, 40, 860, 560), (40, 640, 420, 1160), (480, 640, 860, 1160))
    for box in panels:
        draw.rectangle(box, fill=paper, outline=(15, 15, 15), width=8)
    found = CvDetector().detect(image)
    assert len(found.panels) == 4


def test_cv_splits_panels_across_a_dark_gutter() -> None:
    image = Image.new("RGB", (800, 1000), (28, 28, 32))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 760, 450), fill=(70, 110, 160))
    draw.rectangle((40, 530, 760, 960), fill=(170, 130, 80))
    found = CvDetector().detect(image)
    assert len(found.panels) == 2
    assert found.panels[0].y2 < found.panels[1].y or found.panels[1].y2 < found.panels[0].y


def test_fuse_keeps_a_splash_when_inset_frames_do_not_tile() -> None:
    splash = [Box(20, 20, 960, 960, 0.9)]
    # Clear of every edge of the splash, together a large share of the page,
    # and neither a partition of the splash nor a tiling of the page.
    insets = [
        Box(180, 160, 320, 280, 0.5),
        Box(520, 160, 320, 280, 0.5),
        Box(180, 500, 320, 280, 0.5),
        Box(520, 500, 320, 280, 0.5),
    ]
    fused = fuse_panels(splash, insets, 1000, 1000)
    assert len(fused) == 1
    assert fused[0].score == 0.9
    assert fused[0].w > 900


def test_fuse_splits_a_huge_box_when_guides_reach_the_frame() -> None:
    huge = [Box(0, 0, 1000, 1000, 0.4)]
    guides = [
        Box(0, 0, 480, 300, 0.5),
        Box(520, 0, 480, 300, 0.5),
        Box(0, 700, 480, 250, 0.5),
    ]
    fused = fuse_panels(huge, guides, 1000, 1000)
    assert len(fused) == 3
    assert all(box.score == 0.5 for box in fused)


def test_fuse_replaces_a_huge_box_when_the_guides_tile_the_page() -> None:
    huge = [Box(0, 0, 1000, 1000, 0.4)]
    guides = [
        Box(40, 40, 440, 400, 0.5),
        Box(520, 40, 440, 400, 0.5),
        Box(40, 520, 920, 400, 0.5),
    ]
    fused = fuse_panels(huge, guides, 1000, 1000)
    assert len(fused) == 3
    assert {box.y for box in fused} == {40, 520}
    assert all(box.score == 0.5 for box in fused)


def test_fuse_splits_a_box_across_two_frames() -> None:
    merged = [Box(40, 40, 920, 1400, 0.9)]
    guides = [Box(40, 40, 920, 600, 0.5), Box(40, 720, 920, 720, 0.5)]
    fused = fuse_panels(merged, guides, 1000, 1500)
    assert len(fused) == 2
    assert {box.y for box in fused} == {40, 720}


def test_fuse_keeps_a_model_box_a_single_guide_only_overlaps() -> None:
    model = [Box(10, 10, 400, 500, 0.9), Box(10, 600, 400, 500, 0.9)]
    guides = [Box(12, 12, 390, 490, 0.5), Box(12, 602, 390, 490, 0.5)]
    fused = fuse_panels(model, guides, 500, 1200)
    assert len(fused) == 2
    assert all(box.score == 0.9 for box in fused)


def test_fuse_keeps_a_box_when_inner_frames_do_not_tile_it() -> None:
    model = [Box(100, 100, 400, 500, 0.9)]
    guides = [Box(140, 140, 120, 80, 0.5), Box(180, 300, 100, 90, 0.5)]
    fused = fuse_panels(model, guides, 800, 1000)
    assert len(fused) == 1
    assert fused[0].score == 0.9


def test_fuse_emits_a_shared_partition_once() -> None:
    first = Box(0, 0, 500, 400, 0.8)
    second = Box(20, 10, 500, 400, 0.7)
    guides = [Box(0, 0, 240, 400, 0.5), Box(260, 0, 240, 400, 0.5)]
    fused = fuse_panels([first, second], guides, 600, 500)
    assert len(fused) == 2
    assert {box.x for box in fused} == {0, 260}


def test_fuse_ignores_a_guide_that_covers_the_page() -> None:
    model = [Box(10, 10, 200, 200, 0.9), Box(10, 300, 200, 200, 0.9)]
    guides = [Box(0, 0, 480, 580, 0.5)]
    fused = fuse_panels(model, guides, 500, 600)
    assert len(fused) == 2
    assert all(box.score == 0.9 for box in fused)


def test_aligned_frames_replace_a_broken_detection() -> None:
    primary = [Box(20, 20, 760, 900, 0.4)]
    frames = [
        Box(20, 20, 360, 400, 0.5),
        Box(400, 20, 380, 400, 0.5),
        Box(20, 460, 360, 420, 0.5),
        Box(400, 460, 380, 420, 0.5),
    ]
    fused = merge_panel_guides(primary, frames, [], 800, 1000)
    assert len(fused) == 4
    assert {box.y for box in fused} == {20, 460}


def test_aligned_frames_keep_a_healthy_detection() -> None:
    primary = [
        Box(20, 20, 360, 400, 0.9),
        Box(400, 20, 380, 400, 0.9),
        Box(20, 460, 360, 420, 0.9),
        Box(400, 460, 380, 420, 0.9),
    ]
    frames = [
        Box(30, 40, 200, 180, 0.5),
        Box(430, 40, 200, 180, 0.5),
        Box(30, 500, 200, 180, 0.5),
        Box(430, 500, 200, 180, 0.5),
    ]
    fused = merge_panel_guides(primary, frames, [], 800, 1000)
    assert len(fused) == 4
    assert all(box.score == 0.9 for box in fused)


def test_overlapping_boxes_give_way_to_a_gutter_tiling() -> None:
    primary = [Box(40, 30, 700, 400, 0.8), Box(80, 40, 700, 400, 0.6), Box(40, 500, 760, 400, 0.9)]
    gutters = [Box(40, 20, 760, 420, 0.5), Box(40, 500, 760, 420, 0.5)]
    fused = merge_panel_guides(primary, [], gutters, 840, 960)
    assert len(fused) == 2
    assert all(box.score == 0.5 for box in fused)


def test_fuse_adds_a_frame_the_model_missed() -> None:
    model = [Box(10, 10, 400, 400, 0.8)]
    guides = [Box(10, 10, 400, 400, 0.5), Box(10, 500, 400, 400, 0.5)]
    fused = fuse_panels(model, guides, 500, 1000)
    assert len(fused) == 2
    assert any(box.y >= 500 for box in fused)


def test_fusion_restores_a_confident_box_cut_across_a_wide_hole() -> None:
    image = Image.new("RGB", (400, 400), (40, 80, 140))
    model = [Box(10, 10, 380, 380, 0.92)]
    frames = [Box(20, 20, 150, 360, 0.5), Box(230, 20, 150, 360, 0.5)]
    fused = fuse_detections(image, model, frames, [])
    assert len(fused) == 1
    assert fused[0].score == 0.92
    assert fused[0].w > 300


def test_fusion_splits_a_low_score_box_on_a_gutter() -> None:
    image = Image.new("RGB", (400, 400), "white")
    model = [Box(20, 20, 360, 360, 0.3)]
    frames = [Box(20, 20, 360, 160, 0.5), Box(20, 220, 360, 160, 0.5)]
    fused = fuse_detections(image, model, frames, [])
    assert len(fused) == 2
    assert {box.y for box in fused} == {20, 220}


def test_fusion_adds_a_gutter_panel_the_model_missed() -> None:
    image = Image.new("RGB", (400, 400), "white")
    model = [Box(20, 20, 360, 150, 0.9)]
    gutters = [Box(20, 20, 360, 150, 0.5), Box(20, 210, 360, 160, 0.5)]
    fused = fuse_detections(image, model, [], gutters)
    assert len(fused) == 2
    assert any(box.y >= 200 for box in fused)


def test_fusion_keeps_model_boxes_when_there_are_no_guides() -> None:
    image = Image.new("RGB", (400, 800), (30, 30, 30))
    model = [
        Box(10, 10, 180, 180, 0.9),
        Box(210, 10, 180, 180, 0.8),
        Box(10, 400, 180, 180, 0.85),
        Box(210, 400, 180, 180, 0.88),
    ]
    fused = fuse_detections(image, model, [], [])
    assert len(fused) == 4
    assert {box.score for box in fused} == {0.9, 0.8, 0.85, 0.88}


def test_fusion_leaves_a_margin_strip_out() -> None:
    image = Image.new("RGB", (400, 400), "white")
    model = [Box(20, 20, 360, 300, 0.9)]
    gutters = [Box(20, 340, 80, 40, 0.5)]
    fused = fuse_detections(image, model, [], gutters)
    assert len(fused) == 1
    assert fused[0].score == 0.9


def test_fusion_does_not_split_a_blob_another_box_already_covers() -> None:
    image = Image.new("RGB", (400, 400), "white")
    wide = Box(10, 10, 380, 180, 0.3)
    inner = Box(150, 10, 230, 180, 0.45)
    frame = Box(10, 10, 180, 180, 0.5)
    fused = fuse_detections(image, [wide, inner], [frame], [])
    assert any(box.w > 300 for box in fused)


def test_fusion_keeps_a_wide_box_when_children_cover_one_side() -> None:
    image = Image.new("RGB", (400, 400), (20, 20, 20))
    wide = Box(10, 10, 380, 180, 0.6)
    right_a = Box(220, 20, 150, 160, 0.9)
    right_b = Box(240, 30, 140, 150, 0.85)
    fused = fuse_detections(image, [wide, right_a, right_b], [], [])
    assert any(box.w > 300 for box in fused)


def test_fusion_does_not_add_a_gutter_that_already_holds_a_panel() -> None:
    image = Image.new("RGB", (400, 800), "white")
    model = [Box(40, 400, 160, 300, 0.9)]
    gutters = [Box(20, 390, 340, 360, 0.5)]
    fused = fuse_detections(image, model, [], gutters)
    assert len(fused) == 1
    assert fused[0].score == 0.9


def test_frame_panels_follow_rules_and_ignore_a_blank_page() -> None:
    blank = Image.new("RGB", (400, 600), "white")
    assert frame_panels(blank) == []
    image = Image.new("RGB", (600, 800), (230, 210, 170))
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 570, 370), outline=(0, 0, 0), width=6)
    draw.rectangle((30, 420, 570, 770), outline=(0, 0, 0), width=6)
    found = frame_panels(image)
    assert len(found) == 2


def test_auto_falls_back_when_the_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("TATE_MODEL_PATH", "missing-model.onnx")
    from tate.detect import resolve_detector

    detector = resolve_detector("auto")
    assert detector.name == "cv"
    assert "warning" in capsys.readouterr().err
    with pytest.raises(TateError, match="TATE_MODEL_PATH"):
        resolve_detector("yolo")


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("TATE_NETWORK_TESTS") != "1", reason="set TATE_NETWORK_TESTS=1")
def test_real_model_finds_panels_on_a_synthetic_page() -> None:
    from tate.detect.model import ensure_model
    from tate.detect.yolo import load_session

    detection = YoloDetector(load_session(ensure_model())).detect(grid_page())
    assert len(detection.panels) >= 2
