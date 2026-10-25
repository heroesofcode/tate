import io
import zipfile
from pathlib import Path

from PIL import Image

from tate.detect.page import PageDetector
from tate.pipeline import process_chapter
from tests.pages import grid_page, save_jpeg


def test_package_matches_the_contract(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")  # type: ignore[attr-defined]
    folder = tmp_path / "chapter"
    folder.mkdir()
    save_jpeg(folder / "a.jpg", grid_page(600, 800))
    save_jpeg(folder / "b.jpg", grid_page(600, 800))
    output = tmp_path / "out.tate"
    preview = tmp_path / "preview.png"
    first = process_chapter(
        folder,
        output,
        title="Demo",
        width=1080,
        detector_name="page",
        preview=preview,
        pages=None,
        detector=PageDetector(),
    )
    second = process_chapter(
        folder,
        tmp_path / "out-again.tate",
        title="Demo",
        width=1080,
        detector_name="page",
        detector=PageDetector(),
    )
    assert output.read_bytes() == (tmp_path / "out-again.tate").read_bytes()
    manifest = first.manifest
    assert manifest["format"] == "tate"
    assert manifest["version"] == 1
    assert manifest["title"] == "Demo"
    assert manifest["source"] == "chapter"
    assert manifest["createdAt"] == "2023-11-14T22:13:20Z"
    assert manifest["reading"] == "rtl"
    assert manifest["width"] == 1080
    assert manifest["background"] == "#ffffff"
    assert manifest["detector"] == "page"
    assert manifest["pageCount"] == 2
    assert manifest["cover"] == "cover.webp"
    assert manifest["segments"][0]["gapBefore"] == 0
    assert manifest["segments"][0]["page"] == 0
    assert manifest["segments"][1]["page"] == 1
    assert manifest["segments"][1]["gapBefore"] > manifest["segments"][0]["gapBefore"]
    assert second.page_count == 2

    with zipfile.ZipFile(output) as archive:
        infos = {info.filename: info for info in archive.infolist()}
        assert infos["manifest.json"].compress_type == zipfile.ZIP_DEFLATED
        assert infos["cover.webp"].compress_type == zipfile.ZIP_STORED
        for segment in manifest["segments"]:
            info = infos[segment["file"]]
            assert info.compress_type == zipfile.ZIP_STORED
            with Image.open(io.BytesIO(archive.read(segment["file"]))) as image:
                assert image.size == (segment["width"], segment["height"])
                assert image.size[0] == 1080
                assert image.size[1] <= 4096
        with Image.open(io.BytesIO(archive.read("cover.webp"))) as cover:
            assert cover.size[0] == 360
    with Image.open(preview) as preview_image:
        assert preview_image.size[0] == 480


def test_cv_grid_is_read_right_to_left(tmp_path: Path, monkeypatch: object) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1700000000")  # type: ignore[attr-defined]
    path = tmp_path / "page.jpg"
    save_jpeg(path, grid_page())
    debug = tmp_path / "debug"
    result = process_chapter(
        path,
        tmp_path / "page.tate",
        width=1080,
        detector_name="cv",
        debug_dir=debug,
    )
    assert result.detector == "cv"
    segments = result.manifest["segments"]
    assert len(segments) == 4
    centers = [(box[0] + box[2] / 2, box[1] + box[3] / 2) for box in (item["box"] for item in segments)]
    assert centers[0][0] > centers[1][0]
    assert centers[0][1] < centers[2][1]
    assert centers[2][0] > centers[3][0]
    assert all(item["page"] == 0 for item in segments)
    assert (debug / "0001.jpg").is_file()
