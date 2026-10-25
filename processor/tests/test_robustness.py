"""Odd pages, archives, and documents run through the whole processor."""

from __future__ import annotations

import io
import struct
import zipfile
import zlib
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from tate.cli import main
from tate.detect.page import PageDetector
from tate.errors import TateError
from tate.io.inputs import natural_key
from tate.pipeline import ProcessResult, process_chapter

STRIP = 64
EPOCH = "1700000000"


@pytest.fixture
def epoch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", EPOCH)


def test_extreme_pages_become_a_valid_package(tmp_path: Path, epoch: None) -> None:
    folder = tmp_path / "sizes"
    folder.mkdir()
    # The tall page is first so the cover is the segment that used to exceed WebP's limit.
    _save_rgb(folder / "01-thin.png", (1, 5000), (0, 0, 0))
    _save_rgb(folder / "02-one.png", (1, 1), (255, 0, 0))
    _save_rgb(folder / "03-wide.png", (5000, 1), (0, 0, 255))
    _save_rgb(folder / "04-spread.png", (1800, 200), (20, 20, 20))
    _save_rgb(folder / "05-tiny.png", (16, 16), (7, 8, 9))
    _save_rgb(folder / "06-white.png", (80, 40), (255, 255, 255))
    _save_rgb(folder / "07-black.png", (80, 40), (0, 0, 0))
    result = _process(folder, tmp_path / "sizes.tate")
    _assert_package(result, width=STRIP, source="sizes", pages=7)
    segments = result.manifest["segments"]
    thin = _page_segments(segments, 0)
    assert thin[0]["box"][1] == 0
    assert thin[-1]["box"][1] + thin[-1]["box"][3] == 5000
    assert all(part["gapBefore"] == 0 for part in thin)
    assert all(part["height"] <= 4096 for part in thin)
    assert len(thin) >= 2
    one = _page_segments(segments, 1)
    assert one[0]["box"] == [0, 0, 1, 1]
    wide = _page_segments(segments, 2)
    assert wide[0]["box"][2:] == [5000, 1]
    spread = _page_segments(segments, 3)
    assert spread[0]["box"] == [0, 0, 1800, 200]
    assert len(_page_segments(segments, 4)) == 1
    with zipfile.ZipFile(result.output) as archive:
        white = _open_segment(archive, _page_segments(segments, 5)[0])
        black = _open_segment(archive, _page_segments(segments, 6)[0])
        cover = Image.open(io.BytesIO(archive.read("cover.webp")))
        try:
            assert _center(white) == (255, 255, 255)
            assert _center(black) == (0, 0, 0)
            assert cover.size == (360, 16380)
        finally:
            white.close()
            black.close()
            cover.close()


def test_color_modes_composite_on_white(tmp_path: Path, epoch: None) -> None:
    folder = tmp_path / "modes"
    folder.mkdir()
    _save_mode(folder / "01-l.png", Image.new("L", (48, 48), 180))
    _save_mode(folder / "02-la-clear.png", Image.new("LA", (48, 48), (0, 0)))
    _save_mode(folder / "03-la-black.png", Image.new("LA", (48, 48), (0, 255)))
    _save_mode(folder / "04-rgba-clear.png", Image.new("RGBA", (48, 48), (255, 0, 0, 0)))
    _save_mode(folder / "05-rgba-red.png", Image.new("RGBA", (48, 48), (255, 0, 0, 255)))
    _save_mode(folder / "06-rgba-half.png", Image.new("RGBA", (48, 48), (255, 0, 0, 128)))
    _save_palette(folder / "07-p-clear.gif", index=0, transparent=True)
    _save_palette(folder / "08-p-green.gif", index=1, transparent=False)
    _save_mode(folder / "09-cmyk.jpg", Image.new("CMYK", (48, 48), (0, 0, 0, 255)), fmt="JPEG", quality=90)
    _save_gray16(folder / "10-i16-mid.png", 32768)
    _save_gray16(folder / "11-i16-white.png", 65535)
    _save_gray16(folder / "12-i16-black.png", 0)
    _save_mode(folder / "13-bit.png", Image.new("1", (48, 48), 0))
    result = _process(folder, tmp_path / "modes.tate")
    _assert_package(result, width=STRIP, source="modes", pages=13)
    assert len(result.manifest["segments"]) == 13
    colors: list[tuple[int, int, int]] = []
    with zipfile.ZipFile(result.output) as archive:
        for segment in result.manifest["segments"]:
            image = _open_segment(archive, segment)
            try:
                colors.append(_center(image))
            finally:
                image.close()
        _close(colors[0], (180, 180, 180))
        _close(colors[1], (255, 255, 255))
        _close(colors[2], (0, 0, 0))
        _close(colors[3], (255, 255, 255))
        _close(colors[4], (255, 0, 0))
        assert colors[5][0] > 200 and 90 < colors[5][1] < 170 and 90 < colors[5][2] < 170
        _close(colors[6], (255, 255, 255))
        _close(colors[7], (0, 255, 0))
        assert all(channel < 40 for channel in colors[8])
        _close(colors[9], (128, 128, 128))
        _close(colors[10], (255, 255, 255))
        _close(colors[11], (0, 0, 0))
        _close(colors[12], (0, 0, 0))


def test_exif_rotation_is_respected(tmp_path: Path, epoch: None) -> None:
    path = tmp_path / "turned.jpg"
    _save_exif_jpeg(path)
    result = _process(path, tmp_path / "turned.tate")
    _assert_package(result, width=STRIP, source="turned.jpg", pages=1)
    segment = result.manifest["segments"][0]
    assert segment["box"] == [0, 0, 24, 48]
    with zipfile.ZipFile(result.output) as archive:
        image = _open_segment(archive, segment)
    try:
        blue_rows = _rows_matching(image, lambda pixel: pixel[2] > 180 and pixel[2] > pixel[0] + 50)
        red_rows = _rows_matching(image, lambda pixel: pixel[0] > 180 and pixel[0] > pixel[2] + 50)
    finally:
        image.close()
    assert blue_rows and red_rows
    assert min(blue_rows) < min(red_rows)


def test_animation_keeps_the_first_frame(tmp_path: Path, epoch: None) -> None:
    gif = tmp_path / "anim.gif"
    webp = tmp_path / "anim.webp"
    red = Image.new("RGB", (48, 48), (255, 0, 0))
    green = Image.new("RGB", (48, 48), (0, 255, 0))
    red.save(gif, format="GIF", save_all=True, append_images=[green], duration=80, loop=0)
    red.save(webp, format="WEBP", save_all=True, append_images=[green], duration=80, loop=0, lossless=True)
    red.close()
    green.close()
    for path in (gif, webp):
        result = _process(path, tmp_path / f"{path.stem}.tate")
        _assert_package(result, width=STRIP, source=path.name, pages=1)
        with zipfile.ZipFile(result.output) as archive:
            image = _open_segment(archive, result.manifest["segments"][0])
        try:
            pixel = _center(image)
        finally:
            image.close()
        assert pixel[0] > 200 and pixel[1] < 40


def test_broken_images_fail_in_the_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    jpeg = io.BytesIO()
    png = io.BytesIO()
    Image.new("RGB", (24, 24), (1, 2, 3)).save(jpeg, format="JPEG")
    Image.new("RGB", (24, 24), (1, 2, 3)).save(png, format="PNG")
    truncated_jpeg = tmp_path / "cut.jpg"
    truncated_png = tmp_path / "cut.png"
    empty = tmp_path / "empty.jpg"
    truncated_jpeg.write_bytes(jpeg.getvalue()[:24])
    truncated_png.write_bytes(png.getvalue()[:32])
    empty.write_bytes(b"")
    huge = tmp_path / "huge.png"
    huge.write_bytes(_png_header(7000, 7000))
    for path in (truncated_jpeg, truncated_png, empty, huge):
        _assert_cli_error(path, tmp_path, capsys)
    with pytest.raises(TateError, match="too large"):
        process_chapter(huge, tmp_path / "huge.tate", width=STRIP, detector=PageDetector())


def test_hostile_zip_entries_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    corrupt = tmp_path / "corrupt.cbz"
    _write_corrupt_member(corrupt)
    declared = tmp_path / "declared.cbz"
    _write_declared_size(declared, 90_000_000)
    _assert_cli_error(corrupt, tmp_path, capsys)
    _assert_cli_error(declared, tmp_path, capsys)
    with pytest.raises(TateError, match="too large"):
        process_chapter(declared, tmp_path / "declared.tate", width=STRIP, detector=PageDetector())


def test_nested_unicode_names_are_packaged(tmp_path: Path, epoch: None) -> None:
    folder = tmp_path / "章"
    folder.mkdir()
    _save_rgb(folder / "01-a.png", (48, 48), (255, 0, 0))
    _save_rgb(folder / "02-ページ.png", (48, 48), (0, 0, 255))
    (folder / "notes.txt").write_text("skip me", encoding="utf-8")
    single = tmp_path / "単ページ.png"
    _save_rgb(single, (48, 48), (0, 255, 0))
    folder_result = _process(folder, tmp_path / "folder.tate")
    _assert_package(folder_result, width=STRIP, source="章", pages=2)
    single_result = _process(single, tmp_path / "single.tate")
    _assert_package(single_result, width=STRIP, source="単ページ.png", pages=1)

    archive_path = tmp_path / "章.cbz"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("nested/deeper/b.png", _png_bytes((0, 255, 0)))
        archive.writestr("nested/ページ.jpg", _jpeg_bytes((255, 0, 0)))
        archive.writestr("readme.txt", "not a page")
        archive.writestr("ComicInfo.xml", "<ComicInfo/>")
        archive.writestr("Thumbs.db", "xx")
    result = _process(archive_path, tmp_path / "archive.tate")
    _assert_package(result, width=STRIP, source="章.cbz", pages=2)
    names = ["nested/deeper/b.png", "nested/ページ.jpg"]
    assert sorted(names, key=natural_key) == names
    with zipfile.ZipFile(result.output) as archive:
        first_image = _open_segment(archive, result.manifest["segments"][0])
        second_image = _open_segment(archive, result.manifest["segments"][1])
        try:
            first = _center(first_image)
            second = _center(second_image)
        finally:
            first_image.close()
            second_image.close()
    assert first[1] > 200
    assert second[0] > 200


def test_pdf_pages_of_different_sizes_and_orientations(tmp_path: Path, epoch: None) -> None:
    path = tmp_path / "book.pdf"
    _write_varied_pdf(path)
    result = _process(path, tmp_path / "book.tate")
    _assert_package(result, width=STRIP, source="book.pdf", pages=3)
    segments = result.manifest["segments"]
    assert [segment["page"] for segment in segments] == [0, 1, 2]
    portrait, landscape, turned = (segment["box"] for segment in segments)
    assert portrait[3] > portrait[2]
    assert landscape[2] > landscape[3]
    assert turned[2] > turned[3]
    assert turned[2:] != portrait[2:]


def test_encrypted_and_corrupt_pdfs_fail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # One blank page with the user password "secret", made with Ghostscript pdfwrite.
    encrypted = Path(__file__).parent / "data" / "encrypted.pdf"
    corrupt = tmp_path / "broken.pdf"
    corrupt.write_bytes(b"%PDF-1.4\n1 0 obj\n<<>>\nstream\nnot a page\nendstream\n%%EOF")
    for path in (encrypted, corrupt):
        _assert_cli_error(path, tmp_path, capsys)
        with pytest.raises(TateError, match="could not read PDF"):
            process_chapter(path, tmp_path / "out.tate", width=STRIP, detector=PageDetector())


def _process(path: Path, output: Path) -> ProcessResult:
    return process_chapter(path, output, width=STRIP, detector=PageDetector())


def _assert_package(result: ProcessResult, *, width: int, source: str, pages: int) -> None:
    manifest = result.manifest
    assert manifest["format"] == "tate"
    assert manifest["version"] == 1
    assert manifest["reading"] == "rtl"
    assert manifest["width"] == width
    assert manifest["background"] == "#ffffff"
    assert manifest["detector"] == "page"
    assert manifest["pageCount"] == pages
    assert manifest["cover"] == "cover.webp"
    assert manifest["source"] == source
    assert "/" not in manifest["source"] and "\\" not in manifest["source"]
    assert manifest["createdAt"] == "2023-11-14T22:13:20Z"
    expected_title = source if Path(source).suffix == "" else Path(source).stem
    assert manifest["title"] == expected_title
    assert manifest["segments"]
    assert all(0 <= segment["page"] < pages for segment in manifest["segments"])
    with zipfile.ZipFile(result.output) as archive:
        infos = {info.filename: info for info in archive.infolist()}
        assert infos["manifest.json"].compress_type == zipfile.ZIP_DEFLATED
        assert infos["cover.webp"].compress_type == zipfile.ZIP_STORED
        for segment in manifest["segments"]:
            info = infos[segment["file"]]
            assert info.compress_type == zipfile.ZIP_STORED
            assert segment["width"] == width
            assert 1 <= segment["height"] <= 4096
            image = _open_segment(archive, segment)
            try:
                assert image.size == (segment["width"], segment["height"])
            finally:
                image.close()


def _assert_cli_error(path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / f"{path.stem}-cli.tate"
    with pytest.raises(TateError):
        process_chapter(path, output, width=STRIP, detector=PageDetector())
    code = main(["process", str(path), "--detector", "page", "--width", str(STRIP), "-o", str(output)])
    assert code == 1
    captured = capsys.readouterr()
    assert any(line.startswith("error:") for line in captured.err.splitlines())
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def _page_segments(segments: list[dict[str, object]], page: int) -> list[dict[str, object]]:
    return [segment for segment in segments if segment["page"] == page]


def _open_segment(archive: zipfile.ZipFile, segment: dict[str, object]) -> Image.Image:
    file_name = segment["file"]
    assert isinstance(file_name, str)
    return Image.open(io.BytesIO(archive.read(file_name)))


def _close(pixel: tuple[int, int, int], expected: tuple[int, int, int], tolerance: int = 8) -> None:
    assert all(abs(channel - target) <= tolerance for channel, target in zip(pixel, expected, strict=True))


def _center(image: Image.Image) -> tuple[int, int, int]:
    pixel = image.getpixel((image.width // 2, image.height // 2))
    assert isinstance(pixel, tuple) and len(pixel) >= 3
    return int(pixel[0]), int(pixel[1]), int(pixel[2])


def _rows_matching(image: Image.Image, predicate: object) -> list[int]:
    assert callable(predicate)
    rows: list[int] = []
    for y in range(image.height):
        for x in range(image.width):
            pixel = image.getpixel((x, y))
            assert isinstance(pixel, tuple)
            if predicate(pixel):
                rows.append(y)
                break
    return rows


def _save_rgb(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", size, color)
    image.save(path, format="PNG")
    image.close()


def _save_mode(path: Path, image: Image.Image, *, fmt: str | None = None, quality: int | None = None) -> None:
    options: dict[str, object] = {}
    if fmt is not None:
        options["format"] = fmt
    if quality is not None:
        options["quality"] = quality
    image.save(path, **options)
    image.close()


def _save_palette(path: Path, *, index: int, transparent: bool) -> None:
    image = Image.new("P", (48, 48))
    image.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    image.putdata([index] * (48 * 48))
    if transparent:
        image.save(path, format="GIF", transparency=0)
    else:
        image.save(path, format="GIF")
    image.close()


def _save_gray16(path: Path, value: int) -> None:
    image = Image.new("I", (48, 48), value).convert("I;16")
    image.save(path, format="PNG")
    image.close()


def _save_exif_jpeg(path: Path) -> None:
    image = Image.new("RGB", (48, 24), (255, 0, 0))
    ImageDraw.Draw(image).rectangle((0, 0, 23, 23), fill=(0, 0, 255))
    exif = Image.Exif()
    exif[0x0112] = 6
    image.save(path, format="JPEG", quality=95, exif=exif.tobytes())
    image.close()


def _png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (48, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes(color: tuple[int, int, int], size: tuple[int, int] = (48, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def _png_header(width: int, height: int) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


def _write_corrupt_member(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("page.jpg", b"A" * 4000)
        archive.writestr("other.jpg", _jpeg_bytes((10, 20, 30)))
    blob = bytearray(path.read_bytes())
    start = blob.find(b"page.jpg") + len(b"page.jpg")
    for index in range(start, start + 24):
        blob[index] ^= 0xFF
    path.write_bytes(blob)


def _write_declared_size(path: Path, declared: int) -> None:
    """Forge the uncompressed size so the entry claims to be huge without storing it."""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("page.jpg", _jpeg_bytes((4, 5, 6)))
    blob = bytearray(path.read_bytes())
    local = blob.find(b"PK\x03\x04")
    central = blob.find(b"PK\x01\x02")
    blob[local + 22 : local + 26] = struct.pack("<I", declared)
    blob[central + 24 : central + 28] = struct.pack("<I", declared)
    path.write_bytes(blob)


def _write_varied_pdf(path: Path) -> None:
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument.new()
    try:
        document.new_page(220, 460)
        document.new_page(520, 200)
        turned = document.new_page(220, 460)
        turned.set_rotation(90)
        document.save(str(path))
    finally:
        document.close()
