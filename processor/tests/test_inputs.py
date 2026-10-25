import zipfile
from pathlib import Path

import pytest

from tate.errors import TateError
from tate.io.inputs import (
    ChapterInput,
    InputLimits,
    entry_disposition,
    natural_key,
    parse_page_range,
)
from tests.pages import jpeg_bytes, save_jpeg, solid


def test_natural_order_puts_page2_before_page10() -> None:
    names = ["page10.jpg", "page2.jpg", "page1.jpg"]
    assert sorted(names, key=natural_key) == ["page1.jpg", "page2.jpg", "page10.jpg"]


def test_entry_disposition() -> None:
    assert entry_disposition("page2.jpg") == "ok"
    assert entry_disposition("scans/page10.PNG") == "ok"
    assert entry_disposition("__MACOSX/page.jpg") == "junk"
    assert entry_disposition(".hidden.jpg") == "junk"
    assert entry_disposition("Thumbs.db") == "junk"
    assert entry_disposition("ComicInfo.xml") == "junk"
    assert entry_disposition("../evil.jpg") == "unsafe"
    assert entry_disposition("/etc/passwd.jpg") == "unsafe"
    assert entry_disposition("C:/windows/page.jpg") == "unsafe"


def test_zip_skips_junk_and_sorts_naturally(tmp_path: Path) -> None:
    path = tmp_path / "chapter.cbz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("__MACOSX/._page2.jpg", b"junk")
        archive.writestr(".hidden.jpg", jpeg_bytes())
        archive.writestr("Thumbs.db", b"xx")
        archive.writestr("ComicInfo.xml", b"<x/>")
        archive.writestr("page10.jpg", jpeg_bytes((1, 1, 1)))
        archive.writestr("nested/page2.jpg", jpeg_bytes((2, 2, 2)))
    with ChapterInput.open(path, None) as chapter:
        pages = list(chapter)
    assert [page.name for page in pages] == ["page2.jpg", "page10.jpg"]
    assert pages[0].index == 0
    assert pages[1].index == 1


def test_path_traversal_is_rejected_and_not_written(tmp_path: Path) -> None:
    path = tmp_path / "bad.cbz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../evil.jpg", jpeg_bytes())
        archive.writestr("ok.jpg", jpeg_bytes())
    with pytest.raises(TateError, match="unsafe"), ChapterInput.open(path, None) as chapter:
        list(chapter)
    assert not (tmp_path / "evil.jpg").exists()
    assert not (tmp_path.parent / "evil.jpg").exists()


def test_compression_bomb_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bomb.cbz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("page.jpg", b"\x00" * 2_000_000)
    limits = InputLimits(max_compression_ratio=20, max_entry_bytes=10_000_000)
    with pytest.raises(TateError, match="compression"), ChapterInput.open(path, None, limits) as chapter:
        list(chapter)


def test_oversize_entry_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "big.cbz"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("page.jpg", jpeg_bytes())
    limits = InputLimits(max_entry_bytes=200)
    with pytest.raises(TateError, match="too large"), ChapterInput.open(path, None, limits) as chapter:
        list(chapter)


def test_folder_pages_and_range(tmp_path: Path) -> None:
    folder = tmp_path / "scans"
    folder.mkdir()
    save_jpeg(folder / "page10.jpg", solid((40, 50), (10, 10, 10)))
    save_jpeg(folder / "page2.jpg", solid((40, 50), (20, 20, 20)))
    (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    (folder / "__MACOSX").mkdir()
    save_jpeg(folder / "__MACOSX" / "page2.jpg", solid((40, 50), (1, 1, 1)))
    with ChapterInput.open(folder, "2-2") as chapter:
        pages = list(chapter)
    assert len(pages) == 1
    assert pages[0].index == 1
    assert pages[0].name == "page10.jpg"


def test_page_range_errors() -> None:
    with pytest.raises(TateError, match="invalid"):
        parse_page_range("a-b", 10)
    with pytest.raises(TateError, match="last page"):
        parse_page_range("1-11", 10)


def test_pdf_renders_selected_pages(tmp_path: Path) -> None:
    path = tmp_path / "book.pdf"
    light = solid((300, 420), (240, 240, 240))
    dark = solid((300, 420), (20, 20, 20))
    light.save(path, "PDF", resolution=72, save_all=True, append_images=[dark])
    with ChapterInput.open(path, "1-2") as chapter:
        pages = list(chapter)
    assert len(pages) == 2
    assert pages[0].index == 0
    assert pages[1].index == 1
    assert pages[0].image.height == pytest.approx(2000, rel=0.05)
    assert _mean(pages[0].image) > _mean(pages[1].image) + 20
    with ChapterInput.open(path, "2-2") as chapter:
        only = list(chapter)
    assert len(only) == 1
    assert only[0].index == 1


def test_missing_and_unsupported(tmp_path: Path) -> None:
    with pytest.raises(TateError, match="not found"):
        ChapterInput.open(tmp_path / "nope.cbz", None)
    mystery = tmp_path / "notes.txt"
    mystery.write_text("hello", encoding="utf-8")
    with pytest.raises(TateError, match="unsupported"):
        ChapterInput.open(mystery, None)


def _mean(image: object) -> float:
    import numpy as np
    from PIL import Image

    assert isinstance(image, Image.Image)
    return float(np.asarray(image.resize((32, 32))).mean())
