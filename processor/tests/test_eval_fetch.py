import hashlib
import io
import json
import zipfile
from pathlib import Path
from urllib.request import Request

import pytest
from PIL import Image

from tate.errors import TateError
from tate.eval.checksum import file_sha256, pixel_sha256
from tate.eval.dataset import load_dataset
from tate.eval.fetch import fetch_dataset, page_png
from tate.io.inputs import InputLimits, render_pdf_page


def _png(color: tuple[int, int, int] = (20, 30, 40), size: tuple[int, int] = (12, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source(source_id: str, url: str, digest: str | None, kind: str = "image") -> dict[str, object]:
    fetch: dict[str, object] = {"kind": kind, "url": url}
    if digest is not None:
        fetch["sha256"] = digest
    return {
        "id": source_id,
        "title": "Sample",
        "author": "A",
        "license": "MIT",
        "license_url": "https://example.com/license",
        "attribution": "Sample by A",
        "homepage": "https://example.com",
        "fetch": fetch,
    }


def test_fetch_file_zip_pdf_and_derive_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raster = _png()
    raster_path = tmp_path / "page.png"
    raster_path.write_bytes(raster)
    pdf_path = tmp_path / "page.pdf"
    Image.new("RGB", (30, 40), (240, 230, 220)).save(pdf_path, format="PDF")
    archive_path = tmp_path / "pages.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("scan.png", raster)
    book_path = tmp_path / "book.zip"
    with zipfile.ZipFile(book_path, "w") as archive:
        archive.write(pdf_path, "vol/book.pdf")
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        rendered = render_pdf_page(document[0], InputLimits(), "page.pdf")
        pdf_size = rendered.size
        rendered.close()
    finally:
        document.close()

    dataset_root = tmp_path / "dataset"
    (dataset_root / "pages").mkdir(parents=True)
    sources = [
        _source("raster", raster_path.as_uri(), _sha(raster)),
        _source("pack", archive_path.as_uri(), _sha(archive_path.read_bytes()), kind="zip"),
        _source("volume", pdf_path.as_uri(), _sha(pdf_path.read_bytes())),
        _source("book", book_path.as_uri(), _sha(book_path.read_bytes()), kind="zip"),
    ]
    (dataset_root / "sources.json").write_text(json.dumps(sources) + "\n")
    pages = [
        {
            "id": "base",
            "styles": ["standard"],
            "image": {"source": "raster"},
            "size": [12, 8],
            "panels": [{"box": [1, 1, 4, 4]}],
            "whole": False,
            "balloons": [],
            "ci": True,
        },
        {
            "id": "from-zip",
            "styles": ["standard"],
            "image": {"source": "pack", "member": "scan.png"},
            "size": [12, 8],
            "panels": [{"box": [1, 1, 4, 4]}],
            "whole": False,
            "balloons": [],
            "ci": False,
        },
        {
            "id": "from-pdf",
            "styles": ["standard"],
            "image": {"source": "volume", "pdf_page": 0},
            "size": [pdf_size[0], pdf_size[1]],
            "panels": [{"box": [0, 0, 2, 2]}],
            "whole": True,
            "balloons": [],
            "ci": True,
        },
        {
            "id": "zipped-pdf",
            "styles": ["standard"],
            "image": {"source": "book", "member": "vol/book.pdf", "pdf_page": 0},
            "size": [pdf_size[0], pdf_size[1]],
            "panels": [{"box": [0, 0, 2, 2]}],
            "whole": True,
            "balloons": [],
            "ci": True,
        },
        {
            "id": "turned",
            "styles": ["skewed"],
            "image": {"derive": {"from": "base", "ops": [{"rotate": 90}]}},
            "size": [8, 12],
            "whole": False,
            "ci": True,
        },
    ]
    (dataset_root / "pages" / "demo.json").write_text(
        json.dumps({"work": "demo", "title": "Demo", "direction": "rtl", "pages": pages}) + "\n"
    )
    calls: list[str] = []
    real_open = __import__("urllib.request", fromlist=["urlopen"]).urlopen

    def counting(request: Request, timeout: int = 120) -> object:
        calls.append(request.full_url)
        assert "tate-eval" in request.get_header("User-agent")
        return real_open(request, timeout=timeout)

    monkeypatch.setattr("tate.eval.fetch.urlopen", counting)
    cache = tmp_path / "cache"
    dataset = load_dataset(dataset_root)
    fetch_dataset(dataset, cache=cache, subset="ci")
    assert any(call.startswith("file:") for call in calls)
    assert not any("pages.zip" in call for call in calls)
    base = Image.open(page_png(cache, "base"))
    assert base.size == (12, 8)
    assert base.getpixel((0, 0)) == (20, 30, 40)
    base.close()
    turned = Image.open(page_png(cache, "turned"))
    assert turned.size == (8, 12)
    turned.close()
    pdf_image = Image.open(page_png(cache, "from-pdf"))
    assert pdf_image.size == pdf_size
    pdf_image.close()
    zipped = Image.open(page_png(cache, "zipped-pdf"))
    assert zipped.size == pdf_size
    zipped.close()
    assert len(list((cache / "members" / "book").glob("*.pdf"))) == 1
    assert not page_png(cache, "from-zip").exists()

    calls.clear()
    fetch_dataset(load_dataset(dataset_root), cache=cache, subset="ci")
    assert calls == []


def test_sha256_mismatch_fails(tmp_path: Path) -> None:
    raster = _png()
    path = tmp_path / "page.png"
    path.write_bytes(raster)
    root = tmp_path / "dataset"
    (root / "pages").mkdir(parents=True)
    (root / "sources.json").write_text(json.dumps([_source("raster", path.as_uri(), "ab" * 32)]) + "\n")
    (root / "pages" / "demo.json").write_text(
        json.dumps(
            {
                "work": "demo",
                "title": "Demo",
                "direction": "rtl",
                "pages": [
                    {
                        "id": "base",
                        "styles": ["standard"],
                        "image": {"source": "raster"},
                        "size": [12, 8],
                        "panels": [{"box": [1, 1, 4, 4]}],
                        "whole": False,
                        "balloons": [],
                        "ci": True,
                    }
                ],
            }
        )
        + "\n"
    )
    with pytest.raises(TateError, match="sha256 mismatch"):
        fetch_dataset(load_dataset(root), cache=tmp_path / "cache", subset="all")
    assert not (tmp_path / "cache" / "sources" / "raster").exists()


def test_update_checksums_fills_missing_hashes(tmp_path: Path) -> None:
    raster = _png((1, 2, 3), (10, 6))
    path = tmp_path / "page.png"
    path.write_bytes(raster)
    root = tmp_path / "dataset"
    (root / "pages").mkdir(parents=True)
    (root / "sources.json").write_text(json.dumps([_source("raster", path.as_uri(), None)]) + "\n")
    (root / "pages" / "demo.json").write_text(
        json.dumps(
            {
                "work": "demo",
                "title": "Demo",
                "direction": "ltr",
                "pages": [
                    {
                        "id": "base",
                        "styles": ["color"],
                        "image": {"source": "raster"},
                        "size": [10, 6],
                        "panels": [{"box": [1, 1, 3, 3]}],
                        "whole": False,
                        "balloons": [],
                        "ci": False,
                    }
                ],
            }
        )
        + "\n"
    )
    with pytest.raises(TateError, match="missing sha256"):
        fetch_dataset(load_dataset(root), cache=tmp_path / "cache")
    fetch_dataset(load_dataset(root), cache=tmp_path / "cache", update_checksums=True)
    text = (root / "sources.json").read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\n  " in text
    saved = json.loads(text)
    assert saved[0]["fetch"]["sha256"] == file_sha256(path)
    page = json.loads((root / "pages" / "demo.json").read_text(encoding="utf-8"))["pages"][0]
    with Image.open(path) as image:
        assert page["pixel_sha256"] == pixel_sha256(image)
    assert list(saved[0]).index("fetch") > list(saved[0]).index("id")


def test_download_retries_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raster = _png()
    path = tmp_path / "page.png"
    path.write_bytes(raster)
    root = tmp_path / "dataset"
    (root / "pages").mkdir(parents=True)
    (root / "sources.json").write_text(json.dumps([_source("raster", path.as_uri(), _sha(raster))]) + "\n")
    (root / "pages" / "demo.json").write_text(
        json.dumps(
            {
                "work": "demo",
                "title": "Demo",
                "direction": "rtl",
                "pages": [
                    {
                        "id": "base",
                        "styles": ["standard"],
                        "image": {"source": "raster"},
                        "size": [12, 8],
                        "panels": [{"box": [1, 1, 4, 4]}],
                        "whole": False,
                        "balloons": [],
                        "ci": True,
                    }
                ],
            }
        )
        + "\n"
    )
    real_open = __import__("urllib.request", fromlist=["urlopen"]).urlopen
    attempts = {"n": 0}

    def flaky(request: Request, timeout: int = 120) -> object:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("temporary")
        return real_open(request, timeout=timeout)

    monkeypatch.setattr("tate.eval.fetch.urlopen", flaky)
    monkeypatch.setattr("tate.eval.fetch.time.sleep", lambda _seconds: None)
    fetch_dataset(load_dataset(root), cache=tmp_path / "cache")
    assert attempts["n"] == 2
