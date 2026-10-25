"""Download sources once, then materialise decoded pages in the cache."""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.request import Request

import platformdirs
from PIL import Image

from tate.errors import TateError
from tate.eval.checksum import file_sha256, pixel_sha256
from tate.eval.dataset import ancestor_ids, image_to_json, select_pages, write_sources, write_work
from tate.eval.derive import render_ops
from tate.eval.model import (
    Dataset,
    DeriveImage,
    ImageFetch,
    LoadedPage,
    PageAnn,
    Source,
    SourceImage,
    Work,
    ZipFetch,
)
from tate.io.inputs import IMAGE_SUFFIXES, InputLimits, open_image, render_pdf_page

USER_AGENT = "tate-eval (+https://github.com/heroesofcode/tate)"
_ATTEMPTS = 4
_TIMEOUT_S = 120


def default_cache_dir() -> Path:
    """Platform cache directory ``tate/eval``."""
    return Path(platformdirs.user_cache_dir("tate", appauthor=False)) / "eval"


def page_png(cache: Path, page_id: str) -> Path:
    return cache / "pages" / f"{page_id}.png"


def urlopen(request: Request, timeout: int = _TIMEOUT_S) -> Any:
    """Open ``request``. Tests replace this; it is one download at a time."""
    import urllib.request

    return urllib.request.urlopen(request, timeout=timeout)


@dataclass(frozen=True, slots=True)
class FetchResult:
    sources: int
    pages: int
    updated_files: tuple[str, ...]


def fetch_dataset(
    dataset: Dataset,
    *,
    cache: Path,
    subset: str = "all",
    update_checksums: bool = False,
) -> FetchResult:
    """Download what ``subset`` needs, verify checksums, and write page PNGs.

    Already cached files are reused when their checksum still matches.
    ``--update-checksums`` fills missing source and pixel hashes and rewrites
    those JSON files in a stable order.
    """
    selected = select_pages(dataset, subset)
    order = ancestor_ids(dataset, selected)
    by_id = dataset.page_map()
    needed = [by_id[page_id] for page_id in order]
    source_ids: list[str] = []
    for loaded in needed:
        image = loaded.page.image
        if isinstance(image, SourceImage):
            source_ids.append(image.source)
    unique_sources = list(dict.fromkeys(source_ids))
    sources = dataset.source_map()
    source_hashes: dict[str, str] = {}
    for source_id in unique_sources:
        source_hashes[source_id] = _ensure_source(
            sources[source_id],
            cache,
            allow_missing_sha=update_checksums,
        )

    pixel_hashes: dict[str, str] = {}
    for loaded in needed:
        parent_hash: str | None = None
        if isinstance(loaded.page.image, DeriveImage):
            parent_hash = pixel_hashes[loaded.page.image.from_page]
        pixel_hashes[loaded.page.id] = _ensure_page(loaded, cache, sources, parent_hash)

    updated: list[str] = []
    if update_checksums:
        updated = _write_checksums(dataset, source_hashes, pixel_hashes)
    return FetchResult(len(unique_sources), len(needed), tuple(updated))


def _ensure_source(source: Source, cache: Path, *, allow_missing_sha: bool) -> str:
    if source.fetch.sha256 is None and not allow_missing_sha:
        raise TateError(f"source {source.id} is missing sha256")
    destination = cache / "sources" / source.id
    if destination.is_file():
        digest = file_sha256(destination)
        if source.fetch.sha256 is not None and digest != source.fetch.sha256:
            raise TateError(f"sha256 mismatch for {source.id}")
        return digest
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        _download(source.fetch.url, temporary)
        digest = file_sha256(temporary)
        if source.fetch.sha256 is not None and digest != source.fetch.sha256:
            raise TateError(f"sha256 mismatch for {source.id}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return file_sha256(destination)


def _download(url: str, dest: Path) -> None:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(_ATTEMPTS):
        try:
            with urlopen(request, timeout=_TIMEOUT_S) as response, dest.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            return
        except Exception as exc:
            dest.unlink(missing_ok=True)
            last_error = exc
            if attempt + 1 == _ATTEMPTS:
                break
            time.sleep(0.5 * (2**attempt))
    raise TateError(f"download failed: {url}") from last_error


def _ensure_page(
    loaded: LoadedPage,
    cache: Path,
    sources: dict[str, Source],
    parent_pixels: str | None,
) -> str:
    page = loaded.page
    png = page_png(cache, page.id)
    stamp = png.with_suffix(".stamp")
    recipe = _recipe(page, parent_pixels)
    if png.is_file() and stamp.is_file():
        cached = _read_stamp(stamp)
        if cached is not None and cached[1] == recipe and file_matches_pixels(png, cached[0]):
            if page.pixel_sha256 is not None and page.pixel_sha256 != cached[0]:
                raise TateError(f"pixel_sha256 mismatch for {page.id}")
            return cached[0]
    image = _build_page(page, cache, sources)
    try:
        if image.size != page.size:
            raise TateError(
                f"page {page.id} image is {image.width}x{image.height}, "
                f"dataset says {page.size[0]}x{page.size[1]}"
            )
        digest = pixel_sha256(image)
        if page.pixel_sha256 is not None and digest != page.pixel_sha256:
            raise TateError(f"pixel_sha256 mismatch for {page.id}")
        png.parent.mkdir(parents=True, exist_ok=True)
        image.save(png, format="PNG")
        stamp.write_text(f"{digest}\n{recipe}\n", encoding="utf-8")
    finally:
        image.close()
    return digest


def file_matches_pixels(path: Path, digest: str) -> bool:
    with Image.open(path) as image:
        image.load()
        return pixel_sha256(image) == digest


def _read_stamp(path: Path) -> tuple[str, str] | None:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 2 or len(lines[0]) != 64:
        return None
    return lines[0], lines[1]


def _recipe(page: PageAnn, parent_pixels: str | None) -> str:
    payload = {
        "image": image_to_json(page.image),
        "parent": parent_pixels,
        "size": [page.size[0], page.size[1]],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _build_page(page: PageAnn, cache: Path, sources: dict[str, Source]) -> Image.Image:
    image = page.image
    if isinstance(image, DeriveImage):
        with Image.open(page_png(cache, image.from_page)) as parent:
            parent.load()
            rgb = parent.convert("RGB")
        return render_ops(rgb, image.ops)
    source = sources[image.source]
    blob = cache / "sources" / source.id
    if isinstance(source.fetch, ZipFetch):
        if not image.member:
            raise TateError(f"page {page.id} needs a zip member")
        data, name = _read_member(blob, image.member)
        if image.pdf_page is not None:
            return _render_pdf_bytes(data, image.pdf_page, name)
        return open_image(data, InputLimits(), name)
    if isinstance(source.fetch, ImageFetch) and image.member is not None:
        raise TateError(f"page {page.id} member is only valid for a zip source")
    if image.pdf_page is not None:
        return _render_pdf_path(blob, image.pdf_page, source.id)
    data = _read_capped(blob, InputLimits().max_entry_bytes, source.id)
    return open_image(data, InputLimits(), source.id)


def _read_capped(path: Path, limit: int, label: str) -> bytes:
    size = path.stat().st_size
    if size > limit:
        raise TateError(f"image is too large: {label}")
    return path.read_bytes()


def _read_member(archive_path: Path, member: str) -> tuple[bytes, str]:
    limits = InputLimits()
    try:
        with zipfile.ZipFile(archive_path) as archive:
            try:
                info = archive.getinfo(member)
            except KeyError as exc:
                raise TateError(f"zip is missing {member}") from exc
            if info.flag_bits & 0x1:
                raise TateError(f"encrypted zip entry: {member}")
            if info.file_size < 0 or info.file_size > limits.max_entry_bytes:
                raise TateError(f"zip entry is too large: {member}")
            with archive.open(info, "r") as handle:
                data = handle.read(limits.max_entry_bytes + 1)
    except zipfile.BadZipFile as exc:
        raise TateError(f"could not read zip archive: {archive_path.name}") from exc
    if len(data) > limits.max_entry_bytes:
        raise TateError(f"zip entry is too large: {member}")
    suffix = PurePosixPath(member).suffix.casefold()
    if suffix not in IMAGE_SUFFIXES and suffix != ".pdf":
        raise TateError(f"unsupported zip entry: {member}")
    return data, PurePosixPath(member).name


def _render_pdf_path(path: Path, page_index: int, label: str) -> Image.Image:
    import pypdfium2 as pdfium

    try:
        document: Any = pdfium.PdfDocument(str(path))
    except Exception as exc:
        raise TateError(f"could not read PDF: {label}") from exc
    try:
        return _render_document_page(document, page_index, label)
    finally:
        document.close()


def _render_pdf_bytes(data: bytes, page_index: int, label: str) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as handle:
        handle.write(data)
        handle.flush()
        return _render_pdf_path(Path(handle.name), page_index, label)


def _render_document_page(document: Any, page_index: int, label: str) -> Image.Image:
    count = len(document)
    if page_index < 0 or page_index >= count:
        raise TateError(f"PDF page {page_index} is outside {label} ({count} pages)")
    page: Any = document[page_index]
    try:
        return render_pdf_page(page, InputLimits(), label)
    finally:
        closer: Any = getattr(page, "close", None)
        if callable(closer):
            closer()


def _write_checksums(
    dataset: Dataset,
    source_hashes: dict[str, str],
    pixel_hashes: dict[str, str],
) -> list[str]:
    updated: list[str] = []
    sources_changed = False
    sources: list[Source] = []
    for source in dataset.sources:
        digest = source_hashes.get(source.id)
        if source.fetch.sha256 is None and digest is not None:
            sources.append(replace(source, fetch=replace(source.fetch, sha256=digest)))
            sources_changed = True
        else:
            sources.append(source)
    if sources_changed:
        write_sources(dataset.root / "sources.json", sources)
        updated.append("sources.json")
    for work in dataset.works:
        rewritten, changed = _fill_work(work, pixel_hashes)
        if changed:
            write_work(rewritten)
            updated.append(f"pages/{work.path.name}")
    return updated


def _fill_work(work: Work, pixel_hashes: dict[str, str]) -> tuple[Work, bool]:
    changed = False
    pages: list[PageAnn] = []
    for page in work.pages:
        digest = pixel_hashes.get(page.id)
        if page.pixel_sha256 is None and digest is not None:
            pages.append(replace(page, pixel_sha256=digest))
            changed = True
        else:
            pages.append(page)
    if not changed:
        return work, False
    return replace(work, pages=tuple(pages)), True
