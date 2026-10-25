"""Load page images from a folder, an archive, a PDF, or a single file.

Zip entries are read in memory. Entry names are never used as filesystem paths.
"""

from __future__ import annotations

import io
import math
import os
import re
import warnings
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from tate.errors import TateError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}
_JUNK_NAMES = {"thumbs.db", "desktop.ini", ".ds_store"}
_RANGE = re.compile(r"(\d+)-(\d+)")
_PAGE = re.compile(r"\d+")


@dataclass(frozen=True, slots=True)
class InputLimits:
    """Caps that keep a hostile or huge input from exhausting memory."""

    max_entries: int = 4000
    max_entry_bytes: int = 64 * 1024 * 1024
    max_total_bytes: int = 1024 * 1024 * 1024
    max_image_pixels: int = 40_000_000
    max_compression_ratio: float = 100.0
    max_pdf_pages: int = 2000
    pdf_target_height: int = 2000
    max_page_pixels: int = 8_000_000


@dataclass(frozen=True, slots=True)
class Page:
    """One source page, already decoded to RGB."""

    index: int
    name: str
    image: Image.Image


def natural_key(name: str) -> tuple[tuple[int, str | int], ...]:
    """Sort key that orders `page2` before `page10`."""
    parts: list[tuple[int, str | int]] = []
    for piece in re.split(r"(\d+)", name):
        if piece.isdigit():
            parts.append((0, int(piece)))
        else:
            parts.append((1, piece.casefold()))
    return tuple(parts)


def entry_disposition(name: str) -> str:
    """Classify a zip entry as `ok`, `junk`, or `unsafe`."""
    raw = name.replace("\\", "/")
    if raw.startswith("/") or raw.startswith("//") or re.match(r"^[A-Za-z]:/", raw):
        return "unsafe"
    parts = PurePosixPath(raw).parts
    if any(part == ".." for part in parts):
        return "unsafe"
    if not parts or raw.endswith("/"):
        return "junk"
    for part in parts:
        if part.startswith(".") or part.casefold() == "__macosx":
            return "junk"
    base = parts[-1]
    if base.casefold() in _JUNK_NAMES or base.startswith("."):
        return "junk"
    if PurePosixPath(base).suffix.casefold() not in IMAGE_SUFFIXES:
        return "junk"
    return "ok"


def parse_page_range(spec: str, count: int) -> list[int]:
    """Parse a 1-based `N` or `N-M` range into 0-based indexes."""
    text = spec.strip()
    match = _RANGE.fullmatch(text)
    if match:
        start = int(match.group(1))
        end = int(match.group(2))
    elif _PAGE.fullmatch(text):
        start = end = int(text)
    else:
        raise TateError(f"invalid page range: {spec!r} (expected N or N-M)")
    if start < 1 or end < start:
        raise TateError(f"invalid page range: {spec}")
    if count < 1 or start > count:
        raise TateError(f"page range starts after the last page ({count})")
    if end > count:
        raise TateError(f"page range ends after the last page ({count})")
    return list(range(start - 1, end))


def to_rgb(image: Image.Image) -> Image.Image:
    oriented = ImageOps.exif_transpose(image)
    current = image if oriented is None else oriented
    if current.mode == "RGB":
        return current
    if current.mode == "RGBA":
        background = Image.new("RGB", current.size, (255, 255, 255))
        background.paste(current, mask=current.getchannel("A"))
        return background
    return current.convert("RGB")


def open_image(data: bytes, limits: InputLimits, name: str) -> Image.Image:
    if len(data) > limits.max_entry_bytes:
        raise TateError(f"image is too large: {name}")
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = limits.max_image_pixels
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            try:
                with Image.open(io.BytesIO(data)) as raw:
                    raw.load()
                    if raw.width * raw.height > limits.max_image_pixels:
                        raise TateError(f"image is too large: {name}")
                    frame = raw.copy()
            except TateError:
                raise
            except Image.DecompressionBombError as exc:
                raise TateError(f"image is too large: {name}") from exc
            except (UnidentifiedImageError, OSError) as exc:
                raise TateError(f"could not read image: {name}") from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous
    return to_rgb(frame)


@dataclass
class _Source:
    index: int
    name: str
    load: Callable[[], Image.Image]


class ChapterInput:
    """Selected pages of one input, decoded one at a time."""

    def __init__(self, sources: list[_Source], cleanup: list[Callable[[], None]]) -> None:
        self._sources = sources
        self._cleanup = cleanup

    @classmethod
    def open(
        cls,
        path: Path,
        pages: str | None,
        limits: InputLimits | None = None,
    ) -> ChapterInput:
        chosen = limits if limits is not None else InputLimits()
        if not path.exists():
            raise TateError(f"input not found: {path.name}")
        if path.is_dir():
            return cls._open_directory(path, pages, chosen)
        suffix = path.suffix.casefold()
        if suffix in {".cbz", ".zip"}:
            return cls._open_zip(path, pages, chosen)
        if suffix == ".pdf":
            return cls._open_pdf(path, pages, chosen)
        if suffix in IMAGE_SUFFIXES:
            return cls._open_file(path, pages, chosen)
        raise TateError(f"unsupported input: {path.name}")

    def close(self) -> None:
        for cleanup in self._cleanup:
            cleanup()
        self._cleanup.clear()

    def __enter__(self) -> ChapterInput:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._sources)

    def __iter__(self) -> Iterator[Page]:
        for source in self._sources:
            yield Page(index=source.index, name=source.name, image=source.load())

    @classmethod
    def _open_directory(cls, path: Path, pages: str | None, limits: InputLimits) -> ChapterInput:
        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(path):
            dirnames[:] = [
                name for name in dirnames if not name.startswith(".") and name.casefold() != "__macosx"
            ]
            for filename in filenames:
                if entry_disposition(filename) != "ok":
                    continue
                found.append(Path(dirpath) / filename)
        found.sort(key=lambda item: natural_key(item.relative_to(path).as_posix()))
        return cls._from_files(path, found, pages, limits)

    @classmethod
    def _open_file(cls, path: Path, pages: str | None, limits: InputLimits) -> ChapterInput:
        return cls._from_files(path, [path], pages, limits)

    @classmethod
    def _from_files(
        cls,
        label: Path,
        files: list[Path],
        pages: str | None,
        limits: InputLimits,
    ) -> ChapterInput:
        if not files:
            raise TateError(f"no images found in input: {label.name}")
        indexes = parse_page_range(pages, len(files)) if pages else list(range(len(files)))
        total = 0
        sources: list[_Source] = []
        for index in indexes:
            file = files[index]
            size = file.stat().st_size
            if size > limits.max_entry_bytes:
                raise TateError(f"image is too large: {file.name}")
            total += size
            if total > limits.max_total_bytes:
                raise TateError("input is too large")
            sources.append(_Source(index, file.name, _file_loader(file, limits)))
        return cls(sources, [])

    @classmethod
    def _open_zip(cls, path: Path, pages: str | None, limits: InputLimits) -> ChapterInput:
        if not zipfile.is_zipfile(path):
            raise TateError(f"not a zip archive: {path.name}")
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                if len(infos) > limits.max_entries:
                    raise TateError(f"zip has too many entries: {path.name}")
                accepted: list[zipfile.ZipInfo] = []
                total = 0
                for info in infos:
                    disposition = entry_disposition(info.filename)
                    if disposition == "junk":
                        continue
                    _guard_zip_entry(info, limits)
                    if disposition != "ok":
                        raise TateError(f"unsafe zip entry: {PurePosixPath(info.filename).name}")
                    total += max(0, info.file_size)
                    if total > limits.max_total_bytes:
                        raise TateError("zip uncompressed size exceeds the limit")
                    accepted.append(info)
                accepted.sort(key=lambda info: natural_key(info.filename.replace("\\", "/")))
                if not accepted:
                    raise TateError(f"no images found in input: {path.name}")
                indexes = parse_page_range(pages, len(accepted)) if pages else list(range(len(accepted)))
                blobs: list[tuple[int, str, bytes]] = []
                for index in indexes:
                    info = accepted[index]
                    name = PurePosixPath(info.filename.replace("\\", "/")).name
                    blobs.append((index, name, _read_zip_entry(archive, info, limits)))
        except zipfile.BadZipFile as exc:
            raise TateError(f"could not read zip archive: {path.name}") from exc
        sources = [_Source(index, name, _bytes_loader(data, limits, name)) for index, name, data in blobs]
        return cls(sources, [])

    @classmethod
    def _open_pdf(cls, path: Path, pages: str | None, limits: InputLimits) -> ChapterInput:
        import pypdfium2 as pdfium

        try:
            document: Any = pdfium.PdfDocument(str(path))
        except Exception as exc:
            raise TateError(f"could not read PDF: {path.name}") from exc
        try:
            count = len(document)
            if count > limits.max_pdf_pages:
                raise TateError(f"PDF has too many pages: {path.name}")
            if count < 1:
                raise TateError(f"no images found in input: {path.name}")
            indexes = parse_page_range(pages, count) if pages else list(range(count))
            sources = [
                _Source(index, f"{index + 1}", _pdf_loader(document, index, limits, path.name))
                for index in indexes
            ]
        except Exception:
            document.close()
            raise
        return cls(sources, [document.close])


def _file_loader(path: Path, limits: InputLimits) -> Callable[[], Image.Image]:
    def load() -> Image.Image:
        data = path.read_bytes()
        return open_image(data, limits, path.name)

    return load


def _bytes_loader(data: bytes, limits: InputLimits, name: str) -> Callable[[], Image.Image]:
    def load() -> Image.Image:
        return open_image(data, limits, name)

    return load


def _pdf_loader(
    document: Any,
    index: int,
    limits: InputLimits,
    name: str,
) -> Callable[[], Image.Image]:
    def load() -> Image.Image:
        page = document[index]
        try:
            return render_pdf_page(page, limits, name)
        finally:
            closer = getattr(page, "close", None)
            if callable(closer):
                closer()

    return load


def render_pdf_page(page: Any, limits: InputLimits, name: str) -> Image.Image:
    """Rasterize one PDF page to about `pdf_target_height` pixels, capping pixels."""
    size = page.get_size()
    width_pt = float(size[0])
    height_pt = float(size[1])
    if width_pt <= 0 or height_pt <= 0:
        raise TateError(f"PDF page has no size: {name}")
    scale = limits.pdf_target_height / height_pt
    pixels = width_pt * scale * height_pt * scale
    if pixels > limits.max_page_pixels:
        scale *= math.sqrt(limits.max_page_pixels / pixels)
    bitmap = page.render(scale=scale, fill_color=(255, 255, 255, 255))
    try:
        rendered = bitmap.to_pil()
    finally:
        closer = getattr(bitmap, "close", None)
        if callable(closer):
            closer()
    image = to_rgb(rendered)
    if image.width * image.height > limits.max_image_pixels:
        image.close()
        raise TateError(f"image is too large: {name}")
    return image


def _guard_zip_entry(info: zipfile.ZipInfo, limits: InputLimits) -> None:
    label = PurePosixPath(info.filename.replace("\\", "/")).name or info.filename
    if info.flag_bits & 0x1:
        raise TateError(f"encrypted zip entry: {label}")
    size = info.file_size
    if size < 0 or size > limits.max_entry_bytes:
        raise TateError(f"zip entry is too large: {label}")
    if info.compress_size > 0 and size > 1_000_000:
        ratio = size / info.compress_size
        if ratio > limits.max_compression_ratio:
            raise TateError(f"zip entry looks like a compression bomb: {label}")


def _read_zip_entry(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limits: InputLimits) -> bytes:
    label = PurePosixPath(info.filename.replace("\\", "/")).name
    with archive.open(info, "r") as handle:
        data = handle.read(limits.max_entry_bytes + 1)
    if len(data) > limits.max_entry_bytes:
        raise TateError(f"zip entry is too large: {label}")
    return data
