"""Write a version-1 `.tate` zip and the optional preview strip."""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict

from PIL import Image

from tate.errors import TateError
from tate.layout import BACKGROUND_HEX, BACKGROUND_RGB

WEBP_QUALITY = 85
COVER_WIDTH = 360
PREVIEW_WIDTH = 480
WEBP_MAX_DIMENSION = 16383


class SegmentRecord(TypedDict):
    file: str
    width: int
    height: int
    page: int
    gapBefore: int
    box: list[int]


class Manifest(TypedDict):
    format: str
    version: int
    title: str
    source: str
    createdAt: str
    reading: str
    width: int
    background: str
    detector: str
    pageCount: int
    cover: str
    segments: list[SegmentRecord]


def created_at_iso() -> str:
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    if raw is not None and raw.strip() != "":
        try:
            stamp = int(raw)
        except ValueError as exc:
            raise TateError("SOURCE_DATE_EPOCH must be an integer") from exc
        moment = datetime.fromtimestamp(stamp, tz=UTC)
    else:
        moment = datetime.now(UTC).replace(microsecond=0)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def zip_timestamp() -> tuple[int, int, int, int, int, int]:
    raw = os.environ.get("SOURCE_DATE_EPOCH")
    if raw is not None and raw.strip() != "":
        try:
            stamp = int(raw)
        except ValueError as exc:
            raise TateError("SOURCE_DATE_EPOCH must be an integer") from exc
        moment = time.gmtime(stamp)
    else:
        moment = time.gmtime()
    if moment.tm_year < 1980:
        return (1980, 1, 1, 0, 0, 0)
    if moment.tm_year > 2107:
        return (2107, 12, 31, 23, 59, 59)
    return (
        moment.tm_year,
        moment.tm_mon,
        moment.tm_mday,
        moment.tm_hour,
        moment.tm_min,
        moment.tm_sec,
    )


def segment_name(index: int) -> str:
    number = index + 1
    if number > 9999:
        raise TateError("chapter has more than 9999 segments")
    return f"strip/{number:04d}.webp"


def encode_webp(image: Image.Image, *, quality: int = WEBP_QUALITY) -> bytes:
    buffer = io.BytesIO()
    try:
        image.save(buffer, format="WEBP", quality=quality, method=4)
    except (OSError, ValueError) as exc:
        raise TateError("could not encode an image") from exc
    return buffer.getvalue()


def make_cover(image: Image.Image, cover_width: int = COVER_WIDTH) -> Image.Image:
    """Scale a segment to `cover_width`. The top is kept if WebP cannot hold the height."""
    width = max(1, min(cover_width, WEBP_MAX_DIMENSION))
    height = _scaled_height(image.height, image.width, width)
    source = image
    if height > WEBP_MAX_DIMENSION:
        keep = max(1, int(image.height * WEBP_MAX_DIMENSION / height))
        source = image.crop((0, 0, image.width, min(image.height, keep)))
        height = min(WEBP_MAX_DIMENSION, _scaled_height(source.height, source.width, width))
    resized = source.resize((width, height), Image.Resampling.LANCZOS)
    if source is not image:
        source.close()
    return resized


def _scaled_height(src_h: int, src_w: int, width: int) -> int:
    return max(1, round(src_h * (width / max(1, src_w))))


def build_manifest(
    *,
    title: str,
    source: str,
    width: int,
    detector: str,
    page_count: int,
    segments: list[SegmentRecord],
) -> Manifest:
    return {
        "format": "tate",
        "version": 1,
        "title": title,
        "source": source,
        "createdAt": created_at_iso(),
        "reading": "rtl",
        "width": width,
        "background": BACKGROUND_HEX,
        "detector": detector,
        "pageCount": page_count,
        "cover": "cover.webp",
        "segments": segments,
    }


def write_package(
    path: Path,
    manifest: Manifest,
    cover: bytes,
    segments: Sequence[tuple[str, bytes]],
) -> None:
    """Write the chapter atomically. Images are stored; the manifest is deflated."""
    when = zip_timestamp()
    payload = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    partial = path.with_name(path.name + ".partial")
    try:
        with zipfile.ZipFile(partial, "w") as archive:
            archive.writestr(_zip_info("manifest.json", when, zipfile.ZIP_DEFLATED), payload)
            archive.writestr(_zip_info("cover.webp", when, zipfile.ZIP_STORED), cover)
            for name, blob in segments:
                archive.writestr(_zip_info(name, when, zipfile.ZIP_STORED), blob)
        os.replace(partial, path)
    except TateError:
        if partial.exists():
            partial.unlink()
        raise
    except OSError as exc:
        if partial.exists():
            partial.unlink()
        raise TateError(f"could not write {path.name}") from exc


def write_preview(path: Path, pieces: Sequence[tuple[int, Image.Image]], strip_width: int) -> None:
    """Composite the strip and downscale it to 480 px wide."""
    if strip_width < 1 or not pieces:
        raise TateError("nothing to preview")
    scale = PREVIEW_WIDTH / strip_width
    rendered: list[tuple[int, Image.Image]] = []
    total = 0
    for gap, image in pieces:
        gap_px = round(gap * scale)
        height = max(1, round(image.height * scale))
        resized = image.resize((PREVIEW_WIDTH, height), Image.Resampling.LANCZOS)
        rendered.append((gap_px, resized))
        total += gap_px + height
    canvas = Image.new("RGB", (PREVIEW_WIDTH, max(1, total)), BACKGROUND_RGB)
    cursor = 0
    for gap_px, image in rendered:
        cursor += gap_px
        canvas.paste(image, (0, cursor))
        cursor += image.height
        image.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG")
    canvas.close()


def _zip_info(
    name: str,
    when: tuple[int, int, int, int, int, int],
    compress_type: int,
) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=when)
    info.compress_type = compress_type
    info.external_attr = 0o644 << 16
    return info
