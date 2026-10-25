"""Run detection, layout, and packaging for one chapter."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from tate.debug import draw_debug
from tate.detect import Detector, resolve_detector
from tate.errors import TateError
from tate.io.inputs import ChapterInput
from tate.layout import SegmentDraft, assign_gaps
from tate.layout import layout_page as lay_out
from tate.order import ReadingDirection
from tate.package import (
    Manifest,
    SegmentRecord,
    build_manifest,
    encode_webp,
    make_cover,
    segment_name,
    write_package,
    write_preview,
)


@dataclass(frozen=True, slots=True)
class ProcessResult:
    output: Path
    manifest: Manifest
    detector: str
    page_count: int
    panel_count: int
    output_bytes: int
    elapsed: float


def input_name(path: Path) -> str:
    """File name of the input; a directory such as ``.`` resolves to its real name."""
    name = path.name or path.resolve().name
    return name or "chapter"


def input_stem(path: Path) -> str:
    if path.is_dir():
        return input_name(path)
    return path.stem or "chapter"


def default_output(path: Path) -> Path:
    return Path.cwd() / f"{input_stem(path)}.tate"


def default_title(path: Path) -> str:
    return input_stem(path)


def format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def format_summary(result: ProcessResult) -> str:
    try:
        shown = result.output.relative_to(Path.cwd())
    except ValueError:
        shown = result.output
    return (
        f"pages: {result.page_count}\n"
        f"panels: {result.panel_count}\n"
        f"detector: {result.detector}\n"
        f"output: {shown} ({format_bytes(result.output_bytes)})\n"
        f"time: {result.elapsed:.1f}s\n"
    )


def process_chapter(
    input_path: Path,
    output_path: Path | None = None,
    *,
    title: str | None = None,
    width: int = 1080,
    detector_name: str = "auto",
    preview: Path | None = None,
    debug_dir: Path | None = None,
    pages: str | None = None,
    detector: Detector | None = None,
    direction: ReadingDirection = "rtl",
) -> ProcessResult:
    """Process one chapter. Pass `detector` to skip model loading in tests."""
    started = time.perf_counter()
    if width < 64 or width > 8192:
        raise TateError("width must be between 64 and 8192")
    if direction not in {"rtl", "ltr"}:
        raise TateError("direction must be rtl or ltr")
    chosen = detector if detector is not None else resolve_detector(detector_name)
    destination = output_path if output_path is not None else default_output(input_path)
    if destination.exists() and destination.is_dir():
        raise TateError(f"output is a directory: {destination.name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
    if preview is not None:
        preview.parent.mkdir(parents=True, exist_ok=True)

    drafts: list[SegmentDraft] = []
    page_count = 0
    with ChapterInput.open(input_path, pages) as chapter:
        if len(chapter) == 0:
            raise TateError(f"no images found in input: {input_path.name}")
        for page in chapter:
            page_count += 1
            detection = chosen.detect(page.image)
            laid = lay_out(
                page.image,
                detection,
                page_index=page.index,
                width=width,
                direction=direction,
            )
            if debug_dir is not None:
                overlay = draw_debug(page.image, laid)
                overlay.save(debug_dir / f"{page.index + 1:04d}.jpg", format="JPEG", quality=90)
                overlay.close()
            drafts.extend(laid.segments)
            page.image.close()
    if not drafts:
        raise TateError("no segments produced")

    gaps = assign_gaps(drafts, width)
    cover = encode_webp(make_cover(drafts[0].image))
    records: list[SegmentRecord] = []
    blobs: list[tuple[str, bytes]] = []
    preview_parts: list[tuple[int, Image.Image]] = []
    for index, (draft, gap) in enumerate(zip(drafts, gaps, strict=True)):
        if draft.image.width != width:
            raise TateError("internal error: segment width mismatch")
        name = segment_name(index)
        blobs.append((name, encode_webp(draft.image)))
        records.append(
            {
                "file": name,
                "width": width,
                "height": draft.image.height,
                "page": draft.page,
                "gapBefore": gap,
                "box": [draft.box.x, draft.box.y, draft.box.w, draft.box.h],
            }
        )
        if preview is not None:
            preview_parts.append((gap, draft.image.copy()))
        draft.image.close()

    manifest = build_manifest(
        title=title if title is not None else default_title(input_path),
        source=input_name(input_path),
        width=width,
        detector=chosen.name,
        page_count=page_count,
        segments=records,
        reading=direction,
    )
    write_package(destination, manifest, cover, blobs)
    if preview is not None:
        write_preview(preview, preview_parts, width)
    elapsed = time.perf_counter() - started
    return ProcessResult(
        output=destination,
        manifest=manifest,
        detector=chosen.name,
        page_count=page_count,
        panel_count=sum(1 for draft in drafts if not draft.continuation),
        output_bytes=destination.stat().st_size,
        elapsed=elapsed,
    )
