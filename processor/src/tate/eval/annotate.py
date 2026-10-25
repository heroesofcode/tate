"""Draft ground truth from the detector, and contact sheets for review."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tate.boxes import Box, intersection_area, union_box
from tate.detect import Detector, resolve_detector
from tate.errors import TateError
from tate.eval.checksum import pixel_sha256
from tate.eval.dataset import _expect_id, _parse_work, load_dataset, write_work
from tate.eval.fetch import default_cache_dir, page_png
from tate.eval.metrics import STRIP_WIDTH
from tate.eval.model import LoadedPage, PageAnn, PanelAnn, SourceImage, Work
from tate.io.inputs import ChapterInput
from tate.layout import layout_page
from tate.order import reading_order

_TILE = (720, 960)
_LABEL = 40


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    wrote: int
    skipped: int


def bootstrap(
    dataset_root: Path,
    work_id: str,
    pages_from: Path,
    *,
    detector: Detector | None = None,
    force: bool = False,
    title: str | None = None,
) -> BootstrapResult:
    """Write draft panels and merged text boxes. Existing pages are kept unless ``force``."""
    _expect_id(work_id, "work")
    chosen = detector if detector is not None else resolve_detector("yolo")
    path = dataset_root / "pages" / f"{work_id}.json"
    if path.is_file():
        work = _parse_work(path)
    else:
        work = Work(
            id=work_id,
            title=title if title is not None else work_id,
            direction="rtl",
            path=path,
            pages=(),
        )
    if title is not None:
        work = replace(work, title=title)
    existing = {page.id: page for page in work.pages}
    wrote = 0
    skipped = 0
    drafted: list[PageAnn] = []
    seen: set[str] = set()
    with ChapterInput.open(pages_from, None) as chapter:
        if len(chapter) == 0:
            raise TateError(f"no images found in input: {pages_from.name}")
        for source in chapter:
            page_id = f"{work_id}-{source.index + 1:04d}"
            seen.add(page_id)
            current = existing.get(page_id)
            if current is not None and not force:
                drafted.append(current)
                skipped += 1
                source.image.close()
                continue
            drafted.append(_draft_page(current, page_id, work_id, source.name, source.image, chosen))
            wrote += 1
            source.image.close()
    for page in work.pages:
        if page.id not in seen:
            drafted.append(page)
    write_work(replace(work, path=path, pages=tuple(drafted)))
    return BootstrapResult(wrote, skipped)


def review(
    dataset_root: Path,
    work_id: str,
    *,
    cache: Path,
    out: Path,
    page_ids: Sequence[str] | None = None,
) -> list[Path]:
    """Tile ground-truth overlays into ``review/<work>-NN.jpg`` contact sheets."""
    dataset = load_dataset(dataset_root)
    pages = [page for page in dataset.pages if page.work_id == work_id]
    if page_ids is not None:
        by_id = {page.page.id: page for page in pages}
        missing = [page_id for page_id in page_ids if page_id not in by_id]
        if missing:
            raise TateError(f"unknown page {missing[0]}")
        pages = [by_id[page_id] for page_id in page_ids]
    if not pages:
        raise TateError(f"no pages for {work_id}")
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    chunks = [pages[index : index + 4] for index in range(0, len(pages), 4)]
    for number, chunk in enumerate(chunks, start=1):
        sheet = _contact_sheet(chunk, cache)
        destination = out / f"{work_id}-{number:02d}.jpg"
        sheet.save(destination, format="JPEG", quality=85)
        sheet.close()
        written.append(destination)
    return written


def merge_overlapping(boxes: Sequence[Box]) -> list[Box]:
    """Union boxes that overlap into one box per connected component."""
    remaining = list(boxes)
    merged: list[Box] = []
    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            kept: list[Box] = []
            for other in remaining:
                if intersection_area(current, other) > 0:
                    current = union_box(current, other)
                    changed = True
                else:
                    kept.append(other)
            remaining = kept
        merged.append(current)
    order = reading_order(merged)
    return [merged[index] for index in order]


def bootstrap_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write draft panels and balloons for one work.")
    parser.add_argument("--work", required=True, help="work id")
    parser.add_argument("--pages-from", type=Path, required=True, help="directory, zip, cbz, pdf, or image")
    parser.add_argument("--dataset", default="eval")
    parser.add_argument("--title", help="work title used when the file is created")
    parser.add_argument("--force", action="store_true", help="replace panels on pages that already exist")
    args = parser.parse_args(argv)
    try:
        result = bootstrap(
            Path(str(args.dataset)),
            str(args.work),
            args.pages_from,
            force=bool(args.force),
            title=args.title,
        )
    except TateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(f"wrote: {result.wrote}\nskipped: {result.skipped}\n")
    return 0


def review_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render ground-truth contact sheets for one work.")
    parser.add_argument("--work", required=True)
    parser.add_argument("--pages", help="comma-separated page ids")
    parser.add_argument("--dataset", default="eval")
    parser.add_argument("--cache", type=Path, help="page cache (default: the user cache tate/eval)")
    parser.add_argument("--out", type=Path, default="review")
    args = parser.parse_args(argv)
    page_ids = (
        None if args.pages is None else [part.strip() for part in str(args.pages).split(",") if part.strip()]
    )
    cache = args.cache if args.cache is not None else default_cache_dir()
    try:
        paths = review(
            Path(str(args.dataset)),
            str(args.work),
            cache=cache,
            out=Path(str(args.out)),
            page_ids=page_ids,
        )
    except TateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for path in paths:
        print(path)
    return 0


def _draft_page(
    current: PageAnn | None,
    page_id: str,
    work_id: str,
    source_name: str,
    image: Image.Image,
    detector: Detector,
) -> PageAnn:
    detection = detector.detect(image)
    laid = layout_page(image, detection, page_index=0, width=STRIP_WIDTH)
    try:
        panels = tuple(PanelAnn(box) for box in laid.emitted_panels())
        balloons = tuple(merge_overlapping(detection.texts))
        whole = laid.fallback or laid.spread
        styles = ("standard", "spread") if laid.spread else ("standard",)
        if current is None:
            return PageAnn(
                id=page_id,
                styles=styles,
                image=SourceImage(work_id),
                size=image.size,
                pixel_sha256=pixel_sha256(image),
                panels=panels,
                panels_written=True,
                whole=whole,
                balloons=balloons,
                balloons_written=True,
                ignore=(),
                ignore_written=False,
                ci=False,
                notes=f"draft from {source_name}",
            )
        return replace(
            current,
            size=image.size,
            pixel_sha256=pixel_sha256(image),
            panels=panels,
            panels_written=True,
            balloons=balloons,
            balloons_written=True,
            whole=whole,
        )
    finally:
        for segment in laid.segments:
            segment.image.close()


def _contact_sheet(pages: Sequence[LoadedPage], cache: Path) -> Image.Image:
    width, height = _TILE
    sheet = Image.new("RGB", (width * 2, (height + _LABEL) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=22)
    for slot, loaded in enumerate(pages):
        png = page_png(cache, loaded.page.id)
        if not png.is_file():
            raise TateError(f"page {loaded.page.id} is not in the cache; run tate eval fetch")
        with Image.open(png) as raw:
            raw.load()
            overlay = _ground_truth(raw.convert("RGB"), loaded)
        tile = _fit(overlay, width, height)
        overlay.close()
        column = slot % 2
        row = slot // 2
        origin_x = column * width
        origin_y = row * (height + _LABEL)
        sheet.paste(tile, (origin_x, origin_y + _LABEL))
        tile.close()
        draw.text((origin_x + 8, origin_y + 8), loaded.page.id, fill=(0, 0, 0), font=font)
    return sheet


def _ground_truth(image: Image.Image, loaded: LoadedPage) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    line = max(2, min(canvas.size) // 400)
    font = ImageFont.load_default(size=max(16, min(canvas.size) // 40))
    for box in loaded.geometry.ignore:
        draw.rectangle(_corners(box), outline=(120, 120, 120), width=max(2, line // 2))
    for box in loaded.geometry.balloons:
        draw.rectangle(_corners(box), outline=(30, 90, 220), width=max(2, line // 2))
    for number, panel in enumerate(loaded.geometry.panels, start=1):
        draw.rectangle(_corners(panel.box), outline=(30, 150, 60), width=line)
        draw.text(
            (panel.box.x + line, panel.box.y + line),
            str(number),
            fill=(30, 150, 60),
            font=font,
        )
    image.close()
    return canvas


def _fit(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    copy.close()
    return canvas


def _corners(box: Box) -> list[int]:
    return [box.x, box.y, max(box.x, box.x2 - 1), max(box.y, box.y2 - 1)]
