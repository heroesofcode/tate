"""Run detectors and ``layout_page`` on cached pages."""

from __future__ import annotations

import multiprocessing
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from tate.boxes import Box
from tate.detect import Detector, resolve_detector
from tate.errors import TateError
from tate.eval.checksum import pixel_sha256
from tate.eval.dataset import select_pages
from tate.eval.fetch import page_png
from tate.eval.metrics import STRIP_WIDTH, PageScore, score_page
from tate.eval.model import Dataset, LoadedPage
from tate.layout import layout_page

_DETECTORS: dict[str, Detector] = {}


@dataclass(frozen=True, slots=True)
class _Task:
    detector: str
    page_id: str
    work: str
    styles: tuple[str, ...]
    png: str
    size: tuple[int, int]
    pixel_sha256: str | None
    whole: bool
    panels: tuple[tuple[int, int, int, int], ...]
    balloons: tuple[tuple[int, int, int, int], ...]
    ignore: tuple[tuple[int, int, int, int], ...]


def default_jobs() -> int:
    """CPU count, capped at 4."""
    return max(1, min(4, os.cpu_count() or 1))


def parse_detectors(text: str) -> list[str]:
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names:
        raise TateError("select at least one detector")
    if len(names) != len(set(names)):
        raise TateError("detectors must be unique")
    for name in names:
        if name not in {"yolo", "cv"}:
            raise TateError(f"unknown detector: {name}")
    return names


def evaluate(
    dataset: Dataset,
    *,
    cache: str | os.PathLike[str],
    detectors: Sequence[str],
    subset: str,
    jobs: int,
    instances: Mapping[str, Detector] | None = None,
) -> dict[str, list[PageScore]]:
    """Score ``subset``. ``instances`` runs in-process, which is how tests avoid a model download."""
    if jobs < 1:
        raise TateError("jobs must be at least 1")
    pages = select_pages(dataset, subset)
    if not pages:
        raise TateError(f"no pages in subset {subset}")
    root = Path(cache)
    missing = [page.page.id for page in pages if not page_png(root, page.page.id).is_file()]
    if missing:
        raise TateError(f"page {missing[0]} is not in the cache; run tate eval fetch")
    tasks = [
        _task(page, detector, str(page_png(root, page.page.id))) for page in pages for detector in detectors
    ]
    if instances is not None:
        grouped = _run_instances(tasks, instances)
    elif jobs == 1:
        _init(list(detectors))
        grouped = _collect(tasks, [_run_task(task) for task in tasks])
    else:
        grouped = _run_pool(tasks, list(detectors), jobs)
    for name in grouped:
        grouped[name].sort(key=lambda page: page.page_id)
    return grouped


def _task(page: LoadedPage, detector: str, png: str) -> _Task:
    geometry = page.geometry
    return _Task(
        detector=detector,
        page_id=page.page.id,
        work=page.work_id,
        styles=page.page.styles,
        png=png,
        size=page.page.size,
        pixel_sha256=page.page.pixel_sha256,
        whole=geometry.whole,
        panels=tuple((panel.box.x, panel.box.y, panel.box.w, panel.box.h) for panel in geometry.panels),
        balloons=tuple((box.x, box.y, box.w, box.h) for box in geometry.balloons),
        ignore=tuple((box.x, box.y, box.w, box.h) for box in geometry.ignore),
    )


def _init(names: list[str]) -> None:
    global _DETECTORS
    _DETECTORS = {name: resolve_detector(name) for name in names}


def _run_task(task: _Task) -> PageScore:
    detector = _DETECTORS[task.detector]
    with Image.open(task.png) as raw:
        raw.load()
        image = raw.convert("RGB")
    if image.size != task.size:
        width, height = image.size
        image.close()
        raise TateError(
            f"page {task.page_id} image is {width}x{height}, dataset says {task.size[0]}x{task.size[1]}"
        )
    if task.pixel_sha256 is not None and pixel_sha256(image) != task.pixel_sha256:
        image.close()
        raise TateError(f"pixel_sha256 mismatch for {task.page_id}")
    laid = None
    try:
        started = time.perf_counter()
        detection = detector.detect(image)
        laid = layout_page(image, detection, page_index=0, width=STRIP_WIDTH)
        elapsed = time.perf_counter() - started
        return score_page(
            image,
            page_id=task.page_id,
            work=task.work,
            styles=task.styles,
            whole=task.whole,
            gt_panels=tuple(Box(*box) for box in task.panels),
            balloons=tuple(Box(*box) for box in task.balloons),
            ignore=tuple(Box(*box) for box in task.ignore),
            layout=laid,
            elapsed=elapsed,
        )
    finally:
        image.close()
        if laid is not None:
            for segment in laid.segments:
                segment.image.close()


def _run_instances(tasks: Sequence[_Task], instances: Mapping[str, Detector]) -> dict[str, list[PageScore]]:
    global _DETECTORS
    previous = _DETECTORS
    _DETECTORS = dict(instances)
    try:
        return _collect(tasks, [_run_task(task) for task in tasks])
    finally:
        _DETECTORS = previous


def _run_pool(tasks: Sequence[_Task], detectors: list[str], jobs: int) -> dict[str, list[PageScore]]:
    workers = min(jobs, max(1, len(tasks)))
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init,
        initargs=(detectors,),
    ) as pool:
        results = list(pool.map(_run_task, tasks))
    return _collect(tasks, results)


def _collect(tasks: Sequence[_Task], results: Sequence[PageScore]) -> dict[str, list[PageScore]]:
    grouped: dict[str, list[PageScore]] = {}
    for task, score in zip(tasks, results, strict=True):
        grouped.setdefault(task.detector, []).append(score)
    return grouped
