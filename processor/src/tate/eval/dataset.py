"""Load, validate, and rewrite the ground-truth dataset."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from tate.boxes import Box
from tate.errors import TateError
from tate.eval.derive import apply_geometry
from tate.eval.model import (
    Contrast,
    Dataset,
    DeriveImage,
    Gamma,
    Geometry,
    ImageFetch,
    Jpeg,
    LoadedPage,
    Noise,
    Op,
    PageAnn,
    PanelAnn,
    Rotate,
    Scale,
    Source,
    SourceImage,
    Work,
    Yellow,
    ZipFetch,
)
from tate.eval.styles import STYLES

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA = re.compile(r"^[0-9a-fA-F]{64}$")
_MAX_PIXELS = 40_000_000
_MAX_NOTES = 4000
_MAX_CHAIN = 8

_SOURCE_KEYS = {"id", "title", "author", "license", "license_url", "attribution", "homepage", "fetch"}
_FETCH_KEYS = {"kind", "url", "sha256"}
_WORK_KEYS = {"work", "title", "direction", "pages"}
_PAGE_KEYS = {
    "id",
    "styles",
    "image",
    "size",
    "pixel_sha256",
    "panels",
    "whole",
    "balloons",
    "ignore",
    "ci",
    "notes",
}
_PANEL_KEYS = {"box", "poly"}
_BALLOON_KEYS = {"box"}


def load_dataset(root: Path) -> Dataset:
    """Read ``sources.json`` and ``pages/*.json``. Derived ground truth is resolved here."""
    source_path = root / "sources.json"
    if not source_path.is_file():
        raise TateError(f"missing {source_path.name}")
    sources = _parse_sources(_load_json(source_path, "sources.json"), "sources.json")
    pages_dir = root / "pages"
    works: list[Work] = []
    if pages_dir.is_dir():
        for path in sorted(pages_dir.glob("*.json")):
            works.append(_parse_work(path))
    by_id: dict[str, tuple[Work, PageAnn]] = {}
    for work in works:
        for page in work.pages:
            if page.id in by_id:
                raise TateError(f"duplicate page id {page.id}")
            by_id[page.id] = (work, page)
    known = {source.id for source in sources}
    loaded: list[LoadedPage] = []
    for work in works:
        for page in work.pages:
            geometry = _resolve(page, by_id, ())
            _check_image_ref(page, known, by_id)
            loaded.append(LoadedPage(page, geometry, work.id, work.direction))
    return Dataset(root, tuple(sources), tuple(works), tuple(loaded))


def select_pages(dataset: Dataset, subset: str) -> tuple[LoadedPage, ...]:
    if subset == "all":
        return dataset.pages
    if subset == "ci":
        return tuple(page for page in dataset.pages if page.page.ci)
    raise TateError(f"unknown subset: {subset}")


def ancestor_ids(dataset: Dataset, pages: Sequence[LoadedPage]) -> list[str]:
    """Page ids required to materialise ``pages``, including derive parents."""
    by_id = {page.page.id: page.page for page in dataset.pages}
    needed: list[str] = []
    seen: set[str] = set()
    visiting: set[str] = set()

    def walk(page_id: str) -> None:
        if page_id in seen:
            return
        if page_id in visiting:
            raise TateError(f"derive cycle at {page_id}")
        page = by_id.get(page_id)
        if page is None:
            raise TateError(f"unknown page {page_id}")
        visiting.add(page_id)
        if isinstance(page.image, DeriveImage):
            walk(page.image.from_page)
        visiting.remove(page_id)
        seen.add(page_id)
        needed.append(page_id)

    for loaded in pages:
        walk(loaded.page.id)
    return needed


def dump_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sources_to_json(sources: Sequence[Source]) -> list[dict[str, object]]:
    return [_source_json(source) for source in sources]


def work_to_json(work: Work) -> dict[str, object]:
    return {
        "work": work.id,
        "title": work.title,
        "direction": work.direction,
        "pages": [_page_json(page) for page in work.pages],
    }


def write_sources(path: Path, sources: Sequence[Source]) -> None:
    dump_json(path, sources_to_json(sources))


def write_work(work: Work) -> None:
    dump_json(work.path, work_to_json(work))


def _resolve(page: PageAnn, by_id: Mapping[str, tuple[Work, PageAnn]], stack: tuple[str, ...]) -> Geometry:
    if page.id in stack:
        raise TateError(f"derive cycle at {page.id}")
    if len(stack) > _MAX_CHAIN:
        raise TateError(f"derive chain is too long at {page.id}")
    if isinstance(page.image, SourceImage):
        return Geometry(page.size, page.panels, page.balloons, page.ignore, page.whole)
    parent = by_id.get(page.image.from_page)
    if parent is None:
        raise TateError(f"page {page.id} derives from unknown page {page.image.from_page}")
    parent_geometry = _resolve(parent[1], by_id, (*stack, page.id))
    transformed = apply_geometry(parent_geometry, page.image.ops)
    if transformed.size != page.size:
        raise TateError(
            f"page {page.id} declares size {page.size[0]}x{page.size[1]}, "
            f"derive produces {transformed.size[0]}x{transformed.size[1]}"
        )
    if page.whole != parent_geometry.whole:
        raise TateError(f"page {page.id} whole does not match its source page")
    if page.panels_written:
        _same_panels(page.id, page.panels, transformed.panels)
    if page.balloons_written:
        _same_boxes(page.id, "balloons", page.balloons, transformed.balloons)
    if page.ignore_written:
        _same_boxes(page.id, "ignore", page.ignore, transformed.ignore)
    return Geometry(page.size, transformed.panels, transformed.balloons, transformed.ignore, page.whole)


def _check_image_ref(page: PageAnn, sources: set[str], by_id: Mapping[str, tuple[Work, PageAnn]]) -> None:
    image = page.image
    if isinstance(image, DeriveImage):
        if image.from_page not in by_id:
            raise TateError(f"page {page.id} derives from unknown page {image.from_page}")
        return
    if image.source not in sources:
        raise TateError(f"page {page.id} uses unknown source {image.source}")


def _same_panels(page_id: str, written: Sequence[PanelAnn], transformed: Sequence[PanelAnn]) -> None:
    left = [(panel.box.x, panel.box.y, panel.box.w, panel.box.h) for panel in written]
    right = [(panel.box.x, panel.box.y, panel.box.w, panel.box.h) for panel in transformed]
    if left != right:
        raise TateError(f"page {page_id} panels do not match the transformed ground truth")


def _same_boxes(page_id: str, label: str, written: Sequence[Box], transformed: Sequence[Box]) -> None:
    left = [(box.x, box.y, box.w, box.h) for box in written]
    right = [(box.x, box.y, box.w, box.h) for box in transformed]
    if left != right:
        raise TateError(f"page {page_id} {label} do not match the transformed ground truth")


def _parse_sources(value: object, label: str) -> list[Source]:
    rows = _expect_list(value, label)
    sources: list[Source] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        item = f"{label}[{index}]"
        data = _expect_object(row, item)
        _reject_unknown(data, _SOURCE_KEYS, item)
        source_id = _expect_id(_require(data, "id", item), f"{item}.id")
        if source_id in seen:
            raise TateError(f"duplicate source id {source_id}")
        seen.add(source_id)
        fetch = _parse_fetch(_require(data, "fetch", item), f"{item}.fetch")
        sources.append(
            Source(
                id=source_id,
                title=_expect_text(_require(data, "title", item), f"{item}.title"),
                author=_expect_text(_require(data, "author", item), f"{item}.author"),
                license=_expect_text(_require(data, "license", item), f"{item}.license"),
                license_url=_expect_text(_require(data, "license_url", item), f"{item}.license_url"),
                attribution=_expect_text(_require(data, "attribution", item), f"{item}.attribution"),
                homepage=_expect_text(_require(data, "homepage", item), f"{item}.homepage"),
                fetch=fetch,
            )
        )
    return sources


def _parse_fetch(value: object, label: str) -> ImageFetch | ZipFetch:
    data = _expect_object(value, label)
    _reject_unknown(data, _FETCH_KEYS, label)
    kind = _require(data, "kind", label)
    if kind not in {"image", "zip"}:
        raise TateError(f"{label}.kind must be image or zip")
    url = _expect_text(_require(data, "url", label), f"{label}.url")
    sha = _optional_sha(data.get("sha256"), f"{label}.sha256")
    if kind == "image":
        return ImageFetch(url, sha)
    return ZipFetch(url, sha)


def _parse_work(path: Path) -> Work:
    label = f"pages/{path.name}"
    data = _expect_object(_load_json(path, label), label)
    _reject_unknown(data, _WORK_KEYS, label)
    work_id = _expect_id(_require(data, "work", label), f"{label}.work")
    if path.stem != work_id:
        raise TateError(f"{label} work id {work_id} does not match the file name")
    direction = _require(data, "direction", label)
    if direction not in {"rtl", "ltr"}:
        raise TateError(f"{label}.direction must be rtl or ltr")
    rows = _expect_list(_require(data, "pages", label), f"{label}.pages")
    pages: list[PageAnn] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        page = _parse_page(row, f"{label} pages[{index}]")
        if page.id in seen:
            raise TateError(f"duplicate page id {page.id}")
        seen.add(page.id)
        pages.append(page)
    return Work(
        id=work_id,
        title=_expect_text(_require(data, "title", label), f"{label}.title"),
        direction=str(direction),
        path=path,
        pages=tuple(pages),
    )


def _parse_page(value: object, label: str) -> PageAnn:
    data = _expect_object(value, label)
    _reject_unknown(data, _PAGE_KEYS, label)
    page_id = _expect_id(_require(data, "id", label), f"{label}.id")
    styles = _parse_styles(_require(data, "styles", label), f"{label}.styles")
    width, height = _parse_size(_require(data, "size", label), f"{label}.size")
    image = _parse_image(_require(data, "image", label), f"{label}.image")
    panels_written = "panels" in data
    panels = _parse_panels(data.get("panels", []), f"{label}.panels", width, height) if panels_written else ()
    if not panels_written and isinstance(image, SourceImage):
        raise TateError(f"{label} is missing panels")
    balloons_written = "balloons" in data
    balloons = (
        _parse_box_list(data.get("balloons", []), f"{label}.balloons", width, height)
        if balloons_written
        else ()
    )
    if not balloons_written and isinstance(image, SourceImage):
        raise TateError(f"{label} is missing balloons")
    ignore_written = "ignore" in data
    ignore = _parse_box_list(data["ignore"], f"{label}.ignore", width, height) if ignore_written else ()
    whole = _expect_bool(_require(data, "whole", label), f"{label}.whole")
    ci = _expect_bool(_require(data, "ci", label), f"{label}.ci")
    notes = _optional_notes(data.get("notes"), f"{label}.notes") if "notes" in data else None
    pixel = (
        _optional_sha(data.get("pixel_sha256"), f"{label}.pixel_sha256") if "pixel_sha256" in data else None
    )
    return PageAnn(
        id=page_id,
        styles=styles,
        image=image,
        size=(width, height),
        pixel_sha256=pixel,
        panels=panels,
        panels_written=panels_written or isinstance(image, SourceImage),
        whole=whole,
        balloons=balloons,
        balloons_written=balloons_written or isinstance(image, SourceImage),
        ignore=ignore,
        ignore_written=ignore_written,
        ci=ci,
        notes=notes,
    )


def _parse_image(value: object, label: str) -> SourceImage | DeriveImage:
    data = _expect_object(value, label)
    if "derive" in data and "source" in data:
        raise TateError(f"{label} must be a source or a derive, not both")
    if "derive" in data:
        _reject_unknown(data, {"derive"}, label)
        body = _expect_object(data["derive"], f"{label}.derive")
        _reject_unknown(body, {"from", "ops"}, f"{label}.derive")
        origin = _expect_id(_require(body, "from", f"{label}.derive"), f"{label}.derive.from")
        ops = _parse_ops(_require(body, "ops", f"{label}.derive"), f"{label}.derive.ops")
        return DeriveImage(origin, tuple(ops))
    if "source" not in data:
        raise TateError(f"{label} must contain source or derive")
    _reject_unknown(data, {"source", "member", "pdf_page"}, label)
    source = _expect_id(_require(data, "source", label), f"{label}.source")
    member = _parse_member(data["member"], f"{label}.member") if "member" in data else None
    pdf_page = _expect_int(data["pdf_page"], f"{label}.pdf_page") if "pdf_page" in data else None
    if pdf_page is not None and pdf_page < 0:
        raise TateError(f"{label}.pdf_page must be 0 or greater")
    return SourceImage(source, member, pdf_page)


def _parse_ops(value: object, label: str) -> list[Op]:
    rows = _expect_list(value, label)
    if not rows:
        raise TateError(f"{label} must list at least one transform")
    return [_parse_op(row, f"{label}[{index}]") for index, row in enumerate(rows)]


def _parse_op(value: object, label: str) -> Op:
    data = _expect_object(value, label)
    keys = set(data)
    if keys == {"rotate"}:
        return Rotate(_expect_finite(data["rotate"], f"{label}.rotate"))
    if keys == {"contrast"}:
        factor = _expect_finite(data["contrast"], f"{label}.contrast")
        if factor < 0 or factor > 4:
            raise TateError(f"{label}.contrast must be between 0 and 4")
        return Contrast(factor)
    if keys == {"gamma"}:
        gamma = _expect_finite(data["gamma"], f"{label}.gamma")
        if gamma <= 0 or gamma > 8:
            raise TateError(f"{label}.gamma must be greater than 0 and at most 8")
        return Gamma(gamma)
    if keys == {"noise", "seed"}:
        sigma = _expect_finite(data["noise"], f"{label}.noise")
        if sigma < 0 or sigma > 64:
            raise TateError(f"{label}.noise must be between 0 and 64")
        seed = _expect_int(data["seed"], f"{label}.seed")
        return Noise(sigma, seed)
    if keys == {"jpeg"}:
        quality = _expect_int(data["jpeg"], f"{label}.jpeg")
        if quality < 1 or quality > 95:
            raise TateError(f"{label}.jpeg must be between 1 and 95")
        return Jpeg(quality)
    if keys == {"scale"}:
        factor = _expect_finite(data["scale"], f"{label}.scale")
        if factor <= 0 or factor > 4:
            raise TateError(f"{label}.scale must be greater than 0 and at most 4")
        return Scale(factor)
    if keys == {"yellow"}:
        amount = _expect_finite(data["yellow"], f"{label}.yellow")
        if amount < 0 or amount > 1:
            raise TateError(f"{label}.yellow must be between 0 and 1")
        return Yellow(amount)
    raise TateError(f"{label} is not a known transform")


def _parse_styles(value: object, label: str) -> tuple[str, ...]:
    rows = _expect_list(value, label)
    styles: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, str) or row not in STYLES:
            raise TateError(f"{label}[{index}] is not a known style")
        if row in styles:
            raise TateError(f"{label} repeats {row}")
        styles.append(row)
    return tuple(styles)


def _parse_size(value: object, label: str) -> tuple[int, int]:
    rows = _expect_list(value, label)
    if len(rows) != 2:
        raise TateError(f"{label} must be [width, height]")
    width = _expect_int(rows[0], f"{label}[0]")
    height = _expect_int(rows[1], f"{label}[1]")
    if width < 1 or height < 1:
        raise TateError(f"{label} must be positive")
    if width * height > _MAX_PIXELS:
        raise TateError(f"{label} is too large")
    return width, height


def _parse_panels(value: object, label: str, width: int, height: int) -> tuple[PanelAnn, ...]:
    rows = _expect_list(value, label)
    panels: list[PanelAnn] = []
    for index, row in enumerate(rows):
        item = f"{label}[{index}]"
        data = _expect_object(row, item)
        _reject_unknown(data, _PANEL_KEYS, item)
        box = _parse_xyxy(_require(data, "box", item), f"{item}.box", width, height)
        poly = _parse_poly(data["poly"], f"{item}.poly", box) if "poly" in data else None
        panels.append(PanelAnn(box, poly))
    return tuple(panels)


def _parse_box_list(value: object, label: str, width: int, height: int) -> tuple[Box, ...]:
    rows = _expect_list(value, label)
    boxes: list[Box] = []
    for index, row in enumerate(rows):
        item = f"{label}[{index}]"
        data = _expect_object(row, item)
        _reject_unknown(data, _BALLOON_KEYS, item)
        boxes.append(_parse_xyxy(_require(data, "box", item), f"{item}.box", width, height))
    return tuple(boxes)


def _parse_poly(value: object, label: str, box: Box) -> tuple[tuple[float, float], ...]:
    rows = _expect_list(value, label)
    if len(rows) < 3:
        raise TateError(f"{label} needs at least 3 points")
    points: list[tuple[float, float]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != 2:
            raise TateError(f"{label}[{index}] must be [x, y]")
        points.append(
            (
                _expect_finite(row[0], f"{label}[{index}][0]"),
                _expect_finite(row[1], f"{label}[{index}][1]"),
            )
        )
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    bounds = (math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys)))
    if bounds != (box.x, box.y, box.x2, box.y2):
        raise TateError(f"{label} bounding box does not match box")
    return tuple(points)


def _parse_xyxy(value: object, label: str, width: int, height: int) -> Box:
    rows = _expect_list(value, label)
    if len(rows) != 4:
        raise TateError(f"{label} must be [x1, y1, x2, y2]")
    x1 = _expect_int(rows[0], f"{label}[0]")
    y1 = _expect_int(rows[1], f"{label}[1]")
    x2 = _expect_int(rows[2], f"{label}[2]")
    y2 = _expect_int(rows[3], f"{label}[3]")
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        raise TateError(f"{label} is outside the page or empty")
    return Box(x1, y1, x2 - x1, y2 - y1)


def _parse_member(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value.startswith("/") or "\\" in value:
        raise TateError(f"{label} must be a relative path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise TateError(f"{label} must be a relative path")
    return value


def _source_json(source: Source) -> dict[str, object]:
    fetch: dict[str, object] = {
        "kind": "image" if isinstance(source.fetch, ImageFetch) else "zip",
        "url": source.fetch.url,
    }
    if source.fetch.sha256 is not None:
        fetch["sha256"] = source.fetch.sha256
    return {
        "id": source.id,
        "title": source.title,
        "author": source.author,
        "license": source.license,
        "license_url": source.license_url,
        "attribution": source.attribution,
        "homepage": source.homepage,
        "fetch": fetch,
    }


def _page_json(page: PageAnn) -> dict[str, object]:
    body: dict[str, object] = {
        "id": page.id,
        "styles": list(page.styles),
        "image": image_to_json(page.image),
        "size": [page.size[0], page.size[1]],
    }
    if page.pixel_sha256 is not None:
        body["pixel_sha256"] = page.pixel_sha256
    if page.panels_written or not isinstance(page.image, DeriveImage):
        body["panels"] = [_panel_json(panel) for panel in page.panels]
    body["whole"] = page.whole
    if page.balloons_written or not isinstance(page.image, DeriveImage):
        body["balloons"] = [{"box": _xyxy(box)} for box in page.balloons]
    if page.ignore_written:
        body["ignore"] = [{"box": _xyxy(box)} for box in page.ignore]
    body["ci"] = page.ci
    if page.notes is not None:
        body["notes"] = page.notes
    return body


def image_to_json(image: SourceImage | DeriveImage) -> dict[str, object]:
    if isinstance(image, DeriveImage):
        return {"derive": {"from": image.from_page, "ops": [_op_json(op) for op in image.ops]}}
    body: dict[str, object] = {"source": image.source}
    if image.member is not None:
        body["member"] = image.member
    if image.pdf_page is not None:
        body["pdf_page"] = image.pdf_page
    return body


def _panel_json(panel: PanelAnn) -> dict[str, object]:
    body: dict[str, object] = {"box": _xyxy(panel.box)}
    if panel.poly is not None:
        body["poly"] = [[_json_number(x), _json_number(y)] for x, y in panel.poly]
    return body


def _op_json(op: Op) -> dict[str, object]:
    if isinstance(op, Rotate):
        return {"rotate": _json_number(op.degrees)}
    if isinstance(op, Contrast):
        return {"contrast": _json_number(op.factor)}
    if isinstance(op, Gamma):
        return {"gamma": _json_number(op.gamma)}
    if isinstance(op, Noise):
        return {"noise": _json_number(op.sigma), "seed": op.seed}
    if isinstance(op, Jpeg):
        return {"jpeg": op.quality}
    if isinstance(op, Scale):
        return {"scale": _json_number(op.factor)}
    return {"yellow": _json_number(op.amount)}


def _xyxy(box: Box) -> list[int]:
    return [box.x, box.y, box.x2, box.y2]


def _json_number(value: float) -> int | float:
    if math.isfinite(value) and value.is_integer() and abs(value) < 2**53:
        return int(value)
    return value


def _load_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TateError(f"missing {label}") from exc
    except json.JSONDecodeError as exc:
        raise TateError(f"invalid JSON in {label}: {exc.msg}") from exc
    except OSError as exc:
        raise TateError(f"could not read {label}") from exc


def _expect_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TateError(f"{label} must be an object")
    data: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TateError(f"{label} has a non-string key")
        data[key] = item
    return data


def _expect_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TateError(f"{label} must be a list")
    return list(value)


def _require(data: Mapping[str, object], key: str, label: str) -> object:
    if key not in data:
        raise TateError(f"{label} is missing {key}")
    return data[key]


def _reject_unknown(data: Mapping[str, object], allowed: set[str], label: str) -> None:
    extra = sorted(set(data) - allowed)
    if extra:
        raise TateError(f"{label} has unknown fields: {', '.join(extra)}")


def _expect_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TateError(f"{label} must be a non-empty string")
    return value


def _expect_id(value: object, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise TateError(f"{label} must be a short identifier")
    return value


def _expect_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TateError(f"{label} must be a boolean")
    return value


def _expect_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TateError(f"{label} must be an integer")
    return value


def _expect_finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TateError(f"{label} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise TateError(f"{label} must be finite")
    return number


def _optional_sha(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise TateError(f"{label} must be a sha256 hex string")
    return value.lower()


def _optional_notes(value: object, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TateError(f"{label} must be a string")
    if len(value) > _MAX_NOTES:
        raise TateError(f"{label} is too long")
    return value
