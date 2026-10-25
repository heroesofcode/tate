"""Dataset records. Boxes are half-open ``[x1, y1, x2, y2)`` in page pixels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tate.boxes import Box


@dataclass(frozen=True, slots=True)
class ImageFetch:
    """One downloaded file: a raster image or a PDF."""

    url: str
    sha256: str | None


@dataclass(frozen=True, slots=True)
class ZipFetch:
    """One downloaded archive. Pages name the member they need."""

    url: str
    sha256: str | None


@dataclass(frozen=True, slots=True)
class Source:
    id: str
    title: str
    author: str
    license: str
    license_url: str
    attribution: str
    homepage: str
    fetch: ImageFetch | ZipFetch


@dataclass(frozen=True, slots=True)
class Rotate:
    degrees: float


@dataclass(frozen=True, slots=True)
class Contrast:
    factor: float


@dataclass(frozen=True, slots=True)
class Gamma:
    gamma: float


@dataclass(frozen=True, slots=True)
class Noise:
    sigma: float
    seed: int


@dataclass(frozen=True, slots=True)
class Jpeg:
    quality: int


@dataclass(frozen=True, slots=True)
class Scale:
    factor: float


@dataclass(frozen=True, slots=True)
class Yellow:
    amount: float


Op = Rotate | Contrast | Gamma | Noise | Jpeg | Scale | Yellow


@dataclass(frozen=True, slots=True)
class SourceImage:
    source: str
    member: str | None = None
    pdf_page: int | None = None


@dataclass(frozen=True, slots=True)
class DeriveImage:
    from_page: str
    ops: tuple[Op, ...]


@dataclass(frozen=True, slots=True)
class PanelAnn:
    box: Box
    poly: tuple[tuple[float, float], ...] | None = None


@dataclass(frozen=True, slots=True)
class PageAnn:
    """One page as written in the work file, before derive resolution."""

    id: str
    styles: tuple[str, ...]
    image: SourceImage | DeriveImage
    size: tuple[int, int]
    pixel_sha256: str | None
    panels: tuple[PanelAnn, ...]
    panels_written: bool
    whole: bool
    balloons: tuple[Box, ...]
    balloons_written: bool
    ignore: tuple[Box, ...]
    ignore_written: bool
    ci: bool
    notes: str | None


@dataclass(frozen=True, slots=True)
class Geometry:
    """Ground truth in the final page image's pixel space."""

    size: tuple[int, int]
    panels: tuple[PanelAnn, ...]
    balloons: tuple[Box, ...]
    ignore: tuple[Box, ...]
    whole: bool


@dataclass(frozen=True, slots=True)
class Work:
    id: str
    title: str
    direction: str
    path: Path
    pages: tuple[PageAnn, ...]


@dataclass(frozen=True, slots=True)
class LoadedPage:
    page: PageAnn
    geometry: Geometry
    work_id: str
    direction: str


@dataclass(frozen=True, slots=True)
class Dataset:
    root: Path
    sources: tuple[Source, ...]
    works: tuple[Work, ...]
    pages: tuple[LoadedPage, ...]

    def source_map(self) -> dict[str, Source]:
        return {source.id: source for source in self.sources}

    def page_map(self) -> dict[str, LoadedPage]:
        return {loaded.page.id: loaded for loaded in self.pages}
