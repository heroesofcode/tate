"""Draw synthetic pages, pack a CBZ, and run both offline detectors.

Nothing here is from a published comic. The cv package is the one the reader
pipeline spec imports.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.pages import grid_page, save_jpeg  # noqa: E402

CV_NAME = "chapter-cv.tate"
PAGE_NAME = "chapter-page.tate"
CV_TITLE = "Pipeline Chapter"
PAGE_TITLE = "Pipeline Pages"
PAGE_COUNT = 2


def run_process(cbz: Path, output: Path, detector: str, title: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tate",
            "process",
            str(cbz),
            "-o",
            str(output),
            "--detector",
            detector,
            "--title",
            title,
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def read_manifest(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        payload: object = json.loads(archive.read("manifest.json"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path.name}: manifest is not an object")
    return payload


def segment_count(path: Path, manifest: dict[str, object], detector: str, title: str) -> int:
    if manifest.get("format") != "tate" or manifest.get("version") != 1:
        raise SystemExit(f"{path.name}: not a version-1 tate package")
    if manifest.get("detector") != detector:
        raise SystemExit(f"{path.name}: detector is {manifest.get('detector')}")
    if manifest.get("title") != title:
        raise SystemExit(f"{path.name}: title is {manifest.get('title')}")
    if manifest.get("pageCount") != PAGE_COUNT:
        raise SystemExit(f"{path.name}: pageCount is {manifest.get('pageCount')}")
    segments = manifest.get("segments")
    if not isinstance(segments, list) or not segments:
        raise SystemExit(f"{path.name}: no segments")
    if detector == "page" and len(segments) != PAGE_COUNT:
        raise SystemExit(f"{path.name}: expected one segment per page, got {len(segments)}")
    return len(segments)


def build(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = out_dir / "pages"
    pages.mkdir(exist_ok=True)
    names = [pages / "page1.jpg", pages / "page2.jpg"]
    for path in names:
        save_jpeg(path, grid_page())
    cbz = out_dir / "chapter.cbz"
    with zipfile.ZipFile(cbz, "w") as archive:
        for path in names:
            archive.write(path, path.name)

    cv_out = out_dir / CV_NAME
    page_out = out_dir / PAGE_NAME
    run_process(cbz, cv_out, "cv", CV_TITLE)
    run_process(cbz, page_out, "page", PAGE_TITLE)
    for path in (cv_out, page_out):
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"missing package: {path.name}")

    cv_count = segment_count(cv_out, read_manifest(cv_out), "cv", CV_TITLE)
    page_count = segment_count(page_out, read_manifest(page_out), "page", PAGE_TITLE)
    if cv_count <= page_count:
        raise SystemExit(f"cv produced {cv_count} segments; page detector produced {page_count}")
    print(cv_out)
    return cv_out


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: pipeline_chapter.py OUT_DIR", file=sys.stderr)
        return 2
    build(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
