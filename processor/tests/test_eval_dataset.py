import json
from pathlib import Path

import pytest

from tate.errors import TateError
from tate.eval.dataset import load_dataset


def _source(source_id: str = "scan") -> dict[str, object]:
    return {
        "id": source_id,
        "title": "Sample",
        "author": "A",
        "license": "MIT",
        "license_url": "https://example.com/license",
        "attribution": "Sample by A",
        "homepage": "https://example.com",
        "fetch": {"kind": "image", "url": "https://example.com/page.png", "sha256": "ab" * 32},
    }


def _page(page_id: str = "p1", **extra: object) -> dict[str, object]:
    body: dict[str, object] = {
        "id": page_id,
        "styles": ["standard"],
        "image": {"source": "scan"},
        "size": [100, 80],
        "panels": [{"box": [10, 20, 40, 50]}],
        "whole": False,
        "balloons": [{"box": [12, 22, 20, 28]}],
        "ci": True,
    }
    body.update(extra)
    return body


def _write(
    root: Path, pages: list[dict[str, object]], sources: list[dict[str, object]] | None = None
) -> None:
    root.mkdir(exist_ok=True)
    (root / "sources.json").write_text(json.dumps(sources if sources is not None else [_source()]) + "\n")
    (root / "pages").mkdir(exist_ok=True)
    (root / "pages" / "demo.json").write_text(
        json.dumps({"work": "demo", "title": "Demo", "direction": "rtl", "pages": pages}) + "\n"
    )


def test_a_valid_page_file_loads(tmp_path: Path) -> None:
    page = _page(
        notes="gutter",
        ignore=[{"box": [0, 70, 10, 80]}],
        pixel_sha256="cd" * 32,
    )
    page["panels"] = [
        {"box": [10, 20, 40, 50], "poly": [[10, 20], [40, 20], [40, 50], [10, 50]]},
    ]
    _write(tmp_path, [page])
    dataset = load_dataset(tmp_path)
    loaded = dataset.pages[0]
    assert loaded.work_id == "demo"
    assert loaded.direction == "rtl"
    assert loaded.geometry.panels[0].box.w == 30
    assert loaded.geometry.ignore[0].h == 10
    assert loaded.page.pixel_sha256 == "cd" * 32


def test_unknown_style_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, [_page(styles=["nope"])])
    with pytest.raises(TateError, match="style"):
        load_dataset(tmp_path)


def test_poly_must_match_its_box(tmp_path: Path) -> None:
    page = _page()
    page["panels"] = [{"box": [10, 20, 40, 50], "poly": [[10, 20], [12, 20], [12, 22]]}]
    _write(tmp_path, [page])
    with pytest.raises(TateError, match="bounding box"):
        load_dataset(tmp_path)


def test_box_outside_the_page_is_rejected(tmp_path: Path) -> None:
    _write(tmp_path, [_page(panels=[{"box": [90, 70, 120, 90]}])])
    with pytest.raises(TateError, match="outside"):
        load_dataset(tmp_path)


def test_bad_direction_duplicate_id_and_unknown_field(tmp_path: Path) -> None:
    _write(tmp_path, [_page(), _page()])
    with pytest.raises(TateError, match="duplicate"):
        load_dataset(tmp_path)

    _write_raw = tmp_path / "pages" / "demo.json"
    _write_raw.write_text(
        json.dumps(
            {"work": "demo", "title": "Demo", "direction": "down", "pages": [_page()]},
        )
    )
    with pytest.raises(TateError, match="direction"):
        load_dataset(tmp_path)

    _write_raw.write_text(
        json.dumps(
            {"work": "demo", "title": "Demo", "direction": "ltr", "pages": [_page()], "extra": 1},
        )
    )
    with pytest.raises(TateError, match="unknown fields"):
        load_dataset(tmp_path)


def test_derived_page_uses_transformed_ground_truth(tmp_path: Path) -> None:
    derived = {
        "id": "p2",
        "styles": ["low-contrast"],
        "image": {"derive": {"from": "p1", "ops": [{"rotate": 90}]}},
        "size": [80, 100],
        "whole": False,
        "ci": False,
    }
    _write(tmp_path, [_page(), derived])
    loaded = {page.page.id: page for page in load_dataset(tmp_path).pages}
    box = loaded["p2"].geometry.panels[0].box
    assert (box.x, box.y, box.w, box.h) == (20, 60, 30, 30)
    balloon = loaded["p2"].geometry.balloons[0]
    assert balloon.w > 0 and balloon.h > 0


def test_derived_page_rejects_a_mismatched_box(tmp_path: Path) -> None:
    derived = {
        "id": "p2",
        "styles": ["standard"],
        "image": {"derive": {"from": "p1", "ops": [{"rotate": 90}]}},
        "size": [80, 100],
        "panels": [{"box": [0, 0, 10, 10]}],
        "whole": False,
        "balloons": [],
        "ci": False,
    }
    _write(tmp_path, [_page(), derived])
    with pytest.raises(TateError, match="transformed"):
        load_dataset(tmp_path)


def test_missing_source_and_a_derive_cycle(tmp_path: Path) -> None:
    _write(tmp_path, [_page()], sources=[])
    with pytest.raises(TateError, match="unknown source"):
        load_dataset(tmp_path)

    first = _page("p1", image={"derive": {"from": "p2", "ops": [{"yellow": 0.1}]}}, size=[100, 80])
    second = {
        "id": "p2",
        "styles": ["standard"],
        "image": {"derive": {"from": "p1", "ops": [{"yellow": 0.1}]}},
        "size": [100, 80],
        "whole": False,
        "ci": False,
    }
    _write(tmp_path, [first, second], sources=[_source()])
    with pytest.raises(TateError, match="cycle"):
        load_dataset(tmp_path)
