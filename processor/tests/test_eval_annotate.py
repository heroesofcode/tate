import json
from pathlib import Path

from PIL import Image, ImageDraw

from tate.boxes import Box, Detection
from tate.eval.annotate import bootstrap, merge_overlapping, review, review_main
from tate.eval.fetch import page_png


class _Detector:
    @property
    def name(self) -> str:
        return "yolo"

    def detect(self, image: Image.Image) -> Detection:
        del image
        return Detection(
            (Box(20, 20, 160, 740), Box(220, 20, 160, 740)),
            (Box(30, 40, 40, 30), Box(40, 50, 40, 30)),
        )


def _page() -> Image.Image:
    image = Image.new("RGB", (400, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 180, 760), fill=(30, 30, 30))
    draw.rectangle((220, 20, 380, 760), fill=(30, 30, 30))
    return image


def test_merge_overlapping_text_boxes() -> None:
    merged = merge_overlapping((Box(30, 40, 40, 30), Box(40, 50, 40, 30), Box(200, 10, 10, 10)))
    assert len(merged) == 2
    assert any(box.x == 30 and box.x2 == 80 and box.y2 == 80 for box in merged)


def test_bootstrap_skips_existing_pages_until_forced(tmp_path: Path) -> None:
    folder = tmp_path / "scans"
    folder.mkdir()
    _page().save(folder / "a.png")
    dataset = tmp_path / "dataset"
    first = bootstrap(dataset, "demo", folder, detector=_Detector())
    assert (first.wrote, first.skipped) == (1, 0)
    path = dataset / "pages" / "demo.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    panels = document["pages"][0]["panels"]
    assert len(panels) == 2
    assert panels[0]["box"][0] > panels[1]["box"][0]
    assert len(document["pages"][0]["balloons"]) == 1
    document["pages"][0]["panels"] = [{"box": [1, 1, 20, 20]}]
    path.write_text(json.dumps(document) + "\n")

    second = bootstrap(dataset, "demo", folder, detector=_Detector())
    assert (second.wrote, second.skipped) == (0, 1)
    kept = json.loads(path.read_text(encoding="utf-8"))
    assert kept["pages"][0]["panels"] == [{"box": [1, 1, 20, 20]}]

    bootstrap(dataset, "demo", folder, detector=_Detector(), force=True)
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert len(restored["pages"][0]["panels"]) == 2


def test_review_writes_a_contact_sheet(tmp_path: Path) -> None:
    image = _page()
    dataset = tmp_path / "dataset"
    (dataset / "pages").mkdir(parents=True)
    (dataset / "sources.json").write_text(
        json.dumps(
            [
                {
                    "id": "demo",
                    "title": "Demo",
                    "author": "A",
                    "license": "MIT",
                    "license_url": "https://example.com/license",
                    "attribution": "Demo by A",
                    "homepage": "https://example.com",
                    "fetch": {
                        "kind": "image",
                        "url": "https://example.com/a.png",
                        "sha256": "ab" * 32,
                    },
                }
            ]
        )
        + "\n"
    )
    (dataset / "pages" / "demo.json").write_text(
        json.dumps(
            {
                "work": "demo",
                "title": "Demo",
                "direction": "rtl",
                "pages": [
                    {
                        "id": "demo-0001",
                        "styles": ["standard"],
                        "image": {"source": "demo"},
                        "size": [400, 800],
                        "panels": [{"box": [220, 20, 380, 760]}, {"box": [20, 20, 180, 760]}],
                        "whole": False,
                        "balloons": [{"box": [30, 40, 80, 80]}],
                        "ignore": [{"box": [0, 0, 10, 10]}],
                        "ci": False,
                    }
                ],
            }
        )
        + "\n"
    )
    cache = tmp_path / "cache"
    destination = page_png(cache, "demo-0001")
    destination.parent.mkdir(parents=True)
    image.save(destination)
    sheets = review(dataset, "demo", cache=cache, out=tmp_path / "review", page_ids=["demo-0001"])
    assert len(sheets) == 1
    assert sheets[0].name == "demo-01.jpg"
    with Image.open(sheets[0]) as sheet:
        assert sheet.size == (1440, 2000)

    code = review_main(
        ["--work", "demo", "--dataset", str(dataset), "--cache", str(cache), "--out", str(tmp_path / "again")]
    )
    assert code == 0
    assert (tmp_path / "again" / "demo-01.jpg").is_file()
