import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from tate.boxes import Box, Detection
from tate.cli import main
from tate.errors import TateError
from tate.eval.check import judge, load_thresholds
from tate.eval.diff import format_diff, load_metrics
from tate.eval.metrics import PageScore, score_page
from tate.eval.report import metrics_document, write_report
from tate.layout import PageLayout, SegmentDraft


def _segment(box: Box) -> SegmentDraft:
    return SegmentDraft(Image.new("RGB", (2, 2), "white"), 0, box, False, box)


def _page(page_id: str, *, hit: bool, styles: tuple[str, ...] = ("standard",)) -> PageScore:
    image = Image.new("RGB", (80, 60), "white")
    box = Box(10, 10, 30, 20)
    segments = [_segment(box)] if hit else []
    gt = (box, Box(10, 34, 30, 20)) if hit else (box,)
    if hit:
        segments.append(_segment(gt[1]))
    balloons = (Box(12, 12, 8, 8),) if hit else ()
    return score_page(
        image,
        page_id=page_id,
        work="demo",
        styles=styles,
        whole=False,
        gt_panels=gt,
        balloons=balloons,
        ignore=(),
        layout=PageLayout(segments, [], [], False, False),
        elapsed=0.2,
    )


def _thresholds(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "detector": "cv",
                "overall": {
                    "panel_f1_50": 0.95,
                    "order_exact": 0.95,
                    "balloon_cut_rate_max": 0.02,
                    "content_loss_pages_max": 0,
                    "fallback_rate_max": 0.10,
                },
                "per_style": {"panel_f1_50": 0.90},
                "min_pages_per_style": 3,
            }
        )
        + "\n"
    )


def test_thresholds_pass_and_fail(tmp_path: Path) -> None:
    path = tmp_path / "thresholds.json"
    _thresholds(path)
    thresholds = load_thresholds(path)
    good = [_page("a", hit=True), _page("b", hit=True), _page("c", hit=True)]
    assert judge(thresholds, {"cv": good}) == []

    bad = [_page(f"p{index}", hit=False, styles=("dense",)) for index in range(3)]
    lines = judge(thresholds, {"cv": bad})
    assert any("panel_f1_50" in line and "short by" in line for line in lines)
    assert any("style dense" in line for line in lines)
    assert judge(thresholds, {"yolo": good})[0].startswith("thresholds ask for cv")

    sparse = [_page("only", hit=False, styles=("bleed",))]
    sparse_lines = judge(thresholds, {"cv": sparse})
    assert not any("style bleed" in line for line in sparse_lines)


def test_report_writes_overlays_only_for_failures(tmp_path: Path) -> None:
    passing = _page("ok", hit=True)
    failing = _page("bad", hit=False)
    images = {}
    for page in (passing, failing):
        path = tmp_path / f"{page.page_id}.png"
        Image.new("RGB", (80, 60), "white").save(path)
        images[page.page_id] = path
    out = tmp_path / "report"
    write_report(
        out,
        dataset_name="eval",
        subset="ci",
        scores={"cv": [passing, failing]},
        images=images,
        threshold_lines=["cv overall panel_f1_50 is 0.5000, threshold is 0.9500 (short by 0.4500)"],
        checked=True,
        all_overlays=False,
    )
    assert (out / "report.md").is_file()
    assert (out / "report.html").is_file()
    assert (out / "metrics.json").is_file()
    assert (out / "overlays" / "cv" / "bad.jpg").is_file()
    assert not (out / "overlays" / "cv" / "ok.jpg").exists()
    with Image.open(out / "overlays" / "cv" / "bad.jpg") as overlay:
        assert max(overlay.size) <= 1600
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "overlays/cv/bad.jpg" in html
    assert "<style>" in html
    assert "ok.jpg" not in html

    wide = tmp_path / "wide.png"
    Image.new("RGB", (1800, 40), "white").save(wide)
    images["bad"] = wide
    failing_wide = score_page(
        Image.new("RGB", (1800, 40), "white"),
        page_id="bad",
        work="demo",
        styles=("standard",),
        whole=False,
        gt_panels=(Box(10, 4, 20, 20),),
        balloons=(),
        ignore=(),
        layout=PageLayout([], [], [], False, False),
        elapsed=0.1,
    )
    write_report(
        tmp_path / "wide-report",
        dataset_name="eval",
        subset="all",
        scores={"cv": [failing_wide]},
        images={"bad": wide},
        threshold_lines=[],
        checked=False,
        all_overlays=False,
    )
    with Image.open(tmp_path / "wide-report" / "overlays" / "cv" / "bad.jpg") as overlay:
        assert max(overlay.size) <= 1600

    write_report(
        tmp_path / "all",
        dataset_name="eval",
        subset="all",
        scores={"cv": [passing]},
        images=images,
        threshold_lines=[],
        checked=True,
        all_overlays=True,
    )
    assert (tmp_path / "all" / "overlays" / "cv" / "ok.jpg").is_file()


def test_metrics_diff_prints_a_before_after_table(tmp_path: Path) -> None:
    before = metrics_document("eval", "all", {"cv": [_page("a", hit=False)]})
    after = metrics_document("eval", "all", {"cv": [_page("a", hit=True)]})
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    before_path.write_text(json.dumps(before) + "\n")
    after_path.write_text(json.dumps(after) + "\n")
    text = format_diff(load_metrics(before_path), load_metrics(after_path))
    assert "before" in text and "after" in text and "delta" in text
    assert "panel_f1_50" in text
    assert "+" in text


def test_cv_scores_in_a_process_pool(tmp_path: Path) -> None:
    from tate.eval.dataset import load_dataset
    from tate.eval.run import evaluate

    dataset = tmp_path / "dataset"
    cache = tmp_path / "cache"
    (dataset / "pages").mkdir(parents=True)
    (cache / "pages").mkdir(parents=True)
    pages = []
    for name in ("one", "two"):
        Image.new("RGB", (32, 48), "white").save(cache / "pages" / f"{name}.png")
        pages.append(
            {
                "id": name,
                "styles": ["standard"],
                "image": {"source": "local"},
                "size": [32, 48],
                "panels": [{"box": [2, 2, 10, 10]}],
                "whole": False,
                "balloons": [],
                "ci": True,
            }
        )
    (dataset / "sources.json").write_text(
        json.dumps(
            [
                {
                    "id": "local",
                    "title": "Local",
                    "author": "A",
                    "license": "MIT",
                    "license_url": "https://example.com/license",
                    "attribution": "Local by A",
                    "homepage": "https://example.com",
                    "fetch": {"kind": "image", "url": "https://example.com/a.png", "sha256": "ab" * 32},
                }
            ]
        )
        + "\n"
    )
    (dataset / "pages" / "demo.json").write_text(
        json.dumps({"work": "demo", "title": "Demo", "direction": "rtl", "pages": pages}) + "\n"
    )
    scores = evaluate(load_dataset(dataset), cache=cache, detectors=["cv"], subset="ci", jobs=2)
    assert [page.page_id for page in scores["cv"]] == ["one", "two"]


def test_ltr_work_is_scored_in_ltr_order(tmp_path: Path) -> None:
    from tate.eval.dataset import load_dataset
    from tate.eval.run import evaluate

    class _Panels:
        name = "cv"

        def detect(self, image: Image.Image) -> Detection:
            return Detection(
                (
                    Box(20, 20, 160, 180),
                    Box(220, 20, 160, 180),
                    Box(20, 260, 160, 180),
                    Box(220, 260, 160, 180),
                ),
                (),
            )

    dataset = tmp_path / "dataset"
    cache = tmp_path / "cache" / "pages"
    (dataset / "pages").mkdir(parents=True)
    cache.mkdir(parents=True)
    image = Image.new("RGB", (400, 480), "white")
    pen = ImageDraw.Draw(image)
    for box in ((20, 20, 180, 200), (220, 20, 380, 200), (20, 260, 180, 440), (220, 260, 380, 440)):
        pen.rectangle(box, fill=(40, 40, 40))
    image.save(cache / "west.png")
    image.close()
    (dataset / "sources.json").write_text(
        json.dumps(
            [
                {
                    "id": "local",
                    "title": "Local",
                    "author": "A",
                    "license": "MIT",
                    "license_url": "https://example.com/license",
                    "attribution": "Local by A",
                    "homepage": "https://example.com",
                    "fetch": {"kind": "image", "url": "https://example.com/a.png", "sha256": "ab" * 32},
                }
            ]
        )
        + "\n"
    )
    (dataset / "pages" / "west.json").write_text(
        json.dumps(
            {
                "work": "west",
                "title": "West",
                "direction": "ltr",
                "pages": [
                    {
                        "id": "west",
                        "styles": ["ltr"],
                        "image": {"source": "local"},
                        "size": [400, 480],
                        "panels": [
                            {"box": [20, 20, 180, 200]},
                            {"box": [220, 20, 380, 200]},
                            {"box": [20, 260, 180, 440]},
                            {"box": [220, 260, 380, 440]},
                        ],
                        "whole": False,
                        "balloons": [],
                        "ci": True,
                    }
                ],
            }
        )
        + "\n"
    )
    scores = evaluate(
        load_dataset(dataset),
        cache=tmp_path / "cache",
        detectors=["cv"],
        subset="ci",
        jobs=1,
        instances={"cv": _Panels()},
    )
    page = scores["cv"][0]
    assert page.page_id == "west"
    assert page.order_exact is True
    assert page.wrong_order is False
    assert page.kendall_tau == 1
    assert [box.x for box in page.pred_panels] == [20, 220, 20, 220]


def test_eval_cli_check_and_diff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    image = Image.new("RGB", (40, 30), "white")
    cache = tmp_path / "cache" / "pages"
    cache.mkdir(parents=True)
    image.save(cache / "only.png")
    dataset = tmp_path / "dataset"
    (dataset / "pages").mkdir(parents=True)
    (dataset / "sources.json").write_text("[]\n")
    # A page still needs a source. Point it at a file URL that is never fetched.
    raster = tmp_path / "only.png"
    image.save(raster)
    digest = __import__("hashlib").sha256(raster.read_bytes()).hexdigest()
    (dataset / "sources.json").write_text(
        json.dumps(
            [
                {
                    "id": "local",
                    "title": "Local",
                    "author": "A",
                    "license": "MIT",
                    "license_url": "https://example.com/license",
                    "attribution": "Local by A",
                    "homepage": "https://example.com",
                    "fetch": {"kind": "image", "url": raster.as_uri(), "sha256": digest},
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
                        "id": "only",
                        "styles": ["standard"],
                        "image": {"source": "local"},
                        "size": [40, 30],
                        "panels": [{"box": [4, 4, 20, 20]}, {"box": [4, 16, 20, 28]}],
                        "whole": False,
                        "balloons": [{"box": [6, 6, 10, 10]}],
                        "ci": True,
                    }
                ],
            }
        )
        + "\n"
    )
    thresholds = tmp_path / "thresholds.json"
    _thresholds(thresholds)
    code = main(
        [
            "eval",
            "run",
            "--dataset",
            str(dataset),
            "--cache",
            str(tmp_path / "cache"),
            "--detector",
            "cv",
            "--subset",
            "ci",
            "--jobs",
            "1",
            "--out",
            str(tmp_path / "out"),
            "--check",
            str(thresholds),
        ]
    )
    assert code == 1
    assert (tmp_path / "out" / "report.md").is_file()
    assert (tmp_path / "out" / "metrics.json").is_file()

    other = tmp_path / "other.json"
    document = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    other.write_text(json.dumps(document) + "\n")
    assert main(["eval", "diff", str(tmp_path / "out" / "metrics.json"), str(other)]) == 0
    with pytest.raises(TateError, match="not a tate eval"):
        load_metrics(thresholds)
