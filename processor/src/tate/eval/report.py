"""Write ``report.md``, ``report.html``, ``metrics.json``, and failure overlays."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tate.boxes import Box
from tate.eval.dataset import dump_json
from tate.eval.metrics import (
    LOSS_PAGE,
    MetricBlock,
    PageScore,
    aggregate,
    aggregate_styles,
    aggregate_works,
    ink_masks,
    xyxy,
)

_OVERLAY_LONG_SIDE = 1600
_JPEG_QUALITY = 85


def write_report(
    out: Path,
    *,
    dataset_name: str,
    subset: str,
    scores: Mapping[str, Sequence[PageScore]],
    images: Mapping[str, Path],
    threshold_lines: Sequence[str],
    checked: bool,
    all_overlays: bool,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    overlays = _write_overlays(out, scores, images, all_overlays)
    document = metrics_document(dataset_name, subset, scores)
    dump_json(out / "metrics.json", document)
    markdown = render_markdown(
        dataset_name,
        subset,
        scores,
        threshold_lines,
        checked,
        overlays,
    )
    (out / "report.md").write_text(markdown, encoding="utf-8")
    (out / "report.html").write_text(
        render_html(dataset_name, subset, scores, threshold_lines, checked, overlays),
        encoding="utf-8",
    )


def metrics_document(
    dataset_name: str,
    subset: str,
    scores: Mapping[str, Sequence[PageScore]],
) -> dict[str, object]:
    detectors: dict[str, object] = {}
    for name in sorted(scores):
        pages = list(scores[name])
        detectors[name] = {
            "overall": block_to_json(aggregate(pages)),
            "per_style": {style: block_to_json(block) for style, block in aggregate_styles(pages).items()},
            "per_work": {work: block_to_json(block) for work, block in aggregate_works(pages).items()},
            "pages": {
                page.page_id: page_to_json(page) for page in sorted(pages, key=lambda item: item.page_id)
            },
        }
    return {
        "version": 1,
        "dataset": dataset_name,
        "subset": subset,
        "width": 1080,
        "detectors": detectors,
    }


def block_to_json(block: MetricBlock) -> dict[str, object]:
    return {
        "pages": block.pages,
        "tp_50": block.panel_50.tp,
        "fp_50": block.panel_50.fp,
        "fn_50": block.panel_50.fn,
        "precision_50": block.panel_50.precision,
        "recall_50": block.panel_50.recall,
        "panel_f1_50": block.panel_f1_50,
        "mean_page_f1_50": block.mean_page_f1_50,
        "tp_75": block.panel_75.tp,
        "fp_75": block.panel_75.fp,
        "fn_75": block.panel_75.fn,
        "precision_75": block.panel_75.precision,
        "recall_75": block.panel_75.recall,
        "panel_f1_75": block.panel_f1_75,
        "mean_page_f1_75": block.mean_page_f1_75,
        "order_exact": block.order_exact,
        "order_pages": block.order_pages,
        "mean_kendall_tau": block.mean_kendall_tau,
        "tau_pages": block.tau_pages,
        "balloons": block.balloons,
        "balloons_cut": block.balloons_cut,
        "balloon_cut_rate": block.balloon_cut_rate,
        "content_loss_mean": block.content_loss_mean,
        "content_loss_max": block.content_loss_max,
        "content_loss_pages": block.content_loss_pages,
        "fallback_pages": block.fallback_pages,
        "fallback_rate": block.fallback_rate,
        "whole_pages": block.whole_pages,
        "wrong_split_pages": block.wrong_split_pages,
        "wrong_split_rate": block.wrong_split_rate,
        "time_mean_s": block.time_mean_s,
        "time_p95_s": block.time_p95_s,
    }


def page_to_json(page: PageScore) -> dict[str, object]:
    return {
        "work": page.work,
        "styles": list(page.styles),
        "whole": page.whole,
        "fallback": page.fallback,
        "spread": page.spread,
        "unexpected_fallback": page.unexpected_fallback,
        "wrong_split": page.wrong_split,
        "wrong_order": page.wrong_order,
        "reasons": list(page.reasons()),
        "elapsed_s": page.elapsed,
        "gt_panels": [xyxy(box) for box in page.gt_panels],
        "pred_panels": [xyxy(box) for box in page.pred_panels],
        "crops": [
            {"box": xyxy(crop.box), "continuation": crop.continuation, "order": crop.order}
            for crop in page.crops
        ],
        "balloons_total": page.balloons_total,
        "balloons_cut": page.balloons_cut,
        "cut_balloons": [xyxy(box) for box in page.cut_balloons],
        "content_loss": page.content_loss,
        "tp_50": page.counts_50.tp,
        "fp_50": page.counts_50.fp,
        "fn_50": page.counts_50.fn,
        "f1_50": page.f1_50,
        "tp_75": page.counts_75.tp,
        "fp_75": page.counts_75.fp,
        "fn_75": page.counts_75.fn,
        "f1_75": page.f1_75,
        "order_exact": page.order_exact,
        "kendall_tau": page.kendall_tau,
    }


def render_markdown(
    dataset_name: str,
    subset: str,
    scores: Mapping[str, Sequence[PageScore]],
    threshold_lines: Sequence[str],
    checked: bool,
    overlays: Mapping[tuple[str, str], str],
) -> str:
    page_count = max((len(pages) for pages in scores.values()), default=0)
    parts = [
        "# tate eval report",
        "",
        f"Dataset: `{dataset_name}`  ",
        f"Subset: `{subset}`  ",
        f"Pages: {page_count}",
        "",
        "Rates are fractions. Content loss is uncovered ink divided by ink. "
        f"A page counts as content loss above {LOSS_PAGE:.1%}.",
        "",
    ]
    for name in sorted(scores):
        pages = list(scores[name])
        parts.append(f"## {name}")
        parts.append("")
        parts.append(_metric_table(aggregate(pages)))
        parts.append("")
        parts.append(f"### {name} by style")
        parts.append("")
        parts.append(_group_table(aggregate_styles(pages)))
        parts.append("")
        parts.append(f"### {name} by work")
        parts.append("")
        parts.append(_group_table(aggregate_works(pages)))
        parts.append("")
    if len(scores) >= 2:
        parts.append("## Detector comparison")
        parts.append("")
        parts.append(_comparison_table(scores))
        parts.append("")
    parts.append("## Failing pages")
    parts.append("")
    failing = _failing(scores)
    if not failing:
        parts.append("None.")
    else:
        parts.append("| detector | page | work | reasons | F1 @0.50 | content loss |")
        parts.append("| --- | --- | --- | --- | ---: | ---: |")
        for name, page in failing:
            link = overlays.get((name, page.page_id))
            label = f"[{page.page_id}]({link})" if link else page.page_id
            reasons = ", ".join(page.reasons())
            parts.append(
                f"| {name} | {label} | {page.work} | {reasons} | {page.f1_50:.3f} | {page.content_loss:.4f} |"
            )
    parts.append("")
    parts.append("## Thresholds")
    parts.append("")
    if not checked:
        parts.append("Not checked.")
    elif not threshold_lines:
        parts.append("Passed.")
    else:
        for line in threshold_lines:
            parts.append(f"- {line}")
    parts.append("")
    return "\n".join(parts)


def render_html(
    dataset_name: str,
    subset: str,
    scores: Mapping[str, Sequence[PageScore]],
    threshold_lines: Sequence[str],
    checked: bool,
    overlays: Mapping[tuple[str, str], str],
) -> str:
    body = html.escape(render_markdown(dataset_name, subset, scores, threshold_lines, checked, overlays))
    thumbs = _thumbnails(scores, overlays)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>tate eval report</title>
<style>
body {{ font: 16px/1.45 sans-serif; margin: 2rem auto; max-width: 1080px; color: #1b1b1b; }}
pre {{ white-space: pre-wrap; font: 14px/1.4 ui-monospace, monospace; }}
.thumbs {{ display: flex; flex-wrap: wrap; gap: 12px; }}
.thumbs a {{ color: inherit; text-decoration: none; width: 180px; }}
.thumbs img {{ width: 180px; height: auto; border: 1px solid #ccc; }}
.thumbs span {{ display: block; font-size: 13px; margin-top: 4px; }}
</style>
</head>
<body>
<pre>{body}</pre>
<h2>Overlays</h2>
<div class="thumbs">
{thumbs}
</div>
</body>
</html>
"""


def draw_overlay(image: Image.Image, score: PageScore) -> Image.Image:
    """Ground truth in green, crops in red or orange, cut balloons in magenta, lost ink in blue."""
    _ink, _covered, lost = ink_masks(
        image,
        color="color" in score.styles,
        ignore=score.ignore,
        crops=tuple(crop.box for crop in score.crops),
    )
    pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    tint = np.array([40.0, 90.0, 255.0])
    pixels[lost] = pixels[lost] * 0.45 + tint * 0.55
    canvas = Image.fromarray(np.clip(np.rint(pixels), 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    line = max(2, min(canvas.size) // 400)
    font = ImageFont.load_default(size=max(16, min(canvas.size) // 40))
    for number, box in enumerate(score.gt_panels, start=1):
        _box(draw, box, (30, 150, 60), line)
        _label(draw, box, str(number), (30, 150, 60), font, line)
    for crop in score.crops:
        color = (210, 120, 20) if crop.continuation else (200, 40, 40)
        _box(draw, crop.box, color, line)
        _label(draw, crop.box, str(crop.order), color, font, line)
    for box in score.cut_balloons:
        _box(draw, box, (210, 30, 170), max(2, line // 2))
    return _downscale(canvas)


def _write_overlays(
    out: Path,
    scores: Mapping[str, Sequence[PageScore]],
    images: Mapping[str, Path],
    all_overlays: bool,
) -> dict[tuple[str, str], str]:
    written: dict[tuple[str, str], str] = {}
    for name, pages in scores.items():
        for page in pages:
            if not all_overlays and not page.reasons():
                continue
            source = images.get(page.page_id)
            if source is None:
                continue
            relative = f"overlays/{name}/{page.page_id}.jpg"
            destination = out / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(source) as raw:
                raw.load()
                rgb = raw.convert("RGB")
            overlay = draw_overlay(rgb, page)
            rgb.close()
            overlay.save(destination, format="JPEG", quality=_JPEG_QUALITY)
            overlay.close()
            written[(name, page.page_id)] = relative
    return written


def _downscale(image: Image.Image) -> Image.Image:
    long_side = max(image.size)
    if long_side <= _OVERLAY_LONG_SIDE:
        return image
    scale = _OVERLAY_LONG_SIDE / long_side
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    image.close()
    return resized


def _box(draw: ImageDraw.ImageDraw, box: Box, color: tuple[int, int, int], width: int) -> None:
    draw.rectangle([box.x, box.y, max(box.x, box.x2 - 1), max(box.y, box.y2 - 1)], outline=color, width=width)


def _label(
    draw: ImageDraw.ImageDraw,
    box: Box,
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
    pad: int,
) -> None:
    draw.text((box.x + pad, box.y + pad), text, fill=color, font=font)


def _metric_table(block: MetricBlock) -> str:
    rows = (
        ("pages", str(block.pages)),
        ("panel precision @0.50", _fmt(block.panel_50.precision)),
        ("panel recall @0.50", _fmt(block.panel_50.recall)),
        ("panel F1 @0.50", _fmt(block.panel_f1_50)),
        ("mean page F1 @0.50", _fmt(block.mean_page_f1_50)),
        ("panel F1 @0.75", _fmt(block.panel_f1_75)),
        ("mean page F1 @0.75", _fmt(block.mean_page_f1_75)),
        ("TP / FP / FN @0.50", f"{block.panel_50.tp} / {block.panel_50.fp} / {block.panel_50.fn}"),
        ("order exact", _fmt(block.order_exact)),
        ("mean Kendall tau", _fmt(block.mean_kendall_tau)),
        ("balloon cut rate", _fmt(block.balloon_cut_rate)),
        ("content loss mean", _fmt(block.content_loss_mean, 4)),
        ("content loss max", _fmt(block.content_loss_max, 4)),
        ("pages with content loss > 0.5%", str(block.content_loss_pages)),
        ("fallback rate", _fmt(block.fallback_rate)),
        ("wrong split", _fmt(block.wrong_split_rate)),
        ("time mean s", _fmt(block.time_mean_s)),
        ("time p95 s", _fmt(block.time_p95_s)),
    )
    lines = ["| metric | value |", "| --- | ---: |"]
    lines.extend(f"| {label} | {value} |" for label, value in rows)
    return "\n".join(lines)


def _group_table(groups: Mapping[str, MetricBlock]) -> str:
    if not groups:
        return "None."
    lines = [
        "| group | pages | F1 @0.50 | order exact | balloon cut | loss mean | loss pages | fallback |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, block in groups.items():
        lines.append(
            f"| {name} | {block.pages} | {_fmt(block.panel_f1_50)} | {_fmt(block.order_exact)} | "
            f"{_fmt(block.balloon_cut_rate)} | {_fmt(block.content_loss_mean, 4)} | "
            f"{block.content_loss_pages} | {_fmt(block.fallback_rate)} |"
        )
    return "\n".join(lines)


def _comparison_table(scores: Mapping[str, Sequence[PageScore]]) -> str:
    names = sorted(scores)
    blocks = [aggregate(scores[name]) for name in names]
    lines = ["| metric | " + " | ".join(names) + " |", "| --- | " + " | ".join("---:" for _ in names) + " |"]
    rows = (
        ("panel F1 @0.50", [_fmt(block.panel_f1_50) for block in blocks]),
        ("order exact", [_fmt(block.order_exact) for block in blocks]),
        ("balloon cut rate", [_fmt(block.balloon_cut_rate) for block in blocks]),
        ("content loss mean", [_fmt(block.content_loss_mean, 4) for block in blocks]),
        ("fallback rate", [_fmt(block.fallback_rate) for block in blocks]),
        ("time mean s", [_fmt(block.time_mean_s) for block in blocks]),
    )
    for label, values in rows:
        lines.append(f"| {label} | {' | '.join(values)} |")
    return "\n".join(lines)


def _failing(scores: Mapping[str, Sequence[PageScore]]) -> list[tuple[str, PageScore]]:
    rows = [(name, page) for name, pages in scores.items() for page in pages if page.reasons()]
    return sorted(rows, key=lambda item: _severity(item[1]))


def _severity(page: PageScore) -> tuple[int, int, int, float, str]:
    flags = (
        (1 if page.content_loss > LOSS_PAGE else 0)
        + (1 if page.unexpected_fallback else 0)
        + (1 if page.wrong_split else 0)
        + (1 if page.wrong_order else 0)
    )
    return (
        -(page.counts_50.fn * 3 + page.counts_50.fp),
        -page.balloons_cut,
        -flags,
        -page.content_loss,
        page.page_id,
    )


def _thumbnails(scores: Mapping[str, Sequence[PageScore]], overlays: Mapping[tuple[str, str], str]) -> str:
    parts: list[str] = []
    for name in sorted(scores):
        for page in sorted(scores[name], key=lambda item: item.page_id):
            relative = overlays.get((name, page.page_id))
            if relative is None:
                continue
            href = html.escape(relative, quote=True)
            caption = html.escape(f"{name} {page.page_id}")
            parts.append(f'<a href="{href}"><img src="{href}" alt="{caption}"><span>{caption}</span></a>')
    if not parts:
        return "<p>None.</p>"
    return "\n".join(parts)


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"
