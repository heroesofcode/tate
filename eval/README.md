# tate evaluation

This directory is the ground truth for the processor: how close a panel-to-strip conversion comes to a hand-checked reading of the page. The command is `tate eval`. It does not change `tate process`.

Run it from the repository root so `--dataset eval` and `--check eval/thresholds.json` resolve. The processor itself lives in `processor/`.

```sh
cd processor && uv sync
cd ..
tate eval fetch --dataset eval --subset ci
tate eval run --dataset eval --subset ci --check eval/thresholds.json --detector yolo
```

The regression command, once this dataset has pages, is:

```sh
tate eval run --subset ci --check eval/thresholds.json --detector yolo
```

No CI job runs it yet. Tests in the processor draw their own pages and do not download this dataset.

## Dataset composition

Filled in when the annotated set is published.

| Work | Pages | License | Notes |
| --- | ---: | --- | --- |
| — | — | — | — |

## Files

| Path | Role |
| --- | --- |
| `eval/sources.json` | Where page files are downloaded from, with license and checksum |
| `eval/pages/<work>.json` | One work: pages, boxes, and how to build each image |
| `eval/thresholds.json` | The gate used by `tate eval run --check` |
| `eval/schema/pages.schema.json` | JSON Schema for a work file |

`sources.json` is a list. Each source has `id`, `title`, `author`, `license` (an SPDX identifier or a short name), `license_url`, `attribution` (the exact credit line), `homepage`, and `fetch`:

- `{"kind":"image","url":"...","sha256":"..."}` for one file (a raster image or a PDF)
- `{"kind":"zip","url":"...","sha256":"..."}` for an archive

A work may mix sources. Each page names the source it needs. `sha256` is lowercase hex of the downloaded bytes. It may be omitted until `tate eval fetch --update-checksums` fills it. A present checksum that does not match the file is an error. Checksums are not computed over decoded pixels except for the optional `pixel_sha256` below.

## Page file

```json
{
  "work": "example",
  "title": "Example",
  "direction": "rtl",
  "pages": []
}
```

`direction` is `rtl` or `ltr`. It records the reading direction of the work. Metrics follow the order of the `panels` array, which is the ground-truth reading order.

The file name is `pages/<work>.json`. `work` matches that stem.

Each page:

| Field | Meaning |
| --- | --- |
| `id` | Unique across the dataset. Letters, digits, `.`, `_`, `-`. |
| `styles` | Tags from the vocabulary below. A page may carry several. Unknown tags are an error. |
| `image` | How to obtain the pixels. See below. |
| `size` | `[width, height]` of the decoded page. Checked when the page is fetched and when it is scored. |
| `pixel_sha256` | Optional. SHA-256 of the raw RGB bytes (`Image.tobytes()` on an RGB image: row-major, 3 bytes per pixel, no header). Catches decoder drift. |
| `panels` | Ground-truth reading order. Each item is `{"box":[x1,y1,x2,y2]}`. |
| `whole` | `true` when the correct strip is the page kept as one piece (spread, splash, or a page that must not be cut). `panels` may then hold a single page box. |
| `balloons` | Speech balloons and text blocks that must not be cut. Same box shape. |
| `ignore` | Optional. Ink inside these boxes does not count as content loss (page numbers, scan borders, publisher marks). |
| `ci` | `true` to include the page in `--subset ci`. |
| `notes` | Optional free text. |

Boxes are half-open pixel rectangles `[x1, y1, x2, y2)`: `x2` and `y2` are exclusive, matching the processor. A panel may add `"poly": [[x, y], ...]` (at least three points) when it is not a rectangle. The box must be that polygon's axis-aligned bounding box: `x1 = floor(min x)`, `y1 = floor(min y)`, `x2 = ceil(max x)`, `y2 = ceil(max y)`.

Style tags: `standard`, `dense`, `borderless`, `bleed`, `irregular`, `diagonal`, `spread`, `splash`, `4koma`, `old-scan`, `low-contrast`, `skewed`, `overlap-border`, `balloon-cross`, `color`, `ltr`, `rtl`.

### Image

One of:

- `{"source":"<id>"}` — the source is a single raster file.
- `{"source":"<id>","member":"path/in/zip.jpg"}` — a member of a zip source. The member path is relative, with no `..`.
- `{"source":"<id>","pdf_page":0}` — a PDF source (or a PDF member when `member` is also set). `pdf_page` is 0-based. The page is rasterized with the processor's `render_pdf_page`, so the pixels match `tate process`.
- `{"derive":{"from":"<page id>","ops":[...]}}` — build the page from another dataset page.

Derive ops, applied in order:

| Op | Effect |
| --- | --- |
| `{"rotate": degrees}` | Rotate counter-clockwise. The canvas grows to fit, and the new area is white. 0/90/180/270 match Pillow's transpose. Other angles match Pillow's `rotate(..., expand=True)`. |
| `{"scale": factor}` | Resize to `round(width * factor)` by `round(height * factor)`. |
| `{"contrast": factor}` | `(pixel - 128) * factor + 128`, clamped. `1` is unchanged. Lower values blend toward mid-gray. |
| `{"gamma": g}` | `255 * (pixel / 255) ** g` per channel. |
| `{"noise": sigma, "seed": n}` | Gaussian noise from a NumPy `Generator` (PCG64) seeded with `n`. `sigma` is in 8-bit levels. |
| `{"jpeg": quality}` | Encode as JPEG at `quality` (1–95) and decode. |
| `{"yellow": amount}` | Blend toward aged paper `(242, 228, 196)`. `0` is unchanged, `1` is that color. |

Rotate and scale move the ground truth. Each box's four corners (or each polygon's points, when a polygon is present) are mapped into the new canvas, and the box becomes the axis-aligned half-open bounding box, clamped to the new image. Other ops leave boxes where they are. The derived page's `size` must be the size after the ops. `whole` must match the source page. `panels`, `balloons`, and `ignore` may be omitted; they are then taken from the source page after the geometric ops. If they are present, they must equal that transformed ground truth.

## Fetch

```sh
tate eval fetch [--dataset eval] [--subset all|ci] [--cache DIR] [--update-checksums]
```

The cache defaults to the user cache directory `tate/eval` (`platformdirs`). Downloads are one at a time, with `User-Agent: tate-eval (+https://github.com/heroesofcode/tate)`, four attempts, and backoff. Bodies are streamed to disk. A cached source whose sha256 still matches is not downloaded again. Archives stay on disk; only the members a page names are read.

Decoded pages are stored as PNG at `<cache>/pages/<page id>.png`. A later fetch skips a page when that PNG is present, its pixel hash matches the stamp, and the page's image recipe (including the parent page's pixel hash, for a derived page) is unchanged. `--subset ci` still builds derive parents that are not themselves in the subset.

`--update-checksums` writes missing `sha256` and `pixel_sha256` values back into the JSON. Files are pretty-printed with a stable key order, 2-space indent, UTF-8, and a trailing newline. An existing checksum is never overwritten; a mismatch fails the command.

## Run

```sh
tate eval run [--dataset eval] [--subset all|ci] [--detector yolo,cv] [--out DIR] [--check FILE] [--jobs N] [--cache DIR] [--all-overlays]
```

For each selected page and each detector, this runs `detect` and `layout_page` at strip width 1080, the same call `tate process` makes. Time is wall time around those two calls (`time.perf_counter`). `--jobs` is a process pool (default: CPU count, at most 4). Each worker loads its own detector, including the YOLO session.

`--out` (default `eval-report`) receives:

- `report.md` — overall table per detector, per-style table, per-work table, detector comparison, failing pages, threshold result
- `report.html` — the same text, inline CSS, and thumbnails that link to overlays
- `metrics.json` — version 1, including every page, so two runs can be diffed
- `overlays/<detector>/<page id>.jpg` — only pages that fail, unless `--all-overlays`

A page fails when it has a false positive or false negative panel, a wrong matched order, a cut balloon, content loss above 0.5%, an unexpected fallback, or a wrong split. Overlays are JPEG quality 85, long side at most 1600 px. Ground-truth panels are green with reading-order numbers. Predicted crops are red, continuations orange, with the predicted order number. Cut balloons are magenta. Ink that no crop covers is tinted blue.

```sh
tate eval diff A/metrics.json B/metrics.json
```

prints a before / after / delta table for the overall metrics and for per-style panel F1 at IoU 0.5.

`--check eval/thresholds.json` exits non-zero after the report is written. Each missed threshold is printed as `error:` with the actual value, the limit, and the gap. A value equal to the limit passes.

The initial thresholds are:

```json
{
  "detector": "yolo",
  "overall": {
    "panel_f1_50": 0.95,
    "order_exact": 0.95,
    "balloon_cut_rate_max": 0.02,
    "content_loss_pages_max": 0,
    "fallback_rate_max": 0.10
  },
  "per_style": {"panel_f1_50": 0.90},
  "min_pages_per_style": 3
}
```

`panel_f1_50` and `order_exact` are minimums. The `*_max` keys are maximums. `per_style` is applied only to styles that have at least `min_pages_per_style` pages in the evaluated set. The `detector` field names which run is gated. An undefined rate (no balloons, or no page with two or more panels) fails a threshold that asks for it.

## Annotation

Drafts and contact sheets are scripts, not `tate` subcommands.

```sh
cd processor
uv run python scripts/eval_bootstrap.py --work ID --pages-from ./scans
uv run python scripts/eval_review.py --work ID --pages id1,id2
```

`eval_bootstrap.py` runs the YOLO detector and `layout_page`, then writes `panels` in the order tate emits them and `balloons` from the text class. Overlapping text boxes are merged so one balloon is one box. Pages already present in the work file are left alone unless `--force`. New pages use the work id as their `source` and `styles: ["standard"]` (plus `spread` when the page is a spread). Add a real source entry, fix the boxes, and set `ci` before fetching. A fallback or a spread is drafted as `whole: true`.

`eval_review.py` reads the cache (fetch first) and draws ground-truth panels with order numbers, balloons, and ignore boxes. It tiles those overlays 2×2 into `review/<work>-NN.jpg` (or `--out`) with the page id on each tile. Derived pages show the transformed boxes.

## Metrics

Everything below is computed per detector, then broken down overall, per style tag (a page counts in each of its tags), and per work. Micro counts sum page counts inside the group. A page is one row in a mean.

**Predicted panels.** After `layout_page`:

- If `fallback` or `spread` is set, the prediction is one box covering the page, `[0, 0, width, height]`.
- Otherwise the prediction is `PageLayout.emitted_panels()`: each segment's `panel` box (the panel before text expansion and padding), skipping segments with `continuation`. Pieces of one tall panel collapse to that panel.

Crops used for balloons and content loss are `SegmentDraft.box`, and continuation pieces stay separate.

**IoU and matching.** Intersection over union of half-open boxes. At a threshold (0.5 and 0.75, independently), keep pairs at or above the threshold and match greedily, highest IoU first. Ties break toward the lower predicted index, then the lower ground-truth index. A box is used at most once.

**Whole pages.** The ground-truth region is the only panel box if there is one, otherwise the union of the listed panels, otherwise the full page. The page is correct when there is exactly one predicted panel and it covers at least 95% of that region (`intersection / ground-truth area >= 0.95`). Correct contributes TP=1, FP=0, FN=0. Otherwise FN=1 and FP is the number of predicted panels. This rule is the same at both IoU thresholds. Whole pages are left out of the order metrics.

**Precision, recall, F1.** From the summed TP, FP, and FN: precision is TP/(TP+FP) when TP+FP > 0, else 1. Recall is TP/(TP+FN) when TP+FN > 0, else 1. F1 is the harmonic mean, or 1 when precision and recall are both 0. `panel_f1_50` is the micro F1 at IoU 0.5. Mean page F1 is the unweighted mean of the same F1 computed on each page alone.

**Reading order.** Non-whole pages only, using the IoU 0.5 matches. Kendall tau-a is computed on the predicted indices of the matched panels listed in ground-truth order: `(concordant pairs − discordant pairs) / C(n, 2)`. Pages with fewer than 2 matches are skipped, and they do not enter the mean tau. Order is exact when every ground-truth panel is matched, there is no false positive, and the matched predicted order is the ground-truth order. The order-exact rate's denominator is the non-whole pages that have at least 2 ground-truth panels. A matched subset that is itself in order can have tau 1 while the page is still not order-exact, because of an extra or a missing panel; that page fails for panels, not for order. Order is wrong when tau is defined and below 1.

**Balloon cut.** A balloon is intact when some single crop contains at least 98% of its area. Continuation pieces do not combine. The cut rate is cut balloons / balloons. It is undefined when the group has no balloons.

**Content loss.** Luminance is `0.299 R + 0.587 G + 0.114 B`. A pixel is ink when luminance is below 200. On a page tagged `color`, a pixel is also ink when its Euclidean distance in RGB from the paper color is greater than 30. The paper color is the per-channel median of the border ring (NumPy median: the mean of the two middle values when the count is even). The ring is 2% of the width and 2% of the height, at least 1 pixel, along each edge. Pixels in `ignore` boxes, and pixels in the outer 1% margin (at least 1 pixel on each edge), are not ink. Loss is the ink pixels outside the union of all crop boxes, divided by the ink pixels. It is 0 when there is no ink. The report gives the mean, the max, and the number of pages whose loss is greater than 0.005.

**Fallback.** A page that is not `whole` and whose layout set `fallback`. A spread (`spread` true, `fallback` false) is not a fallback. The rate divides by all evaluated pages.

**Wrong split.** A `whole` page whose layout is neither a fallback nor a spread and that emits more than one panel. The rate divides by the number of `whole` pages, and is undefined when there are none.

**Time.** Mean and the linear 95th percentile of the per-page seconds. Sort the samples and interpolate at rank `(n − 1) * 0.95`.

## Threshold file

`overall` and `per_style` may contain:

| Key | Comparison |
| --- | --- |
| `panel_f1_50` | actual must be ≥ |
| `order_exact` | actual must be ≥ |
| `balloon_cut_rate_max` | actual must be ≤ |
| `content_loss_pages_max` | count of pages with loss > 0.5% must be ≤ |
| `fallback_rate_max` | actual must be ≤ |

Unknown keys are an error, so a typo cannot pass the gate quietly.
