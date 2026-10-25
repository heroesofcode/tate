# tate processor

Command-line tool that turns a manga chapter into a `.tate` package: a vertical strip of panels in reading order, sized for a phone. The default is Japanese order (right to left, top to bottom). `--direction ltr` reads Western comics left to right. The package contract is [docs/format.md](../docs/format.md).

The tool runs on a normal Linux PC, on CPU, with no GPU.

## Install

From a checkout:

```sh
cd processor
uv sync
uv run tate --help
```

From Git, into an isolated environment:

```sh
uv tool install "git+https://github.com/heroesofcode/tate#subdirectory=processor"
```

pipx accepts the same URL:

```sh
pipx install "git+https://github.com/heroesofcode/tate#subdirectory=processor"
```

## Usage

```sh
tate process chapter.cbz
tate process chapter.pdf -o chapter.tate --title "Chapter 1" --pages 1-30
tate process ./scans --detector cv --width 1080 --direction ltr --preview strip.png --debug debug/
```

`INPUT` may be a `.cbz` or `.zip` of images, a `.pdf`, a folder of images, or one image. Images are ordered naturally (`page2` before `page10`). Junk entries (`__MACOSX`, dotfiles, `Thumbs.db`) are ignored.

The output defaults to `<input stem>.tate` in the current directory. A short summary (pages, panels, detector, output size, time) is printed on success. Bad input exits non-zero with an `error:` line.

| Option | Meaning |
| --- | --- |
| `-o, --output` | `.tate` path to write |
| `--title` | Chapter title (default: input name) |
| `--width` | Strip width in pixels (default `1080`) |
| `--detector` | `auto` (default), `yolo`, `cv`, or `page` |
| `--direction` | `rtl` (default, right to left) or `ltr` (left to right) |
| `--pages` | 1-based range `N` or `N-M` |
| `--preview PNG` | Whole strip as one PNG, 480 px wide |
| `--debug DIR` | Each page with panel boxes, text boxes, and reading-order numbers |
| `--quiet` | Suppress the summary |

`SOURCE_DATE_EPOCH` (Unix seconds) fixes `createdAt` and the zip timestamps so the same input produces the same bytes.

## Detectors

`auto` tries `yolo`, then falls back to `cv` if the model cannot be downloaded, and says so on stderr. `yolo` and `cv` do not fall back. `page` skips detection and emits each page as one segment (covers, splash pages, or when you want the page left whole).

Panels are ordered right to left and top to bottom. `--direction ltr` mirrors that order: rows still run top to bottom, panels in a row run left to right, and a tall panel that blocks a horizontal cut is read as the left column first. Two or more side-by-side strips, each a stack of four or more panels, are read a strip at a time (right strip first, or the left strip first with `--direction ltr`). A speech-balloon box that overlaps a panel is included in that panel's crop so the balloon is not sliced. After the panels are chosen, each crop grows until the crops together cover the page, so a margin, a caption, or a balloon that crosses a frame is still emitted. A block of strokes separated from every panel by a gutter (a missed panel, a title column) is emitted as its own segment. A box that covers several panels is split on the paper gutters and frame rules between them. Strokes are pixels darker than the page's paper tone, so a tan scan is not treated as one field of ink. If detection looks wrong (too little of the ink covered, no real panel split, or heavy overlaps), the whole page is kept. A landscape spread — two facing pages meeting at a centre fold, or a wide page with no panel split — stays in one piece and is scaled to the strip width. A landscape page that is itself a grid of panels is laid out panel by panel. A panel narrower than the strip is scaled up by at most 1.6× and centered on white. Segments taller than 4096 px are split.

### yolo

ONNX Runtime on `CPUExecutionProvider`. The model is "manga panel & text detector (YOLO26-nano)": class 0 is a panel, class 1 is text. The first run downloads it into the user cache directory (`platformdirs`). `TATE_MODEL_PATH` points at a local file instead.

Input is `1x3x1024x1024` float RGB in `0..1`, letterboxed with pad 114. Output `output0` is `1x300x6`: `x1, y1, x2, y2, conf, class` in letterboxed pixels, already NMS'd.

### cv

Classical OpenCV fallback. No model and no download. Gutters are regions the colour of the page background, so a white sheet, yellowed paper, and a dark margin all separate panels. Long dark rules enclose rectangles, including a panel that bleeds off the scan, and a small caption box sharing an edge is folded back into its panel. Text boxes are left empty; balloons are not specially protected in this mode.

### page

Each page is one segment.

## Model credit and licence

The YOLO weights are **not** part of this repository and are not bundled in the wheel. They are downloaded at runtime from a pinned URL and rejected if the sha256 does not match `e66667bc6d5f00013ff27efc15d21e521825369d44dfd5d7f6e43cda2ca512b7`.

- Weights: [leoxs22/manga-panel-detector-yolo26n](https://huggingface.co/leoxs22/manga-panel-detector-yolo26n), trained on Manga109-s, licensed **AGPL-3.0**.
- ONNX export used here: [mednasserallah/manga-panel-detector-yolo26n-onnx](https://huggingface.co/mednasserallah/manga-panel-detector-yolo26n-onnx) at revision `f6b7f3cb2bada1ec7d8b16501f9c984890811445`, file `manga_panel_detector_fp32_1024.onnx`.

tate itself is MIT. Choosing the YOLO detector runs those AGPL-3.0 weights on your machine. The `cv` and `page` detectors do not use them.

## Evaluation

`tate eval` scores the panel strip against a ground-truth dataset. The file format, metrics, and the fetch / run / diff commands are in [eval/README.md](../eval/README.md).

The regression command, once annotated pages are in the dataset, is:

```sh
tate eval run --subset ci --check eval/thresholds.json --detector yolo
```

There is no CI job for that command yet.

## Development

```sh
uv run ruff check
uv run ruff format --check
uv run mypy src
uv run pytest
```

`pytest -m network` calls the real model and is skipped unless `TATE_NETWORK_TESTS=1`. The other tests stay offline and draw their own pages.

`uv run python scripts/pipeline_chapter.py OUT_DIR` draws synthetic pages, packs a CBZ, and runs `tate process` twice (`--detector cv`, then `--detector page`). The reader spec `web/e2e/pipeline.spec.ts` imports the `cv` package when `TATE_E2E_PACKAGE` points at it.
