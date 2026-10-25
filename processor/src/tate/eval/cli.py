"""``tate eval`` subcommands."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from tate.eval.check import judge, load_thresholds
from tate.eval.dataset import load_dataset
from tate.eval.diff import format_diff, load_metrics
from tate.eval.fetch import default_cache_dir, fetch_dataset, page_png
from tate.eval.report import write_report
from tate.eval.run import default_jobs, evaluate, parse_detectors


def register_eval(subparsers: Any) -> None:
    parser = subparsers.add_parser("eval", help="score the panel strip against ground truth")
    commands = parser.add_subparsers(dest="eval_command", required=True)

    fetch = commands.add_parser("fetch", help="download sources and decode pages into the cache")
    _add_dataset(fetch)
    fetch.add_argument("--subset", choices=("all", "ci"), default="all")
    fetch.add_argument("--cache", type=Path, help="page cache (default: the user cache tate/eval)")
    fetch.add_argument(
        "--update-checksums",
        action="store_true",
        help="fill missing sha256 and pixel_sha256 fields",
    )

    run = commands.add_parser("run", help="score cached pages and write a report")
    _add_dataset(run)
    run.add_argument("--subset", choices=("all", "ci"), default="all")
    run.add_argument("--detector", default="yolo,cv", help="comma-separated list: yolo, cv")
    run.add_argument("--out", type=Path, default="eval-report", help="report directory")
    run.add_argument("--check", type=Path, help="thresholds JSON; exit non-zero when one is missed")
    run.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="worker processes (default: CPUs, max 4)",
    )
    run.add_argument("--cache", type=Path, help="page cache (default: the user cache tate/eval)")
    run.add_argument("--all-overlays", action="store_true", help="draw an overlay for every page")

    diff = commands.add_parser("diff", help="compare two metrics.json files")
    diff.add_argument("before", type=Path)
    diff.add_argument("after", type=Path)


def run_eval(args: Any) -> int:
    command = str(args.eval_command)
    if command == "fetch":
        return _fetch(args)
    if command == "run":
        return _run(args)
    if command == "diff":
        sys.stdout.write(format_diff(load_metrics(args.before), load_metrics(args.after)))
        return 0
    raise SystemExit(2)


def _fetch(args: Any) -> int:
    root = Path(args.dataset)
    cache = Path(args.cache) if args.cache is not None else default_cache_dir()
    result = fetch_dataset(
        load_dataset(root),
        cache=cache,
        subset=str(args.subset),
        update_checksums=bool(args.update_checksums),
    )
    sys.stdout.write(f"sources: {result.sources}\npages: {result.pages}\ncache: {cache}\n")
    if result.updated_files:
        sys.stdout.write("updated: " + ", ".join(result.updated_files) + "\n")
    return 0


def _run(args: Any) -> int:
    root = Path(args.dataset)
    cache = Path(args.cache) if args.cache is not None else default_cache_dir()
    dataset = load_dataset(root)
    detectors = parse_detectors(str(args.detector))
    scores = evaluate(
        dataset,
        cache=cache,
        detectors=detectors,
        subset=str(args.subset),
        jobs=default_jobs() if args.jobs is None else int(args.jobs),
    )
    checked = args.check is not None
    lines: list[str] = []
    if checked:
        lines = judge(load_thresholds(Path(args.check)), scores)
    images = {page.page.id: page_png(cache, page.page.id) for page in dataset.pages}
    write_report(
        Path(args.out),
        dataset_name=root.as_posix(),
        subset=str(args.subset),
        scores=scores,
        images=images,
        threshold_lines=lines,
        checked=checked,
        all_overlays=bool(args.all_overlays),
    )
    if lines:
        for line in lines:
            print(f"error: {line}", file=sys.stderr)
        return 1
    if checked:
        sys.stdout.write("thresholds: passed\n")
    return 0


def _add_dataset(parser: Any) -> None:
    parser.add_argument("--dataset", type=Path, default="eval", help="dataset directory")
