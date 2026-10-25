"""Command line for the tate processor."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tate.errors import TateError
from tate.eval.cli import register_eval, run_eval
from tate.pipeline import format_summary, process_chapter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tate",
        description="Turn a manga chapter into a vertical strip.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    process = sub.add_parser("process", help="build a .tate chapter from images, a CBZ, or a PDF")
    process.add_argument("input", type=Path, help="cbz, zip, pdf, image, or a folder of images")
    process.add_argument(
        "-o",
        "--output",
        type=Path,
        help="output .tate path (default: <stem>.tate in the current directory)",
    )
    process.add_argument("--title", help="chapter title (default: input name)")
    process.add_argument("--width", type=int, default=1080, help="strip width in pixels (default: 1080)")
    process.add_argument(
        "--detector",
        choices=("auto", "yolo", "cv", "page"),
        default="auto",
        help="panel detector (default: auto)",
    )
    process.add_argument("--preview", type=Path, metavar="PNG", help="also write a 480px-wide preview PNG")
    process.add_argument(
        "--debug",
        type=Path,
        metavar="DIR",
        help="write each page with panel boxes and reading-order numbers",
    )
    process.add_argument("--pages", help="1-based page range to read, as N or N-M")
    process.add_argument("--quiet", action="store_true", help="only print errors")
    register_eval(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "eval":
            return run_eval(args)
        if args.command != "process":
            parser.error(f"unknown command {args.command}")
        result = process_chapter(
            args.input,
            args.output,
            title=args.title,
            width=args.width,
            detector_name=args.detector,
            preview=args.preview,
            debug_dir=args.debug,
            pages=args.pages,
        )
    except TateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not args.quiet:
        sys.stdout.write(format_summary(result))
    return 0
