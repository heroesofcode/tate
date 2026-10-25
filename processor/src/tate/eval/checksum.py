"""Checksums for downloaded files and decoded page pixels."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image


def file_sha256(path: Path) -> str:
    """SHA-256 of a file on disk, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pixel_sha256(image: Image.Image) -> str:
    """SHA-256 of raw RGB pixels, row-major, three bytes per pixel, no header."""
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    return hashlib.sha256(rgb.tobytes()).hexdigest()
