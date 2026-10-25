"""Download the pinned YOLO weights into the user cache and check the digest."""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import platformdirs

from tate.errors import TateError

MODEL_URL = (
    "https://huggingface.co/mednasserallah/manga-panel-detector-yolo26n-onnx/resolve/"
    "f6b7f3cb2bada1ec7d8b16501f9c984890811445/manga_panel_detector_fp32_1024.onnx"
)
MODEL_SHA256 = "e66667bc6d5f00013ff27efc15d21e521825369d44dfd5d7f6e43cda2ca512b7"
MODEL_FILENAME = "manga_panel_detector_fp32_1024.onnx"
MAX_MODEL_BYTES = 32 * 1024 * 1024


def model_cache_path() -> Path:
    return Path(platformdirs.user_cache_dir("tate", appauthor=False)) / MODEL_FILENAME


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_verified(data: bytes, dest: Path, expected_sha256: str) -> None:
    """Write `data` to `dest` only when its sha256 matches. The replace is atomic."""
    digest = hashlib.sha256(data).hexdigest()
    if digest != expected_sha256:
        raise TateError(f"detector model sha256 mismatch (got {digest})")
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(f".{dest.name}.{os.getpid()}.part")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, dest)
    finally:
        if temporary.exists():
            temporary.unlink()


def ensure_model() -> Path:
    """Return a local model path, downloading the pinned file on first use.

    `TATE_MODEL_PATH` overrides the cache and skips the download. A cached file
    with the wrong digest is replaced.
    """
    override = os.environ.get("TATE_MODEL_PATH")
    if override:
        path = Path(override)
        if not path.is_file():
            raise TateError(f"TATE_MODEL_PATH is not a file ({path.name})")
        return path
    dest = model_cache_path()
    if dest.is_file() and file_sha256(dest) == MODEL_SHA256:
        return dest
    write_verified(_download_model(), dest, MODEL_SHA256)
    return dest


def _download_model() -> bytes:
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "tate/0.1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            chunks: list[bytes] = []
            total = 0
            while True:
                block = response.read(256 * 1024)
                if not block:
                    break
                total += len(block)
                if total > MAX_MODEL_BYTES:
                    raise TateError("detector model download exceeded the size limit")
                chunks.append(block)
    except TateError:
        raise
    except Exception as exc:
        raise TateError("could not download the panel detector model") from exc
    return b"".join(chunks)
