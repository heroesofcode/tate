"""Panel detectors. Each one returns boxes in source-page pixels."""

from __future__ import annotations

import sys
from typing import Any, Protocol

from PIL import Image

from tate.boxes import Detection
from tate.detect.cv import CvDetector
from tate.detect.model import ensure_model
from tate.detect.page import PageDetector
from tate.detect.yolo import YoloDetector, load_session
from tate.errors import TateError

__all__ = [
    "CvDetector",
    "Detector",
    "PageDetector",
    "YoloDetector",
    "resolve_detector",
]


class Detector(Protocol):
    @property
    def name(self) -> str: ...

    def detect(self, image: Image.Image) -> Detection: ...


def resolve_detector(name: str, *, session: Any | None = None) -> Detector:
    """Build the requested detector. `auto` falls back to OpenCV if YOLO is unavailable."""
    if name == "page":
        return PageDetector()
    if name == "cv":
        return CvDetector()
    if name == "yolo":
        return YoloDetector(session if session is not None else load_session(ensure_model()))
    if name == "auto":
        if session is not None:
            return YoloDetector(session)
        try:
            return YoloDetector(load_session(ensure_model()))
        except TateError as exc:
            print(f"warning: {exc}; using cv detector", file=sys.stderr)
            return CvDetector()
    raise TateError(f"unknown detector: {name}")
