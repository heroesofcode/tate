"""Detector that treats each page as a single segment."""

from __future__ import annotations

from PIL import Image

from tate.boxes import Box, Detection


class PageDetector:
    @property
    def name(self) -> str:
        return "page"

    def detect(self, image: Image.Image) -> Detection:
        return Detection((Box(0, 0, image.width, image.height),), ())
