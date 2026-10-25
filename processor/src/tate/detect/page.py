"""Detector that treats each page as a single segment."""

from __future__ import annotations

from PIL import Image

from tate.boxes import Detection


class PageDetector:
    """Emit no panel boxes.

    A rectangle the size of the page is split on the gutters inside it, the
    same as any other detection. With no boxes, the page stays one segment.
    """

    @property
    def name(self) -> str:
        return "page"

    def detect(self, image: Image.Image) -> Detection:
        return Detection((), ())
