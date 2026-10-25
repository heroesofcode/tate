"""YOLO26-nano manga panel and text detector, via ONNX Runtime on CPU."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from PIL import Image

from tate.boxes import Box, Detection
from tate.errors import TateError

INPUT_SIZE = 1024
PAD_COLOR = 114
PANEL_CLASS = 0
TEXT_CLASS = 1
PANEL_CONF = 0.25
TEXT_CONF = 0.25


@dataclass(frozen=True, slots=True)
class Letterbox:
    ratio: float
    pad_x: int
    pad_y: int


def load_session(path: Path) -> Any:
    import onnxruntime as ort

    options: Any = ort.SessionOptions()
    options.log_severity_level = 3
    try:
        return ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise TateError(f"could not load the panel detector model ({path.name})") from exc


def letterbox_image(image: Image.Image, size: int = INPUT_SIZE) -> tuple[np.ndarray[Any, Any], Letterbox]:
    """Fit an RGB image in a square canvas. Pad value matches the export (114)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    if height < 1 or width < 1:
        raise TateError("empty image")
    ratio = size / max(height, width)
    resized_w = min(size, max(1, int(width * ratio)))
    resized_h = min(size, max(1, int(height * ratio)))
    pad_x = (size - resized_w) // 2
    pad_y = (size - resized_h) // 2
    resized = cast(
        np.ndarray[Any, Any],
        cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR),
    )
    canvas = np.full((size, size, 3), PAD_COLOR, dtype=np.uint8)
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized
    tensor = np.transpose(canvas.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(tensor), Letterbox(ratio, pad_x, pad_y)


def map_detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    score: float,
    letter: Letterbox,
    page_w: int,
    page_h: int,
) -> Box | None:
    if letter.ratio <= 0:
        return None
    left = (x1 - letter.pad_x) / letter.ratio
    top = (y1 - letter.pad_y) / letter.ratio
    right = (x2 - letter.pad_x) / letter.ratio
    bottom = (y2 - letter.pad_y) / letter.ratio
    box = Box(
        round(left),
        round(top),
        round(right - left),
        round(bottom - top),
        score,
    ).clamp(page_w, page_h)
    if box.w < 4 or box.h < 4:
        return None
    return box


class YoloDetector:
    def __init__(self, session: Any) -> None:
        self._session = session

    @property
    def name(self) -> str:
        return "yolo"

    def detect(self, image: Image.Image) -> Detection:
        tensor, letter = letterbox_image(image)
        try:
            inputs = self._session.get_inputs()
            input_name = str(inputs[0].name) if inputs else "images"
            raw = self._session.run(None, {input_name: tensor})
        except TateError:
            raise
        except Exception as exc:
            raise TateError("panel detector failed") from exc
        if not raw:
            raise TateError("panel detector returned no output")
        prediction = np.asarray(raw[0])
        if prediction.ndim == 3 and int(prediction.shape[0]) == 1:
            prediction = prediction[0]
        if prediction.ndim != 2 or int(prediction.shape[1]) != 6:
            shape = tuple(int(value) for value in prediction.shape)
            raise TateError(f"unexpected detector output shape {shape}")
        panels: list[Box] = []
        texts: list[Box] = []
        page_w, page_h = image.size
        floor = min(PANEL_CONF, TEXT_CONF)
        for row in prediction:
            confidence = float(row[4])
            if confidence < floor:
                continue
            class_id = round(float(row[5]))
            mapped = map_detection(
                float(row[0]),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                confidence,
                letter,
                page_w,
                page_h,
            )
            if mapped is None:
                continue
            if class_id == PANEL_CLASS and confidence >= PANEL_CONF:
                panels.append(mapped)
            elif class_id == TEXT_CLASS and confidence >= TEXT_CONF:
                texts.append(mapped)
        return Detection(tuple(panels), tuple(texts))
