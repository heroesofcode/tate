"""Deterministic page transforms. Rotate and scale also move the ground truth."""

from __future__ import annotations

import io
import math
from collections.abc import Callable, Sequence
from typing import assert_never

import numpy as np
from PIL import Image

from tate.boxes import Box
from tate.errors import TateError
from tate.eval.model import (
    Contrast,
    Gamma,
    Geometry,
    Jpeg,
    Noise,
    Op,
    PanelAnn,
    Rotate,
    Scale,
    Yellow,
)

Point = tuple[float, float]
Mapper = Callable[[float, float], Point]


def apply_geometry(geometry: Geometry, ops: Sequence[Op]) -> Geometry:
    """Return ground truth after ``ops``, in the transformed page's pixel space."""
    current = geometry
    for op in ops:
        current = _geometry_op(current, op)
    return current


def render_ops(image: Image.Image, ops: Sequence[Op]) -> Image.Image:
    """Apply ``ops`` to ``image``. The caller owns the returned image."""
    current = image
    for op in ops:
        produced = _render_op(current, op)
        if produced is not current and current is not image:
            current.close()
        current = produced
    return current


def _geometry_op(geometry: Geometry, op: Op) -> Geometry:
    if isinstance(op, Rotate):
        mapper, size = rotation_mapper(geometry.size[0], geometry.size[1], op.degrees)
        return _map_geometry(geometry, mapper, size)
    if isinstance(op, Scale):
        mapper, size = scale_mapper(geometry.size[0], geometry.size[1], op.factor)
        return _map_geometry(geometry, mapper, size)
    return geometry


def _map_geometry(geometry: Geometry, mapper: Mapper, size: tuple[int, int]) -> Geometry:
    return Geometry(
        size=size,
        panels=tuple(_map_panel(panel, mapper, size) for panel in geometry.panels),
        balloons=tuple(_map_box(box, mapper, size) for box in geometry.balloons),
        ignore=tuple(_map_box(box, mapper, size) for box in geometry.ignore),
        whole=geometry.whole,
    )


def _map_panel(panel: PanelAnn, mapper: Mapper, size: tuple[int, int]) -> PanelAnn:
    if panel.poly is not None:
        mapped = tuple(mapper(x, y) for x, y in panel.poly)
        box = _bounds(mapped, size)
        return PanelAnn(box, mapped)
    return PanelAnn(_map_box(panel.box, mapper, size), None)


def _map_box(box: Box, mapper: Mapper, size: tuple[int, int]) -> Box:
    corners = (
        (float(box.x), float(box.y)),
        (float(box.x2), float(box.y)),
        (float(box.x2), float(box.y2)),
        (float(box.x), float(box.y2)),
    )
    return _bounds(tuple(mapper(x, y) for x, y in corners), size)


def _bounds(points: Sequence[Point], size: tuple[int, int]) -> Box:
    """Axis-aligned half-open bounds of ``points``, clamped to ``size``."""
    width, height = size
    x1 = math.floor(min(point[0] for point in points))
    y1 = math.floor(min(point[1] for point in points))
    x2 = math.ceil(max(point[0] for point in points))
    y2 = math.ceil(max(point[1] for point in points))
    x1 = min(max(x1, 0), max(0, width - 1))
    y1 = min(max(y1, 0), max(0, height - 1))
    x2 = min(max(x2, x1 + 1), width)
    y2 = min(max(y2, y1 + 1), height)
    return Box(x1, y1, x2 - x1, y2 - y1)


def rotation_mapper(width: int, height: int, degrees: float) -> tuple[Mapper, tuple[int, int]]:
    """Map source points into a canvas expanded the way Pillow rotates with ``expand``."""
    turned = degrees % 360.0
    if abs(turned - 0.0) < 1e-9 or abs(turned - 360.0) < 1e-9:
        return _identity, (width, height)
    if abs(turned - 90.0) < 1e-9:
        return lambda x, y: (y, width - x), (height, width)
    if abs(turned - 180.0) < 1e-9:
        return lambda x, y: (width - x, height - y), (width, height)
    if abs(turned - 270.0) < 1e-9:
        return lambda x, y: (height - y, x), (height, width)
    return _affine_rotation(width, height, turned)


def scale_mapper(width: int, height: int, factor: float) -> tuple[Mapper, tuple[int, int]]:
    new_w = max(1, round(width * factor))
    new_h = max(1, round(height * factor))
    scale_x = new_w / width
    scale_y = new_h / height
    return lambda x, y: (x * scale_x, y * scale_y), (new_w, new_h)


def _identity(x: float, y: float) -> Point:
    return x, y


def _affine_rotation(width: int, height: int, degrees: float) -> tuple[Mapper, tuple[int, int]]:
    """Inverse of Pillow's destination-to-source rotate matrix, so corners land on the new canvas."""
    angle = -math.radians(degrees % 360.0)
    center_x = width / 2
    center_y = height / 2
    matrix = [
        round(math.cos(angle), 15),
        round(math.sin(angle), 15),
        0.0,
        round(-math.sin(angle), 15),
        round(math.cos(angle), 15),
        0.0,
    ]

    def toward_source(x: float, y: float) -> Point:
        a, b, c, d, e, f = matrix
        return a * x + b * y + c, d * x + e * y + f

    matrix[2], matrix[5] = toward_source(-center_x, -center_y)
    matrix[2] += center_x
    matrix[5] += center_y
    corners = ((0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height)))
    projected = [toward_source(x, y) for x, y in corners]
    min_x = min(point[0] for point in projected)
    min_y = min(point[1] for point in projected)
    new_w = math.ceil(max(point[0] for point in projected)) - math.floor(min_x)
    new_h = math.ceil(max(point[1] for point in projected)) - math.floor(min_y)
    matrix[2], matrix[5] = toward_source(-(new_w - width) / 2.0, -(new_h - height) / 2.0)
    a, b, c, d, e, f = matrix
    det = a * e - b * d
    if det == 0:
        raise TateError("rotation transform is degenerate")

    def mapper(x: float, y: float) -> Point:
        src_x = x - c
        src_y = y - f
        return (e * src_x - b * src_y) / det, (-d * src_x + a * src_y) / det

    return mapper, (new_w, new_h)


def _render_op(image: Image.Image, op: Op) -> Image.Image:
    if isinstance(op, Rotate):
        turned = op.degrees % 360.0
        if abs(turned) < 1e-9 or abs(turned - 360.0) < 1e-9:
            return image.copy()
        return image.rotate(
            op.degrees,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(255, 255, 255),
        )
    if isinstance(op, Scale):
        _mapper, size = scale_mapper(image.width, image.height, op.factor)
        return image.resize(size, Image.Resampling.LANCZOS)
    if isinstance(op, Contrast):
        return _blend_mid_gray(image, op.factor)
    if isinstance(op, Gamma):
        return _apply_gamma(image, op.gamma)
    if isinstance(op, Noise):
        return _apply_noise(image, op.sigma, op.seed)
    if isinstance(op, Jpeg):
        return _apply_jpeg(image, op.quality)
    if isinstance(op, Yellow):
        return _apply_yellow(image, op.amount)
    assert_never(op)


def _blend_mid_gray(image: Image.Image, factor: float) -> Image.Image:
    """``factor`` 1 keeps the page; lower values blend every channel toward 128."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    blended = (pixels - 128.0) * factor + 128.0
    return Image.fromarray(np.clip(np.rint(blended), 0, 255).astype(np.uint8), mode="RGB")


def _apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
    """``out = 255 * (pixel / 255) ** gamma`` per channel."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    powered = np.power(pixels, gamma) * 255.0
    return Image.fromarray(np.clip(np.rint(powered), 0, 255).astype(np.uint8), mode="RGB")


def _apply_noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    """Add Gaussian noise from a NumPy ``Generator`` (PCG64) seeded with ``seed``."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    noise = np.random.default_rng(seed).normal(0.0, sigma, size=pixels.shape)
    merged = pixels + noise
    return Image.fromarray(np.clip(np.rint(merged), 0, 255).astype(np.uint8), mode="RGB")


def _apply_jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        decoded.load()
        return decoded.convert("RGB")


def _apply_yellow(image: Image.Image, amount: float) -> Image.Image:
    """Blend toward aged paper ``(242, 228, 196)``. ``amount`` 0 is unchanged."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    paper = np.array([242.0, 228.0, 196.0], dtype=np.float64)
    blended = pixels * (1.0 - amount) + paper * amount
    return Image.fromarray(np.clip(np.rint(blended), 0, 255).astype(np.uint8), mode="RGB")
