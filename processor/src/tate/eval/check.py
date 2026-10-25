"""Compare an evaluation run with ``eval/thresholds.json``."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tate.errors import TateError
from tate.eval.metrics import LOSS_PAGE, MetricBlock, PageScore, aggregate, aggregate_styles

_MIN_KEYS = {"panel_f1_50", "order_exact"}
_MAX_KEYS = {"balloon_cut_rate_max", "content_loss_pages_max", "fallback_rate_max"}
_KNOWN = _MIN_KEYS | _MAX_KEYS


@dataclass(frozen=True, slots=True)
class Thresholds:
    detector: str
    overall: dict[str, float]
    per_style: dict[str, float]
    min_pages_per_style: int
    path: Path


def load_thresholds(path: Path) -> Thresholds:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TateError(f"missing {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise TateError(f"invalid JSON in {path.name}: {exc.msg}") from exc
    if not isinstance(raw, dict):
        raise TateError(f"{path.name} must be an object")
    allowed = {"detector", "overall", "per_style", "min_pages_per_style"}
    extra = sorted(set(raw) - allowed)
    if extra:
        raise TateError(f"{path.name} has unknown fields: {', '.join(str(key) for key in extra)}")
    for key in allowed:
        if key not in raw:
            raise TateError(f"{path.name} is missing {key}")
    detector = raw["detector"]
    if detector not in {"yolo", "cv"}:
        raise TateError(f"{path.name} detector must be yolo or cv")
    minimum = raw["min_pages_per_style"]
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise TateError(f"{path.name} min_pages_per_style must be an integer >= 1")
    return Thresholds(
        detector=str(detector),
        overall=_parse_limits(raw["overall"], f"{path.name} overall"),
        per_style=_parse_limits(raw["per_style"], f"{path.name} per_style"),
        min_pages_per_style=minimum,
        path=path,
    )


def judge(thresholds: Thresholds, scores: Mapping[str, Sequence[PageScore]]) -> list[str]:
    """Lines describing each missed threshold. An empty list means the gate passed."""
    pages = scores.get(thresholds.detector)
    if pages is None:
        return [f"thresholds ask for {thresholds.detector}, but this run did not include it"]
    failures: list[str] = []
    overall = aggregate(pages)
    failures.extend(_check_block(thresholds.detector, "overall", overall, thresholds.overall))
    styles = aggregate_styles(pages)
    for style, block in styles.items():
        if block.pages < thresholds.min_pages_per_style:
            continue
        failures.extend(_check_block(thresholds.detector, f"style {style}", block, thresholds.per_style))
    return failures


def _check_block(detector: str, scope: str, block: MetricBlock, limits: Mapping[str, float]) -> list[str]:
    failures: list[str] = []
    for key, limit in limits.items():
        actual = _actual(block, key)
        label = f"{detector} {scope} {key}"
        if actual is None:
            failures.append(f"{label} is undefined, threshold is {_format_number(limit)}")
            continue
        if key in _MIN_KEYS and actual < limit:
            failures.append(
                f"{label} is {_format_number(actual)}, threshold is {_format_number(limit)} "
                f"(short by {_format_number(limit - actual)})"
            )
        elif key in _MAX_KEYS and actual > limit:
            failures.append(
                f"{label} is {_format_number(actual)}, maximum is {_format_number(limit)} "
                f"(over by {_format_number(actual - limit)})"
            )
    return failures


def _actual(block: MetricBlock, key: str) -> float | None:
    if key == "panel_f1_50":
        return block.panel_f1_50
    if key == "order_exact":
        return block.order_exact
    if key == "balloon_cut_rate_max":
        return block.balloon_cut_rate
    if key == "content_loss_pages_max":
        return float(block.content_loss_pages)
    if key == "fallback_rate_max":
        return block.fallback_rate
    return None


def _parse_limits(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, dict):
        raise TateError(f"{label} must be an object")
    extra = sorted(set(value) - _KNOWN)
    if extra:
        raise TateError(f"{label} has unknown fields: {', '.join(str(key) for key in extra)}")
    limits: dict[str, float] = {}
    for key in _KNOWN:
        if key not in value:
            continue
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, int | float):
            raise TateError(f"{label}.{key} must be a number")
        parsed = float(number)
        if not math.isfinite(parsed) or parsed < 0:
            raise TateError(f"{label}.{key} must be zero or greater")
        if key != "content_loss_pages_max" and parsed > 1:
            raise TateError(f"{label}.{key} must be between 0 and 1")
        limits[key] = parsed
    return limits


def loss_page_limit() -> float:
    """Pages above this content-loss fraction are counted in ``content_loss_pages``."""
    return LOSS_PAGE


def _format_number(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9 and abs(value) < 1e12:
        return str(round(value))
    return f"{value:.4f}"
