"""Print a before/after table for two ``metrics.json`` files."""

from __future__ import annotations

import json
from pathlib import Path

from tate.errors import TateError

_OVERALL = (
    "panel_f1_50",
    "panel_f1_75",
    "mean_page_f1_50",
    "order_exact",
    "mean_kendall_tau",
    "balloon_cut_rate",
    "content_loss_mean",
    "content_loss_max",
    "content_loss_pages",
    "fallback_rate",
    "wrong_split_pages",
    "time_mean_s",
    "time_p95_s",
)


def load_metrics(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise TateError(f"missing {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise TateError(f"invalid JSON in {path.name}: {exc.msg}") from exc
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise TateError(f"{path.name} is not a tate eval metrics file")
    return {str(key): value for key, value in raw.items()}


def format_diff(before: dict[str, object], after: dict[str, object]) -> str:
    left = _detectors(before)
    right = _detectors(after)
    names = sorted(set(left) | set(right))
    if not names:
        return "no detectors\n"
    lines = [
        f"{'metric':<24} {'before':>12} {'after':>12} {'delta':>12}",
        f"{'-' * 24} {'-' * 12} {'-' * 12} {'-' * 12}",
    ]
    for name in names:
        lines.append(f"[{name}]")
        before_block = _overall(left.get(name))
        after_block = _overall(right.get(name))
        for key in _OVERALL:
            lines.append(_row(key, before_block.get(key), after_block.get(key)))
        styles = sorted(set(_styles(left.get(name))) | set(_styles(right.get(name))))
        for style in styles:
            label = f"style {style} panel_f1_50"
            lines.append(
                _row(
                    label,
                    _style_f1(left.get(name), style),
                    _style_f1(right.get(name), style),
                )
            )
    return "\n".join(lines) + "\n"


def _row(label: str, before: object, after: object) -> str:
    return f"{label:<24} {_cell(before):>12} {_cell(after):>12} {_delta(before, after):>12}"


def _cell(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "n/a"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return f"{float(value):.4f}"


def _delta(before: object, after: object) -> str:
    if isinstance(before, bool) or isinstance(after, bool):
        return "n/a"
    if not isinstance(before, int | float) or not isinstance(after, int | float):
        return "n/a"
    return f"{float(after) - float(before):+.4f}"


def _detectors(document: dict[str, object]) -> dict[str, dict[str, object]]:
    raw = document.get("detectors")
    if not isinstance(raw, dict):
        return {}
    found: dict[str, dict[str, object]] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, dict):
            found[key] = {str(item): item_value for item, item_value in value.items()}
    return found


def _overall(detector: dict[str, object] | None) -> dict[str, object]:
    if detector is None:
        return {}
    raw = detector.get("overall")
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items()}


def _styles(detector: dict[str, object] | None) -> dict[str, object]:
    if detector is None:
        return {}
    raw = detector.get("per_style")
    if not isinstance(raw, dict):
        return {}
    return {str(key): value for key, value in raw.items()}


def _style_f1(detector: dict[str, object] | None, style: str) -> object:
    block = _styles(detector).get(style)
    if not isinstance(block, dict):
        return None
    return block.get("panel_f1_50")
