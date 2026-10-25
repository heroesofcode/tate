"""Style tags a ground-truth page may carry. Unknown tags are an error."""

from __future__ import annotations

STYLES: frozenset[str] = frozenset(
    {
        "standard",
        "dense",
        "borderless",
        "bleed",
        "irregular",
        "diagonal",
        "spread",
        "splash",
        "4koma",
        "old-scan",
        "low-contrast",
        "skewed",
        "overlap-border",
        "balloon-cross",
        "color",
        "ltr",
        "rtl",
    }
)
