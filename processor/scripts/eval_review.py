"""Render ground-truth contact sheets for one work. Not part of the tate command."""

from __future__ import annotations

import sys

from tate.eval.annotate import review_main

if __name__ == "__main__":
    sys.exit(review_main())
