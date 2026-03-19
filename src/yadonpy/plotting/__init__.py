"""Lightweight plotting helpers (SVG-first).

This subpackage is a small, dependency-light port of the post-processing
plotting utilities used in yuzc's `yzc_gmx_gen`.

Design goals
------------
* Headless-safe (matplotlib Agg backend).
* Vector output by default (SVG).
* Reasonable defaults for MD time-series plots.
"""

from __future__ import annotations

from .style import apply_matplotlib_style, golden_figsize
from .legend import place_legend

__all__ = [
    "apply_matplotlib_style",
    "golden_figsize",
    "place_legend",
]
