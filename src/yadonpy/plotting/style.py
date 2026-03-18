from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt

try:
    from cycler import cycler  # noqa: F401
except Exception:  # pragma: no cover
    cycler = None


@dataclass(frozen=True)
class PlotStyle:
    font_family: Optional[str]
    color_cycle: List[str]
    colormap_name: str = "viridis"


# Golden ratio (width / height)
PHI: float = (1.0 + 5.0**0.5) / 2.0


def _pick_font() -> Optional[str]:
    """Pick a preferred font if available.

    Preference order:
      1) Helvetica
      2) Arial
      2) Times New Roman
      3) None (matplotlib default)
    """
    try:
        import matplotlib.font_manager as fm

        names = {f.name for f in fm.fontManager.ttflist}
        if "Helvetica" in names:
            return "Helvetica"
        if "Arial" in names:
            return "Arial"
        if "DejaVu Sans" in names:
            return "DejaVu Sans"
        if "Times New Roman" in names:
            return "Times New Roman"
    except Exception:
        pass
    return None


def _premium_cycle(n: int = 8) -> List[str]:
    """A vivid-but-not-gaudy qualitative palette."""

    base = [
        "#0077BB",  # blue
        "#33BBEE",  # cyan
        "#009988",  # teal/green
        "#EE7733",  # orange
        "#CC3311",  # red
        "#EE3377",  # magenta
        "#AA3377",  # purple
        "#228833",  # green
        "#BBBBBB",  # neutral gray
        "#4477AA",  # muted blue
    ]
    if n <= len(base):
        return base[:n]
    cols: List[str] = []
    core = [c for c in base if c != "#BBBBBB"]
    while len(cols) < n:
        cols.extend(core)
    return cols[:n]


def golden_figsize(width: float = 8.0) -> tuple[float, float]:
    """Convenience helper for a golden-ratio canvas."""
    return (float(width), float(width) / PHI)


def get_default_style(n_colors: int = 8) -> PlotStyle:
    return PlotStyle(font_family=_pick_font(), color_cycle=_premium_cycle(n_colors), colormap_name="viridis")


def apply_matplotlib_style(n_colors: int = 8) -> PlotStyle:
    """Apply plotting defaults suitable for noisy MD time-series.

    - Headless-safe.
    - SVG output: keep text selectable/searchable (svg.fonttype='none').
    """
    st = get_default_style(n_colors)

    rc = {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "svg.fonttype": "none",
        "figure.figsize": (8.0, 5.0),
        "lines.linewidth": 1.6,
        "lines.markersize": 4.0,
        "axes.linewidth": 1.0,
        "grid.alpha": 0.18,
        "grid.linestyle": "-",
    }
    if st.font_family:
        rc["font.family"] = st.font_family
    matplotlib.rcParams.update(rc)

    # Keep Matplotlib's default color cycle to avoid busy multi-curve plots.
    return st
