"""Legend-placement helpers for generated analysis plots.

YadonPy creates many SVG diagnostics automatically. These helpers keep legends
readable by trying preferred locations, measuring overlap, and falling back to
safe defaults when matplotlib cannot infer a good placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from matplotlib.axes import Axes
from matplotlib.legend import Legend


# Preferred legend corners (in order):
# upper left -> lower left -> upper right -> lower right
DEFAULT_PREFERRED_LOCS: Tuple[str, ...] = (
    "upper left",
    "lower left",
    "upper right",
    "lower right",
)


@dataclass(frozen=True)
class LegendPlacementResult:
    loc: str
    score: float


def _sample_line_xy(line, max_points: int = 1200) -> Optional[np.ndarray]:
    x = line.get_xdata(orig=False)
    y = line.get_ydata(orig=False)
    if x is None or y is None:
        return None
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(x.size, y.size)
    if n <= 0:
        return None
    x = x[:n]
    y = y[:n]
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        x = x[idx]
        y = y[idx]
    return np.column_stack([x, y])


def _sample_collection_xy(coll, max_points: int = 2000) -> Optional[np.ndarray]:
    if not hasattr(coll, "get_offsets"):
        return None
    off = coll.get_offsets()
    if off is None:
        return None
    xy = np.asarray(off, dtype=float)
    if xy.size == 0 or xy.shape[1] < 2:
        return None
    xy = xy[:, :2]
    if xy.shape[0] > max_points:
        idx = np.linspace(0, xy.shape[0] - 1, max_points).astype(int)
        xy = xy[idx]
    return xy


def _bbox_contains_points(bbox, pts_display: np.ndarray) -> np.ndarray:
    x0, y0 = bbox.x0, bbox.y0
    x1, y1 = bbox.x1, bbox.y1
    return (
        (pts_display[:, 0] >= x0)
        & (pts_display[:, 0] <= x1)
        & (pts_display[:, 1] >= y0)
        & (pts_display[:, 1] <= y1)
    )


def _overlap_score(ax: Axes, legend_bbox, *, other_axes: Optional[Sequence[Axes]] = None) -> float:
    """Estimate overlap by counting sampled plotted points under the legend box."""
    score = 0.0
    axes = [ax] + list(other_axes or [])

    for a in axes:
        for line in a.lines:
            xy = _sample_line_xy(line)
            if xy is None:
                continue
            pts = line.axes.transData.transform(xy)
            score += float(_bbox_contains_points(legend_bbox, pts).sum())

    for a in axes:
        for coll in a.collections:
            xy = _sample_collection_xy(coll)
            if xy is None:
                continue
            pts = coll.axes.transData.transform(xy)
            score += float(_bbox_contains_points(legend_bbox, pts).sum())

    return float(score)


def choose_legend_location(
    ax: Axes,
    *,
    other_axes: Optional[Sequence[Axes]] = None,
    handles=None,
    labels=None,
    preferred_locs: Sequence[str] = DEFAULT_PREFERRED_LOCS,
    legend_kwargs: Optional[dict] = None,
) -> LegendPlacementResult:
    """Pick a legend location following preferred corner order and minimal overlap."""
    legend_kwargs = dict(legend_kwargs or {})
    fig = ax.figure

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    best_loc = str(preferred_locs[0]) if preferred_locs else "upper left"
    best_score = float("inf")

    for loc in preferred_locs:
        leg = ax.legend(handles=handles, labels=labels, loc=loc, **legend_kwargs)
        fig.canvas.draw()
        bbox = leg.get_window_extent(renderer=renderer)
        score = _overlap_score(ax, bbox, other_axes=other_axes)
        leg.remove()

        if score < best_score:
            best_score = score
            best_loc = str(loc)
        if best_score <= 0:
            break

    return LegendPlacementResult(loc=best_loc, score=best_score)


def place_legend(
    ax: Axes,
    *,
    handles=None,
    labels=None,
    other_axes: Optional[Sequence[Axes]] = None,
    preferred_locs: Sequence[str] = DEFAULT_PREFERRED_LOCS,
    **legend_kwargs,
) -> Legend:
    """Place a legend while minimizing overlap."""
    res = choose_legend_location(
        ax,
        handles=handles,
        labels=labels,
        preferred_locs=preferred_locs,
        legend_kwargs=legend_kwargs,
    )
    return ax.legend(handles=handles, labels=labels, loc=res.loc, **legend_kwargs)
