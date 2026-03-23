"""SVG-first plotting utilities for GROMACS outputs.

This module is intentionally lightweight: it plots common .xvg time-series
and analysis curves (RDF, MSD, density, pressure, etc.).

API
---
* :func:`plot_xvg_svg`      - plot one XVG (all series) into a single SVG
* :func:`plot_xvg_split_svg` - plot one SVG per series column

Notes
-----
* Uses matplotlib Agg backend for headless environments.
* Default output is SVG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # safe on headless clusters
import matplotlib.pyplot as plt

import numpy as np

from ...plotting.style import apply_matplotlib_style, golden_figsize
from ...plotting.legend import place_legend
from .xvg import read_xvg


def _as_path(p) -> Path:
    return p if isinstance(p, Path) else Path(str(p))


def plot_xvg_svg(
    xvg_path,
    *,
    out_svg=None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    caption: Optional[str] = None,
    cols: Optional[Sequence[str]] = None,
    legend: bool = True,
    grid: bool = True,
    fig_width: float = 8.0,
) -> Path:
    """Plot a GROMACS XVG to a single SVG.

    Args:
        xvg_path: input .xvg file
        out_svg: output .svg path (default: alongside xvg, same stem)
        title: plot title (default: xvg stem)
        xlabel/ylabel: override axis labels; falls back to XVG metadata
        caption: optional caption rendered at the bottom
        cols: list of data columns to plot (excluding 'x'); default: all
        legend: show legend if multiple series
        grid: show grid
        fig_width: width in inches (height uses golden ratio)

    Returns:
        Path to the written SVG
    """
    xvg_path = _as_path(xvg_path)
    out_svg = _as_path(out_svg) if out_svg is not None else (xvg_path.with_suffix(".svg"))
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    x = read_xvg(xvg_path)
    df = x.df

    # Determine columns
    ycols = [c for c in df.columns if c != "x"]
    if cols is not None:
        cols_set = set(cols)
        ycols = [c for c in ycols if c in cols_set]
    if not ycols:
        raise ValueError(f"No plottable columns in {xvg_path}")

    apply_matplotlib_style(n_colors=max(8, len(ycols)))
    plt.figure(figsize=golden_figsize(float(fig_width)))

    t = np.asarray(df["x"].to_numpy(dtype=float))
    for c in ycols:
        y = np.asarray(df[c].to_numpy(dtype=float))
        plt.plot(t, y, label=str(c))

    plt.title(title or xvg_path.stem)
    plt.xlabel(xlabel if xlabel is not None else (x.xlabel or "x"))
    plt.ylabel(ylabel if ylabel is not None else (x.ylabel or "y"))
    if grid:
        plt.grid(True)
    if legend and len(ycols) > 1:
        place_legend(plt.gca())
    # Layout: reserve a small bottom margin for an optional caption
    if caption:
        fig = plt.gcf()
        fig.text(0.5, 0.01, str(caption), ha='center', va='bottom', fontsize=8)
        plt.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.close()
    return out_svg


def plot_xvg_split_svg(
    xvg_path,
    *,
    out_dir=None,
    title_prefix: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    cols: Optional[Sequence[str]] = None,
    grid: bool = True,
    fig_width: float = 8.0,
) -> list[Path]:
    """Plot one SVG per series column from an XVG.

    Output filenames are:
        <stem>__<col>.svg
    """
    xvg_path = _as_path(xvg_path)
    out_dir = _as_path(out_dir) if out_dir is not None else xvg_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    x = read_xvg(xvg_path)
    df = x.df
    ycols = [c for c in df.columns if c != "x"]
    if cols is not None:
        cols_set = set(cols)
        ycols = [c for c in ycols if c in cols_set]
    if not ycols:
        raise ValueError(f"No plottable columns in {xvg_path}")

    out: list[Path] = []
    for c in ycols:
        out_svg = out_dir / f"{xvg_path.stem}__{str(c).replace(' ', '_')}.svg"
        out.append(
            plot_xvg_svg(
                xvg_path,
                out_svg=out_svg,
                title=f"{title_prefix or xvg_path.stem}: {c}",
                xlabel=xlabel,
                ylabel=ylabel,
                cols=[c],
                legend=False,
                grid=grid,
                fig_width=fig_width,
            )
        )
    return out


def plot_xy_svg(
    x: np.ndarray,
    y: np.ndarray,
    *,
    out_svg,
    title: str,
    xlabel: str,
    ylabel: str,
    caption: Optional[str] = None,
    label: Optional[str] = None,
    grid: bool = True,
    fig_width: float = 8.0,
) -> Path:
    """Plot a simple x-y series into an SVG."""
    out_svg = _as_path(out_svg)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    apply_matplotlib_style()
    plt.figure(figsize=golden_figsize(float(fig_width)))
    plt.plot(np.asarray(x, dtype=float), np.asarray(y, dtype=float), label=label)
    plt.title(str(title))
    plt.xlabel(str(xlabel))
    plt.ylabel(str(ylabel))
    if grid:
        plt.grid(True)
    if label:
        place_legend(plt.gca())
    if caption:
        fig = plt.gcf()
        fig.text(0.5, 0.01, str(caption), ha='center', va='bottom', fontsize=8)
        plt.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.close()
    return out_svg
