"""Automatic plotting utilities (SVG-first).

This module ports the *behavior* (not the exact code) of `yzc_gmx_gen`:
every workflow stage writes human-readable SVG plots next to the raw XVG/CSV.

Design goals
------------
* Headless-safe (matplotlib Agg).
* Never crash a workflow: plotting should be best-effort.
* Default output: SVG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ...plotting.style import apply_matplotlib_style, golden_figsize
from ...plotting.legend import place_legend
from .xvg import read_xvg
from .plot import plot_xvg_svg, plot_xvg_split_svg


def _as_path(p) -> Path:
    return p if isinstance(p, Path) else Path(str(p))


def _moving_average_1d(y: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    w = int(max(1, window))
    if w <= 1 or y.size < 3:
        return y
    # pad edges to avoid large boundary artifacts
    pad = w // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(ypad, kernel, mode="valid")


def plot_thermo_stage(
    thermo_xvg: Path,
    *,
    out_dir: Path,
    title_prefix: str,
) -> Dict[str, str]:
    """Create standard thermo plots for a stage.

    Returns dict of created svg paths.
    """
    out_dir = _as_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: Dict[str, str] = {}
    try:
        svg = plot_xvg_svg(thermo_xvg, out_svg=out_dir / "thermo.svg", title=f"{title_prefix} thermo")
        created["thermo_svg"] = str(svg)
    except Exception:
        pass
    try:
        svgs = plot_xvg_split_svg(thermo_xvg, out_dir=out_dir, title_prefix=title_prefix)
        created["thermo_split_svgs"] = str(len(svgs))
    except Exception:
        pass
    return created


def plot_density_time(
    xvg: Path,
    *,
    out_svg: Path,
    title: str,
) -> Optional[Path]:
    try:
        return plot_xvg_svg(xvg, out_svg=out_svg, title=title, xlabel="Time (ps)", ylabel="Density")
    except Exception:
        return None


def plot_tg_curve(
    temperatures_k: Sequence[float],
    densities: Sequence[float],
    *,
    split_index: int,
    low_fit: Tuple[float, float],
    high_fit: Tuple[float, float],
    tg_k: float,
    out_svg: Path,
) -> Optional[Path]:
    """Plot density vs temperature with piecewise linear fits and Tg intersection."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        T = np.asarray(list(temperatures_k), dtype=float)
        rho = np.asarray(list(densities), dtype=float)
        out_svg = _as_path(out_svg)
        out_svg.parent.mkdir(parents=True, exist_ok=True)

        apply_matplotlib_style()
        plt.figure(figsize=golden_figsize(8.0))
        plt.plot(T, rho, marker="o", label="mean density")

        # fits
        m1, b1 = float(low_fit[0]), float(low_fit[1])
        m2, b2 = float(high_fit[0]), float(high_fit[1])
        T1 = np.linspace(float(T.min()), float(T[split_index - 1]), 50)
        T2 = np.linspace(float(T[split_index]), float(T.max()), 50)
        plt.plot(T1, m1 * T1 + b1, linestyle="--", label="low-T fit")
        plt.plot(T2, m2 * T2 + b2, linestyle="--", label="high-T fit")

        plt.axvline(float(tg_k), linestyle=":", label=f"Tg = {tg_k:.1f} K")
        plt.title("Tg fit (density vs T)")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Density (kg/m$^3$)")
        plt.grid(True)
        place_legend(plt.gca())
        plt.tight_layout()
        plt.savefig(out_svg, format="svg")
        plt.close()
        return out_svg
    except Exception:
        return None


def plot_msd(
    msd_xvg: Path,
    *,
    out_dir: Path,
    group: str,
    window: Optional[int] = None,
    fit_t_start_ps: Optional[float] = None,
    fit_t_end_ps: Optional[float] = None,
) -> Dict[str, str]:
    """Create MSD plots (linear + loglog) like yzc-gmx-gen."""
    out_dir = _as_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: Dict[str, str] = {}

    df = read_xvg(msd_xvg).df
    t_ps = np.asarray(df["x"].to_numpy(dtype=float))
    ycol = [c for c in df.columns if c != "x"][0]
    msd = np.asarray(df[ycol].to_numpy(dtype=float))
    if window is None:
        window = int(min(51, max(11, len(msd) // 50)))
    msd_plot = _moving_average_1d(msd, window=int(window))

    # linear (time in ns)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        apply_matplotlib_style()
        out_svg = out_dir / f"msd_{group}.svg"
        plt.figure(figsize=golden_figsize(8.0))
        plt.plot(t_ps / 1000.0, msd_plot, label=f"MSD (smoothed, w={window})")
        plt.plot(t_ps / 1000.0, msd, alpha=0.25, label="MSD (raw)")
        plt.title(f"MSD - {group}")
        plt.xlabel("Time (ns)")
        plt.ylabel("MSD (nm$^2$)")
        plt.grid(True)
        place_legend(plt.gca())
        plt.tight_layout()
        plt.savefig(out_svg, format="svg")
        plt.close()
        created["msd_svg"] = str(out_svg)
    except Exception:
        pass

    # loglog
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        apply_matplotlib_style()
        out_svg = out_dir / f"msd_loglog_{group}.svg"
        plt.figure(figsize=golden_figsize(8.0))
        t_ns = t_ps / 1000.0
        plt.loglog(t_ns, msd_plot, label=f"MSD (smoothed, w={window})")
        plt.loglog(t_ns, msd, alpha=0.25, label="MSD (raw)")
        plt.title(f"MSD (log-log) - {group}")
        plt.xlabel("Time (ns)")
        plt.ylabel("MSD (nm$^2$)")
        plt.grid(True)
        place_legend(plt.gca())
        plt.tight_layout()
        plt.savefig(out_svg, format="svg")
        plt.close()
        created["msd_loglog_svg"] = str(out_svg)
    except Exception:
        pass

    return created


def plot_msd_overlay(
    *,
    msd_xvgs: Dict[str, Path],
    out_svg: Path,
    title: str,
    loglog: bool = False,
) -> Optional[Path]:
    """Overlay MSD of multiple groups."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_svg = _as_path(out_svg)
        out_svg.parent.mkdir(parents=True, exist_ok=True)

        apply_matplotlib_style(n_colors=max(8, len(msd_xvgs)))
        plt.figure(figsize=golden_figsize(8.0))
        for name in sorted(msd_xvgs.keys()):
            fp = msd_xvgs[name]
            df = read_xvg(fp).df
            t_ps = np.asarray(df["x"].to_numpy(dtype=float))
            ycol = [c for c in df.columns if c != "x"][0]
            msd = np.asarray(df[ycol].to_numpy(dtype=float))
            w = int(min(51, max(11, len(msd) // 50)))
            msd_plot = _moving_average_1d(msd, w)
            t_ns = t_ps / 1000.0
            if loglog:
                plt.loglog(t_ns, msd_plot, label=name)
            else:
                plt.plot(t_ns, msd_plot, label=name)
        plt.title(title)
        plt.xlabel("Time (ns)")
        plt.ylabel("MSD (nm$^2$)")
        plt.grid(True)
        place_legend(plt.gca())
        plt.tight_layout()
        plt.savefig(out_svg, format="svg")
        plt.close()
        return out_svg
    except Exception:
        return None


def plot_rdf_cn(
    *,
    rdf_xvg: Path,
    cn_xvg: Optional[Path],
    out_svg: Path,
    title: str,
) -> Optional[Path]:
    """Plot RDF (and optional coordination number) on a dual-y axis."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_svg = _as_path(out_svg)
        out_svg.parent.mkdir(parents=True, exist_ok=True)

        df = read_xvg(rdf_xvg).df
        r = np.asarray(df["x"].to_numpy(dtype=float))
        ycol = [c for c in df.columns if c != "x"][0]
        g = np.asarray(df[ycol].to_numpy(dtype=float))

        apply_matplotlib_style()
        fig, ax1 = plt.subplots(figsize=golden_figsize(8.0))
        ax1.plot(r, g, label="RDF")
        ax1.set_xlabel("r (nm)")
        ax1.set_ylabel("g(r)")
        ax1.grid(True)

        ax2 = None
        if cn_xvg is not None and Path(cn_xvg).exists():
            df2 = read_xvg(cn_xvg).df
            r2 = np.asarray(df2["x"].to_numpy(dtype=float))
            y2col = [c for c in df2.columns if c != "x"][0]
            cn = np.asarray(df2[y2col].to_numpy(dtype=float))
            ax2 = ax1.twinx()
            ax2.plot(r2, cn, linestyle="--", label="CN")
            ax2.set_ylabel("CN")

        fig.suptitle(title)
        # legend combining axes
        handles, labels = ax1.get_legend_handles_labels()
        if ax2 is not None:
            h2, l2 = ax2.get_legend_handles_labels()
            handles += h2
            labels += l2
        if handles:
            ax1.legend(handles, labels, loc="best")

        fig.tight_layout()
        fig.savefig(out_svg, format="svg")
        plt.close(fig)
        return out_svg
    except Exception:
        return None



def plot_rg(rg_xvg: Path, *, out_dir: Path, group: str, window: Optional[int] = None) -> Dict[str, str]:
    """Plot Rg time series with moving-average smoothing and convergence cue."""
    out_dir = _as_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: Dict[str, str] = {}
    df = read_xvg(rg_xvg).df
    if 'x' not in df.columns or len(df.columns) < 2:
        return created
    t_ps = np.asarray(df['x'].to_numpy(dtype=float))
    ycol = [c for c in df.columns if c != 'x'][0]
    rg = np.asarray(df[ycol].to_numpy(dtype=float))
    if window is None:
        window = int(min(101, max(21, len(rg)//60)))
    rg_ma = _moving_average_1d(rg, window=int(window))
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        apply_matplotlib_style()
        out_svg = out_dir / f"rg_{group}.svg"
        plt.figure(figsize=golden_figsize(8.0))
        t_ns = t_ps / 1000.0
        plt.plot(t_ns, rg, label=f"{group} (raw)")
        plt.plot(t_ns, rg_ma, label=f"{group} (MA)")
        # highlight last segment used for simple convergence judgement
        frac_last = 0.2
        if len(t_ns) > 5:
            i0 = int(len(t_ns) * (1 - frac_last))
            plt.axvspan(float(t_ns[i0]), float(t_ns[-1]), alpha=0.10)
            # annotate std of last segment
            try:
                sd = float(np.std(rg[i0:]))
                plt.text(0.02, 0.95, f"std(last {int(frac_last*100)}%)={sd:.4f} nm", transform=plt.gca().transAxes, va='top')
            except Exception:
                pass
        plt.xlabel('Time (ns)')
        plt.ylabel('Rg (nm)')
        plt.title(f'Rg convergence: {group}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_svg)
        plt.close()
        created['rg_svg'] = str(out_svg)
    except Exception:
        return created
    return created
