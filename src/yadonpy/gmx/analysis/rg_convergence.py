"""Polymer Rg convergence gate (ported from yzc-gmx-gen).

This module provides a robust, diagnostics-friendly convergence check for the
polymer radius of gyration (Rg) time series.

Design goals
------------
- Zero external dependencies beyond NumPy + Matplotlib.
- Headless-safe plotting (SVG) for HPC clusters.
- Compatible with YadonPy's `AnalyzeResult` API.

The logic is adapted from yuzc's yzc-gmx-gen toolkit (Rg gate + diagnostic plot).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # safe on headless clusters
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RgConvergenceResult:
    # overall decision: True if either convergence criterion is satisfied
    ok: bool

    # which criterion triggered convergence: "trend", "sd_max", "both", or "none"
    converged_by: str

    # per-criterion flags
    ok_trend: bool
    ok_sd_max: bool

    # plateau start time (ps)
    plateau_start_time: float

    # rolling window points used for plotting/metrics
    window_points: int

    # Trend-based criterion thresholds
    slope_threshold_per_ps: float
    rel_std_threshold: float
    sma_sd_threshold: float

    # sd_max-based criterion threshold (RadonPy-like)
    rg_sd_crit: float

    # final-window stats (computed on [plateau_start_time, end])
    mean: float
    std: float
    rel_std: float
    slope: float

    # RadonPy-like: std of rolling-mean values in the final window
    sma_sd: float
    sma_sd_rel: float

    # sd_max in the final window (max std among components if available; else std(rg))
    sd_max: float
    sd_max_rel: float


def _rolling_mean_std(y: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return rolling mean/std with same length as y; leading values are NaN."""
    y = np.asarray(y, dtype=float)
    window = int(window)
    if window < 2:
        mu = y.astype(float).copy()
        sd = np.full_like(mu, np.nan, dtype=float)
        sd[:] = 0.0
        return mu, sd

    mu = np.full_like(y, np.nan, dtype=float)
    sd = np.full_like(y, np.nan, dtype=float)
    for i in range(window - 1, len(y)):
        seg = y[i - window + 1 : i + 1]
        mu[i] = float(np.mean(seg))
        sd[i] = float(np.std(seg))
    return mu, sd


def _window_lin_slope(t: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of y(t)."""
    if len(t) < 2:
        return float("nan")
    A = np.vstack([t, np.ones_like(t)]).T
    slope, _intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)


def find_rg_convergence(
    t: np.ndarray,
    rg: np.ndarray,
    rg_components: Optional[np.ndarray] = None,
    *,
    min_window_frac: float = 0.25,
    step_frac: float = 0.02,
    window_frac: float = 0.10,
    slope_threshold_per_ps: float = 1e-3,
    rel_std_threshold: float = 0.10,
    sma_sd_threshold: float = 0.05,
    rg_sd_crit: float = 0.10,
) -> RgConvergenceResult:
    """Find the earliest time after which Rg is converged.

    Criterion (in the window [start, end]):

      1) |slope| <= slope_threshold_per_ps
      2) std/|mean| <= rel_std_threshold
      3) sma_sd <= |mean| * sma_sd_threshold

    Alternative (sd_max-based) criterion (RadonPy-like):

      4) sd_max <= |mean| * rg_sd_crit

    Where sma_sd is the standard deviation of the rolling-mean series inside the
    considered window. sd_max is the maximum standard deviation among Rg
    components (if available); otherwise std(rg).
    """
    t = np.asarray(t, dtype=float)
    rg = np.asarray(rg, dtype=float)

    comps: Optional[np.ndarray] = None
    if rg_components is not None:
        comps = np.asarray(rg_components, dtype=float)
        if comps.ndim != 2 or len(comps) != len(rg):
            raise ValueError("rg_components must be a 2D array with the same length as rg")

    if t.ndim != 1 or rg.ndim != 1 or len(t) != len(rg):
        raise ValueError("find_rg_convergence expects 1D t and rg of equal length")

    # Filter invalid points (NaN/inf) to make the convergence check robust.
    mask = np.isfinite(t) & np.isfinite(rg)
    if comps is not None:
        # Only keep rows where all components are finite.
        mask = mask & np.all(np.isfinite(comps), axis=1)
    if not np.all(mask):
        t = t[mask]
        rg = rg[mask]
        if comps is not None:
            comps = comps[mask]

    n = len(t)
    if n < 20:
        return RgConvergenceResult(
            ok=False,
            converged_by="insufficient_data",
            ok_trend=False,
            ok_sd_max=False,
            plateau_start_time=float("nan"),
            window_points=0,
            slope_threshold_per_ps=float(slope_threshold_per_ps),
            rel_std_threshold=float(rel_std_threshold),
            sma_sd_threshold=float(sma_sd_threshold),
            rg_sd_crit=float(rg_sd_crit),
            mean=float("nan"),
            std=float("nan"),
            rel_std=float("nan"),
            slope=float("nan"),
            sma_sd=float("nan"),
            sma_sd_rel=float("nan"),
            sd_max=float("nan"),
            sd_max_rel=float("nan"),
        )

    min_window = int(max(10, math.floor(float(min_window_frac) * n)))
    step = int(max(1, math.floor(float(step_frac) * n)))
    w = int(max(5, math.floor(float(window_frac) * n)))
    w = min(w, n)

    rg_mu, _rg_sd = _rolling_mean_std(rg, w)

    def eval_segment(start: int) -> RgConvergenceResult:
        tt = t[start:]
        yy = rg[start:]
        if len(tt) < min_window:
            return RgConvergenceResult(
                ok=False,
                converged_by="none",
                ok_trend=False,
                ok_sd_max=False,
                plateau_start_time=float(tt[0]) if len(tt) else float("nan"),
                window_points=int(w),
                slope_threshold_per_ps=float(slope_threshold_per_ps),
                rel_std_threshold=float(rel_std_threshold),
                sma_sd_threshold=float(sma_sd_threshold),
                rg_sd_crit=float(rg_sd_crit),
                mean=float("nan"),
                std=float("nan"),
                rel_std=float("nan"),
                slope=float("nan"),
                sma_sd=float("nan"),
                sma_sd_rel=float("nan"),
                sd_max=float("nan"),
                sd_max_rel=float("nan"),
            )

        mean = float(np.mean(yy))
        std = float(np.std(yy))
        rel_std = float(std / abs(mean)) if mean != 0 else float("inf")
        slope = _window_lin_slope(tt, yy)

        mu_seg = rg_mu[start:]
        mu_seg = mu_seg[~np.isnan(mu_seg)]
        if len(mu_seg) >= max(5, min_window // 4):
            sma_sd = float(np.std(mu_seg[-w:])) if len(mu_seg) >= w else float(np.std(mu_seg))
        else:
            sma_sd = float("nan")
        sma_sd_rel = float(sma_sd / abs(mean)) if mean != 0 and not math.isnan(sma_sd) else float("nan")

        ok_trend = (abs(slope) <= float(slope_threshold_per_ps)) and (rel_std <= float(rel_std_threshold))
        if not math.isnan(sma_sd) and mean != 0:
            ok_trend = ok_trend and (sma_sd <= abs(mean) * float(sma_sd_threshold))
        else:
            ok_trend = False

        if comps is None:
            sd_max = float(std)
        else:
            seg = comps[start:]
            if not seg.size:
                sd_max = float("nan")
            else:
                # Use NaN-safe statistics. Some engines can output NaN components
                # if the selected group is empty or the trajectory is too short.
                sd_cols = np.nanstd(seg, axis=0)
                if np.all(np.isnan(sd_cols)):
                    sd_max = float("nan")
                else:
                    # Avoid RuntimeWarning from np.nanmax on all-NaN slices
                    sd_cols2 = np.where(np.isnan(sd_cols), float("-inf"), sd_cols)
                    v = float(np.max(sd_cols2))
                    sd_max = float("nan") if v == float("-inf") else float(v)

        if mean != 0 and not math.isnan(sd_max):
            sd_max_rel = float(sd_max / abs(mean))
            ok_sd_max = bool(sd_max <= abs(mean) * float(rg_sd_crit))
        else:
            sd_max_rel = float("nan")
            ok_sd_max = False

        ok = bool(ok_sd_max or ok_trend)
        # Prefer sd_max strategy: as long as we enter a plateau by sd_max, pass.
        if ok_sd_max and ok_trend:
            converged_by = "sd_max"
        elif ok_sd_max:
            converged_by = "sd_max"
        elif ok_trend:
            converged_by = "trend"
        else:
            converged_by = "none"

        return RgConvergenceResult(
            ok=bool(ok),
            converged_by=str(converged_by),
            ok_trend=bool(ok_trend),
            ok_sd_max=bool(ok_sd_max),
            plateau_start_time=float(tt[0]),
            window_points=int(w),
            slope_threshold_per_ps=float(slope_threshold_per_ps),
            rel_std_threshold=float(rel_std_threshold),
            sma_sd_threshold=float(sma_sd_threshold),
            rg_sd_crit=float(rg_sd_crit),
            mean=float(mean),
            std=float(std),
            rel_std=float(rel_std),
            slope=float(slope),
            sma_sd=float(sma_sd),
            sma_sd_rel=float(sma_sd_rel),
            sd_max=float(sd_max),
            sd_max_rel=float(sd_max_rel),
        )

    best = eval_segment(n - min_window)
    if best.ok:
        for s in range(0, n - min_window, step):
            cand = eval_segment(s)
            if cand.ok:
                best = cand
                break

    return best


def plot_rg_convergence_svg(
    *,
    t: np.ndarray,
    rg: np.ndarray,
    rg_components: Optional[np.ndarray] = None,
    res: RgConvergenceResult,
    out_svg: Path,
    title: str = "Polymer Rg convergence",
    xlabel: str = "Time (ps)",
    ylabel: str = "Rg (nm)",
) -> None:
    """Create a diagnostics SVG figure with Rg + convergence metrics & criteria."""
    out_svg = Path(out_svg)
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    t = np.asarray(t, dtype=float)
    rg = np.asarray(rg, dtype=float)
    comps: Optional[np.ndarray] = None
    if rg_components is not None:
        comps = np.asarray(rg_components, dtype=float)

    w = max(2, int(res.window_points) if res.window_points else max(5, int(0.1 * len(rg))))
    w = min(w, len(rg))
    rg_mu, rg_sd = _rolling_mean_std(rg, w)

    slope_roll = np.full_like(rg, np.nan, dtype=float)
    for i in range(w - 1, len(rg)):
        seg_t = t[i - w + 1 : i + 1]
        seg_mu = rg_mu[i - w + 1 : i + 1]
        if np.any(np.isnan(seg_mu)):
            continue
        slope_roll[i] = _window_lin_slope(seg_t, seg_mu)

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_std_roll = rg_sd / np.abs(rg_mu)

    if comps is None:
        sd_max_roll = rg_sd.copy()
    else:
        sd_cols = []
        for j in range(comps.shape[1]):
            _mu_j, sd_j = _rolling_mean_std(comps[:, j], w)
            sd_cols.append(sd_j)
        sd_stack = np.vstack(sd_cols) if sd_cols else np.full((1, len(rg)), np.nan, dtype=float)
        if np.all(np.isnan(sd_stack)):
            sd_max_roll = np.full(sd_stack.shape[1], np.nan, dtype=float)
        else:
            # Avoid RuntimeWarning from np.nanmax when some time slices are all-NaN
            sd_stack2 = np.where(np.isnan(sd_stack), float("-inf"), sd_stack)
            sd_max_roll = np.max(sd_stack2, axis=0)
            sd_max_roll = np.where(sd_max_roll == float("-inf"), np.nan, sd_max_roll)

    with np.errstate(divide="ignore", invalid="ignore"):
        sd_max_roll_rel = sd_max_roll / np.abs(rg_mu)

    start_idx = 0
    if not math.isnan(res.plateau_start_time):
        start_idx = int(np.searchsorted(t, res.plateau_start_time, side="left"))
        start_idx = min(max(start_idx, 0), len(t) - 1)

    fig = plt.figure(figsize=(9.0, 7.0))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 2], hspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, rg, linewidth=1.2, label="Rg (raw)")
    ax1.plot(t, rg_mu, linewidth=1.4, label=f"Rg (rolling mean, w={w})")
    if start_idx > 0:
        ax1.axvline(t[start_idx], linestyle="--", linewidth=1.0, label="plateau start")
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, np.abs(slope_roll), linewidth=1.2, label="|slope(rolling mean)|")
    ax2.axhline(res.slope_threshold_per_ps, linestyle="--", linewidth=1.0, label="slope crit")
    ax2.plot(t, rel_std_roll, linewidth=1.2, label="rel std (rolling)")
    ax2.axhline(res.rel_std_threshold, linestyle="--", linewidth=1.0, label="rel std crit")
    ax2.plot(t, sd_max_roll_rel, linewidth=1.2, label="sd_max rel (rolling)")
    ax2.axhline(res.rg_sd_crit, linestyle=":", linewidth=1.0, label="sd_max crit")
    if start_idx > 0:
        ax2.axvline(t[start_idx], linestyle="--", linewidth=1.0)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("metrics")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    # annotate summary
    txt = (
        f"ok={res.ok}  by={res.converged_by}\n"
        f"mean={res.mean:.4g} nm  std={res.std:.3g}  rel_std={res.rel_std:.3g}\n"
        f"slope={res.slope:.3g}/ps  sma_sd_rel={res.sma_sd_rel:.3g}  sd_max_rel={res.sd_max_rel:.3g}"
    )
    ax1.text(0.02, 0.02, txt, transform=ax1.transAxes, fontsize=9, va="bottom", ha="left")

    # tight_layout can emit warnings with some backends/layouts; ignore them.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            plt.tight_layout()
        except Exception:
            pass
    plt.savefig(out_svg, format="svg")
    plt.close(fig)
