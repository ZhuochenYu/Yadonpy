"""Simple plateau detection utilities (inspired by yzc-gmx-gen).

These are used by yadonpy's equilibration gate to decide whether key
observables (density, polymer Rg) have reached a steady plateau.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass
class PlateauResult:
    ok: bool
    slope: float
    std: float
    mean: float
    rel_std: float
    window_start_time_ps: float


def check_plateau(
    t_ps: np.ndarray,
    y: np.ndarray,
    *,
    tail_frac: float = 0.2,
    slope_threshold_per_ps: float = 1e-7,
    rel_std_threshold: float = 0.02,
) -> PlateauResult:
    """Check whether the last tail_frac of the series is on a plateau."""
    t_ps = np.asarray(t_ps, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("check_plateau expects 1D y")

    n = len(t_ps)
    if n < 10:
        return PlateauResult(
            ok=False,
            slope=float("nan"),
            std=float("nan"),
            mean=float("nan"),
            rel_std=float("nan"),
            window_start_time_ps=float("nan"),
        )

    start = int(max(0, math.floor((1.0 - float(tail_frac)) * n)))
    tt = t_ps[start:]
    yy = y[start:]
    if len(tt) < 5:
        return PlateauResult(
            ok=False,
            slope=float("nan"),
            std=float("nan"),
            mean=float("nan"),
            rel_std=float("nan"),
            window_start_time_ps=float(tt[0]) if len(tt) else float("nan"),
        )

    # linear regression slope
    A = np.vstack([tt, np.ones_like(tt)]).T
    slope, _intercept = np.linalg.lstsq(A, yy, rcond=None)[0]

    mean = float(np.mean(yy))
    std = float(np.std(yy))
    rel_std = float(std / abs(mean)) if mean != 0 else float("inf")

    ok = (abs(float(slope)) <= float(slope_threshold_per_ps))
    if mean != 0:
        ok = ok and ((std / abs(mean)) <= float(rel_std_threshold))

    return PlateauResult(
        ok=bool(ok),
        slope=float(slope),
        std=float(std),
        mean=float(mean),
        rel_std=float(rel_std),
        window_start_time_ps=float(tt[0]),
    )


def find_plateau_start(
    t_ps: np.ndarray,
    y: np.ndarray,
    *,
    min_window_frac: float = 0.2,
    step_frac: float = 0.02,
    slope_threshold_per_ps: float = 1e-7,
    rel_std_threshold: float = 0.02,
) -> PlateauResult:
    """Try to find the earliest time after which the series is on a plateau."""
    t_ps = np.asarray(t_ps, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        raise ValueError("find_plateau_start expects 1D y")

    n = len(t_ps)
    if n < 20:
        return PlateauResult(
            ok=False,
            slope=float("nan"),
            std=float("nan"),
            mean=float("nan"),
            rel_std=float("nan"),
            window_start_time_ps=float("nan"),
        )

    min_window = int(max(10, math.floor(float(min_window_frac) * n)))
    step = int(max(1, math.floor(float(step_frac) * n)))

    best = check_plateau(
        t_ps,
        y,
        tail_frac=min_window / n,
        slope_threshold_per_ps=slope_threshold_per_ps,
        rel_std_threshold=rel_std_threshold,
    )
    if best.ok:
        # Move start earlier if possible
        for start in range(0, n - min_window, step):
            tt = t_ps[start:]
            yy = y[start:]
            if len(tt) < min_window:
                continue
            cand = check_plateau(
                tt,
                yy,
                tail_frac=1.0,
                slope_threshold_per_ps=slope_threshold_per_ps,
                rel_std_threshold=rel_std_threshold,
            )
            if cand.ok:
                cand.window_start_time_ps = float(tt[0])
                best = cand
                break

    return best
