"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .xvg import read_xvg

import re

def parse_gmx_current_sigmas(stdout: str) -> dict:
    """Best-effort parse of EH/GK conductivity from ``gmx current`` stdout.

    Returns dict: {eh_sigma_S_m, gk_sigma_S_m, fit_start_ps, fit_end_ps}
    Values may be None.
    """
    if not stdout:
        return {"eh_sigma_S_m": None, "gk_sigma_S_m": None, "fit_start_ps": None, "fit_end_ps": None}

    lines = [ln.strip() for ln in str(stdout).splitlines() if ln.strip()]

    def _first_float(s: str) -> float | None:
        m = re.search(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", s)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    eh = None
    gk = None
    # Prefer method-tagged lines; walk from bottom to get final summary values.
    for ln in reversed(lines):
        low = ln.lower()
        if ("s/m" not in low) and ("s m" not in low) and ("s*m" not in low):
            continue
        if ("cond" not in low) and ("sigma" not in low):
            continue
        val = _first_float(low)
        if val is None:
            continue
        if eh is None and ("einstein" in low or "helfand" in low or "nernst" in low):
            eh = val
            continue
        if gk is None and ("green" in low or "kubo" in low or "g-k" in low or "gk" in low):
            gk = val
            continue

    # Fit-range parse (version dependent)
    bfit = None
    efit = None
    for ln in lines:
        low = ln.lower()
        if "fit" not in low or "ps" not in low:
            continue
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*ps", low)
        if not m:
            continue
        try:
            bfit = float(m.group(1)); efit = float(m.group(2))
        except Exception:
            pass
        break

    # Fallback: any conductivity-looking line
    if eh is None and gk is None:
        for ln in reversed(lines):
            low = ln.lower()
            if ("s/m" not in low) and ("s m" not in low) and ("s*m" not in low):
                continue
            if ("cond" not in low) and ("sigma" not in low):
                continue
            val = _first_float(low)
            if val is not None:
                eh = val
                break

    return {"eh_sigma_S_m": eh, "gk_sigma_S_m": gk, "fit_start_ps": bfit, "fit_end_ps": efit}


def linear_fit_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, r2) for y = slope*x + intercept."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 3:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), float(r2)


@dataclass(frozen=True)
class EHFit:
    """Einstein–Helfand conductivity fit result."""

    sigma_S_m: float
    window_start_ps: float
    window_end_ps: float
    slope_per_ps: float
    intercept: float
    r2: float
    note: str = ""


@dataclass(frozen=True)
class EHWindowSelection:
    window_start_ps: float
    window_end_ps: float
    explanation: str


def select_eh_fit_window_from_dsp(
    dsp_xvg: Path,
    *,
    start_frac: float = 0.15,
    end_frac: float = 0.85,
    window_frac: float = 0.25,
    min_window_ps: float = 200.0,
    min_r2_accept: float = 0.98,
    slope_tol_rel: float = 0.25,
) -> EHWindowSelection:
    """Auto-select a robust linear window on `gmx current -dsp` output.

    Ported (simplified) from yuzc's yzc-gmx-gen transport analysis.
    """
    xvg = read_xvg(dsp_xvg).df
    t_ps = np.asarray(xvg["x"].to_numpy(dtype=float))
    # first y-column
    ycol = [c for c in xvg.columns if c != "x"][0]
    series = np.asarray(xvg[ycol].to_numpy(dtype=float))

    if t_ps.size < 10:
        raise ValueError(f"Too few points in {dsp_xvg}")
    if start_frac >= end_frac:
        raise ValueError("start_frac must be < end_frac")

    t0_lim = float(t_ps[0] + float(start_frac) * (t_ps[-1] - t_ps[0]))
    t1_lim = float(t_ps[0] + float(end_frac) * (t_ps[-1] - t_ps[0]))
    dt_ps = float(np.median(np.diff(t_ps)))
    dt_ps = max(dt_ps, 1e-9)
    span_ps = float(t1_lim - t0_lim)
    win_ps = max(float(min_window_ps), float(window_frac) * span_ps)
    win_n = max(8, int(round(win_ps / dt_ps)))

    recs: list[dict[str, float]] = []
    for i0 in range(0, t_ps.size - win_n):
        i1 = i0 + win_n
        b = float(t_ps[i0])
        e = float(t_ps[i1 - 1])
        if b < t0_lim or e > t1_lim:
            continue
        slope, intercept, r2 = linear_fit_r2(t_ps[i0:i1], series[i0:i1])
        if not np.isfinite(slope) or not np.isfinite(r2):
            continue
        if slope <= 0.0:
            continue
        recs.append({"b": b, "e": e, "slope": float(slope), "r2": float(r2)})

    if not recs:
        raise ValueError(f"No valid positive-slope windows found in {dsp_xvg}")

    r2_vals = np.array([r["r2"] for r in recs], dtype=float)
    r2_thr = float(max(float(min_r2_accept), float(np.percentile(r2_vals, 90))))
    good = [r for r in recs if r["r2"] >= r2_thr]
    if not good:
        best = max(recs, key=lambda r: r["r2"])
        return EHWindowSelection(
            window_start_ps=float(best["b"]),
            window_end_ps=float(best["e"]),
            explanation=(
                f"Auto EH window: no window passed r2>=thr (thr={r2_thr:.3f}); using best-r2 window."
            ),
        )

    slopes = np.array([r["slope"] for r in good], dtype=float)
    med_slope = float(np.median(slopes))
    if not np.isfinite(med_slope) or med_slope <= 0:
        med_slope = float(np.mean(slopes))

    stable = [r for r in good if abs(r["slope"] - med_slope) / max(med_slope, 1e-30) <= float(slope_tol_rel)]
    if not stable:
        stable = good

    # merge overlaps and pick longest
    stable = sorted(stable, key=lambda r: (r["b"], r["e"]))
    merged: list[tuple[float, float]] = []
    cur_b, cur_e = stable[0]["b"], stable[0]["e"]
    for r in stable[1:]:
        b, e = r["b"], r["e"]
        if b <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((float(cur_b), float(cur_e)))
            cur_b, cur_e = b, e
    merged.append((float(cur_b), float(cur_e)))
    best_b, best_e = max(merged, key=lambda be: (be[1] - be[0], be[1]))

    # enforce minimum length
    if (best_e - best_b) < float(min_window_ps):
        best = max(stable, key=lambda r: r["r2"])
        best_b, best_e = float(best["b"]), float(best["e"])

    return EHWindowSelection(
        window_start_ps=float(best_b),
        window_end_ps=float(best_e),
        explanation=(
            f"Auto EH window: usable [{t0_lim:.1f},{t1_lim:.1f}] ps; win≈{win_ps:.1f} ps; "
            f"r2_thr={r2_thr:.3f}; chosen [{best_b:.1f},{best_e:.1f}] ps."
        ),
    )


def conductivity_from_current_dsp(
    dsp_xvg: Path,
    *,
    force_bfit_ps: Optional[float] = None,
    force_efit_ps: Optional[float] = None,
    time_unit_factor: float = 1e12,
) -> EHFit:
    """Compute conductivity from `gmx current -dsp` curve via EH linear fit.

    GROMACS scales the `-dsp` output by the EH prefactor (6 V k_B T), so the slope
    in the linear regime is the static conductivity. XVG time is typically in ps,
    so we multiply the slope by 1e12 to convert to per-second slope (S/m).
    """
    xvg = read_xvg(dsp_xvg).df
    t_ps = np.asarray(xvg["x"].to_numpy(dtype=float))
    ycol = [c for c in xvg.columns if c != "x"][0]
    series = np.asarray(xvg[ycol].to_numpy(dtype=float))

    if t_ps.size < 10:
        raise ValueError(f"Too few points in {dsp_xvg}")

    if force_bfit_ps is not None and force_efit_ps is not None:
        b, e = float(force_bfit_ps), float(force_efit_ps)
        note = "forced"
    else:
        sel = select_eh_fit_window_from_dsp(dsp_xvg)
        b, e = float(sel.window_start_ps), float(sel.window_end_ps)
        note = sel.explanation

    m = (t_ps >= b) & (t_ps <= e)
    if int(m.sum()) < 5:
        raise ValueError(f"EH fit window too small on {dsp_xvg}: [{b},{e}] ps")

    slope, intercept, r2 = linear_fit_r2(t_ps[m], series[m])
    sigma = float(slope) * float(time_unit_factor)
    return EHFit(
        sigma_S_m=float(sigma),
        window_start_ps=float(b),
        window_end_ps=float(e),
        slope_per_ps=float(slope),
        intercept=float(intercept),
        r2=float(r2),
        note=str(note),
    )



def plot_eh_fit_svg(dsp_xvg: Path, fit: EHFit, *, out_svg: Path, title: str = "EH fit") -> Path:
    """Plot `gmx current -dsp` curve and the fitted linear window to SVG."""
    from .plot import _as_path  # reuse helper
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from ...plotting.style import apply_matplotlib_style, golden_figsize
    from ...plotting.legend import place_legend

    out_svg = _as_path(out_svg)
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    df = read_xvg(dsp_xvg).df
    t = np.asarray(df["x"].to_numpy(dtype=float))
    ycol = [c for c in df.columns if c != "x"][0]
    y = np.asarray(df[ycol].to_numpy(dtype=float))

    apply_matplotlib_style()
    plt.figure(figsize=golden_figsize(8.0))
    plt.plot(t, y, label=ycol)

    # highlight fit window
    m = (t >= float(fit.window_start_ps)) & (t <= float(fit.window_end_ps))
    if int(m.sum()) >= 2:
        plt.plot(t[m], y[m], label="fit window")
        # fitted line
        yhat = float(fit.slope_per_ps) * t[m] + float(fit.intercept)
        plt.plot(t[m], yhat, label=f"linear fit (r2={fit.r2:.3f})")

    plt.title(f"{title}\nσ={fit.sigma_S_m:.4g} S/m, [{fit.window_start_ps:.1f},{fit.window_end_ps:.1f}] ps")
    plt.xlabel("Time (ps)")
    plt.ylabel(str(ycol))
    plt.grid(True)
    place_legend(plt.gca())
    plt.tight_layout()
    plt.savefig(out_svg, format="svg")
    plt.close()
    return out_svg
