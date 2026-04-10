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
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from ...plotting.style import apply_matplotlib_style, golden_figsize
from ...plotting.legend import place_legend
from .structured import detect_first_shell
from .xvg import read_xvg
from .plot import plot_xvg_svg



# ----------------------
# Thermo plot annotations
# ----------------------
# GROMACS `gmx energy` legends usually contain only the term names.
# For human readability, we attach units + a short physical meaning note.
THERMO_TERM_INFO: dict[str, dict[str, str]] = {
    "Temperature": {"ylabel": "Temperature (K)", "meaning": "Instantaneous temperature from kinetic energy."},
    "Pressure": {"ylabel": "Pressure (bar)", "meaning": "Instantaneous pressure of the simulation box."},
    "Density": {"ylabel": "Density (kg/m$^3$)", "meaning": "Mass density of the simulation box."},
    "Volume": {"ylabel": "Volume (nm$^3$)", "meaning": "Instantaneous simulation box volume."},
    "Potential": {"ylabel": "Potential energy (kJ/mol)", "meaning": "Potential energy reported by GROMACS energy module."},
    "Kinetic En.": {"ylabel": "Kinetic energy (kJ/mol)", "meaning": "Kinetic energy reported by GROMACS energy module."},
    "Total Energy": {"ylabel": "Total energy (kJ/mol)", "meaning": "Total energy = potential + kinetic."},
    "Enthalpy": {"ylabel": "Enthalpy (kJ/mol)", "meaning": "Enthalpy H = E + P*V (GROMACS definition)."},

    # Common alternative spellings across GROMACS versions
    "Kinetic En": {"ylabel": "Kinetic energy (kJ/mol)", "meaning": "Kinetic energy reported by GROMACS energy module."},
    "Total-Energy": {"ylabel": "Total energy (kJ/mol)", "meaning": "Total energy = potential + kinetic."},
}

def _thermo_label(term: str) -> tuple[str, str]:
    """Return (ylabel, meaning) for a thermo series name."""
    info = THERMO_TERM_INFO.get(str(term), None)
    if info is None:
        return f"{term}", "Time series extracted from GROMACS energy output."
    return str(info.get("ylabel") or term), str(info.get("meaning") or "Time series extracted from GROMACS energy output.")


import re as _re

def _fs_safe_label(s: str) -> str:
    """Make a filesystem-safe token (for plot filenames)."""
    s = str(s).strip().replace(" ", "_")
    # Keep ascii letters/digits/underscore/dash/dot, replace others with underscore
    s = _re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = _re.sub(r"_+", "_", s).strip("_")
    return s or "series"

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


def plot_npt_convergence(
    thermo_xvg: Path,
    *,
    out_svg: Path,
    title: str,
    frac_last: float = 0.2,
) -> Optional[Path]:
    """Plot NPT convergence as overlaid relative deviations from the final plateau.

    We intentionally plot all series on a shared relative scale instead of their
    raw units so density / volume / box lengths remain visually comparable in a
    single figure.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        thermo_xvg = _as_path(thermo_xvg)
        out_svg = _as_path(out_svg)
        out_svg.parent.mkdir(parents=True, exist_ok=True)

        xvg = read_xvg(thermo_xvg)
        df = xvg.df
        if "x" not in df.columns:
            return None

        series_specs = [
            ("Density", "Density", "kg/m$^3$"),
            ("Volume", "Volume", "nm$^3$"),
            ("Box-X", "a", "nm"),
            ("Box-Y", "b", "nm"),
            ("Box-Z", "c", "nm"),
        ]
        available = [(col, label, unit) for (col, label, unit) in series_specs if col in df.columns]
        if len(available) < 2:
            return None

        t_ps = np.asarray(df["x"].to_numpy(dtype=float), dtype=float)
        if t_ps.size < 2:
            return None
        t_ns = t_ps / 1000.0

        apply_matplotlib_style(n_colors=max(8, len(available)))
        fig, ax = plt.subplots(figsize=golden_figsize(9.0))

        baseline_notes: list[str] = []
        n_last = int(max(5, round(float(max(0.05, frac_last)) * float(t_ps.size))))
        n_last = min(max(1, n_last), int(t_ps.size))

        for col, label, unit in available:
            y = np.asarray(df[col].to_numpy(dtype=float), dtype=float)
            if y.size != t_ps.size:
                continue
            baseline = float(np.mean(y[-n_last:])) if y.size else 0.0
            if not np.isfinite(baseline):
                continue
            if abs(baseline) > 1.0e-12:
                y_rel = 100.0 * (y - baseline) / baseline
                y_label = f"{label} ({baseline:.4g} {unit})"
            else:
                y_rel = y - baseline
                y_label = f"{label} (final≈0 {unit})"
            if y_rel.size >= 9:
                w = int(min(101, max(9, y_rel.size // 80)))
                y_plot = _moving_average_1d(y_rel, w)
            else:
                y_plot = y_rel
            ax.plot(t_ns, y_plot, linewidth=1.6, label=y_label)
            baseline_notes.append(f"{label}={baseline:.5g} {unit}")

        if not baseline_notes:
            plt.close(fig)
            return None

        ax.axhline(0.0, linestyle="--", linewidth=0.9, alpha=0.45, color="black")
        ax.set_title(str(title))
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Deviation from final-window mean (%)")
        ax.grid(True, alpha=0.25)
        place_legend(ax)
        ax.text(
            0.99,
            0.01,
            "Final-window means: " + " | ".join(baseline_notes),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(out_svg, format="svg")
        plt.close(fig)
        return out_svg
    except Exception:
        return None



def plot_thermo_stage(
    thermo_xvg: Path,
    *,
    out_dir: Path,
    title_prefix: str,
    frac_last: float = 0.2,
) -> Dict[str, str]:
    """Create standard thermo plots for a stage (annotated).

    Output files go under ``out_dir`` (typically ``<stage>/plots``):
      - thermo.svg                     (overlay; units may differ across series)
      - thermo__<Term>.svg             (one per term, with units + meaning caption)
      - ABOUT.md                       (explains why this folder exists)

    Returns:
        dict of created SVG paths / counts (best-effort).
    """
    out_dir = _as_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Explain this directory once (human-friendly).
    about = out_dir / "ABOUT.md"
    if not about.exists():
        about.write_text(
            """# plots/

This folder contains **quick-look SVG plots** automatically generated by YadonPy.

Why it exists:
- ``thermo.xvg`` (and other xvg/csv) files are machine-friendly but not convenient to inspect.
- The SVGs are meant for **fast sanity checks** (equilibration, stability, trends) without opening xmgrace.
- They are **non-essential** for continuing workflows: deleting this folder will not break restart/analysis.

Files:
- ``thermo.svg``: overlay of multiple thermo series (note: different series may have different units).
- ``thermo__<Term>.svg``: one plot per thermo term, with explicit axis units and a short physical meaning note.
""" + "\n",
            encoding="utf-8",
        )

    created: Dict[str, str] = {}
    thermo_xvg = _as_path(thermo_xvg)

    # Read once; if energy output is in -xvg none mode, columns may be y1/y2...
    try:
        x = read_xvg(thermo_xvg)
        df = x.df
        ycols = [c for c in df.columns if c != "x"]
        if not ycols:
            return created
    except Exception:
        return created

    # 1) overlay plot (best-effort)
    try:
        overlay_caption = (
            "Overlay of multiple thermo series from GROMACS energy output. "
            "Different series may have different units; use split plots for units + meaning."
        )
        svg = plot_xvg_svg(
            thermo_xvg,
            out_svg=out_dir / "thermo.svg",
            title=f"{title_prefix} thermo (overlay)",
            xlabel=(x.xlabel or "Time (ps)"),
            ylabel="Thermo series (units vary)",
            caption=overlay_caption,
        )
        created["thermo_svg"] = str(svg)
    except Exception:
        pass

    # 2) split per term (annotated)
    n_created = 0
    for c in ycols:
        try:
            ylabel, meaning = _thermo_label(str(c))
            caption = f"Physical meaning: {meaning}"
            out_svg = out_dir / f"{thermo_xvg.stem}__{_fs_safe_label(str(c))}.svg"
            plot_xvg_svg(
                thermo_xvg,
                out_svg=out_svg,
                title=f"{title_prefix}: {c}",
                xlabel=(x.xlabel or "Time (ps)"),
                ylabel=ylabel,
                caption=caption,
                cols=[c],
                legend=False,
            )
            n_created += 1
        except Exception:
            continue
    created["thermo_split_svgs"] = str(n_created)

    # 3) NPT convergence overlay (density / volume / box lengths)
    try:
        npt_svg = plot_npt_convergence(
            thermo_xvg,
            out_svg=out_dir / "npt_convergence.svg",
            title=f"{title_prefix} NPT convergence",
            frac_last=float(frac_last),
        )
        if npt_svg is not None:
            created["npt_convergence_svg"] = str(npt_svg)
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
        return plot_xvg_svg(xvg, out_svg=out_svg, title=title, xlabel="Time (ps)", ylabel="Density (kg/m$^3$)", caption="Physical meaning: mass density of the simulation box.")
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
        t_ns = t_ps / 1000.0
        plt.plot(t_ns, msd_plot, label=f"MSD (smoothed, w={window})")
        plt.plot(t_ns, msd, alpha=0.25, label="MSD (raw)")
        if fit_t_start_ps is not None and fit_t_end_ps is not None:
            plt.axvspan(float(fit_t_start_ps) / 1000.0, float(fit_t_end_ps) / 1000.0, alpha=0.15, label="fit window")
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
        if fit_t_start_ps is not None and fit_t_end_ps is not None:
            fit_mask = (t_ps >= float(fit_t_start_ps)) & (t_ps <= float(fit_t_end_ps)) & (t_ps > 0.0) & (msd > 0.0)
            if np.any(fit_mask):
                plt.loglog(t_ns[fit_mask], msd[fit_mask], linewidth=2.0, label="fit window")
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


def plot_msd_series(
    *,
    t_ps: np.ndarray,
    msd_nm2: np.ndarray,
    out_dir: Path,
    group: str,
    fit_t_start_ps: Optional[float] = None,
    fit_t_end_ps: Optional[float] = None,
    confidence: Optional[str] = None,
    status: Optional[str] = None,
    geometry: Optional[str] = None,
    alpha_mean: Optional[float] = None,
    selection_basis: Optional[str] = None,
    D_m2_s: Optional[float] = None,
    warning: Optional[str] = None,
    window: Optional[int] = None,
) -> Dict[str, str]:
    out_dir = _as_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    created: Dict[str, str] = {}
    t_ps = np.asarray(t_ps, dtype=float)
    msd = np.asarray(msd_nm2, dtype=float)
    if t_ps.size == 0 or msd.size == 0:
        return created
    if window is None:
        window = int(min(51, max(11, len(msd) // 50)))
    msd_plot = _moving_average_1d(msd, int(window))
    note_lines: list[str] = []
    headline = " | ".join([str(x) for x in (confidence, status, geometry) if x])
    if headline:
        note_lines.append(headline)
    if alpha_mean is not None:
        try:
            note_lines.append(f"alpha_mean={float(alpha_mean):.3f}")
        except Exception:
            pass
    if D_m2_s is not None:
        try:
            note_lines.append(f"D={float(D_m2_s):.3e} m^2/s")
        except Exception:
            pass
    if selection_basis:
        note_lines.append(str(selection_basis))
    if warning:
        note_lines.append(f"warning={warning}")
    note = "\n".join(note_lines) if note_lines else None
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        apply_matplotlib_style()
        out_svg = out_dir / f"msd_{_fs_safe_label(group)}.svg"
        plt.figure(figsize=golden_figsize(8.0))
        t_ns = t_ps / 1000.0
        plt.plot(t_ns, msd_plot, label=f"MSD (smoothed, w={window})")
        plt.plot(t_ns, msd, alpha=0.25, label="MSD (raw)")
        if fit_t_start_ps is not None and fit_t_end_ps is not None:
            plt.axvspan(float(fit_t_start_ps) / 1000.0, float(fit_t_end_ps) / 1000.0, alpha=0.15, label="fit window")
        plt.title(f"MSD - {group}")
        if note:
            plt.text(
                0.98,
                0.98,
                note,
                transform=plt.gca().transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
            )
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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        apply_matplotlib_style()
        out_svg = out_dir / f"msd_loglog_{_fs_safe_label(group)}.svg"
        plt.figure(figsize=golden_figsize(8.0))
        t_ns = t_ps / 1000.0
        mask = (t_ns > 0.0) & (msd_plot > 0.0) & (msd > 0.0)
        plt.loglog(t_ns[mask], msd_plot[mask], label=f"MSD (smoothed, w={window})")
        plt.loglog(t_ns[mask], msd[mask], alpha=0.25, label="MSD (raw)")
        if fit_t_start_ps is not None and fit_t_end_ps is not None:
            fit_mask = (t_ps >= float(fit_t_start_ps)) & (t_ps <= float(fit_t_end_ps)) & (t_ps > 0.0) & (msd > 0.0)
            if np.any(fit_mask):
                plt.loglog(t_ns[fit_mask], msd[fit_mask], linewidth=2.0, label="fit window")
                if alpha_mean is not None:
                    try:
                        x_ref = float(np.sqrt(t_ns[fit_mask][0] * t_ns[fit_mask][-1]))
                        y_ref = float(np.exp(np.interp(np.log(x_ref), np.log(t_ns[fit_mask]), np.log(msd_plot[fit_mask]))))
                        guide_x = np.asarray([float(np.min(t_ns[fit_mask])), float(np.max(t_ns[fit_mask]))], dtype=float)
                        guide_y = y_ref * (guide_x / max(x_ref, 1.0e-30))
                        plt.loglog(guide_x, guide_y, linestyle="--", alpha=0.55, label="slope=1 guide")
                    except Exception:
                        pass
        plt.title(f"MSD (log-log) - {group}")
        if note:
            plt.text(
                0.98,
                0.98,
                note,
                transform=plt.gca().transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
            )
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




def _plot_overlay_from_xvgs(
    *,
    xvg_map: Dict[str, Path],
    out_svg: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    loglog: bool = False,
    smooth: bool = False,
) -> Optional[Path]:
    """Overlay the first y-series from multiple XVG files into one SVG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_svg = _as_path(out_svg)
        out_svg.parent.mkdir(parents=True, exist_ok=True)

        apply_matplotlib_style(n_colors=max(8, len(xvg_map)))
        plt.figure(figsize=golden_figsize(8.0))
        for name in sorted(xvg_map.keys()):
            fp = _as_path(xvg_map[name])
            df = read_xvg(fp).df
            if "x" not in df.columns:
                continue
            ycols = [c for c in df.columns if c != "x"]
            if not ycols:
                continue
            x = np.asarray(df["x"].to_numpy(dtype=float))
            y = np.asarray(df[ycols[0]].to_numpy(dtype=float))
            if smooth and y.size >= 9:
                w = int(min(21, max(7, y.size // 80)))
                y = _moving_average_1d(y, w)
            if loglog:
                # avoid log(0)
                mask = (x > 0) & (y > 0)
                x = x[mask]
                y = y[mask]
                if x.size == 0:
                    continue
                plt.loglog(x, y, label=str(name))
            else:
                plt.plot(x, y, label=str(name))
        plt.title(str(title))
        plt.xlabel(str(xlabel))
        plt.ylabel(str(ylabel))
        plt.grid(True)
        place_legend(plt.gca())
        plt.tight_layout()
        plt.savefig(out_svg, format="svg")
        plt.close()
        return out_svg
    except Exception:
        return None


def plot_rdf_cn_summary(
    *,
    rdf_xvgs: Dict[str, Path],
    cn_xvgs: Optional[Dict[str, Path]],
    out_svg: Path,
    title: str,
) -> Optional[Path]:
    """Create a two-panel summary figure with all RDF curves and all CN curves."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_svg = _as_path(out_svg)
        out_svg.parent.mkdir(parents=True, exist_ok=True)

        apply_matplotlib_style(n_colors=max(8, len(rdf_xvgs)))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 9.0), sharex=True)

        for name in sorted(rdf_xvgs.keys()):
            fp = _as_path(rdf_xvgs[name])
            df = read_xvg(fp).df
            if "x" not in df.columns:
                continue
            ycols = [c for c in df.columns if c != "x"]
            if not ycols:
                continue
            r = np.asarray(df["x"].to_numpy(dtype=float))
            g = np.asarray(df[ycols[0]].to_numpy(dtype=float))
            if g.size >= 9:
                w = int(min(21, max(7, g.size // 80)))
                g = _moving_average_1d(g, w)
            ax1.plot(r, g, label=str(name))

        cn_xvgs = cn_xvgs or {}
        for name in sorted(cn_xvgs.keys()):
            fp = _as_path(cn_xvgs[name])
            df = read_xvg(fp).df
            if "x" not in df.columns:
                continue
            ycols = [c for c in df.columns if c != "x"]
            if not ycols:
                continue
            r = np.asarray(df["x"].to_numpy(dtype=float))
            cn = np.asarray(df[ycols[0]].to_numpy(dtype=float))
            if cn.size >= 9:
                w = int(min(21, max(7, cn.size // 80)))
                cn = _moving_average_1d(cn, w)
            ax2.plot(r, cn, label=str(name))

        ax1.set_ylabel("g(r)")
        ax1.set_title(f"{title} - RDF")
        ax1.grid(True)
        place_legend(ax1)

        ax2.set_xlabel("r (nm)")
        ax2.set_ylabel("CN")
        ax2.set_title(f"{title} - CN")
        ax2.grid(True)
        if cn_xvgs:
            place_legend(ax2)

        fig.tight_layout()
        fig.savefig(out_svg, format="svg")
        plt.close(fig)
        return out_svg
    except Exception:
        return None


def plot_msd_summary(
    *,
    msd_xvgs: Dict[str, Path],
    out_svg: Path,
    title: str,
) -> Optional[Path]:
    """Create a single summary figure with linear and log-log MSD overlays."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_svg = _as_path(out_svg)
        out_svg.parent.mkdir(parents=True, exist_ok=True)

        apply_matplotlib_style(n_colors=max(8, len(msd_xvgs)))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 9.0))
        for name in sorted(msd_xvgs.keys()):
            fp = _as_path(msd_xvgs[name])
            df = read_xvg(fp).df
            if "x" not in df.columns:
                continue
            ycols = [c for c in df.columns if c != "x"]
            if not ycols:
                continue
            t_ps = np.asarray(df["x"].to_numpy(dtype=float))
            msd = np.asarray(df[ycols[0]].to_numpy(dtype=float))
            if msd.size >= 9:
                w = int(min(51, max(11, msd.size // 50)))
                msd = _moving_average_1d(msd, w)
            t_ns = t_ps / 1000.0
            ax1.plot(t_ns, msd, label=str(name))
            mask = (t_ns > 0) & (msd > 0)
            if np.any(mask):
                ax2.loglog(t_ns[mask], msd[mask], label=str(name))

        ax1.set_title(f"{title} - linear")
        ax1.set_xlabel("Time (ns)")
        ax1.set_ylabel("MSD (nm$^2$)")
        ax1.grid(True)
        place_legend(ax1)

        ax2.set_title(f"{title} - log-log")
        ax2.set_xlabel("Time (ns)")
        ax2.set_ylabel("MSD (nm$^2$)")
        ax2.grid(True)
        place_legend(ax2)

        fig.tight_layout()
        fig.savefig(out_svg, format="svg")
        plt.close(fig)
        return out_svg
    except Exception:
        return None


def plot_msd_series_summary(
    *,
    msd_series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    out_svg: Path,
    title: str,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_svg = _as_path(out_svg)
        out_svg.parent.mkdir(parents=True, exist_ok=True)

        apply_matplotlib_style(n_colors=max(8, len(msd_series)))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 9.0))
        for name in sorted(msd_series.keys()):
            t_ps, msd = msd_series[name]
            t_ps = np.asarray(t_ps, dtype=float)
            msd = np.asarray(msd, dtype=float)
            if t_ps.size == 0 or msd.size == 0:
                continue
            if msd.size >= 9:
                w = int(min(51, max(11, msd.size // 50)))
                msd = _moving_average_1d(msd, w)
            t_ns = t_ps / 1000.0
            ax1.plot(t_ns, msd, label=str(name))
            mask = (t_ns > 0) & (msd > 0)
            if np.any(mask):
                ax2.loglog(t_ns[mask], msd[mask], label=str(name))

        ax1.set_title(f"{title} - linear")
        ax1.set_xlabel("Time (ns)")
        ax1.set_ylabel("MSD (nm$^2$)")
        ax1.grid(True)
        place_legend(ax1)

        ax2.set_title(f"{title} - log-log")
        ax2.set_xlabel("Time (ns)")
        ax2.set_ylabel("MSD (nm$^2$)")
        ax2.grid(True)
        place_legend(ax2)

        fig.tight_layout()
        fig.savefig(out_svg, format="svg")
        plt.close(fig)
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
        cn_shell = None
        if cn_xvg is not None and Path(cn_xvg).exists():
            df2 = read_xvg(cn_xvg).df
            r2 = np.asarray(df2["x"].to_numpy(dtype=float))
            y2col = [c for c in df2.columns if c != "x"][0]
            cn = np.asarray(df2[y2col].to_numpy(dtype=float))
            shell_info = detect_first_shell(r, g, cn) if r.size == cn.size else {"r_shell_nm": None, "cn_shell": None, "status": "failed", "confidence": "failed"}
            ax2 = ax1.twinx()
            ax2.plot(r2, cn, linestyle="--", label="CN")
            ax2.set_ylabel("CN")
            cn_max = float(np.nanmax(cn)) if cn.size else 0.0
            ax2.set_ylim(0.0, max(1.0, cn_max * 1.1))
            r_shell = shell_info.get("r_shell_nm")
            cn_shell = shell_info.get("cn_shell")
        else:
            shell_info = detect_first_shell(r, g, np.zeros_like(r))
        r_shell = shell_info.get("r_shell_nm")
        cn_shell = shell_info.get("cn_shell")
        if r_shell is not None:
            ax1.axvline(float(r_shell), linestyle=":", alpha=0.8)
            if ax2 is not None and cn_shell is not None:
                try:
                    ax2.plot([float(r_shell)], [float(cn_shell)], marker="o")
                except Exception:
                    pass
            # put a compact note in axes coordinates
            note = f"{shell_info.get('status', 'unknown')} | {shell_info.get('confidence', 'unknown')}\nr={float(r_shell):.3f} nm"
            if cn_shell is not None:
                note += f"\nCN={float(cn_shell):.2f}"
            ax1.text(
                0.98,
                0.98,
                note,
                transform=ax1.transAxes,
                ha="right",
                va="top",
            )

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


def plot_rdf_cn_series(
    *,
    r_nm: np.ndarray,
    g_r: np.ndarray,
    cn_curve: Optional[np.ndarray],
    out_svg: Path,
    title: str,
    shell: Optional[dict[str, object]] = None,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_svg = _as_path(out_svg)
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        r = np.asarray(r_nm, dtype=float)
        g = np.asarray(g_r, dtype=float)
        cn = np.asarray(cn_curve, dtype=float) if cn_curve is not None else None
        shell_info = shell if isinstance(shell, dict) else detect_first_shell(r, g, cn if cn is not None else np.zeros_like(r))

        apply_matplotlib_style()
        fig, ax1 = plt.subplots(figsize=golden_figsize(8.0))
        ax1.plot(r, g, label="RDF")
        ax1.set_xlabel("r (nm)")
        ax1.set_ylabel("g(r)")
        ax1.grid(True)

        ax2 = None
        if cn is not None and cn.size == r.size:
            ax2 = ax1.twinx()
            ax2.plot(r, cn, linestyle="--", label="CN")
            ax2.set_ylabel("CN")
            cn_max = float(np.nanmax(cn)) if cn.size else 0.0
            ax2.set_ylim(0.0, max(1.0, cn_max * 1.1))

        r_shell = shell_info.get("r_shell_nm")
        cn_shell = shell_info.get("cn_shell")
        if r_shell is not None:
            ax1.axvline(float(r_shell), linestyle=":", alpha=0.8)
            if ax2 is not None and cn_shell is not None:
                ax2.plot([float(r_shell)], [float(cn_shell)], marker="o")
            note = f"{shell_info.get('status', 'unknown')} | {shell_info.get('confidence', 'unknown')}\nr={float(r_shell):.3f} nm"
            if cn_shell is not None:
                note += f"\nCN={float(cn_shell):.2f}"
            ax1.text(0.98, 0.98, note, transform=ax1.transAxes, ha="right", va="top")

        fig.suptitle(title)
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


def plot_rdf_cn_series_summary(
    *,
    rdf_series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    cn_series: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]],
    out_svg: Path,
    title: str,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        out_svg = _as_path(out_svg)
        out_svg.parent.mkdir(parents=True, exist_ok=True)

        apply_matplotlib_style(n_colors=max(8, len(rdf_series)))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 9.0), sharex=True)

        for name in sorted(rdf_series.keys()):
            r, g = rdf_series[name]
            r = np.asarray(r, dtype=float)
            g = np.asarray(g, dtype=float)
            if g.size >= 9:
                w = int(min(21, max(7, g.size // 80)))
                g = _moving_average_1d(g, w)
            ax1.plot(r, g, label=str(name))

        cn_series = cn_series or {}
        cn_max = 0.0
        for name in sorted(cn_series.keys()):
            r, cn = cn_series[name]
            r = np.asarray(r, dtype=float)
            cn = np.asarray(cn, dtype=float)
            if cn.size >= 9:
                w = int(min(21, max(7, cn.size // 80)))
                cn = _moving_average_1d(cn, w)
            if cn.size:
                cn_max = max(cn_max, float(np.nanmax(cn)))
            ax2.plot(r, cn, label=str(name))

        ax1.set_ylabel("g(r)")
        ax1.set_title(f"{title} - RDF")
        ax1.grid(True)
        place_legend(ax1)

        ax2.set_xlabel("r (nm)")
        ax2.set_ylabel("CN")
        ax2.set_title(f"{title} - CN")
        ax2.set_ylim(0.0, max(1.0, cn_max * 1.1))
        ax2.grid(True)
        if cn_series:
            place_legend(ax2)

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
