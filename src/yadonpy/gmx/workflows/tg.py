"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from ...runtime import resolve_restart
from ...io.mol2 import write_mol2_from_top_gro_parmed

import numpy as np

from ..analysis.thermo import summarize_terms_xvg
from ..analysis.auto_plot import plot_density_time, plot_tg_curve, plot_thermo_stage
from ..engine import GromacsRunner
from ..mdp_templates import NPT_MDP, MdpSpec, default_mdp_params
from ._util import RunResources, atomic_write_json, load_json, pbc_mol_fix_inplace, safe_mkdir


@dataclass(frozen=True)
class TgResult:
    temperatures_k: list[float]
    density_mean: list[float]
    specific_volume_cm3_g: list[float]
    fit_metric: str
    fit_values: list[float]
    split_index: int
    tg_k: float
    low_fit: tuple[float, float]  # m,b
    high_fit: tuple[float, float]
    low_sse: float
    high_sse: float
    total_sse: float


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (m,b,sse) for y = m*x + b."""
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    y_hat = m * x + b
    sse = float(np.sum((y - y_hat) ** 2))
    return float(m), float(b), sse


def _tg_fit_series(*, density_kg_m3: np.ndarray, fit_metric: str) -> tuple[np.ndarray, np.ndarray]:
    with np.errstate(divide="raise", invalid="raise"):
        specific_volume_cm3_g = 1000.0 / density_kg_m3
    metric = str(fit_metric).strip().lower()
    if metric == "density":
        return density_kg_m3, specific_volume_cm3_g
    if metric == "specific_volume":
        return specific_volume_cm3_g, specific_volume_cm3_g
    raise ValueError(f"Unsupported Tg fit metric: {fit_metric}")


def fit_tg_piecewise_linear(
    temperatures_k: Sequence[float],
    density: Sequence[float],
    *,
    fit_metric: str = "density",
) -> TgResult:
    """Auto-split piecewise linear fit and compute Tg (intersection)."""
    T = np.asarray(temperatures_k, dtype=float)
    rho = np.asarray(density, dtype=float)
    if T.size < 5:
        raise ValueError("Need at least 5 temperature points for robust Tg fit")
    fit_values, specific_volume = _tg_fit_series(density_kg_m3=rho, fit_metric=fit_metric)

    best = None
    # enforce at least 2 points per side
    for k in range(2, T.size - 2):
        m1, b1, sse1 = _linear_fit(T[:k], fit_values[:k])
        m2, b2, sse2 = _linear_fit(T[k:], fit_values[k:])
        if abs(m1 - m2) < 1e-12:
            continue
        tg = (b2 - b1) / (m1 - m2)
        sse = sse1 + sse2
        # prefer Tg within the explored temperature range
        penalty = 0.0
        if tg < min(T) or tg > max(T):
            penalty = 1e6
        score = sse + penalty
        if best is None or score < best[0]:
            best = (score, k, tg, (m1, b1), (m2, b2), sse1, sse2)
    if best is None:
        raise ValueError("Failed to fit Tg (degenerate slopes)")
    _, k, tg, low_fit, high_fit, low_sse, high_sse = best
    return TgResult(
        temperatures_k=list(map(float, T.tolist())),
        density_mean=list(map(float, rho.tolist())),
        specific_volume_cm3_g=list(map(float, specific_volume.tolist())),
        fit_metric=str(fit_metric).strip().lower(),
        fit_values=list(map(float, fit_values.tolist())),
        split_index=int(k),
        tg_k=float(tg),
        low_fit=(float(low_fit[0]), float(low_fit[1])),
        high_fit=(float(high_fit[0]), float(high_fit[1])),
        low_sse=float(low_sse),
        high_sse=float(high_sse),
        total_sse=float(low_sse + high_sse),
    )


class TgJob:
    """Glass transition temperature scan (GROMACS-only).

    For each temperature point, run an NPT simulation and collect mean density
    (last fraction of the time series). Then fit Tg by automatic two-line split.
    """

    def __init__(
        self,
        *,
        gro: Path,
        top: Path,
        out_dir: Path,
        temperatures_k: Sequence[float],
        pressure_bar: float = 1.0,
        dt_ps: float = 0.002,
        npt_ns: float = 2.0,
        frac_last: float = 0.5,
        fit_metric: str = "density",
        runner: Optional[GromacsRunner] = None,
        resources: RunResources = RunResources(),
        auto_plot: bool = True,
    ):
        self.gro = gro
        self.top = top
        self.out_dir = out_dir
        self.temperatures_k = list(map(float, temperatures_k))
        self.pressure_bar = float(pressure_bar)
        self.dt_ps = float(dt_ps)
        self.npt_ns = float(npt_ns)
        self.frac_last = float(frac_last)
        self.fit_metric = str(fit_metric).strip().lower()
        self.runner = runner or GromacsRunner()
        self.resources = resources
        self.auto_plot = bool(auto_plot)

    def run(self, *, restart: Optional[bool] = None) -> Path:
        out = safe_mkdir(self.out_dir)
        rst_flag = resolve_restart(restart)
        summary_path = out / "summary.json"
        summary: dict = load_json(summary_path) or {
            "job": "TgJob",
            "out_dir": str(out),
            "prepared_system": {
                "gro": str(self.gro),
                "top": str(self.top),
            },
            "resources": {
                "ntmpi": int(self.resources.ntmpi) if self.resources.ntmpi is not None else None,
                "ntomp": int(self.resources.ntomp) if self.resources.ntomp is not None else None,
                "use_gpu": bool(self.resources.use_gpu),
                "gpu_id": self.resources.gpu_id,
            },
            "scan": {
                "temperatures_k": self.temperatures_k,
                "pressure_bar": float(self.pressure_bar),
                "dt_ps": float(self.dt_ps),
                "npt_ns": float(self.npt_ns),
                "frac_last": float(self.frac_last),
                "fit_metric": self.fit_metric,
            },
            "points": [],
        }

        p = default_mdp_params()
        p["dt"] = self.dt_ps
        p["ref_p"] = self.pressure_bar

        def ns_to_steps(ns: float) -> int:
            return int((ns * 1000.0) / float(p["dt"]))

        current_gro = self.gro
        current_cpt: Optional[Path] = None
        densities: list[float] = []
        temps: list[float] = []

        for i, T in enumerate(self.temperatures_k, start=1):
            tag = f"T{i:02d}_{int(round(T))}K"
            d = safe_mkdir(out / tag)
            deffnm = "md"
            stage_sum = d / "summary.json"

            if rst_flag and stage_sum.exists() and (d / f"{deffnm}.edr").exists():
                rec = load_json(stage_sum) or {}
                rho = rec.get("density", {}).get("mean")
                if rho is not None:
                    temps.append(float(T))
                    densities.append(float(rho))
                current_gro = d / f"{deffnm}.gro" if (d / f"{deffnm}.gro").exists() else current_gro
                current_cpt = d / f"{deffnm}.cpt" if (d / f"{deffnm}.cpt").exists() else current_cpt
                continue

            mdp = MdpSpec(
                NPT_MDP,
                {
                    **p,
                    "ref_t": float(T),
                    "ref_p": self.pressure_bar,
                    "nsteps": max(ns_to_steps(self.npt_ns), 1000),
                },
            ).write(d / "npt.mdp")

            tpr = d / f"{deffnm}.tpr"
            self.runner.grompp(mdp=mdp, gro=current_gro, top=self.top, out_tpr=tpr, cpt=current_cpt, cwd=d)
            self.runner.mdrun(
                tpr=tpr,
                deffnm=deffnm,
                cwd=d,
                ntomp=self.resources.ntomp,
                ntmpi=self.resources.ntmpi,
                use_gpu=bool(self.resources.use_gpu),
                gpu_id=self.resources.gpu_id,
                append=True,
            )

            # PBC hygiene (best-effort): keep molecules contiguous for downstream tools.
            pbc_mol_fix_inplace(self.runner, tpr=tpr, traj_or_gro=d / f"{deffnm}.gro", cwd=d)
            if (d / f"{deffnm}.xtc").exists():
                pbc_mol_fix_inplace(self.runner, tpr=tpr, traj_or_gro=d / f"{deffnm}.xtc", cwd=d)

            current_gro = d / f"{deffnm}.gro"
            current_cpt = d / f"{deffnm}.cpt" if (d / f"{deffnm}.cpt").exists() else None

            # Optional: export system-level MOL2 for this Tg point (best-effort).
            mol2_path = write_mol2_from_top_gro_parmed(
                top_path=self.top,
                gro_path=current_gro,
                out_mol2=d / f"{deffnm}.mol2",
                overwrite=True,
            )

            edr = d / f"{deffnm}.edr"
            xvg = d / "density.xvg"
            terms = ["Density", "Temperature", "Pressure", "Volume"]
            self.runner.energy_xvg(edr=edr, out_xvg=xvg, terms=terms, cwd=d, allow_missing=True)
            stats = summarize_terms_xvg(xvg=xvg, terms=terms, frac_last=self.frac_last)
            rho_stat = stats.get("Density")
            if rho_stat is None:
                raise RuntimeError("Density term not found in energy output")

            temps.append(float(T))
            densities.append(float(rho_stat.mean))

            rec = {
                "temperature_k": float(T),
                "dir": str(d),
                "density": rho_stat.__dict__,
                "specific_volume_cm3_g": (1000.0 / float(rho_stat.mean)) if float(rho_stat.mean) != 0.0 else None,
                "thermo": {k: v.__dict__ for k, v in stats.items()},
                "mol2": str(mol2_path) if mol2_path else None,
            }

            # --- auto plots (yzc-gmx-gen style)
            if self.auto_plot:
                plots_dir = d / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                # Density time-series
                den_svg = plot_density_time(xvg, out_svg=plots_dir / "density_time.svg", title=f"{tag} density")
                if den_svg is not None:
                    rec.setdefault("plots", {})["density_time_svg"] = str(den_svg)
                # Also include common thermo curves from the same XVG
                try:
                    rec.setdefault("plots", {}).update(plot_thermo_stage(xvg, out_dir=plots_dir, title_prefix=tag))
                except Exception:
                    pass

            atomic_write_json(stage_sum, rec)
            summary["points"].append(rec)

        # Tg fit
        tg = fit_tg_piecewise_linear(temps, densities, fit_metric=self.fit_metric)
        summary["fit"] = {
            "fit_metric": tg.fit_metric,
            "split_index": tg.split_index,
            "tg_k": tg.tg_k,
            "low_fit": {"m": tg.low_fit[0], "b": tg.low_fit[1]},
            "high_fit": {"m": tg.high_fit[0], "b": tg.high_fit[1]},
            "low_sse": tg.low_sse,
            "high_sse": tg.high_sse,
            "total_sse": tg.total_sse,
        }
        summary["curve"] = {
            "temperatures_k": tg.temperatures_k,
            "density_mean": tg.density_mean,
            "specific_volume_cm3_g": tg.specific_volume_cm3_g,
            "fit_values": tg.fit_values,
        }

        # Final Tg curve plot
        if self.auto_plot:
            plots_dir = out / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            tg_svg = plot_tg_curve(
                tg.temperatures_k,
                tg.density_mean,
                split_index=tg.split_index,
                low_fit=tg.low_fit,
                high_fit=tg.high_fit,
                tg_k=tg.tg_k,
                out_svg=plots_dir / "tg_density_vs_T.svg",
            )
            if tg_svg is not None:
                summary.setdefault("plots", {})["tg_curve_svg"] = str(tg_svg)

        # Write a CSV for convenience
        csv = out / "density_vs_T.csv"
        lines = ["T_K,density_kg_m3,specific_volume_cm3_g"]
        for T, rho, sv in zip(tg.temperatures_k, tg.density_mean, tg.specific_volume_cm3_g):
            lines.append(f"{T},{rho},{sv}")
        csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

        atomic_write_json(summary_path, summary)
        return summary_path
