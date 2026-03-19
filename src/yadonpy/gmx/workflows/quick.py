"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..analysis.plot import plot_xvg_svg, plot_xvg_split_svg
from ..analysis.thermo import summarize_terms_xvg
from ..engine import GromacsRunner
from ..mdp_templates import (
    MINIM_STEEP_MDP,
    MINIM_STEEP_HBONDS_MDP,
    MINIM_CG_MDP,
    NVT_MDP,
    MdpSpec,
    default_mdp_params,
)
from ._util import RunResources, atomic_write_json, load_json, pbc_mol_fix_inplace, safe_mkdir


@dataclass(frozen=True)
class QuickRelaxResult:
    out_dir: Path
    gro: Path
    cpt: Optional[Path]
    edr: Optional[Path]
    xtc: Optional[Path]
    summary_json: Path


class QuickRelaxJob:
    """YadonPy-style quick-min / quick-md implemented with GROMACS."""

    def __init__(
        self,
        *,
        gro: Path,
        top: Path,
        out_dir: Path,
        runner: Optional[GromacsRunner] = None,
        resources: RunResources = RunResources(),
        do_quick_md: bool = True,
        quick_md_ns: float = 0.05,
        temperature_k: float = 298.15,
        frac_last: float = 0.5,
    ):
        self.gro = gro
        self.top = top
        self.out_dir = out_dir
        self.runner = runner or GromacsRunner()
        self.resources = resources
        self.do_quick_md = do_quick_md
        self.quick_md_ns = quick_md_ns
        self.temperature_k = temperature_k
        self.frac_last = frac_last

    def run(self, *, restart: bool = True) -> QuickRelaxResult:
        out = safe_mkdir(self.out_dir)
        summary_path = out / "summary.json"

        if restart:
            existing = load_json(summary_path)
            if existing and (out / "quick.gro").exists():
                return QuickRelaxResult(
                    out_dir=out,
                    gro=out / "quick.gro",
                    cpt=(out / "quick.cpt") if (out / "quick.cpt").exists() else None,
                    edr=(out / "quick.edr") if (out / "quick.edr").exists() else None,
                    xtc=(out / "quick.xtc") if (out / "quick.xtc").exists() else None,
                    summary_json=summary_path,
                )

        params = default_mdp_params()
        # ---------- minim (robust: steep/none -> steep/h-bonds -> cg w/ fallback) ----------
        # Non-dynamical integrators should run on CPU (no GPU offload).
        # 1) steep (constraints=none)
        em1_mdp = MdpSpec(
            MINIM_STEEP_MDP,
            {**params, "nsteps": 50000, "emtol": 1000.0, "emstep": 0.01},
        ).write(out / "em_steep.mdp")
        em1_tpr = out / "em_steep.tpr"
        self.runner.grompp(mdp=em1_mdp, gro=self.gro, top=self.top, out_tpr=em1_tpr, cwd=out)
        self.runner.mdrun(
            tpr=em1_tpr,
            deffnm="em_steep",
            cwd=out,
            ntomp=self.resources.ntomp,
            ntmpi=self.resources.ntmpi,
            use_gpu=False,
            nb="gpu",
            gpu_id=None,
            append=True,
        )

        # 2) steep (constraints=h-bonds)
        em2_mdp = MdpSpec(
            MINIM_STEEP_HBONDS_MDP,
            {**params, "nsteps": 50000, "emtol": 1000.0, "emstep": 0.001},
        ).write(out / "em_steep_hbonds.mdp")
        em2_tpr = out / "em_steep_hbonds.tpr"
        self.runner.grompp(mdp=em2_mdp, gro=out / "em_steep.gro", top=self.top, out_tpr=em2_tpr, cwd=out)
        self.runner.mdrun(
            tpr=em2_tpr,
            deffnm="em_steep_hbonds",
            cwd=out,
            ntomp=self.resources.ntomp,
            ntmpi=self.resources.ntmpi,
            use_gpu=False,
            nb="gpu",
            gpu_id=None,
            append=True,
        )

        # 3) CG (final); fallback to steep/h-bonds if CG fails due to constraint issues.
        minim_mdp = MdpSpec(
            MINIM_CG_MDP,
            {**params, "nsteps": 50000, "emtol": 1000.0, "emstep": 0.001},
        ).write(out / "minim.mdp")
        minim_tpr = out / "minim.tpr"
        self.runner.grompp(mdp=minim_mdp, gro=out / "em_steep_hbonds.gro", top=self.top, out_tpr=minim_tpr, cwd=out)
        try:
            self.runner.mdrun(
                tpr=minim_tpr,
                deffnm="minim",
                cwd=out,
                ntomp=self.resources.ntomp,
                ntmpi=self.resources.ntmpi,
                use_gpu=False,
                nb="gpu",
                gpu_id=None,
                append=True,
            )
        except Exception as e:
            msg = str(e)
            if "Minimizer 'cg' can not handle" in msg or "constraint failures" in msg or "LINCS" in msg:
                # Overwrite minim.tpr with steep/h-bonds and run as minim.* so downstream steps keep working.
                fb_mdp = MdpSpec(
                    MINIM_STEEP_HBONDS_MDP,
                    {**params, "nsteps": 100000, "emtol": 1000.0, "emstep": 0.001},
                ).write(out / "minim_fallback_steep_hbonds.mdp")
                self.runner.grompp(mdp=fb_mdp, gro=out / "em_steep_hbonds.gro", top=self.top, out_tpr=minim_tpr, cwd=out)
                self.runner.mdrun(
                    tpr=minim_tpr,
                    deffnm="minim",
                    cwd=out,
                    ntomp=self.resources.ntomp,
                    ntmpi=self.resources.ntmpi,
                    use_gpu=False,
                    nb="gpu",
                    gpu_id=None,
                    append=True,
                )
            else:
                raise
        # PBC hygiene (best-effort): keep molecules contiguous for downstream tools.
        pbc_mol_fix_inplace(self.runner, tpr=minim_tpr, traj_or_gro=out / "minim.gro", cwd=out)
        if (out / "minim.xtc").exists():
            pbc_mol_fix_inplace(self.runner, tpr=minim_tpr, traj_or_gro=out / "minim.xtc", cwd=out)
        current_gro = out / "minim.gro"
        current_cpt = (out / "minim.cpt") if (out / "minim.cpt").exists() else None

        # ---------- quick md (nvt) ----------
        if self.do_quick_md:
            nsteps = int((self.quick_md_ns * 1000.0) / params["dt"])  # ns->ps / dt(ps)
            nvt_mdp = MdpSpec(
                NVT_MDP,
                {
                    **params,
                    "nsteps": max(nsteps, 1000),
                    "ref_t": self.temperature_k,
                },
            ).write(out / "quick.mdp")
            tpr = out / "quick.tpr"
            self.runner.grompp(mdp=nvt_mdp, gro=current_gro, top=self.top, out_tpr=tpr, cpt=current_cpt, cwd=out)
            self.runner.mdrun(
                tpr=tpr,
                deffnm="quick",
                cwd=out,
                ntomp=self.resources.ntomp,
                ntmpi=self.resources.ntmpi,
                use_gpu=bool(self.resources.use_gpu),
                gpu_id=self.resources.gpu_id,
                append=True,
            )
            # PBC hygiene (best-effort)
            pbc_mol_fix_inplace(self.runner, tpr=tpr, traj_or_gro=out / "quick.gro", cwd=out)
            if (out / "quick.xtc").exists():
                pbc_mol_fix_inplace(self.runner, tpr=tpr, traj_or_gro=out / "quick.xtc", cwd=out)
            current_gro = out / "quick.gro"
            current_cpt = (out / "quick.cpt") if (out / "quick.cpt").exists() else None

        # ---------- summary ----------
        summary: dict = {
            "job": "QuickRelaxJob",
            "out_dir": str(out),
            "outputs": {
                "gro": str(current_gro),
                "cpt": str(current_cpt) if current_cpt else None,
                "edr": str(out / ("quick.edr" if self.do_quick_md else "minim.edr"))
                if (out / ("quick.edr" if self.do_quick_md else "minim.edr")).exists()
                else None,
            },
        }

        edr = out / ("quick.edr" if self.do_quick_md else "minim.edr")
        if edr.exists():
            xvg = out / "thermo.xvg"
            res_energy = self.runner.energy_xvg(edr=edr, out_xvg=xvg, terms=["Temperature", "Pressure", "Density", "Volume"], cwd=out, allow_missing=True)
            summary["thermo_missing_terms"] = res_energy.get("missing_terms")
            summary["thermo"] = {
                k: v.__dict__
                for k, v in summarize_terms_xvg(
                    xvg=xvg,
                    terms=["Temperature", "Pressure", "Density", "Volume"],
                    frac_last=self.frac_last,
                ).items()
            }

            # Plots (SVG-first)
            try:
                    plots_dir = out / "plots"
                    plots_dir.mkdir(parents=True, exist_ok=True)
                    plot_xvg_svg(xvg, out_svg=plots_dir / "thermo.svg", title="quick thermo")
                    plot_xvg_split_svg(xvg, out_dir=plots_dir, title_prefix="quick")
            except Exception as _pe:
                summary["thermo_plot_warning"] = str(_pe)

        atomic_write_json(summary_path, summary)
        return QuickRelaxResult(
            out_dir=out,
            gro=current_gro,
            cpt=current_cpt,
            edr=(out / ("quick.edr" if self.do_quick_md else "minim.edr")) if edr.exists() else None,
            xtc=(out / ("quick.xtc" if self.do_quick_md else "minim.xtc")) if (out / ("quick.xtc" if self.do_quick_md else "minim.xtc")).exists() else None,
            summary_json=summary_path,
        )
