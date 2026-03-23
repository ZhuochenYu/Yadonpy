"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


from ...runtime import resolve_restart
from ...io.mol2 import write_mol2_from_top_gro_parmed

from ..analysis.xvg import read_xvg
from ..analysis.auto_plot import _as_path  # small helper
from ..engine import GromacsRunner
from ..mdp_templates import DEFORM_NVT_MDP, MdpSpec, default_mdp_params
from ._util import RunResources, atomic_write_json, pbc_mol_fix_inplace, read_gro_box_nm, safe_mkdir


@dataclass(frozen=True)
class ElongationResult:
    strain: list[float]
    sigma_xx_gpa: list[float]
    sigma_dev_gpa: list[float]


class ElongationJob:
    """Uniaxial elongation using GROMACS `deform`.

    Notes:
    - GROMACS reports pressures in bar in the energy file.
    - We output `sigma_xx = -Pres-XX` (GPa) as a common tensile-stress sign convention.
    - We also output `sigma_dev = -(Pres-XX - (Pres-YY+Pres-ZZ)/2)` (GPa).
    """

    def __init__(
        self,
        *,
        gro: Path,
        top: Path,
        out_dir: Path,
        direction: Literal["x", "y", "z"] = "x",
        temperature_k: float = 298.15,
        dt_ps: float = 0.002,
        strain_rate_per_ps: float = 1e-5,
        final_strain: float = 0.2,
        frac_last: float = 1.0,
        runner: Optional[GromacsRunner] = None,
        resources: RunResources = RunResources(),
        auto_plot: bool = True,
    ):
        self.gro = gro
        self.top = top
        self.out_dir = out_dir
        self.direction = direction
        self.temperature_k = float(temperature_k)
        self.dt_ps = float(dt_ps)
        self.strain_rate_per_ps = float(strain_rate_per_ps)
        self.final_strain = float(final_strain)
        self.frac_last = float(frac_last)
        self.runner = runner or GromacsRunner()
        self.resources = resources
        self.auto_plot = bool(auto_plot)

    def run(self, *, restart: Optional[bool] = None) -> Path:
        out = safe_mkdir(self.out_dir)
        rst_flag = resolve_restart(restart)
        summary_path = out / "summary.json"
        if rst_flag and summary_path.exists() and (out / "md.edr").exists():
            return summary_path

        # Compute deform rate in nm/ps based on initial box length
        Lx, Ly, Lz = read_gro_box_nm(self.gro)
        deform_x = deform_y = deform_z = 0.0
        if self.direction == "x":
            deform_x = self.strain_rate_per_ps * Lx
        elif self.direction == "y":
            deform_y = self.strain_rate_per_ps * Ly
        else:
            deform_z = self.strain_rate_per_ps * Lz

        total_time_ps = self.final_strain / self.strain_rate_per_ps
        nsteps = int(max(total_time_ps / self.dt_ps, 1000))

        p = default_mdp_params()
        p["dt"] = self.dt_ps
        p["ref_t"] = self.temperature_k
        p["nsteps"] = nsteps
        p["deform_x"] = deform_x
        p["deform_y"] = deform_y
        p["deform_z"] = deform_z

        mdp = MdpSpec(DEFORM_NVT_MDP, p).write(out / "deform_nvt.mdp")
        tpr = out / "md.tpr"
        self.runner.grompp(mdp=mdp, gro=self.gro, top=self.top, out_tpr=tpr, cwd=out)
        self.runner.mdrun(
            tpr=tpr,
            deffnm="md",
            cwd=out,
            ntomp=self.resources.ntomp,
            ntmpi=self.resources.ntmpi,
            use_gpu=bool(self.resources.use_gpu),
            gpu_id=self.resources.gpu_id,
            append=True,
        )

        # PBC hygiene (best-effort): keep molecules contiguous for downstream tools.
        pbc_mol_fix_inplace(self.runner, tpr=tpr, traj_or_gro=out / "md.gro", cwd=out)
        # Optional: export system-level MOL2 (best-effort).
        mol2_path = write_mol2_from_top_gro_parmed(top_path=self.top, gro_path=out / "md.gro", out_mol2=out / "md.mol2", overwrite=True)
        if (out / "md.xtc").exists():
            pbc_mol_fix_inplace(self.runner, tpr=tpr, traj_or_gro=out / "md.xtc", cwd=out)

        # Extract box and pressure tensor
        edr = out / "md.edr"
        xvg = out / "stress_box.xvg"
        # 'Time' is implicit x-axis, so do not request it.
        self.runner.energy_xvg(
            edr=edr,
            out_xvg=xvg,
            terms=["Box-X", "Box-Y", "Box-Z", "Pres-XX", "Pres-YY", "Pres-ZZ"],
            cwd=out,
        )
        df = read_xvg(xvg).df
        L0 = float(df["Box-X"].iloc[0] if self.direction == "x" else df["Box-Y"].iloc[0] if self.direction == "y" else df["Box-Z"].iloc[0])
        L = df["Box-X"] if self.direction == "x" else df["Box-Y"] if self.direction == "y" else df["Box-Z"]
        strain = (L - L0) / L0

        # Convert bar -> GPa: 1 bar = 1e5 Pa; 1 GPa = 1e9 Pa => 1 bar = 1e-4 GPa
        bar_to_gpa = 1e-4
        pxx = df["Pres-XX"].to_numpy(dtype=float)
        pyy = df["Pres-YY"].to_numpy(dtype=float)
        pzz = df["Pres-ZZ"].to_numpy(dtype=float)
        sigma_xx = (-pxx) * bar_to_gpa
        sigma_dev = (-(pxx - 0.5 * (pyy + pzz))) * bar_to_gpa

        # Save CSV
        csv = out / "stress_strain.csv"
        lines = ["strain,sigma_xx_GPa,sigma_dev_GPa"]
        for e, s1, s2 in zip(strain.to_numpy(dtype=float), sigma_xx, sigma_dev):
            lines.append(f"{e},{s1},{s2}")
        csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Plot stress-strain (SVG)
        stress_svg = None
        if self.auto_plot:
            try:
                import matplotlib

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                from ...plotting.style import apply_matplotlib_style, golden_figsize
                from ...plotting.legend import place_legend

                apply_matplotlib_style()
                out_svg = _as_path(out / "stress_strain.svg")
                plt.figure(figsize=golden_figsize(8.0))
                plt.plot(strain.to_numpy(dtype=float), sigma_xx, label="sigma_xx")
                plt.plot(strain.to_numpy(dtype=float), sigma_dev, label="sigma_dev")
                plt.title("Stress–strain")
                plt.xlabel("Strain")
                plt.ylabel("Stress (GPa)")
                plt.grid(True)
                place_legend(plt.gca())
                plt.tight_layout()
                plt.savefig(out_svg, format="svg")
                plt.close()
                stress_svg = str(out_svg)
            except Exception:
                stress_svg = None

        summary = {
            "job": "ElongationJob",
            "out_dir": str(out),
            "direction": self.direction,
            "temperature_k": self.temperature_k,
            "dt_ps": self.dt_ps,
            "strain_rate_per_ps": self.strain_rate_per_ps,
            "final_strain": self.final_strain,
            "files": {
                "gro": str(out / "md.gro") if (out / "md.gro").exists() else None,
                "mol2": str(mol2_path) if mol2_path else None,
                "edr": str(edr) if edr.exists() else None,
                "csv": str(csv),
                "stress_strain_svg": stress_svg,
            },
            "results": {
                "strain": list(map(float, strain.to_numpy(dtype=float).tolist())),
                "sigma_xx_gpa": list(map(float, sigma_xx.tolist())),
                "sigma_dev_gpa": list(map(float, sigma_dev.tolist())),
            },
        }
        atomic_write_json(summary_path, summary)
        return summary_path
