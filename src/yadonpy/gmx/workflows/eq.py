"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Optional, Sequence

from ..analysis.plot import plot_xvg_svg, plot_xvg_split_svg
from ..analysis.thermo import bulk_modulus_gpa_from_kappa_t, cp_molar_from_enthalpy, kappa_t_from_volume, summarize_terms_xvg
from ..analysis.xvg import read_xvg
from ..engine import GromacsRunner
from ..mdp_templates import (
    MINIM_STEEP_MDP,
    MINIM_STEEP_HBONDS_MDP,
    MINIM_CG_MDP,
    NVT_MDP,
    NPT_MDP,
    MdpSpec,
    default_mdp_params,
)
from ._util import RunResources, atomic_write_json, load_json, pbc_mol_fix_inplace, safe_mkdir


@dataclass(frozen=True)
class EqStage:
    """One stage in a multi-stage equilibration."""

    name: str
    kind: str  # minim | nvt | npt | md
    mdp: MdpSpec


class EquilibrationJob:
    """Multi-stage equilibration workflow (GROMACS-only).

    Typical usage:
        stages = EquilibrationJob.default_stages(...)
        EquilibrationJob(..., stages=stages).run()
    """

    def __init__(
        self,
        *,
        gro: Path,
        top: Path,
        out_dir: Path,
        stages: Sequence[EqStage],
        runner: Optional[GromacsRunner] = None,
        resources: RunResources = RunResources(),
        frac_last: float = 0.5,
    ):
        self.gro = gro
        self.top = top
        self.out_dir = out_dir
        self.stages = list(stages)
        self.runner = runner or GromacsRunner()
        self.resources = resources
        self.frac_last = frac_last

    def _log(self, msg: str) -> None:
        """Lightweight logger for this workflow.

        Some helpers inside `run()` emit informational messages. We route all
        logging through the underlying `GromacsRunner` so verbosity is
        controlled consistently.
        """
        try:
            self.runner._log(msg)
        except Exception:
            # Last-resort: never fail the workflow due to logging.
            print(msg)

    @staticmethod
    def default_stages(
        *,
        temperature_k: float = 298.15,
        pressure_bar: float = 1.0,
        dt_ps: float = 0.002,
        nvt_ns: float = 0.2,
        npt_ns: float = 0.5,
        prod_ns: float = 1.0,
    ) -> list[EqStage]:
        p = default_mdp_params()
        p["dt"] = dt_ps
        p["ref_t"] = temperature_k
        p["ref_p"] = pressure_bar

        def ns_to_steps(ns: float) -> int:
            return int((ns * 1000.0) / float(p["dt"]))

        stages: list[EqStage] = []
        # Follow yzc-gmx-gen: do a robust steep/none minimization first, then a CG minimization
        # with h-bonds constraints. The workflow keeps the stage directory name as 01_minim and
        # produces the final minimized structure as 01_minim/md.gro.
        stages.append(
            EqStage(
                "01_minim",
                "minim",
                # For CG EM, use a smaller step size by default to reduce constraint rotation.
                # (steep stages override emstep explicitly)
                MdpSpec(MINIM_CG_MDP, {**p, "nsteps": 50000, "emtol": 1000.0, "emstep": 0.001}),
            )
        )
        stages.append(
            EqStage(
                "02_nvt",
                "nvt",
                MdpSpec(
                    NVT_MDP,
                    {
                        **p,
                        "nsteps": max(ns_to_steps(nvt_ns), 1000),
                        "ref_t": temperature_k,
                    },
                ),
            )
        )

        # Per user request:
        #   - Completely disable Parrinello–Rahman (PR) pressure coupling.
        #   - Use the modern stochastic cell-rescaling barostat (C-rescale;
        #     Bernetti & Bussi 2020) for *all* NPT/production-like stages.
        #
        # We follow the "soft -> normal" idea by using a larger tau_p during
        # the first NPT stage.
        stages.append(
            EqStage(
                "03_npt",
                "npt",
                MdpSpec(
                    NPT_MDP,
                    {
                        **p,
                        "pcoupl": "C-rescale",
                        "tau_p": 10.0,
                        "nsteps": max(ns_to_steps(npt_ns), 1000),
                        "ref_t": temperature_k,
                        "ref_p": pressure_bar,
                    },
                ),
            )
        )

        stages.append(
            EqStage(
                "04_md",
                "md",
                MdpSpec(
                    NPT_MDP,
                    {
                        **p,
                        "pcoupl": "C-rescale",
                        "tau_p": float(p.get("tau_p", 2.0)),
                        "nsteps": max(ns_to_steps(prod_ns), 1000),
                        "ref_t": temperature_k,
                        "ref_p": pressure_bar,
                    },
                ),
            )
        )
        return stages

    def run(self, *, restart: bool = True) -> Path:
        out = safe_mkdir(self.out_dir)
        summary_path = out / "summary.json"
        summary: dict = load_json(summary_path) or {"job": "EquilibrationJob", "out_dir": str(out), "stages": []}

        current_gro = self.gro
        current_cpt: Optional[Path] = None

        def _choose_threads(stage_gpu: bool) -> tuple[Optional[int], Optional[int]]:
            """Choose (ntomp, ntmpi) for a stage.

            Heuristic: when using a single GPU, running many thread-MPI ranks
            on the same GPU often causes excessive CUDA context/memory use and
            instability. We therefore collapse to ntmpi=1 and multiply ntomp
            to keep total CPU threads constant.
            """
            ntomp = self.resources.ntomp
            ntmpi = self.resources.ntmpi
            if stage_gpu and ntmpi is not None and int(ntmpi) > 1:
                base_omp = int(ntomp) if ntomp is not None else 1
                total_threads = int(ntmpi) * base_omp
                self._log(f"[INFO] GPU stage: overriding -ntmpi {ntmpi} -> 1 and -ntomp {base_omp} -> {total_threads}.")
                return total_threads, 1
            return ntomp, ntmpi

        def _mdrun_em_minimal(*, deffnm: str, cwd: Path, ntomp: int, gpu_id: Optional[str]) -> None:
            """Run energy minimization with a minimal, user-friendly mdrun command.

            By design we keep EM command-line args lean to improve portability and
            avoid surprising GPU/PME/update offload settings. Requested default:
              gmx mdrun -deffnm <name> -ntmpi 1 -ntomp <xx> -nb gpu -gpu_id <id> -v
            """
            args = ["mdrun", "-deffnm", str(deffnm)]
            # Always add -v when supported for live progress.
            if self.runner._tool_has_option("mdrun", "-v", cwd=cwd):
                args += ["-v"]
            # Thread-MPI layout: EM runs as a single MPI rank.
            if self.runner._tool_has_option("mdrun", "-ntmpi", cwd=cwd):
                args += ["-ntmpi", "1"]
            if self.runner._tool_has_option("mdrun", "-ntomp", cwd=cwd):
                args += ["-ntomp", str(int(ntomp))]
            # Nonbonded on GPU (if available in this GROMACS build).
            if self.runner._tool_has_option("mdrun", "-nb", cwd=cwd):
                args += ["-nb", "gpu"]
            if gpu_id is not None and str(gpu_id).strip() != "" and self.runner._tool_has_option("mdrun", "-gpu_id", cwd=cwd):
                args += ["-gpu_id", str(gpu_id).strip()]
            rc, tail = self.runner._run_capture_tee(args, cwd=cwd)
            if rc != 0:
                raise RuntimeError(f"GROMACS mdrun (EM) failed (rc={rc})\n  cwd: {cwd}\n  cmd: {' '.join(args)}\n  tail:\n{tail}\n")

        for st in self.stages:
            stage_dir = safe_mkdir(out / st.name)
            deffnm = "md"  # keep consistent within stage dir
            stage_summary_path = stage_dir / "summary.json"

            out_gro = stage_dir / f"{deffnm}.gro"
            out_cpt = stage_dir / f"{deffnm}.cpt"
            out_tpr = stage_dir / f"{deffnm}.tpr"

            stage_has_outputs = bool(out_gro.exists())

            # ----------------------
            # GPU policy per stage
            # ----------------------
            # GROMACS cannot offload PME on GPU for non-dynamical integrators
            # (e.g., steepest descent minimization). In practice, "em" should
            # run on CPU only, while NVT/NPT/MD can use GPU offload.
            stage_use_gpu = bool(self.resources.use_gpu) and (st.kind not in ("minim", "em"))

            # ----------------------
            # Robust stage restart logic
            # ----------------------
            # 1) Stage complete: keep outputs and move on.
            if restart and out_gro.exists():
                current_gro = out_gro
                current_cpt = out_cpt if out_cpt.exists() else None
                # If summary exists, the postprocess was already done.
                if stage_summary_path.exists():
                    continue
                # Otherwise, fall through to regenerate the summary only (no rerun).
                stage_has_outputs = True

            # 2) Stage interrupted but checkpoint exists: continue without re-running grompp.
            #    This is the standard GROMACS restart mode: mdrun -cpi md.cpt -append.
            if (not stage_has_outputs) and restart and out_tpr.exists() and out_cpt.exists() and (not out_gro.exists()):
                ntomp_sel, ntmpi_sel = _choose_threads(stage_use_gpu)
                self.runner.mdrun(
                    tpr=out_tpr,
                    deffnm=deffnm,
                    cwd=stage_dir,
                    ntomp=ntomp_sel,
                    ntmpi=ntmpi_sel,
                    use_gpu=stage_use_gpu,
                    nb=("gpu" if st.kind in ("minim", "em") else None),
                    gpu_id=self.resources.gpu_id,
                    append=True,
                    cpi=out_cpt,
                )
                current_gro = out_gro
                current_cpt = out_cpt if out_cpt.exists() else None
            else:
                # 3) Fresh stage run: if stale partial outputs exist, remove them to avoid
                #    confusing GROMACS (e.g., "-append" to an incompatible file set).
                if (not stage_has_outputs) and restart:
                    for p in stage_dir.glob(f"{deffnm}.*"):
                        try:
                            p.unlink()
                        except Exception:
                            pass

                if not stage_has_outputs:
                    # Special-case minimization: mirror yzc-gmx-gen with a robust
                    # steepest-descent (constraints=none) pre-minimization followed by
                    # the main CG minimization (constraints=h-bonds). We keep the final
                    # outputs as md.* in the stage directory for backward compatibility.
                    gro_for_main = current_gro

                    if st.kind in ("minim", "em"):
                        steep_deffnm = "md_steep"
                        steep_gro = stage_dir / f"{steep_deffnm}.gro"
                        steep_tpr = stage_dir / f"{steep_deffnm}.tpr"
                        if not steep_gro.exists():
                            steep_mdp = MdpSpec(
                                MINIM_STEEP_MDP,
                                {**st.mdp.params, "nsteps": 50000, "emtol": 1000.0, "emstep": 0.01},
                            ).write(stage_dir / "steep.mdp")
                            self.runner.grompp(
                                mdp=steep_mdp,
                                gro=current_gro,
                                top=self.top,
                                out_tpr=steep_tpr,
                                cpt=current_cpt,
                                cwd=stage_dir,
                            )
                            # Energy minimization: minimal mdrun args for portability.
                            ntomp_sel, _ = _choose_threads(False)
                            _mdrun_em_minimal(
                                deffnm=steep_deffnm,
                                cwd=stage_dir,
                                ntomp=int(ntomp_sel or 1),
                                gpu_id=self.resources.gpu_id,
                            )

                        # Bridge minimization (steep, constraints=h-bonds).
                        # This step is tolerant to constraint problems that CG cannot handle,
                        # and dramatically improves robustness for rough packed structures.
                        bridge_deffnm = "md_steep_hbonds"
                        bridge_gro = stage_dir / f"{bridge_deffnm}.gro"
                        bridge_tpr = stage_dir / f"{bridge_deffnm}.tpr"
                        if not bridge_gro.exists():
                            bridge_mdp = MdpSpec(
                                MINIM_STEEP_HBONDS_MDP,
                                {
                                    **st.mdp.params,
                                    "nsteps": 50000,
                                    "emtol": 1000.0,
                                    # Use a smaller step size to avoid large bond rotations.
                                    "emstep": 0.001,
                                },
                            ).write(stage_dir / "steep_hbonds.mdp")
                            self.runner.grompp(
                                mdp=bridge_mdp,
                                gro=steep_gro if steep_gro.exists() else current_gro,
                                top=self.top,
                                out_tpr=bridge_tpr,
                                cpt=None,
                                cwd=stage_dir,
                            )
                            # Energy minimization (bridge): minimal mdrun args.
                            ntomp_sel, _ = _choose_threads(False)
                            _mdrun_em_minimal(
                                deffnm=bridge_deffnm,
                                cwd=stage_dir,
                                ntomp=int(ntomp_sel or 1),
                                gpu_id=self.resources.gpu_id,
                            )

                        # Main minimization starts from the bridge result if available.
                        gro_for_main = bridge_gro if bridge_gro.exists() else (steep_gro if steep_gro.exists() else current_gro)

                    mdp_path = st.mdp.write(stage_dir / f"{st.kind}.mdp")
                    tpr = out_tpr
                    self.runner.grompp(
                        mdp=mdp_path,
                        gro=gro_for_main,
                        top=self.top,
                        out_tpr=tpr,
                        cpt=current_cpt,
                        cwd=stage_dir,
                    )
                    ntomp_sel, ntmpi_sel = _choose_threads(stage_use_gpu)
                    try:
                        self.runner.mdrun(
                            tpr=tpr,
                            deffnm=deffnm,
                            cwd=stage_dir,
                            ntomp=ntomp_sel,
                            ntmpi=ntmpi_sel,
                            use_gpu=stage_use_gpu,
                            nb=("gpu" if st.kind in ("minim", "em") else None),
                            gpu_id=self.resources.gpu_id,
                            append=True,
                        )
                    except Exception as e:
                        # If CG fails due to constraint problems (common for rough packed systems),
                        # fall back to a steep minimization with constraints in the same stage so
                        # the workflow can proceed.
                        msg = str(e)
                        if st.kind in ("minim", "em") and ("Minimizer 'cg' can not handle" in msg or "constraint failures" in msg or "LINCS" in msg):
                            self._log("[WARN] CG minimization failed due to constraint issues. Falling back to steep (constraints=h-bonds).")
                            # Overwrite md.tpr with a steep/h-bonds minim and run as deffnm=md.
                            fb_mdp = MdpSpec(
                                MINIM_STEEP_HBONDS_MDP,
                                {
                                    **st.mdp.params,
                                    "nsteps": int(max(100000, int(st.mdp.params.get("nsteps", 50000)))),
                                    "emtol": float(st.mdp.params.get("emtol", 1000.0)),
                                    "emstep": 0.001,
                                },
                            ).write(stage_dir / "minim_fallback_steep_hbonds.mdp")
                            # Re-run grompp to the standard out_tpr (md.tpr)
                            self.runner.grompp(
                                mdp=fb_mdp,
                                gro=gro_for_main,
                                top=self.top,
                                out_tpr=tpr,
                                cpt=None,
                                cwd=stage_dir,
                            )
                            # Energy minimization fallback: minimal mdrun args.
                            ntomp_sel, _ = _choose_threads(False)
                            _mdrun_em_minimal(
                                deffnm=deffnm,
                                cwd=stage_dir,
                                ntomp=int(ntomp_sel or 1),
                                gpu_id=self.resources.gpu_id,
                            )
                        else:
                            raise

                    # Update pointers
                    current_gro = out_gro
                    current_cpt = out_cpt if out_cpt.exists() else None


            # ----------------------
            # PBC hygiene for visualization / downstream conversion
            # ----------------------
            # A common source of "broken bonds" in viewers after NVT/NPT is simply
            # PBC-wrapped coordinates (molecules split across box). Bonds are not
            # actually breaking in GROMACS, but downstream tools (mol2 viewers,
            # OpenBabel, etc.) may infer impossible bonding if coordinates are wrapped.
            #
            # We therefore rewrite BOTH the final stage structure and trajectory with:
            #   gmx trjconv -pbc mol -center -ur compact
            # in-place (md.gro/md.xtc). Best-effort only.
            pbc_gro = pbc_mol_fix_inplace(self.runner, tpr=out_tpr, traj_or_gro=out_gro, cwd=stage_dir)
            pbc_xtc = {"applied": False, "error": None}
            out_xtc = stage_dir / f"{deffnm}.xtc"
            if out_xtc.exists():
                pbc_xtc = pbc_mol_fix_inplace(self.runner, tpr=out_tpr, traj_or_gro=out_xtc, cwd=stage_dir)

            # Postprocess (thermo)
            stage_record: dict = {
                "name": st.name,
                "kind": st.kind,
                "dir": str(stage_dir),
                "gro": str(current_gro),
                "edr": str(stage_dir / f"{deffnm}.edr") if (stage_dir / f"{deffnm}.edr").exists() else None,
                "pbc_mol": {
                    "gro": pbc_gro,
                    "xtc": pbc_xtc,
                },
            }

            edr = stage_dir / f"{deffnm}.edr"
            if edr.exists():
                xvg = stage_dir / "thermo.xvg"
                # Request a broad set of terms. Missing terms will be silently skipped by the summarizer.
                terms = [
                    "Temperature",
                    "Pressure",
                    "Density",
                    "Volume",
                    "Potential",
                    "Kinetic En.",
                    "Total Energy",
                    "Enthalpy",
                ]
                try:
                    res_energy = self.runner.energy_xvg(edr=edr, out_xvg=xvg, terms=terms, cwd=stage_dir, allow_missing=True)
                    stage_record["thermo_missing_terms"] = res_energy.get("missing_terms")
                    thermo_stats = summarize_terms_xvg(xvg=xvg, terms=terms, frac_last=self.frac_last)
                    stage_record["thermo"] = {k: v.__dict__ for k, v in thermo_stats.items()}

                    # Plots (SVG-first)
                    try:
                            plots_dir = stage_dir / "plots"
                            plots_dir.mkdir(parents=True, exist_ok=True)
                            plot_xvg_svg(xvg, out_svg=plots_dir / "thermo.svg", title=f"{stage_dir.name} thermo")
                            # Split per term
                            plot_xvg_split_svg(xvg, out_dir=plots_dir, title_prefix=f"{stage_dir.name}")
                    except Exception as _pe:
                        stage_record["thermo_plot_warning"] = str(_pe)

                    # Fluctuation properties (best-effort)
                    df = read_xvg(xvg).df
                    if "Volume" in df.columns and "Temperature" in df.columns:
                        v_nm3 = df["Volume"].to_numpy(dtype=float)
                        # Use target temperature from stats if present; else fallback to mdp setting
                        t_mean = float(thermo_stats.get("Temperature").mean) if thermo_stats.get("Temperature") else None
                        if t_mean is not None:
                            stage_record["kappa_t_1_pa"] = float(kappa_t_from_volume(v_nm3, t_mean, frac_last=self.frac_last))
                            stage_record["bulk_modulus_gpa"] = float(bulk_modulus_gpa_from_kappa_t(stage_record["kappa_t_1_pa"]))

                    if "Enthalpy" in df.columns and "Temperature" in df.columns:
                        # Cp in J/(mol*K), assuming Enthalpy is in kJ/mol (GROMACS default).
                        t_mean = float(thermo_stats.get("Temperature").mean) if thermo_stats.get("Temperature") else None
                        if t_mean is not None:
                            stage_record["cp_molar_j_mol_k"] = float(cp_molar_from_enthalpy(df["Enthalpy"].to_numpy(dtype=float), t_mean, frac_last=self.frac_last))
                except Exception as e:
                    stage_record["thermo_error"] = str(e)

            atomic_write_json(stage_summary_path, stage_record)
            summary["stages"].append(stage_record)

        atomic_write_json(summary_path, summary)
        return summary_path

    # ----------------------
    # Convenience getters
    # ----------------------
    def _stage_dirs(self) -> list[Path]:
        return [self.out_dir / st.name for st in self.stages]

    def final_stage_dir(self) -> Path:
        for d in reversed(self._stage_dirs()):
            if (d / "md.tpr").exists():
                return d
            if (d / "md.gro").exists():
                return d
        return self._stage_dirs()[-1] if self.stages else self.out_dir

    def final_gro(self) -> Path:
        d = self.final_stage_dir()
        p = d / "md.gro"
        if not p.exists():
            raise FileNotFoundError(f"Final gro not found under {d}")
        return p

    def final_outputs(self) -> tuple[Path, Path, Path]:
        """Return (tpr, xtc, edr) for the final stage."""
        d = self.final_stage_dir()
        tpr = d / "md.tpr"
        xtc = d / "md.xtc"
        edr = d / "md.edr"
        if not tpr.exists():
            raise FileNotFoundError(f"Missing {tpr}")
        if not xtc.exists():
            raise FileNotFoundError(f"Missing {xtc}")
        if not edr.exists():
            raise FileNotFoundError(f"Missing {edr}")
        return tpr, xtc, edr


    def final_trr(self) -> Path:
        """Return trajectory with velocities (TRR) for the final stage, if present."""
        d = self.final_stage_dir()
        p = d / "md.trr"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p} (enable nstxout/nstvout in mdp to write TRR)")
        return p