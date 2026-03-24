"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ...io.mol2 import write_mol2_from_top_gro_parmed
from ...runtime import resolve_restart
from ..analysis.auto_plot import plot_thermo_stage
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
from ._util import RunResources, atomic_write_json, load_json, normalize_gro_molecules_inplace, pbc_mol_fix_inplace, safe_mkdir


def _file_signature(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        st = Path(path).stat()
        return {"path": str(path), "size": int(st.st_size), "mtime": float(st.st_mtime)}
    except FileNotFoundError:
        return {"path": str(path), "missing": True}


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
        ndx: Optional[Path] = None,
        provenance_ndx: Optional[Path] = None,
        out_dir: Path,
        stages: Sequence[EqStage],
        runner: Optional[GromacsRunner] = None,
        resources: RunResources = RunResources(),
        frac_last: float = 0.5,
    ):
        self.gro = gro
        self.top = top
        self.ndx = ndx
        self.provenance_ndx = provenance_ndx if provenance_ndx is not None else ndx
        self.out_dir = out_dir
        self.stages = list(stages)
        self.runner = runner or GromacsRunner()
        self.resources = resources
        self.frac_last = frac_last

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        seconds = float(max(0.0, seconds))
        if seconds < 60.0:
            return f"{seconds:.1f}s"
        minutes, sec = divmod(seconds, 60.0)
        if minutes < 60.0:
            return f"{int(minutes)}m {sec:.1f}s"
        hours, minutes = divmod(minutes, 60.0)
        return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"

    @staticmethod
    def _compact_path(path_like: object) -> str:
        try:
            p = Path(path_like)
            parts = list(p.parts)
            if len(parts) <= 4:
                return str(p)
            return str(Path(*parts[-4:]))
        except Exception:
            return str(path_like)

    @staticmethod
    def _gro_box_lengths_nm(gro: Path) -> tuple[float, float, float] | None:
        try:
            lines = [line.strip() for line in Path(gro).read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                return None
            raw = lines[-1].split()
            vals = [abs(float(token)) for token in raw]
        except Exception:
            return None
        if len(vals) >= 9:
            ax, by, cz, ay, az, bx, bz, cx, cy = vals[:9]
            avec = (ax, ay, az)
            bvec = (bx, by, bz)
            cvec = (cx, cy, cz)
            return (
                math.sqrt(sum(v * v for v in avec)),
                math.sqrt(sum(v * v for v in bvec)),
                math.sqrt(sum(v * v for v in cvec)),
            )
        if len(vals) >= 3:
            return (vals[0], vals[1], vals[2])
        return None

    @staticmethod
    def _gro_atom_count(gro: Path) -> int | None:
        try:
            lines = Path(gro).read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return None
        if len(lines) < 2:
            return None
        try:
            return int(lines[1].strip())
        except Exception:
            return None

    @staticmethod
    def _cleanup_stage_backups(stage_dir: Path, *, deffnm: str = "md") -> None:
        for artifact in stage_dir.glob(f"#{deffnm}*#"):
            try:
                artifact.unlink()
            except Exception:
                pass

    @staticmethod
    def _apply_box_safe_cutoffs(mdp: MdpSpec, *, gro: Path) -> tuple[MdpSpec, dict[str, object] | None]:
        box_nm = EquilibrationJob._gro_box_lengths_nm(gro)
        if not box_nm:
            return mdp, None
        min_box_nm = min(box_nm)
        if min_box_nm <= 0.0:
            return mdp, None

        cutoff_cap_nm = round(min_box_nm * 0.45, 4)
        if cutoff_cap_nm <= 0.0:
            return mdp, None

        params = dict(mdp.params)
        adjusted: dict[str, dict[str, float]] = {}
        for key in ("rlist", "rcoulomb", "rvdw"):
            value = params.get(key)
            try:
                current = float(value)
            except Exception:
                continue
            if current > cutoff_cap_nm:
                params[key] = cutoff_cap_nm
                adjusted[key] = {"old": current, "new": cutoff_cap_nm}

        if not adjusted:
            return mdp, None

        return (
            MdpSpec(mdp.template, params),
            {
                "box_nm": [round(v, 4) for v in box_nm],
                "min_box_nm": round(min_box_nm, 4),
                "cutoff_cap_nm": cutoff_cap_nm,
                "cutoffs": adjusted,
            },
        )

    @staticmethod
    def _detect_invalid_minimization(stage_dir: Path, *, deffnm: str = "md") -> str | None:
        log_path = Path(stage_dir) / f"{deffnm}.log"
        if not log_path.exists():
            return None
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            return f"failed to read minimization log: {exc}"

        for raw in text.lower().splitlines():
            line = raw.strip()
            if "force on at least one atom is not finite" in line:
                return "GROMACS reported a non-finite force"
            if "atoms are overlapping" in line:
                return "GROMACS reported overlapping atoms"
            if line.startswith("maximum force") and "inf" in line:
                return "maximum force became infinite"
            if line.startswith("norm of force") and "inf" in line:
                return "force norm became infinite"
            if line.startswith("potential energy") and "inf" in line:
                return "potential energy became infinite"
        return None

    def _section(self, title: str, detail: str | None = None) -> None:
        self._log('=' * 78)
        self._log(f"[SECTION] {title}")
        if detail:
            self._log(f"[NOTE] {detail}")

    def _item(self, label: str, value: object) -> None:
        self._log(f"[ITEM] {label:<18}: {value}")

    def _stage_begin(self, idx: int, total: int, stage: EqStage, stage_dir: Path) -> float:
        frac = 100.0 * float(idx) / float(max(total, 1))
        self._log('-' * 78)
        self._log(f"[STEP] Stage {idx}/{total} | {stage.name} ({stage.kind}) | {frac:5.1f}%")
        self._log(f"[ITEM] stage_dir          : {self._compact_path(stage_dir)}")
        return time.perf_counter()

    def _stage_done(self, idx: int, total: int, stage: EqStage, t0: float, *, detail: str | None = None) -> None:
        msg = f"[DONE] Stage {idx}/{total} | {stage.name} | elapsed={self._format_elapsed(time.perf_counter() - t0)}"
        if detail:
            msg += f" | {detail}"
        self._log(msg)

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
                        "gen_vel": "yes",
                        "gen_temp": temperature_k,
                        "gen_seed": -1,
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
                        "gen_vel": "no",
                        "gen_temp": temperature_k,
                        "gen_seed": -1,
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
                        "gen_vel": "no",
                        "gen_temp": temperature_k,
                        "gen_seed": -1,
                    },
                ),
            )
        )
        return stages

    def run(self, *, restart: Optional[bool] = None) -> Path:
        out = safe_mkdir(self.out_dir)
        rst_flag = resolve_restart(restart)
        t_all = time.perf_counter()
        self._section("GROMACS equilibration workflow", detail=f"restart={bool(rst_flag)} | stages={len(self.stages)}")
        self._item("out_dir", self._compact_path(out))
        self._item("input_gro", self._compact_path(self.gro))
        self._item("topology", self._compact_path(self.top))
        if self.ndx is not None:
            self._item("index", self._compact_path(self.ndx))
        self._item("resources", f"ntmpi={self.resources.ntmpi} | ntomp={self.resources.ntomp} | gpu={bool(self.resources.use_gpu)} | gpu_id={self.resources.gpu_id}")
        summary_path = out / "summary.json"
        summary: dict = load_json(summary_path) or {"job": "EquilibrationJob", "out_dir": str(out), "stages": []}
        summary["provenance"] = {
            "input_gro_sig": _file_signature(self.gro),
            "input_top_sig": _file_signature(self.top),
            "input_ndx_sig": _file_signature(self.provenance_ndx),
            "resources": {
                "ntmpi": self.resources.ntmpi,
                "ntomp": self.resources.ntomp,
                "gpu": bool(self.resources.use_gpu),
                "gpu_id": self.resources.gpu_id,
            },
        }

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

        def _is_constraint_failure(msg: object) -> bool:
            text = str(msg or "").lower()
            needles = (
                "lincs",
                "constraint failure",
                "constraint failures",
                "too many lincs warnings",
                "constrained together",
                "update groups can not be used",
            )
            return any(token in text for token in needles)

        def _constraints_mode(mdp: MdpSpec) -> str | None:
            try:
                for raw in mdp.render().splitlines():
                    line = raw.split(";", 1)[0].strip()
                    if not line or "=" not in line:
                        continue
                    key, value = [part.strip().lower() for part in line.split("=", 1)]
                    if key == "constraints":
                        return value
            except Exception:
                return None
            return None

        def _canonicalize_stage_gro(*, out_tpr: Path, out_gro: Path, stage_dir: Path) -> dict[str, object]:
            pbc_gro = pbc_mol_fix_inplace(self.runner, tpr=out_tpr, traj_or_gro=out_gro, cwd=stage_dir)
            whole_gro = normalize_gro_molecules_inplace(top=self.top, gro=out_gro)
            if whole_gro.get("applied"):
                self._log(
                    f"[INFO] Whole-molecule canonicalization applied to {out_gro.name} | normalized_molecules={whole_gro.get('normalized_molecules', 0)}"
                )
            elif whole_gro.get("error"):
                self._log(f"[WARN] Whole-molecule canonicalization skipped for {out_gro.name}: {whole_gro.get('error')}")
            return {"pbc_gro": pbc_gro, "whole_gro": whole_gro}

        def _mdrun_em_minimal(*, deffnm: str, cwd: Path, ntomp: int, gpu_id: Optional[str], use_gpu: bool = False) -> None:
            """Run energy minimization with a minimal, user-friendly mdrun command.

            By design we keep EM command-line args lean to improve portability and
            avoid surprising GPU/PME/update offload settings. Requested default:
              gmx mdrun -deffnm <name> -ntmpi 1 -ntomp <xx> -nb gpu -gpu_id <id> -v
            """
            args = ["mdrun", "-deffnm", str(deffnm)]
            # Always add -v when supported for live progress.
            if self.runner._tool_has_option("mdrun", "-v", cwd=cwd):
                args += ["-v"]
            # Requested default: -stepout 10000 for deterministic step prints.
            args += ["-stepout", "10000"]
            # Thread-MPI layout: EM runs as a single MPI rank.
            if self.runner._tool_has_option("mdrun", "-ntmpi", cwd=cwd):
                args += ["-ntmpi", "1"]
            if self.runner._tool_has_option("mdrun", "-ntomp", cwd=cwd):
                args += ["-ntomp", str(int(ntomp))]
            # Nonbonded on GPU (if available in this GROMACS build).
            if use_gpu and self.runner._tool_has_option("mdrun", "-nb", cwd=cwd):
                args += ["-nb", "gpu"]
            if use_gpu and gpu_id is not None and str(gpu_id).strip() != "" and self.runner._tool_has_option("mdrun", "-gpu_id", cwd=cwd):
                args += ["-gpu_id", str(gpu_id).strip()]
            rc, tail = self.runner._run_capture_tee(args, cwd=cwd)
            # Very old GROMACS builds may not support -stepout; retry once without it.
            if rc != 0 and "-stepout" in args and "unknown option" in (tail or "").lower() and "-stepout" in (tail or "").lower():
                try:
                    j = args.index("-stepout")
                    del args[j:j+2]
                    self._log("[WARN] Detected unsupported -stepout in EM mdrun. Retrying without -stepout.")
                except Exception:
                    pass
                rc, tail = self.runner._run_capture_tee(args, cwd=cwd)
            if rc != 0:
                raise RuntimeError(f"GROMACS mdrun (EM) failed (rc={rc})\n  cwd: {cwd}\n  cmd: {' '.join(args)}\n  tail:\n{tail}\n")

        total_stages = len(self.stages)
        for idx, st in enumerate(self.stages, start=1):
            stage_dir = safe_mkdir(out / st.name)
            t_stage = self._stage_begin(idx, total_stages, st, stage_dir)
            deffnm = "md"  # keep consistent within stage dir
            stage_summary_path = stage_dir / "summary.json"

            out_gro = stage_dir / f"{deffnm}.gro"
            out_cpt = stage_dir / f"{deffnm}.cpt"
            out_tpr = stage_dir / f"{deffnm}.tpr"
            stage_cutoff_events: list[dict[str, object]] = []

            def _write_stage_mdp(*, mdp: MdpSpec, gro: Path, filename: str, label: str) -> Path:
                prepared_mdp, cutoff_info = self._apply_box_safe_cutoffs(mdp, gro=gro)
                if cutoff_info:
                    self._log(
                        f"[WARN] Auto-shrinking nonbond cutoffs for {st.name}:{label} | "
                        f"min_box={cutoff_info['min_box_nm']:.4f} nm | cap={cutoff_info['cutoff_cap_nm']:.4f} nm"
                    )
                    stage_cutoff_events.append(
                        {
                            "label": label,
                            "input_gro": str(gro),
                            **cutoff_info,
                        }
                    )
                return prepared_mdp.write(stage_dir / filename)

            stage_has_outputs = bool(out_gro.exists())
            if not rst_flag:
                # Fresh rebuild requested: do not reuse any per-stage artifacts.
                # Clear the whole stage directory up front so GROMACS will not create
                # #md.log.# backups against outputs generated from stale inputs.
                for p in list(stage_dir.iterdir()):
                    try:
                        if p.is_dir():
                            shutil.rmtree(p, ignore_errors=True)
                        else:
                            p.unlink()
                    except Exception:
                        pass
                stage_has_outputs = False

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
            if rst_flag and out_gro.exists():
                invalid_existing_minim = None
                if st.kind in ("minim", "em"):
                    invalid_existing_minim = self._detect_invalid_minimization(stage_dir, deffnm=deffnm)
                if invalid_existing_minim is not None:
                    self._log(
                        f"[WARN] Existing minimization output for {st.name} is invalid "
                        f"({invalid_existing_minim}). Re-running this stage."
                    )
                    for p in stage_dir.glob(f"{deffnm}.*"):
                        try:
                            p.unlink()
                        except Exception:
                            pass
                    try:
                        stage_summary_path.unlink()
                    except Exception:
                        pass
                    stage_has_outputs = False
                else:
                    current_gro = out_gro
                    current_cpt = out_cpt if out_cpt.exists() else None
                    self._log(f"[SKIP] Existing stage output detected | gro={out_gro.name}")
                    _canonicalize_stage_gro(out_tpr=out_tpr, out_gro=out_gro, stage_dir=stage_dir)
                    # If summary exists, the postprocess was already done.
                    if stage_summary_path.exists():
                        self._stage_done(idx, total_stages, st, t_stage, detail="reused existing outputs + summary")
                        continue
                    # Otherwise, fall through to regenerate the summary only (no rerun).
                    stage_has_outputs = True

            # 2) Stage interrupted but checkpoint exists: continue without re-running grompp.
            #    This is the standard GROMACS restart mode: mdrun -cpi md.cpt -append.
            if (not stage_has_outputs) and rst_flag and out_tpr.exists() and out_cpt.exists() and (not out_gro.exists()):
                self._log(f"[RUN] Resuming interrupted stage from checkpoint | cpt={out_cpt.name}")
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
                if (not stage_has_outputs) and rst_flag:
                    for p in stage_dir.glob(f"{deffnm}.*"):
                        try:
                            p.unlink()
                        except Exception:
                            pass

                if not stage_has_outputs:
                    self._log(f"[RUN] Fresh stage execution | restart={bool(rst_flag)} | gpu={bool(stage_use_gpu)}")
                    # Special-case minimization: mirror yzc-gmx-gen with a robust
                    # steepest-descent (constraints=none) pre-minimization followed by
                    # the main CG minimization (constraints=h-bonds). We keep the final
                    # outputs as md.* in the stage directory for backward compatibility.
                    gro_for_main = current_gro

                    if st.kind in ("minim", "em"):
                        target_constraints = _constraints_mode(st.mdp)
                        if target_constraints == "none":
                            mdp_path = _write_stage_mdp(mdp=st.mdp, gro=current_gro, filename=f"{st.kind}.mdp", label="main")
                            gro_for_main = current_gro
                        else:
                            bridge_failed = False
                            steep_deffnm = "md_steep"
                            steep_gro = stage_dir / f"{steep_deffnm}.gro"
                            steep_tpr = stage_dir / f"{steep_deffnm}.tpr"
                            if not steep_gro.exists():
                                steep_mdp = _write_stage_mdp(
                                    mdp=MdpSpec(
                                        MINIM_STEEP_MDP,
                                        {**st.mdp.params, "nsteps": 50000, "emtol": 1000.0, "emstep": 0.01},
                                    ),
                                    gro=current_gro,
                                    filename="steep.mdp",
                                    label="steep",
                                )
                                self.runner.grompp(
                                    mdp=steep_mdp,
                                    gro=current_gro,
                                    top=self.top,
                                    ndx=self.ndx,
                                    out_tpr=steep_tpr,
                                    cpt=current_cpt,
                                    cwd=stage_dir,
                                )
                                ntomp_sel, _ = _choose_threads(False)
                                _mdrun_em_minimal(
                                    deffnm=steep_deffnm,
                                    cwd=stage_dir,
                                    ntomp=int(ntomp_sel or 1),
                                    gpu_id=self.resources.gpu_id,
                                    use_gpu=False,
                                )

                            bridge_deffnm = "md_steep_hbonds"
                            bridge_gro = stage_dir / f"{bridge_deffnm}.gro"
                            bridge_tpr = stage_dir / f"{bridge_deffnm}.tpr"
                            if not bridge_gro.exists():
                                try:
                                    bridge_input_gro = steep_gro if steep_gro.exists() else current_gro
                                    bridge_mdp = _write_stage_mdp(
                                        mdp=MdpSpec(
                                            MINIM_STEEP_HBONDS_MDP,
                                            {
                                                **st.mdp.params,
                                                "nsteps": 50000,
                                                "emtol": 1000.0,
                                                "emstep": 0.0005,
                                                "lincs_iter": max(int(st.mdp.params.get("lincs_iter", 2)), 4),
                                                "lincs_order": max(int(st.mdp.params.get("lincs_order", 8)), 12),
                                            },
                                        ),
                                        gro=bridge_input_gro,
                                        filename="steep_hbonds.mdp",
                                        label="steep_hbonds",
                                    )
                                    self.runner.grompp(
                                        mdp=bridge_mdp,
                                        gro=bridge_input_gro,
                                        top=self.top,
                                        ndx=self.ndx,
                                        out_tpr=bridge_tpr,
                                        cpt=None,
                                        cwd=stage_dir,
                                    )
                                    ntomp_sel, _ = _choose_threads(False)
                                    _mdrun_em_minimal(
                                        deffnm=bridge_deffnm,
                                        cwd=stage_dir,
                                        ntomp=int(ntomp_sel or 1),
                                        gpu_id=self.resources.gpu_id,
                                        use_gpu=False,
                                    )
                                except Exception as e:
                                    if _is_constraint_failure(e):
                                        bridge_failed = True
                                        self._log("[WARN] Constraint-sensitive steep_hbonds bridge failed. Continuing from the unconstrained steep result.")
                                    else:
                                        raise

                            gro_for_main = bridge_gro if bridge_gro.exists() else (steep_gro if steep_gro.exists() else current_gro)
                            if bridge_failed:
                                self._log("[WARN] Replacing the final CG minimization with an unconstrained steep minimization for this stage.")
                                mdp_path = _write_stage_mdp(
                                    mdp=MdpSpec(
                                        MINIM_STEEP_MDP,
                                        {
                                            **st.mdp.params,
                                            "nsteps": int(max(100000, int(st.mdp.params.get("nsteps", 50000)))),
                                            "emtol": float(st.mdp.params.get("emtol", 1000.0)),
                                            "emstep": 0.0005,
                                        },
                                    ),
                                    gro=gro_for_main,
                                    filename=f"{st.kind}.mdp",
                                    label="main_fallback",
                                )
                            else:
                                mdp_path = _write_stage_mdp(mdp=st.mdp, gro=gro_for_main, filename=f"{st.kind}.mdp", label="main")
                    else:
                        mdp_path = _write_stage_mdp(mdp=st.mdp, gro=gro_for_main, filename=f"{st.kind}.mdp", label="main")
                    tpr = out_tpr
                    self.runner.grompp(
                        mdp=mdp_path,
                        gro=gro_for_main,
                        top=self.top,
                        ndx=self.ndx,
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
                        if st.kind in ("minim", "em") and _is_constraint_failure(msg):
                            self._log("[WARN] Constraint-sensitive minimization failed. Falling back to steep (constraints=none).")
                            fb_mdp = _write_stage_mdp(
                                mdp=MdpSpec(
                                    MINIM_STEEP_MDP,
                                    {
                                        **st.mdp.params,
                                        "nsteps": int(max(100000, int(st.mdp.params.get("nsteps", 50000)))),
                                        "emtol": float(st.mdp.params.get("emtol", 1000.0)),
                                        "emstep": 0.0005,
                                    },
                                ),
                                gro=gro_for_main,
                                filename="minim_fallback_steep_none.mdp",
                                label="constraint_fallback",
                            )
                            self.runner.grompp(
                                mdp=fb_mdp,
                                gro=gro_for_main,
                                top=self.top,
                                ndx=self.ndx,
                                out_tpr=tpr,
                                cpt=None,
                                cwd=stage_dir,
                            )
                            ntomp_sel, _ = _choose_threads(False)
                            _mdrun_em_minimal(
                                deffnm=deffnm,
                                cwd=stage_dir,
                                ntomp=int(ntomp_sel or 1),
                                gpu_id=self.resources.gpu_id,
                                use_gpu=False,
                            )
                        else:
                            raise

                    if st.kind in ("minim", "em"):
                        invalid_minim = self._detect_invalid_minimization(stage_dir, deffnm=deffnm)
                        if invalid_minim is not None:
                            raise RuntimeError(
                                f"Invalid energy minimization detected in stage {st.name}: {invalid_minim}. "
                                f"See {stage_dir / f'{deffnm}.log'}. "
                                "This usually means the packed structure still contains severe atom overlaps; "
                                "increase the initial pack volume or lower the initial pack density before retrying."
                            )

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
            #   gmx trjconv -pbc mol
            # in-place (md.gro/md.xtc). Best-effort only.
            canonical_gro = _canonicalize_stage_gro(out_tpr=out_tpr, out_gro=out_gro, stage_dir=stage_dir)
            pbc_gro = canonical_gro["pbc_gro"]
            whole_gro = canonical_gro["whole_gro"]
            pbc_xtc = {"applied": False, "error": None}
            out_xtc = stage_dir / f"{deffnm}.xtc"
            if out_xtc.exists():
                pbc_xtc = pbc_mol_fix_inplace(self.runner, tpr=out_tpr, traj_or_gro=out_xtc, cwd=stage_dir)
            # ----------------------
            # Optional MOL2 export (system-level) via ParmEd
            # ----------------------
            # For visualization/interoperability, convert the final stage gro to mol2 using the stage topology.
            # This is best-effort and will not fail the workflow if conversion is unavailable.
            mol2_path = None
            mol2_skipped = None
            atom_count = self._gro_atom_count(out_gro)
            if atom_count is not None and atom_count > 99999:
                mol2_skipped = f"skipped stage mol2 export for large system ({atom_count} atoms)"
                self._log(f"[INFO] {mol2_skipped}")
            else:
                mol2_path = write_mol2_from_top_gro_parmed(
                    top_path=self.top,
                    gro_path=out_gro,
                    out_mol2=stage_dir / f"{deffnm}.mol2",
                    overwrite=True,
                )

            self._cleanup_stage_backups(stage_dir, deffnm=deffnm)



            # Postprocess (thermo)
            stage_record: dict = {
                "name": st.name,
                "kind": st.kind,
                "dir": str(stage_dir),
                "gro": str(current_gro),
                "edr": str(stage_dir / f"{deffnm}.edr") if (stage_dir / f"{deffnm}.edr").exists() else None,
                "mol2": str(mol2_path) if mol2_path else None,
                "mol2_skipped": mol2_skipped,
                "pbc_mol": {
                    "gro": pbc_gro,
                    "xtc": pbc_xtc,
                },
                "whole_molecule_gro": whole_gro,
                "auto_cutoff_adjustments": stage_cutoff_events,
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
                    "Box-X",
                    "Box-Y",
                    "Box-Z",
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

                    # Plots (SVG-first, annotated)
                    try:
                        plots_dir = stage_dir / "plots"
                        plots_dir.mkdir(parents=True, exist_ok=True)
                        stage_record.setdefault("plots", {}).update(
                            plot_thermo_stage(xvg, out_dir=plots_dir, title_prefix=f"{stage_dir.name}")
                        )
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
            self._stage_done(idx, total_stages, st, t_stage, detail=f"output={out_gro.name}")

        atomic_write_json(summary_path, summary)
        self._log("=" * 78)
        self._log(f"[DONE] GROMACS equilibration workflow | elapsed={self._format_elapsed(time.perf_counter() - t_all)} | summary={self._compact_path(summary_path)}")
        self._log("=" * 78)
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
