"""YadonPy equilibration presets.

These presets are thin wrappers over the pure-GROMACS workflows in
:mod:`yadonpy.gmx.workflows`.

Key design goals
---------------
- Accept an amorphous cell returned by :func:`yadonpy.core.poly.amorphous_cell`.
- Provide a consistent (restart, gpu, gpu_id) interface.
"""

from __future__ import annotations

import shutil
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from ...gmx.workflows._util import RunResources
from ...gmx.workflows.eq import EquilibrationJob
from ...io.gromacs_system import SystemExportResult, export_system_from_cell_meta
from ...workflow import ResumeManager, StepSpec
from ..analyzer import AnalyzeResult


def _parse_gpu_args(gpu: int, gpu_id: Optional[int]) -> tuple[bool, Optional[str]]:
    """Parse legacy + new GPU semantics.

    New semantics (preferred):
      - gpu: 1 (default) -> enable GPU
      - gpu: 0 -> disable GPU
      - gpu_id: int -> select GPU device id used by GROMACS (-gpu_id)

    Backward compatibility:
      - if gpu not in {0, 1} and gpu_id is None -> treat gpu as gpu_id and enable GPU.
    """
    try:
        g = int(gpu)
    except Exception:
        g = 1

    gid = gpu_id
    if gid is None and g not in (0, 1):
        gid = g
        g = 1

    use_gpu = bool(g)
    gid_s = str(int(gid)) if (use_gpu and gid is not None) else None
    return use_gpu, gid_s


def _next_additional_round(work_dir: Path, *, restart: bool) -> tuple[int, Path]:
    """Pick (round_idx, out_dir) for an additional equilibration run.

    Rules:
    - If restart=True and the latest round exists but is incomplete, reuse it.
    - Otherwise create a new round directory.
    """
    # Keep top-level work_dir tidy with numbered module folders.
    base = Path(work_dir) / "04_eq_additional"
    base.mkdir(parents=True, exist_ok=True)

    rounds = []
    for d in base.glob("round_*"):
        if d.is_dir():
            try:
                idx = int(d.name.split("_")[-1])
            except Exception:
                continue
            rounds.append((idx, d))
    rounds.sort(key=lambda x: x[0])

    if rounds and restart:
        idx, d = rounds[-1]
        # consider complete if final md files exist
        if not ((d / "04_md" / "md.tpr").exists() and (d / "04_md" / "md.xtc").exists() and (d / "04_md" / "md.edr").exists() and (d / "04_md" / "md.gro").exists()):
            return idx, d
        return idx + 1, base / f"round_{idx + 1:02d}"

    if not rounds:
        return 0, base / "round_00"

    idx, _ = rounds[-1]
    return idx + 1, base / f"round_{idx + 1:02d}"


def _find_latest_equilibrated_gro(work_dir: Path) -> Optional[Path]:
    """Find the latest equilibrated coordinate file under work_dir.

    Priority:
      1) Additional rounds (highest round idx)
      2) Main EQ21 run
    """
    wd = Path(work_dir)
    add_base = wd / "04_eq_additional"
    if add_base.exists():
        rounds = []
        for d in add_base.glob("round_*"):
            if not d.is_dir():
                continue
            try:
                idx = int(d.name.split("_")[-1])
            except Exception:
                continue
            gro = d / "04_md" / "md.gro"
            if gro.exists():
                rounds.append((idx, gro))
        if rounds:
            rounds.sort(key=lambda x: x[0])
            return rounds[-1][1]

    gro = wd / "03_eq" / "04_md" / "md.gro"
    if gro.exists():
        return gro
    return None


@dataclass
class EQ21step:
    """YadonPy-style equilibration preset wrapper (GROMACS engine)."""

    ac: object
    work_dir: Union[str, Path] = "./"
    ff_name: str = "gaff2_mod"
    charge_method: str = "RESP"
    system_name: str = "system"
    include_h_atomtypes: bool = False
    charge_scale: Optional[Any] = None

    def __post_init__(self):
        self.work_dir = Path(self.work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # Two system exports are maintained (organized under work_dir/02_system):
        #   - scaled (production): work_dir/02_system/
        #   - raw (no scaling, reference): work_dir/02_system/01_raw_non_scaled/
        # The workflow runs on the scaled system by default.
        self._export: Optional[SystemExportResult] = None  # scaled
        self._export_raw: Optional[SystemExportResult] = None
        self._job: Optional[EquilibrationJob] = None
        self._resume = ResumeManager(self.work_dir, enabled=True)

    def _load_export_from_disk(self, sys_dir: Path) -> SystemExportResult:
        meta_path = sys_dir / "system_meta.json"
        box_nm = 0.0
        species: list[dict] = []
        if meta_path.exists():
            try:
                import json

                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                box_nm = float(meta.get("box_nm") or 0.0)
                species = list(meta.get("species") or [])
            except Exception:
                pass
        return SystemExportResult(
            system_gro=sys_dir / "system.gro",
            system_top=sys_dir / "system.top",
            system_ndx=sys_dir / "system.ndx",
            molecules_dir=sys_dir / "molecules",
            system_meta=meta_path,
            box_nm=box_nm,
            species=species,
        )

    def _ensure_system_exported(self) -> SystemExportResult:
        if self._export is not None and self._export_raw is not None:
            return self._export

        # Keep the exported GROMACS system in a single, sortable module folder.
        # - Scaled (production) system lives at: work_dir/02_system/
        # - Raw/unscaled system is stored under: work_dir/02_system/01_raw_non_scaled/
        sys_root = self.work_dir / "02_system"
        raw_dir = sys_root / "01_raw_non_scaled"

        spec = StepSpec(
            name="export_system",
            outputs=[
                sys_root / "system.gro",
                sys_root / "system.top",
                sys_root / "system.ndx",
                sys_root / "system_meta.json",
                raw_dir / "system.gro",
                raw_dir / "system.top",
                raw_dir / "system.ndx",
                raw_dir / "system_meta.json",
            ],
            inputs={
                "ff_name": self.ff_name,
                "charge_method": self.charge_method,
                "charge_scale": str(self.charge_scale),
                "include_h_atomtypes": bool(self.include_h_atomtypes),
            },
            description="Export mixed system into GROMACS gro/top/ndx (scaled + raw)",
        )

        if self._resume.is_done(spec):
            exp_scaled = self._load_export_from_disk(sys_root)
            exp_raw = self._load_export_from_disk(raw_dir)
        else:
            def _run_pair() -> tuple[SystemExportResult, SystemExportResult]:
                # 1) Scaled system (per-species scaling spec)
                exp_scaled_local = export_system_from_cell_meta(
                    cell_mol=self.ac,
                    out_dir=sys_root,
                    ff_name=self.ff_name,
                    charge_method=self.charge_method,
                    charge_scale=self.charge_scale,
                    include_h_atomtypes=self.include_h_atomtypes,
                )
                # 2) Raw (no scaling) system (kept for debugging/reference)
                exp_raw_local = export_system_from_cell_meta(
                    cell_mol=self.ac,
                    out_dir=raw_dir,
                    ff_name=self.ff_name,
                    charge_method=self.charge_method,
                    charge_scale=1.0,
                    include_h_atomtypes=self.include_h_atomtypes,
                )
                return exp_raw_local, exp_scaled_local

            exp_raw, exp_scaled = self._resume.run(spec, _run_pair)

        self._export_raw = exp_raw
        self._export = exp_scaled
        return exp_scaled

    def exec(
        self,
        *,
        temp: float,
        press: float,
        mpi: int = 1,
        omp: int = 1,
        gpu: int = 1,
        gpu_id: Optional[int] = None,
        sim_time: float = 1.0,
        time: Optional[float] = None,
        charge_scale: Optional[Any] = None,
        restart: bool = True,
    ):
        """Run the EQ21 multi-stage equilibration.

        Parameters
        ----------
        restart:
            If True, skip already-finished substeps based on files in work_dir.
        gpu:
            GPU switch (1=on, 0=off). For backward compatibility, values other than
            0/1 will be treated as gpu_id.
        gpu_id:
            Which GPU device id to use for GROMACS (-gpu_id).
        """
        self._resume.enabled = bool(restart)

        if charge_scale is not None and str(charge_scale) != str(self.charge_scale):
            self.charge_scale = charge_scale
            self._export = None
            self._export_raw = None

        exp = self._ensure_system_exported()

        run_dir = self.work_dir / "03_eq"
        if not restart and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        if time is not None:
            sim_time = float(time)

        stages = EquilibrationJob.default_stages(
            temperature_k=float(temp),
            pressure_bar=float(press),
            nvt_ns=0.2,
            npt_ns=0.5,
            prod_ns=float(sim_time),
        )

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(ntmpi=int(mpi), ntomp=int(omp), use_gpu=use_gpu, gpu_id=gid)
        job = EquilibrationJob(gro=exp.system_gro, top=exp.system_top, out_dir=run_dir, stages=stages, resources=res)

        final_dir = run_dir / stages[-1].name if stages else run_dir
        eq_spec = StepSpec(
            name="equilibration_eq21",
            outputs=[final_dir / "md.tpr", final_dir / "md.xtc", final_dir / "md.edr", final_dir / "md.gro"],
            inputs={
                "temp": float(temp),
                "press": float(press),
                "sim_time": float(sim_time),
                "mpi": int(mpi),
                "omp": int(omp),
                "gpu": int(gpu),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
            },
            description="EQ21step multi-stage equilibration",
        )

        self._resume.run(eq_spec, lambda: job.run(restart=bool(restart)))
        self._job = job
        return self.ac

    def analyze(self) -> AnalyzeResult:
        if self._job is None:
            raise RuntimeError("EQ21step.exec() must be called before analyze().")
        exp = self._ensure_system_exported()
        tpr, xtc, edr = self._job.final_outputs()
        trr = None
        try:
            trr = self._job.final_trr()
        except Exception:
            trr = None
        return AnalyzeResult(
            work_dir=self.work_dir,
            tpr=tpr,
            xtc=xtc,
            trr=trr,
            edr=edr,
            top=exp.system_top,
            ndx=exp.system_ndx,
        )


class Additional(EQ21step):
    """Additional equilibration rounds.

    The additional rounds are stored under:
      work_dir/04_eq_additional/round_XX/

    This keeps each round self-contained and makes restart behavior robust.
    """

    def exec(
        self,
        *,
        temp: float,
        press: float,
        mpi: int = 1,
        omp: int = 1,
        gpu: int = 1,
        gpu_id: Optional[int] = None,
        sim_time: float = 1.0,
        time: Optional[float] = None,
        restart: bool = True,
    ):
        self._resume.enabled = bool(restart)

        if time is not None:
            sim_time = float(time)

        exp = self._ensure_system_exported()

        # Prefer continuing from the latest equilibrated structure already present in work_dir.
        # This is important when the user creates a new `Additional(...)` instance in scripts
        # (common in examples), in which case self._job is None.
        start_gro = _find_latest_equilibrated_gro(self.work_dir) or exp.system_gro

        # 判定标签：若已有可用的平衡结构（即 start_gro 不是原始 system.gro），
        # 则无需重复建盒子/EM/NVT/NPT 等流程。
        # 只在原有基础上重复跑最终平衡阶段（04_md）。
        _skip_rebuild = bool(start_gro != exp.system_gro)

        round_idx, run_dir = _next_additional_round(self.work_dir, restart=bool(restart))

        stages = EquilibrationJob.default_stages(
            temperature_k=float(temp),
            pressure_bar=float(press),
            nvt_ns=0.1,
            npt_ns=0.2,
            prod_ns=float(sim_time),
        )

        # 判定标签：只保留最终平衡阶段（04_md）
        if _skip_rebuild and stages:
            stages = [stages[-1]]

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(ntmpi=int(mpi), ntomp=int(omp), use_gpu=use_gpu, gpu_id=gid)
        job = EquilibrationJob(gro=start_gro, top=exp.system_top, out_dir=run_dir, stages=stages, resources=res)

        final_dir = run_dir / stages[-1].name if stages else run_dir
        spec = StepSpec(
            name=f"equilibration_additional_{round_idx:02d}",
            outputs=[final_dir / "md.tpr", final_dir / "md.xtc", final_dir / "md.edr", final_dir / "md.gro"],
            inputs={
                "temp": float(temp),
                "press": float(press),
                "sim_time": float(sim_time),
                "mpi": int(mpi),
                "omp": int(omp),
                "gpu": int(gpu),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
            },
            description="Additional equilibration round",
        )

        self._resume.run(spec, lambda: job.run(restart=bool(restart)))
        self._job = job
        return self.ac


@dataclass
class NPT(EQ21step):
    """NPT production run after equilibration has converged."""

    def exec(
        self,
        *,
        temp: float,
        press: float,
        mpi: int = 1,
        omp: int = 1,
        gpu: int = 1,
        gpu_id: Optional[int] = None,
        time: float = 5.0,
        restart: bool = True,
    ):
        self._resume.enabled = bool(restart)
        exp = self._ensure_system_exported()

        # Start from the latest equilibrated structure if available.
        start_gro = _find_latest_equilibrated_gro(self.work_dir) or exp.system_gro

        run_dir = self.work_dir / "05_npt_production"
        if not restart and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        # Build a single-stage NPT MD using the same NPT MDP template as equilibration.
        from ...gmx.mdp_templates import NPT_MDP, MdpSpec, default_mdp_params
        from ...gmx.workflows.eq import EqStage

        p = default_mdp_params()
        p["ref_t"] = float(temp)
        p["ref_p"] = float(press)

        # Production trajectory: write one frame every 1 ps.
        # dt is in ps; nstxout is in steps.
        nst_per_1ps = max(int(round(1.0 / float(p["dt"]))), 1)
        p["nstxout"] = nst_per_1ps

        def ns_to_steps(ns: float) -> int:
            return int((ns * 1000.0) / float(p["dt"]))

        stages = [
            EqStage(
                "01_npt",
                "md",
                MdpSpec(NPT_MDP, {**p, "nsteps": max(ns_to_steps(float(time)), 1000)}),
            )
        ]

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(ntmpi=int(mpi), ntomp=int(omp), use_gpu=use_gpu, gpu_id=gid)
        job = EquilibrationJob(gro=start_gro, top=exp.system_top, out_dir=run_dir, stages=stages, resources=res)

        final_dir = run_dir / stages[-1].name
        spec = StepSpec(
            name="npt_production",
            outputs=[final_dir / "md.tpr", final_dir / "md.xtc", final_dir / "md.edr", final_dir / "md.gro"],
            inputs={
                "temp": float(temp),
                "press": float(press),
                "time": float(time),
                "mpi": int(mpi),
                "omp": int(omp),
                "gpu": int(gpu),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
            },
            description="NPT production run",
        )

        self._resume.run(spec, lambda: job.run(restart=bool(restart)))
        self._job = job
        return self.ac



@dataclass
class NVT(EQ21step):
    """NVT production run with density fixed to the equilibrium average.

    Workflow:
      1) Read density time series from the *previous* equilibrated NPT/MD edr.
      2) Take the mean density over the last `density_frac_last` fraction.
      3) Compute the scaling factor from the last density value to the target mean density:
             s = (rho_last / rho_mean)^(1/3)
      4) Scale the starting gro (box + coordinates) with `gmx editconf -scale s s s`.
      5) Run a single-stage NVT production MD.

    Notes:
      - We intentionally scale from the *last* density in the series to the target mean,
        so we do not need to know system mass explicitly.
      - This behavior is inspired by yuzc's in-house yzc-gmx-gen workflow.
    """

    def exec(
        self,
        *,
        temp: float,
        mpi: int = 1,
        omp: int = 1,
        gpu: int = 1,
        gpu_id: Optional[int] = None,
        time: float = 5.0,
        restart: bool = True,
        density_control: bool = True,
        density_frac_last: float = 0.3,
    ):
        self._resume.enabled = bool(restart)
        exp = self._ensure_system_exported()

        # Start from the latest equilibrated structure if available.
        start_gro = _find_latest_equilibrated_gro(self.work_dir) or exp.system_gro

        run_dir = self.work_dir / "05_nvt_production"
        if not restart and run_dir.exists():
            shutil.rmtree(run_dir, ignore_errors=True)

        # Density control (optional): scale the starting gro to the mean density over the last X% of equilibrium.
        scaled_gro = start_gro
        if density_control:
            from ...gmx.engine import GromacsRunner
            from ...gmx.analysis.xvg import read_xvg
            from ...gmx.analysis.thermo import stats_from_xvg
            import uuid

            runner = GromacsRunner()
            tmp = None

            # Try to locate the previous edr (same directory as start_gro).
            prev_dir = Path(start_gro).parent
            prev_edr = prev_dir / "md.edr"

            # Fallback: search under work_dir for the newest md.edr.
            if not prev_edr.exists():
                candidates = list(Path(self.work_dir).rglob("md.edr"))
                candidates = [c for c in candidates if c.is_file()]
                if candidates:
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    prev_edr = candidates[0]

            if prev_edr.exists():
                # Detect density term name (can vary across GROMACS versions, e.g. "Density (kg/m^3)").
                mapping = runner.list_energy_terms(edr=prev_edr)
                dens_term = None
                for k in mapping.keys():
                    if "density" in k.lower():
                        dens_term = k
                        break

                if dens_term is not None:
                    tmp = run_dir / f"_yadonpy_density_tmp_{uuid.uuid4().hex[:8]}.xvg"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    runner.energy_xvg(edr=prev_edr, out_xvg=tmp, terms=[dens_term])

                    df = read_xvg(tmp).df
                    dens_cols = [c for c in df.columns if c != "x"]
                    if dens_cols:
                        col = dens_cols[0]
                        rho_last = float(df[col].iloc[-1])
                        st = stats_from_xvg(tmp, col=col, frac_last=float(density_frac_last))
                        rho_mean = float(st.mean)

                        # Avoid divide-by-zero or pathological scaling.
                        if rho_last > 0 and rho_mean > 0:
                            s = float((rho_last / rho_mean) ** (1.0 / 3.0))
                            scaled_gro = run_dir / "start_scaled_to_eq_density.gro"
                            runner.run(
                                [
                                    "editconf",
                                    "-f",
                                    str(start_gro),
                                    "-o",
                                    str(scaled_gro),
                                    "-scale",
                                    f"{s}",
                                    f"{s}",
                                    f"{s}",
                                ],
                                cwd=run_dir,
                                check=True,
                                capture=True,
                            )
                # Best-effort cleanup
                if tmp is not None:
                    try:
                        tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
                    except Exception:
                        pass

        # Build a single-stage NVT MD.
        from ...gmx.mdp_templates import NVT_MDP, MdpSpec, default_mdp_params
        from ...gmx.workflows.eq import EqStage

        p = default_mdp_params()
        p["ref_t"] = float(temp)

        # Production trajectory: write one frame every 1 ps.
        nst_per_1ps = max(int(round(1.0 / float(p["dt"]))), 1)
        p["nstxout"] = nst_per_1ps

        def ns_to_steps(ns: float) -> int:
            return int((ns * 1000.0) / float(p["dt"]))

        stages = [
            EqStage(
                "01_nvt",
                "md",
                MdpSpec(NVT_MDP, {**p, "nsteps": max(ns_to_steps(float(time)), 1000)}),
            )
        ]

        use_gpu, gid = _parse_gpu_args(gpu, gpu_id)
        res = RunResources(ntmpi=int(mpi), ntomp=int(omp), use_gpu=use_gpu, gpu_id=gid)
        job = EquilibrationJob(gro=scaled_gro, top=exp.system_top, out_dir=run_dir, stages=stages, resources=res)

        final_dir = run_dir / stages[-1].name
        spec = StepSpec(
            name="nvt_production",
            outputs=[final_dir / "md.tpr", final_dir / "md.xtc", final_dir / "md.edr", final_dir / "md.gro"],
            inputs={
                "temp": float(temp),
                "time": float(time),
                "mpi": int(mpi),
                "omp": int(omp),
                "gpu": int(gpu),
                "gpu_id": int(gpu_id) if gpu_id is not None else None,
                "density_control": bool(density_control),
                "density_frac_last": float(density_frac_last),
            },
            description="NVT production run (density fixed to equilibrium mean)",
        )

        self._resume.run(spec, lambda: job.run(restart=bool(restart)))
        self._job = job
        return self.ac
