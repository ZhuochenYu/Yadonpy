from __future__ import annotations

import os
from pathlib import Path

from yadonpy.core import poly, utils, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.io import write_gmx
from yadonpy.io.mol2 import write_mol2
from yadonpy.runtime import set_run_options
from yadonpy.sim.preset import eq


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return float(default)
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return int(default)
    return int(raw)


BASE_DIR = Path(__file__).resolve().parent
restart_status = _env_bool("RESTART_STATUS", False)
run_md = _env_bool("RUN_MD", True)

work_root = Path(os.environ.get("WORK_DIR", str(BASE_DIR / "work_segment_branch"))).resolve()

mpi = _env_int("MPI", 1)
omp = _env_int("OMP", 8)
gpu = _env_int("GPU", 0)
gpu_id = _env_int("GPU_ID", 0)
temp_k = _env_float("TEMP_K", 300.0)
press_bar = _env_float("PRESS_BAR", 1.0)
prod_ns = _env_float("PROD_NS", 0.2)


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()
    set_run_options(restart=restart_status)

    wd = workdir(work_root, restart=restart_status)
    ff = GAFF2_mod()

    # Main-chain label convention:
    #   * / [1*]  -> consumed during linear segment/polymer growth
    #   [2*]      -> preserved by seg_gen and later used as a branch site
    monomer_a = ff.ff_assign(ff.mol("*CCO*", require_ready=False, prefer_db=False), report=False)
    monomer_b = ff.ff_assign(ff.mol("*COC*", require_ready=False, prefer_db=False), report=False)
    branchable = ff.ff_assign(ff.mol("*C([2*])C*", require_ready=False, prefer_db=False), report=False)
    branch_unit = ff.ff_assign(ff.mol("*CO*", require_ready=False, prefer_db=False), report=False)
    solvent = ff.ff_assign(ff.mol("CCOC(=O)OC", require_ready=False, prefer_db=False), report=False)
    terminator = utils.mol_from_smiles("[H][*]")

    if not all([monomer_a, monomer_b, branchable, branch_unit, solvent]):
        raise RuntimeError("Could not assign force-field parameters to one or more example species.")

    segment_aab = poly.seg_gen(
        [monomer_a, monomer_a, monomer_b],
        name="segment_aab",
        work_dir=wd.child("01_segment_aab"),
    )
    branchable_segment = poly.seg_gen(
        [branchable, branchable, monomer_a],
        name="branchable_segment",
        work_dir=wd.child("02_branchable_segment"),
    )

    # Cap the tail so this side-chain segment has exactly one attach linker.
    side_segment = poly.seg_gen(
        [branch_unit],
        cap_tail="[H][*]",
        name="side_segment",
        work_dir=wd.child("03_side_segment"),
    )

    # Pre-branch one deterministic site while preserving the other [2*] sites
    # for later post-polymerization grafting.
    prebranched_segment = poly.branch_segment_rw(
        branchable_segment,
        [side_segment],
        mode="pre",
        position=2,
        exact_map={"position": 2, "site_index": 0, "branch": 0},
        name="prebranched_segment",
        work_dir=wd.child("04_prebranch"),
    )

    # Long block construction from segments; this is intentionally a light
    # wrapper around the existing random-walk engine.
    block_polymer = poly.block_segment_rw(
        [segment_aab, prebranched_segment],
        [3, 2],
        name="segment_block_polymer",
        work_dir=wd.child("05_block_polymer"),
    )

    # Post-branch remaining [2*] sites statistically.  Use ds=[1.0] to consume
    # all available [2*] sites with side_segment in this demonstrator.
    branched_polymer = poly.branch_segment_rw(
        block_polymer,
        [side_segment],
        mode="post",
        position=2,
        ds=[1.0],
        name="segment_branched_polymer",
        work_dir=wd.child("06_postbranch"),
    )
    branched_polymer = poly.terminate_rw(
        branched_polymer,
        terminator,
        name="segment_branched_polymer",
        work_dir=wd.child("07_terminate"),
    )

    branched_polymer = ff.ff_assign(branched_polymer, report=False)
    if not branched_polymer:
        raise RuntimeError("Can not assign force-field parameters for the branched segment polymer.")

    write_mol2(mol=branched_polymer, out_dir=wd / "00_molecules")
    write_gmx(mol=branched_polymer, out_dir=wd / "90_polymer_gmx")

    ac = poly.amorphous_cell(
        [branched_polymer, solvent],
        [1, 12],
        density=0.05,
        work_dir=wd.child("08_build_cell"),
        retry=3,
        retry_step=500,
    )

    if not run_md:
        print("[OK] Segment/branch polymer and amorphous cell were built. Set RUN_MD=1 for EQ/production.")
        raise SystemExit(0)

    eqmd = eq.EQ21step(ac, work_dir=wd)
    ac = eqmd.exec(temp=temp_k, press=press_bar, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)
    analy = eqmd.analyze()
    analy.get_all_prop(temp=temp_k, press=press_bar, save=True)
    result = analy.check_eq()

    for _ in range(4):
        if result:
            break
        eqmd = eq.Additional(ac, work_dir=wd)
        ac = eqmd.exec(temp=temp_k, press=press_bar, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)
        analy = eqmd.analyze()
        analy.get_all_prop(temp=temp_k, press=press_bar, save=True)
        result = analy.check_eq()

    if not result:
        raise RuntimeError("Equilibration did not converge; inspect analysis/equilibrium.json before production.")

    prod = eq.NPT(ac, work_dir=wd)
    ac = prod.exec(temp=temp_k, press=press_bar, time=prod_ns, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)
    analy = prod.analyze()
    analy.get_all_prop(temp=temp_k, press=press_bar, save=True)
    analy.msd()
