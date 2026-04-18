from __future__ import annotations

"""CMC random copolymer + DTD electrolyte example backed by MolDB monomers.

The anionic glucose monomers are expected to be RESP-ready in MolDB with
``polyelectrolyte_mode=True``. This keeps the expensive monomer QM step out of
the system-build script and directly exercises the MolDB -> ITP export path.

Use ``YADONPY_BUILD_ONLY=1`` to stop after amorphous-cell construction.
Use ``YADONPY_EXPORT_ONLY=1`` to stop after exporting ``02_system``.
These modes are useful for checking topology / ITP generation on machines
without GROMACS.
"""

import os
from pathlib import Path

from yadonpy.runtime import set_run_options
from yadonpy.core import utils, poly, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2, MERZ
from yadonpy.sim import qm
from yadonpy.sim.preset import eq


def _env_flag(name: str, default: bool = False) -> bool:
    token = str(os.environ.get(name, "")).strip().lower()
    if not token:
        return bool(default)
    return token in {"1", "true", "t", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return int(default)
    return int(raw)


def _load_ready_from_moldb(
    ff,
    smiles: str,
    *,
    label: str,
    bonded: str | None = None,
    polyelectrolyte_mode: bool = False,
    repo_db_dir: Path,
):
    last_exc: Exception | None = None
    for db_dir, db_label in ((None, "default"), (repo_db_dir, "repo")):
        try:
            mol = ff.mol_rdkit(
                smiles,
                name=label,
                db_dir=db_dir,
                charge="RESP",
                require_ready=True,
                prefer_db=True,
                polyelectrolyte_mode=polyelectrolyte_mode,
                polyelectrolyte_detection="auto",
            )
            mol = ff.ff_assign(
                mol,
                bonded=bonded,
                polyelectrolyte_mode=polyelectrolyte_mode,
            )
            if not mol:
                raise RuntimeError(f"Cannot assign force field parameters for {label}.")
            print(f"[MolDB] loaded {label} from {db_label} db")
            return mol
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(
        f"{label} is expected to be RESP-ready in MolDB. "
        "Please refresh it with examples/07_moldb_precompute_and_reuse/"
        "03_refresh_cmc_glucose_polyelectrolytes.py."
    ) from last_exc


# ---------------- user inputs ----------------
restart_status = True
build_only = _env_flag("YADONPY_BUILD_ONLY", default=False)
export_only = _env_flag("YADONPY_EXPORT_ONLY", default=False)
smoke_mode = _env_flag("YADONPY_SMOKE", default=False)

set_run_options(restart=restart_status)

ff = GAFF2()
ion_ff = MERZ()

# ---- CMC monomers (two connection points '*...*') ----
glucose_smiles = "*OC1OC(CO)C(*)C(O)C1O"
glucose_2_smiles = "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]"
glucose_3_smiles = "*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O"
glucose_6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"

DTD_smiles = "O=S1(=O)OC=CO1"

feed_ratio = [12, 26, 27, 35]
feed_prob = poly.ratio_to_prob(feed_ratio)

target_mw = 10000.0
ter_smiles = "[H][*]"

# ---- Solvents ----
EC_smiles = "O=C1OCCO1"
EMC_smiles = "CCOC(=O)OC"
DEC_smiles = "CCOC(=O)OCC"

# ---- Salt / ions ----
Li_smiles = "[Li+]"
PF6_smiles = "F[P-](F)(F)(F)(F)F"
Na_smiles = "[Na+]"

temp = 318.15
press = 1.0
mpi = 1
omp = 14
gpu = 1
gpu_id = 0

omp_psi4 = _env_int("YADONPY_PSI4_OMP", 20)
mem_mb = _env_int("YADONPY_PSI4_MEMORY_MB", 20000)

BASE_DIR = Path(__file__).resolve().parent
REPO_DB_DIR = BASE_DIR.parents[1] / "moldb"
_work_dir_override = str(os.environ.get("YADONPY_WORK_DIR", "")).strip()
work_dir = Path(_work_dir_override).expanduser() if _work_dir_override else (BASE_DIR / "work_dir_dtd_moldb")


def _formal_charge(mol) -> int:
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


def main() -> int:
    doctor(print_report=True)
    ensure_initialized()

    wd = workdir(work_dir, restart=restart_status)
    cmc_rw_dir = wd.child("CMC_rw")
    cmc_term_dir = wd.child("CMC_term")
    ac_build_dir = wd.child("00_build_cell")

    # ---------------- build monomers ----------------
    glucose = utils.mol_from_smiles(glucose_smiles)
    glucose = ff.ff_assign(glucose)
    if not glucose:
        raise RuntimeError("Can not assign force field parameters for glucose.")

    glucose_2 = _load_ready_from_moldb(
        ff,
        glucose_2_smiles,
        label="glucose_2",
        polyelectrolyte_mode=True,
        repo_db_dir=REPO_DB_DIR,
    )
    glucose_3 = _load_ready_from_moldb(
        ff,
        glucose_3_smiles,
        label="glucose_3",
        polyelectrolyte_mode=True,
        repo_db_dir=REPO_DB_DIR,
    )
    glucose_6 = _load_ready_from_moldb(
        ff,
        glucose_6_smiles,
        label="glucose_6",
        polyelectrolyte_mode=True,
        repo_db_dir=REPO_DB_DIR,
    )

    # termination
    ter1 = utils.mol_from_smiles(ter_smiles)
    qm.assign_charges(
        ter1,
        charge="RESP",
        opt=True,
        work_dir=wd,
        omp=omp_psi4,
        memory=mem_mb,
        log_name="ter1",
    )

    # DP from target polymer Mw
    dp = poly.calc_n_from_mol_weight(
        [glucose, glucose_2, glucose_3, glucose_6],
        target_mw,
        ratio=feed_prob,
        terminal1=ter1,
    )
    print(f"[CMC] estimated DP from target Mw = {dp}")

    # random copolymerization (self-avoiding RW), then terminate
    chain_len = 12 if smoke_mode else 50
    CMC = poly.random_copolymerize_rw(
        [glucose, glucose_2, glucose_3, glucose_6],
        chain_len,
        ratio=feed_prob,
        tacticity="atactic",
        work_dir=cmc_rw_dir,
    )
    CMC = poly.terminate_rw(CMC, ter1, work_dir=cmc_term_dir)
    CMC = ff.ff_assign(CMC, polyelectrolyte_mode=True)
    if not CMC:
        raise RuntimeError("Can not assign force field parameters for CMC.")
    q_poly = _formal_charge(CMC)

    # ---------------- build solvents / additive ----------------
    EC = ff.ff_assign(utils.mol_from_smiles(EC_smiles))
    EMC = ff.ff_assign(utils.mol_from_smiles(EMC_smiles))
    DEC = ff.ff_assign(utils.mol_from_smiles(DEC_smiles))
    if not EC or not EMC or not DEC:
        raise RuntimeError("Can not assign force field parameters for carbonate solvents.")

    DTD = _load_ready_from_moldb(ff, DTD_smiles, label="DTD", repo_db_dir=REPO_DB_DIR)

    # ---------------- ions ----------------
    Li = ion_ff.mol(Li_smiles)
    Li = ion_ff.ff_assign(Li)
    if not Li:
        raise RuntimeError("Can not assign MERZ force field parameters for Li+.")

    Na = ion_ff.mol(Na_smiles)
    Na = ion_ff.ff_assign(Na)
    if not Na:
        raise RuntimeError("Can not assign MERZ force field parameters for Na+.")

    PF6 = _load_ready_from_moldb(ff, PF6_smiles, label="PF6", bonded="DRIH", repo_db_dir=REPO_DB_DIR)

    # ---------------- compute counts ----------------
    n_cmc = 1 if smoke_mode else 8
    n_na = abs(q_poly) * n_cmc
    if smoke_mode:
        counts = [n_cmc, 6, 6, 6, 2, 2, n_na, 1]
    else:
        counts = [n_cmc, 40, 50, 20, 10, 10, n_na, 4]
    charge_scale = [0.7, 1.0, 1.0, 1.0, 0.7, 0.7, 0.7, 1.0]

    print(f"[FORMULATION] smoke_mode={smoke_mode} q_poly={q_poly} counts={counts}")

    # ---------------- pack amorphous cell ----------------
    ac = poly.amorphous_cell(
        [CMC, EC, EMC, DEC, Li, PF6, Na, DTD],
        counts,
        charge_scale=charge_scale,
        polyelectrolyte_mode=True,
        density=0.05,
        neutralize=False,
        work_dir=ac_build_dir,
    )

    if build_only:
        print(f"[BUILD-ONLY] Finished cell construction at {ac_build_dir}")
        return 0

    # ---------------- run equilibration preset ----------------
    eqmd = eq.EQ21step(ac, work_dir=wd)
    if export_only:
        exported = eqmd.ensure_system_exported()
        print(f"[EXPORT-ONLY] Exported 02_system at {exported.system_top.parent}")
        return 0
    ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)

    analy = eqmd.analyze()
    _ = analy.get_all_prop(temp=temp, press=press, save=True)
    result = analy.check_eq()

    for _i in range(4):
        if result:
            break
        eqmd = eq.Additional(ac, work_dir=wd)
        ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)
        analy = eqmd.analyze()
        _ = analy.get_all_prop(temp=temp, press=press, save=True)
        result = analy.check_eq()

    if not result:
        print("[WARNING] Did not reach an equilibrium state after EQ21 + Additional cycles.")

    # ---------------- Production NPT + analysis ----------------
    npt = eq.NPT(ac, work_dir=wd)
    ac = npt.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id, time=20)

    analy = npt.analyze()
    _ = analy.get_all_prop(temp=temp, press=press, save=True)
    _ = analy.rdf(center_mol=Li)
    msd = analy.msd()
    _ = analy.sigma(temp_k=temp, msd=msd)
    _ = analy.den_dis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
