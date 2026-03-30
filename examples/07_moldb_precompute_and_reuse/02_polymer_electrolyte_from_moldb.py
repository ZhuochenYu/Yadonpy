from __future__ import annotations

"""Example 07 / Step 2: Polymer electrolyte workflow using MolDB.

Prerequisite:
  - Run 01_build_moldb.py first so the example species are already "ready" in
    MolDB.

Notes:
    - Example 07 now assumes the precompute step has already filled MolDB with
        a broad electrolyte library, including PF6-.
"""

from pathlib import Path

from yadonpy.runtime import set_run_options
from yadonpy.core import poly, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.merz import MERZ
from yadonpy.sim.preset import eq
from yadonpy.io.mol2 import write_mol2
from yadonpy.io.gmx import write_gmx


# ---------------- user inputs ----------------
restart_status = True
set_run_options(restart=restart_status)

ff = GAFF2_mod()  # default: GAFF2_mod
cation_ff = MERZ()

# ---- two monomers (both must have two connection points '*...*') ----
smiles_A = r"*CCO*"
smiles_B = r"*COC*"

# ---- solvents / ions (no '*') ----
solvent_smiles_A = "CCOC(=O)OC"  # EMC
solvent_smiles_B = "O=C1OCCO1"  # EC

cation_smiles_A = "[Li+]"

ter_smiles = "[H][*]"

ratio = [0.7, 0.3]
reac_ratio: list[float] = []

# MD settings
temp = 300.0
press = 1.0
mpi = 1
omp = 16

gpu = 1
gpu_id = 0

sim_time_ns = 5.0

# Packing
density_g_cm3 = 0.05
counts = [4, 20, 20, 20]
charge_scale = [1.0, 1.0, 1.0, 0.8]

BASE_DIR = Path(__file__).resolve().parent
work_dir = BASE_DIR / "work_dir"


def _load_ready_from_moldb(ff, smiles: str, *, label: str, bonded: str | None = None):
    try:
        mol = ff.mol(smiles, charge="RESP", require_ready=True, prefer_db=True)
    except Exception as exc:
        raise RuntimeError(
            f"{label} is expected to be precomputed in MolDB by "
            "examples/07_moldb_precompute_and_reuse/01_build_moldb.py."
        ) from exc
    mol = ff.ff_assign(mol, bonded=bonded, report=False)
    if not mol:
        raise RuntimeError(f"Cannot assign force field parameters for MolDB-backed {label}.")
    return mol


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    work_dir = workdir(work_dir, restart=restart_status)
    poly_rw_dir = work_dir.child("copoly_rw")
    poly_term_dir = work_dir.child("copoly_term")
    ac_build_dir = work_dir.child("00_build_cell")

    # --- load precomputed molecules from MolDB (as lightweight handles) ---
    monomer_A = _load_ready_from_moldb(ff, smiles_A, label="monomer_A")
    monomer_B = _load_ready_from_moldb(ff, smiles_B, label="monomer_B")
    ter1 = _load_ready_from_moldb(ff, ter_smiles, label="ter1")
    solvent_A = _load_ready_from_moldb(ff, solvent_smiles_A, label="solvent_A")
    solvent_B = _load_ready_from_moldb(ff, solvent_smiles_B, label="solvent_B")

    # --- Li+ from MERZ ---
    cation_A = cation_ff.mol(cation_smiles_A)
    if not cation_ff.ff_assign(cation_A):
        raise RuntimeError("Cannot assign MERZ parameters to Li+")

    # --- Optional: reuse a MolDB-backed anion (example: PF6-) ---
    # anion_smiles_A = "F[P-](F)(F)(F)(F)F"
    # anion_A = _load_ready_from_moldb(ff, anion_smiles_A, label="PF6", bonded="DRIH")

    # --- Build polymer (restart-friendly, manual API style) ---
    dp = max(1, int(poly.calc_n_from_num_atoms([monomer_A, monomer_B], 1000, ratio=ratio, terminal1=ter1)))
    copoly = poly.random_copolymerize_rw(
        [monomer_A, monomer_B],
        dp,
        ratio=ratio,
        reac_ratio=reac_ratio,
        tacticity='atactic',
        name='copoly',
        work_dir=poly_rw_dir,
    )
    copoly = poly.terminate_rw(copoly, ter1, name='copoly', work_dir=poly_term_dir)
    if not ff.ff_assign(copoly):
        raise RuntimeError("Cannot assign force field parameters to copoly")

    # Optional export
    write_mol2(mol=copoly, out_dir=work_dir / "00_molecules", name="copoly")
    write_gmx(mol=copoly, out_dir=work_dir / "90_copoly_gmx", mol_name="copoly")
    write_mol2(mol=solvent_A, out_dir=work_dir / "00_molecules", name="EMC")
    write_gmx(mol=solvent_A, out_dir=work_dir / "90_EMC_gmx", mol_name="EMC")
    write_mol2(mol=solvent_B, out_dir=work_dir / "00_molecules", name="EC")
    write_gmx(mol=solvent_B, out_dir=work_dir / "90_EC_gmx", mol_name="EC")
    write_mol2(mol=cation_A, out_dir=work_dir / "00_molecules", name="Li")
    write_gmx(mol=cation_A, out_dir=work_dir / "90_Li_gmx", mol_name="Li")

    # --- build amorphous cell ---
    ac = poly.amorphous_cell(
        [copoly, solvent_A, solvent_B, cation_A],
        counts,
        charge_scale=charge_scale,
        density=density_g_cm3,
        work_dir=ac_build_dir,
    )

    # --- EQ21 + check equilibrium ---
    eqmd = eq.EQ21step(ac, work_dir=work_dir)
    ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)

    analy = eqmd.analyze()
    _ = analy.get_all_prop(temp=temp, press=press, save=True)
    ok = analy.check_eq()

    for _i in range(4):
        if ok:
            break
        eqmd = eq.Additional(ac, work_dir=work_dir)
        ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)
        analy = eqmd.analyze()
        _ = analy.get_all_prop(temp=temp, press=press, save=True)
        ok = analy.check_eq()

    if not ok:
        raise SystemExit("[ERROR] Did not reach an equilibrium state.")

    # --- Production NPT + analysis ---
    npt = eq.NPT(ac, work_dir=work_dir)
    ac = npt.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id, time=sim_time_ns)

    analy = npt.analyze()
    _ = analy.get_all_prop(temp=temp, press=press, save=True)

    center_molecule = cation_A
    _ = analy.rdf(center_molecule)
    msd = analy.msd()
    sigma = analy.sigma(msd=msd)

    print("msd:", msd)
    print("sigma:", sigma)
