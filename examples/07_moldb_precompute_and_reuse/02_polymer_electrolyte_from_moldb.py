from __future__ import annotations

"""Example 07 / Step 2: Polymer electrolyte workflow using MolDB.

Prerequisite:
  - Run 01_build_moldb.py first so the workflow-local monomers/solvents are
    "ready" in MolDB.
  - Run 03_rebuild_reference_moldb_species.py only when you also want the
    larger reference solvent/salt set in the same MolDB.

Notes:
    - This example intentionally avoids PF6- in the CSV. If you need PF6-,
        precompute it with Example 01 and then reuse it through the MolDB-backed
        `ff.mol(...)` plus `ff.ff_assign(..., bonded='DRIH')` pattern.
"""

from pathlib import Path

from yadonpy.runtime import set_run_options
from yadonpy.core import poly, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.merz import MERZ
from yadonpy.moldb import MolDB
from yadonpy.sim.preset import eq
from yadonpy.io.mol2 import write_mol2
from yadonpy.io.gmx import write_gmx
from yadonpy.core import poly


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

# QM resources for MolDB autocalculate (Psi4)
omp_psi4 = 64
mem_mb = 20000

gpu = 1
gpu_id = 0

sim_time_ns = 5.0

# Packing
density_g_cm3 = 0.05
counts = [4, 20, 20, 20]
charge_scale = [1.0, 1.0, 1.0, 0.8]

BASE_DIR = Path(__file__).resolve().parent
work_dir = BASE_DIR / "work_dir"
work_root = work_dir / "00_autocalculate_moldb"


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    work_dir = workdir(work_dir, restart=restart_status)
    poly_rw_dir = work_dir.child("copoly_rw")
    poly_term_dir = work_dir.child("copoly_term")
    ac_build_dir = work_dir.child("00_build_cell")

    # --- Ensure MolDB has required molecules (idempotent) ---
    db = MolDB()
    db.read_calc_temp = str(BASE_DIR / "template.csv")
    db.autocalculate(work_dir=work_root, omp=omp_psi4, mem=mem_mb, add_to_moldb=True)

    # --- load precomputed molecules from MolDB (as lightweight handles) ---
    monomer_A = ff.mol(smiles_A)
    monomer_B = ff.mol(smiles_B)
    ter1 = ff.mol(ter_smiles)
    solvent_A = ff.mol(solvent_smiles_A)
    solvent_B = ff.mol(solvent_smiles_B)

    ok = all([
        ff.ff_assign(monomer_A),
        ff.ff_assign(monomer_B),
        ff.ff_assign(ter1),
        ff.ff_assign(solvent_A),
        ff.ff_assign(solvent_B),
    ])
    if not ok:
        raise RuntimeError("MolDB entry not ready. Please run 01_build_moldb.py first.")

    # --- Li+ from MERZ ---
    cation_A = cation_ff.mol(cation_smiles_A)
    if not cation_ff.ff_assign(cation_A):
        raise RuntimeError("Cannot assign MERZ parameters to Li+")

    # --- Optional: reuse a MolDB-backed anion (example: PF6-) ---
    # anion_smiles_A = "F[P-](F)(F)(F)(F)F"
    # anion_A = ff.mol(anion_smiles_A, charge='RESP', require_ready=True, prefer_db=True)
    # anion_A = ff.ff_assign(anion_A, bonded='DRIH')

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
