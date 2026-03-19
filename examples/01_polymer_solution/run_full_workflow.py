from __future__ import annotations

# Example 01: Polymer solution workflow in one script (SMILES -> RESP -> polymer -> cell -> EQ21 -> analysis)
#
# Restart logic
#   restart_status=True  : resume/skip finished steps based on files in work_dir
#   restart_status=False : force re-run (keep work_dir, but steps will overwrite their own outputs)

import os
import shutil
from pathlib import Path

from yadonpy.core import utils, poly
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.merz import MERZ
from yadonpy.sim import qm
from yadonpy.sim.preset import eq
from yadonpy.io.mol2 import write_mol2_from_rdkit


# ---------------- user inputs ----------------
restart_status = False

ff = GAFF2_mod()  # default: GAFF2_mod
cation_ff = MERZ()

# ---- two monomers (both must have two connection points '*...*') ----
smiles_A = r"*CCO*"  # updated per request
smiles_B = r"*COC*"

# ---- solvents / ions (no '*') ----
solvent_smiles_A = "CCOC(=O)OC"  # EMC (example)
solvent_smiles_B = "O=C1OCCO1"  # EC (example)

cation_smiles_A = "[Li+]"  # cation (MERZ)
anion_smiles_A = "F[P-](F)(F)(F)(F)F"  # PF6- (GAFF2-family; default uses GAFF2_mod)

# Termination unit (one '*').
# - Hydrogen termination: "[H][*]" or "[*][H]"
# - Other terminations: "*C", "*O", ...
ter_smiles = "[H][*]"

# composition in the copolymer chain (mole fraction)
ratio = [0.7, 0.3]  # A:B
reac_ratio: list[float] = []  # optional Mayo–Lewis biasing, e.g. [rA, rB]

# MD settings
temp = 300.0
press = 1.0
mpi = 1
omp = 16

# GPU semantics:
#   gpu=1 enables GPU, gpu=0 disables GPU.
#   gpu_id selects which GPU GROMACS uses when GPU is enabled.
gpu = 1
gpu_id = 3

sim_time_ns = 5.0

# Packing
density_g_cm3 = 0.05
counts = [4, 20, 20]  # [polymer, solventA, solventB, cation, anion]
charge_scale = [1.0, 1.0, 1.0, 0.8, 0.8]  # aligned with the species list below

# QM/RESP settings
omp_psi4 = 64
mem_mb = 20000
mem = mem_mb

BASE_DIR = Path(__file__).resolve().parent
work_dir = BASE_DIR / "work_dir"


if __name__ == '__main__':
    doctor(print_report=True)
    ensure_initialized()  # idempotent

    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    # polymer####################################################################################################
    # --- build monomers ---
    monomer_A = utils.mol_from_smiles(smiles_A)
    monomer_B = utils.mol_from_smiles(smiles_B)

    # --- conformation search & RESP for each monomer (separate log_name!) ---
    monomer_A, energy = qm.conformation_search(
        monomer_A, ff=ff, work_dir=work_dir,
        psi4_omp=omp_psi4, mpi=mpi, omp=omp, memory=mem, log_name=None
    )
    qm.assign_charges(
        monomer_A, charge='RESP', opt=False, work_dir=work_dir,
        omp=omp_psi4, memory=mem, log_name=None
    )

    monomer_B, energy = qm.conformation_search(
        monomer_B, ff=ff, work_dir=work_dir,
        psi4_omp=omp_psi4, mpi=mpi, omp=omp, memory=mem, log_name=None
    )
    qm.assign_charges(
        monomer_B, charge='RESP', opt=False, work_dir=work_dir,
        omp=omp_psi4, memory=mem, log_name=None
    )

    # --- termination unit RESP (same as sample) ---
    ter1 = utils.mol_from_smiles(ter_smiles)
    qm.assign_charges(
        ter1, charge='RESP', opt=True, work_dir=work_dir,
        omp=omp_psi4, memory=mem, log_name=None
    )

    # --- decide DP from target atom count (works for copolymers too) ---
    # natom_target=1000 here matches RadonPy sample logic
    dp = poly.calc_n_from_num_atoms([monomer_A, monomer_B], 1000, ratio=ratio, terminal1=ter1)

    # --- random copolymerization by self-avoiding random walk ---
    copoly = poly.random_copolymerize_rw(
        [monomer_A, monomer_B], dp, ratio=ratio, reac_ratio=reac_ratio, tacticity='atactic'
    )

    # terminate
    copoly = poly.terminate_rw(copoly, ter1)

    # No explicit naming needed: yadonpy will infer names from Python variable
    # names (copoly, solvent_A, ...) when generating artifacts/logs.
    result = ff.ff_assign(copoly)
    if not result:
        raise RuntimeError('Can not assign force field parameters.')
    write_mol2_from_rdkit(mol=copoly, out_dir=work_dir / '00_molecules')
    # ############################################################################################################

    # solvent#####################################################################################################
    solvent_A = utils.mol_from_smiles(solvent_smiles_A)
    solvent_B = utils.mol_from_smiles(solvent_smiles_B)
    # No explicit naming: names are inferred from variable names when needed.

    solvent_A, energy = qm.conformation_search(
        solvent_A, ff=ff, work_dir=work_dir,
        psi4_omp=omp_psi4, mpi=mpi, omp=omp,
        memory=mem, log_name=None
    )
    qm.assign_charges(
        solvent_A, charge='RESP', opt=False, work_dir=work_dir,
        omp=omp_psi4, memory=mem, log_name=None
    )
    result = ff.ff_assign(solvent_A)
    write_mol2_from_rdkit(mol=solvent_A, out_dir=work_dir / '00_molecules')

    solvent_B, energy = qm.conformation_search(
        solvent_B, ff=ff, work_dir=work_dir,
        psi4_omp=omp_psi4, mpi=mpi, omp=omp,
        memory=mem, log_name=None
    )
    qm.assign_charges(
        solvent_B, charge='RESP', opt=False, work_dir=work_dir,
        omp=omp_psi4, memory=mem, log_name=None
    )
    result = ff.ff_assign(solvent_B)
    write_mol2_from_rdkit(mol=solvent_B, out_dir=work_dir / '00_molecules')
    # ############################################################################################################
    # (No salt) polymer solution example: skip cation/anion

    # build amorphous cell########################################################################################
    ac = poly.amorphous_cell(
        [copoly, solvent_A, solvent_B],
        counts,
        charge_scale=charge_scale,
        density=density_g_cm3,
    )
    # ############################################################################################################

    # EQ21 + check equilibrium####################################################################################
    eqmd = eq.EQ21step(ac, work_dir=work_dir)
    ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id, restart=restart_status)

    analy = eqmd.analyze()
    prop_data = analy.get_all_prop(temp=temp, press=press, save=True)
    result = analy.check_eq()

    # Additional equilibration MD
    for i in range(4):
        if result:
            break
        eqmd = eq.Additional(ac, work_dir=work_dir)
        ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id, restart=restart_status)
        analy = eqmd.analyze()
        prop_data = analy.get_all_prop(temp=temp, press=press, save=True)
        result = analy.check_eq()

    if not result:
        print('[ERROR: Did not reach an equilibrium state.]')
        raise SystemExit(1)

    # Production NPT (sim_time_ns) + analysis#####################################################################
    npt = eq.NPT(ac, work_dir=work_dir)
    ac = npt.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id, time=sim_time_ns, restart=restart_status)

    # Polymer solution example: no ions, so we only report basic thermo + MSD.
    analy = npt.analyze()
    prop_data = analy.get_all_prop(temp=temp, press=press, save=True)
    msd = analy.msd()
