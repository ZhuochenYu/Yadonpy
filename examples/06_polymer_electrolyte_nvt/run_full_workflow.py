from __future__ import annotations

# Example 06: Polymer electrolyte workflow in one script (SMILES -> RESP -> polymer -> cell -> EQ21 -> analysis)
#
# Restart logic
#   restart_status=True  : resume/skip finished steps based on files in work_dir
#   restart_status=False : force re-run (keep work_dir, but steps will overwrite their own outputs)

from pathlib import Path

from yadonpy.runtime import set_run_options
from yadonpy.core import utils, poly, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.merz import MERZ
from yadonpy.sim import qm
from yadonpy.sim.preset import eq


# ---------------- user inputs ----------------
restart_status = False
set_run_options(restart=restart_status)

ff = GAFF2_mod()  # default: GAFF2_mod
cation_ff = MERZ()

# ---- two monomers (both must have two connection points '*...*') ----
smiles_A = r"*CCO*"  # updated per request
smiles_B = r"*COC*"

# ---- solvents / ions (no '*') ----
solvent_smiles_A = "CCOC(=O)OC"  # EMC (example)
solvent_smiles_B = "O=C1OCCO1"  # EC (example)

cation_smiles_A = "[Li+]"  # cation_A (MERZ)
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
counts = [4, 20, 20, 20, 20]  # [polymer, solventA, solventB, cation_A, anion_A]
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

    work_dir = workdir(work_dir, restart=restart_status)
    poly_rw_dir = work_dir.child("copoly_rw")
    poly_term_dir = work_dir.child("copoly_term")
    ac_build_dir = work_dir.child("00_build_cell")

    # polymer####################################################################################################
    # --- build monomers ---
    monomer_A = utils.mol_from_smiles(smiles_A)
    monomer_B = utils.mol_from_smiles(smiles_B)
    # No explicit naming: yadonpy will infer names from variable names when needed.

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
    # (name inferred later)
    qm.assign_charges(
        ter1, charge='RESP', opt=True, work_dir=work_dir,
        omp=omp_psi4, memory=mem, log_name=None
    )

    # --- restart-friendly random copolymerization + termination (manual API style) ---
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
    copoly = ff.ff_assign(copoly)
    if not copoly:
        raise RuntimeError('Can not assign force field parameters for copoly.')
    # ############################################################################################################

    # solvent#####################################################################################################
    solvent_A = utils.mol_from_smiles(solvent_smiles_A)
    solvent_B = utils.mol_from_smiles(solvent_smiles_B)
    # (names inferred later)

    solvent_A, energy = qm.conformation_search(
        solvent_A, ff=ff, work_dir=work_dir,
        psi4_omp=omp_psi4, mpi=mpi, omp=omp,
        memory=mem, log_name=None
    )
    qm.assign_charges(
        solvent_A, charge='RESP', opt=False, work_dir=work_dir,
        omp=omp_psi4, memory=mem, log_name=None
    )
    solvent_A = ff.ff_assign(solvent_A)
    if not solvent_A:
        raise RuntimeError('Can not assign force field parameters for solvent_A.')

    solvent_B, energy = qm.conformation_search(
        solvent_B, ff=ff, work_dir=work_dir,
        psi4_omp=omp_psi4, mpi=mpi, omp=omp,
        memory=mem, log_name=None
    )
    qm.assign_charges(
        solvent_B, charge='RESP', opt=False, work_dir=work_dir,
        omp=omp_psi4, memory=mem, log_name=None
    )
    solvent_B = ff.ff_assign(solvent_B)
    if not solvent_B:
        raise RuntimeError('Can not assign force field parameters for solvent_B.')
    # ############################################################################################################

    # cation_A######################################################################################################
    cation_A = cation_ff.mol(cation_smiles_A)
    # (name inferred later)
    cation_A = cation_ff.ff_assign(cation_A)
    if not cation_A:
        raise RuntimeError('Can not assign MERZ parameters to cation_A')
    # ############################################################################################################

    # anion_A#######################################################################################################
    try:
        anion_A = ff.mol(anion_smiles_A, charge='RESP', require_ready=True, prefer_db=True)
        anion_A = ff.ff_assign(anion_A, bonded='DRIH')
    except Exception as exc:
        raise RuntimeError(
            "PF6 is expected to be precomputed in MolDB for this example. "
            "Please build it first with examples/01_Li_salt/run_pf6_to_moldb.py."
        ) from exc
    if not anion_A:
        raise RuntimeError('Can not assign force field parameters for MolDB-backed PF6.')
    # ############################################################################################################

    # build amorphous cell########################################################################################
    ac = poly.amorphous_cell(
        [copoly, solvent_A, solvent_B, cation_A, anion_A],
        counts,
        charge_scale=charge_scale,
        density=density_g_cm3,
        work_dir=ac_build_dir,
    )
    # ############################################################################################################

    # EQ21 + check equilibrium####################################################################################
    eqmd = eq.EQ21step(ac, work_dir=work_dir)
    ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)

    analy = eqmd.analyze()
    prop_data = analy.get_all_prop(temp=temp, press=press, save=True)
    result = analy.check_eq()

    # Additional equilibration MD
    for i in range(4):
        if result:
            break
        eqmd = eq.Additional(ac, work_dir=work_dir)
        ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)
        analy = eqmd.analyze()
        prop_data = analy.get_all_prop(temp=temp, press=press, save=True)
        result = analy.check_eq()

    if not result:
        print('[ERROR: Did not reach an equilibrium state.]')
        raise SystemExit(1)

    # Production NVT (sim_time_ns) with density fixed to equilibrium mean + analysis###############################
    nvt = eq.NVT(ac, work_dir=work_dir)
    ac = nvt.exec(
        temp=temp,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        time=sim_time_ns,
        
        density_control=True,
        density_frac_last=0.30,
    )

    center_molecule = cation_A
    analy = nvt.analyze()
    # We still report basic thermo (including pressure) from edr.
    prop_data = analy.get_all_prop(temp=temp, press=press, save=True)

    rdf = analy.rdf(center_molecule)
    msd = analy.msd()
    sigma = analy.sigma(msd=msd)
