from __future__ import annotations

# Example 05: CMC-Na random copolymer + 1M LiPF6 in EC/EMC/DEC (1:1:1 mass ratio)
#
# Targets:
#   - Polymer: CMC (random copolymer from 4 glucose-based monomers), Mw ~ 10000 g/mol, 4 chains
#   - Solvent: EC / EMC / DEC with equal mass
#   - Salt: 1 M LiPF6 (min 20 ion pairs)
#   - Counter-ion: Na+ to neutralize polymer formal charge (CMC-Na)
#   - Charge scaling: polymer and all ions scaled by 0.8
#   - Center molecule for RDF: Li+

from pathlib import Path

from yadonpy.runtime import set_run_options
from yadonpy.core import molecular_weight, utils, poly, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2_mod, MERZ
from yadonpy.sim import qm
from yadonpy.sim.preset import eq


# ---------------- user inputs ----------------
restart_status = False
set_run_options(restart=restart_status)

ff = GAFF2_mod()  # or GAFF2() for classic GAFF2 (different parameter DB)
ion_ff = MERZ()

# ---- CMC monomers (two connection points '*...*') ----
glucose_smiles   = "*OC1OC(CO)C(*)C(O)C1O"
glucose_2_smiles = "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]"
glucose_3_smiles = "*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O"
glucose_6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"

# feed ratio (integer)
feed_ratio = [12, 26, 27, 35]
feed_prob = poly.ratio_to_prob(feed_ratio)

# target polymer molecular weight (g/mol)
target_mw = 10000.0

# termination unit (one '*')
ter_smiles = "[H][*]"

# ---- Solvents ----
EC_smiles  = "O=C1OCCO1"
EMC_smiles = "CCOC(=O)OC"
DEC_smiles = "CCOC(=O)OCC"

# ---- Salt / ions ----
Li_smiles  = "[Li+]"                 # MERZ
PF6_smiles = "F[P-](F)(F)(F)(F)F"    # GAFF2_mod
Na_smiles  = "[Na+]"                 # MERZ (counter-ion for CMC-)

# ---- Formulation ----
n_CMC = 4
# Solvent feed: total solvent mass = 6 wt% of polymer feed mass
solvent_wt_over_polymer = 0.06
salt_molarity_M = 1.0      # LiPF6 concentration in mol/L
min_salt_pairs = 20

# packing density (g/cm^3) used to estimate volume for 1M salt count
density_target_g_cm3 = 1.2
density_pack_g_cm3 = 0.05  # low packing density to ensure polymer fits in initial box

# charge scaling aligned with species list: [CMC, EC, EMC, DEC, Li, PF6, Na]
charge_scale = [0.8, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8]

# MD settings
temp = 300.0
press = 1.0
mpi = 1
omp = 16
gpu = 1
gpu_id = 0

# QM settings
omp_psi4 = 32
mem_mb = 20000

BASE_DIR = Path(__file__).resolve().parent
work_dir = BASE_DIR / "work_dir"


def mol_formal_charge(mol) -> int:
    """Sum RDKit atom formal charges."""

    q = 0
    for a in mol.GetAtoms():
        q += int(a.GetFormalCharge())
    return int(q)


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    work_dir = workdir(work_dir, restart=restart_status)
    cmc_rw_dir = work_dir.child("CMC_rw")
    cmc_term_dir = work_dir.child("CMC_term")
    ac_build_dir = work_dir.child("00_build_cell")

    # ---------------- build monomers ----------------
    glucose   = utils.mol_from_smiles(glucose_smiles)
    glucose_2 = utils.mol_from_smiles(glucose_2_smiles)
    glucose_3 = utils.mol_from_smiles(glucose_3_smiles)
    glucose_6 = utils.mol_from_smiles(glucose_6_smiles)

    # Conformation search + RESP for monomers (anionic monomers included)
    # IMPORTANT:
    #   qm.conformation_search historically returned a *new* RDKit Mol instance.
    #   If we don't rebind the result back, later polymerization would still
    #   see the old (uncharged) objects. Keep the monomer list updated.
    monomers = [glucose, glucose_2, glucose_3, glucose_6]
    for i, mon in enumerate(monomers):
        mon, _ = qm.conformation_search(
            mon, ff=ff, work_dir=work_dir,
            psi4_omp=omp_psi4, mpi=mpi, omp=omp, memory=mem_mb, log_name=None
        )
        qm.assign_charges(
            mon,
            charge="RESP",
            opt=False,
            work_dir=work_dir,
            omp=omp_psi4,
            memory=mem_mb,
            log_name=None,
            polyelectrolyte_mode=True,
        )
        monomers[i] = mon

    glucose, glucose_2, glucose_3, glucose_6 = monomers

    # Quick sanity: make sure charges exist (prevents silent all-zero ITPs)
    try:
        qabs = sum(abs(a.GetDoubleProp('AtomicCharge')) for a in glucose.GetAtoms() if a.HasProp('AtomicCharge'))
        if qabs < 1.0e-6:
            raise RuntimeError('Monomer charges are missing after QM. Check the psi4/psiresp-base installation and restart cache.')
    except Exception:
        pass

    # termination
    ter1 = utils.mol_from_smiles(ter_smiles)
    qm.assign_charges(ter1, charge="RESP", opt=True, work_dir=work_dir, omp=omp_psi4, memory=mem_mb, log_name=None)

    # DP from target polymer Mw
    dp = poly.calc_n_from_mol_weight(
        [glucose, glucose_2, glucose_3, glucose_6],
        target_mw,
        ratio=feed_prob,
        terminal1=ter1,
    )

    # random copolymerization (self-avoiding RW), then terminate
    CMC = poly.random_copolymerize_rw(
        [glucose, glucose_2, glucose_3, glucose_6],
        int(dp),
        ratio=feed_prob,
        tacticity='atactic',
        work_dir=cmc_rw_dir,
    )
    CMC = poly.terminate_rw(CMC, ter1, work_dir=cmc_term_dir)
    CMC = ff.ff_assign(CMC)
    if not CMC:
        raise RuntimeError('Can not assign force field parameters for CMC.')

    # Sanity-check atomtype coverage (helps diagnose RDF coverage later).
    try:
        atypes = sorted({a.GetProp('ff_type') for a in CMC.GetAtoms() if a.HasProp('ff_type')})
        print(f"[CMC] unique ff_type = {atypes}")
        if ('oh' not in atypes) or ('o' not in atypes):
            print("[WARN] CMC atomtypes missing expected 'oh'/'o'. RDF/ndx coverage may be limited. "
                  "Check that hydrogens/valences are present and ff_assign succeeded as intended.")
    except Exception:
        pass

    # ---------------- build solvents ----------------
    solvent_specs = [
        ("EC", utils.mol_from_smiles(EC_smiles)),
        ("EMC", utils.mol_from_smiles(EMC_smiles)),
        ("DEC", utils.mol_from_smiles(DEC_smiles)),
    ]
    solvent_map = {}
    for solvent_name, solvent in solvent_specs:
        solvent, _ = qm.conformation_search(
            solvent, ff=ff, work_dir=work_dir,
            psi4_omp=omp_psi4, mpi=mpi, omp=omp, memory=mem_mb, log_name=None
        )
        qm.assign_charges(solvent, charge="RESP", opt=False, work_dir=work_dir, omp=omp_psi4, memory=mem_mb, log_name=None)
        solvent = ff.ff_assign(solvent)
        if not solvent:
            raise RuntimeError(f"Can not assign force field parameters for {solvent_name}.")
        solvent_map[solvent_name] = solvent
    EC = solvent_map["EC"]
    EMC = solvent_map["EMC"]
    DEC = solvent_map["DEC"]

    # ---------------- ions ----------------
    Li = ion_ff.mol(Li_smiles)
    Li = ion_ff.ff_assign(Li)
    if not Li:
        raise RuntimeError("Can not assign MERZ force field parameters for Li+.")

    Na = ion_ff.mol(Na_smiles)
    Na = ion_ff.ff_assign(Na)
    if not Na:
        raise RuntimeError("Can not assign MERZ force field parameters for Na+.")

    try:
        PF6 = ff.mol(PF6_smiles, charge='RESP', require_ready=True, prefer_db=True)
        PF6 = ff.ff_assign(PF6, bonded='DRIH')
    except Exception as exc:
        raise RuntimeError(
            "PF6 is expected to be precomputed in MolDB for this example. "
            "Please build it first with examples/01_Li_salt/run_pf6_to_moldb.py."
        ) from exc
    if not PF6:
        raise RuntimeError('Can not assign force field parameters for MolDB-backed PF6.')

    # ---------------- compute counts ----------------
    # polymer molecular weight from RDKit
    mw_CMC = molecular_weight(CMC, strict=True)
    NA = 6.02214076e23

    # total polymer mass (g) for n_CMC molecules
    m_poly_g = mw_CMC * n_CMC / NA
    # total solvent mass (g) based on polymer feed mass
    m_solvent_total_g = m_poly_g * solvent_wt_over_polymer
    m_total_g = m_poly_g + m_solvent_total_g
    m_solvent_each_g = m_solvent_total_g / 3.0

    # solvent MWs
    mw_EC = molecular_weight(EC, strict=True)
    mw_EMC = molecular_weight(EMC, strict=True)
    mw_DEC = molecular_weight(DEC, strict=True)

    # molecule counts implied by 1:1:1 mass ratio (lower bound 20 each)
    n_EC = max(20, int(round((m_solvent_each_g / mw_EC) * NA)))
    n_EMC = max(20, int(round((m_solvent_each_g / mw_EMC) * NA)))
    n_DEC = max(20, int(round((m_solvent_each_g / mw_DEC) * NA)))

    # estimate volume from total mass and target density
    vol_cm3 = m_total_g / density_target_g_cm3
    vol_L = vol_cm3 / 1000.0

    # 1 M LiPF6 -> ion pairs
    n_LiPF6 = int(round(salt_molarity_M * vol_L * NA))
    if n_LiPF6 < min_salt_pairs:
        n_LiPF6 = min_salt_pairs

    # Na+ neutralization: polymer formal charge (per chain) * n_CMC
    q_poly = mol_formal_charge(CMC)
    n_Na = int(abs(q_poly) * n_CMC) if q_poly != 0 else 0

    counts = [n_CMC, n_EC, n_EMC, n_DEC, n_LiPF6, n_LiPF6, n_Na]

    print("[FORMULATION]")
    print(f"  feed_ratio = {feed_ratio} -> prob = {feed_prob}")
    print(f"  dp = {dp}, Mw(CMC)~{mw_CMC:.1f} g/mol, n_CMC={n_CMC}, formal_charge={q_poly}")
    print(f"  solvents: n_EC={n_EC}, n_EMC={n_EMC}, n_DEC={n_DEC} (1:1:1 mass)")
    print(f"  salt: LiPF6={n_LiPF6} pairs (1M, min {min_salt_pairs})")
    print(f"  Na+: {n_Na} (neutralize polymer)")
    print(f"  density={density_target_g_cm3} g/cm3 => V~{vol_L:.3e} L")

    # ---------------- pack amorphous cell ----------------
    ac = poly.amorphous_cell(
        [CMC, EC, EMC, DEC, Li, PF6, Na],
        counts,
        charge_scale=charge_scale,
        polyelectrolyte_mode=True,
        density=density_pack_g_cm3,
        neutralize=False,
        work_dir=ac_build_dir,
    )

    # ---------------- run equilibration preset ----------------
    eqmd = eq.EQ21step(ac, work_dir=work_dir)
    ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)

    analy = eqmd.analyze()
    _ = analy.get_all_prop(temp=temp, press=press, save=True)
    result = analy.check_eq()

    # Additional equilibration MD (up to 4 cycles)
    for i in range(4):
        if result:
            break
        eqmd = eq.Additional(ac, work_dir=work_dir)
        ac = eqmd.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id)
        analy = eqmd.analyze()
        _ = analy.get_all_prop(temp=temp, press=press, save=True)
        result = analy.check_eq()

    if not result:
        print('[WARNING: Did not reach an equilibrium state after EQ21 + Additional cycles.]')

    # ---------------- Production NPT + analysis ----------------
    # IMPORTANT:
    #   Do NOT analyze from the last equilibration stage directory (EQ/Additional).
    #   Production (NPT/NVT) generates the trajectory used for RDF/MSD/sigma.
    npt = eq.NPT(ac, work_dir=work_dir)
    ac = npt.exec(temp=temp, press=press, mpi=mpi, omp=omp, gpu=gpu, gpu_id=gpu_id, time=20)

    center_molecule = Li
    analy = npt.analyze()
    _ = analy.get_all_prop(temp=temp, press=press, save=True)

    rdf = analy.rdf(center_mol=center_molecule)
    msd = analy.msd()
    sigma = analy.sigma(temp_k=temp, msd=msd)
    density_distributionr = analy.den_dis()
