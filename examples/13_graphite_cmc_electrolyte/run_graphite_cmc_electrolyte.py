from __future__ import annotations

from pathlib import Path

import yadonpy as yp
from yadonpy.core import molecular_weight, poly, register_cell_species_metadata, stack_cell_blocks, utils, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2_mod, MERZ
from yadonpy.interface import make_orthorhombic_pack_cell, plan_direct_electrolyte_counts
from yadonpy.io.gromacs_system import export_system_from_cell_meta
from yadonpy.io.mol2 import write_mol2
from yadonpy.runtime import set_run_options
from yadonpy.sim import qm


restart = True
set_run_options(restart=restart)

ff = GAFF2_mod()
ion_ff = MERZ()

glucose_smiles = "*OC1OC(CO)C(*)C(O)C1O"
glucose_2_smiles = "*OC1OC(CO)C(*)C(O)C1OCC(=O)[O-]"
glucose_3_smiles = "*OC1OC(CO)C(*)C(OCC(=O)[O-])C1O"
glucose_6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"
feed_ratio = [12, 26, 27, 35]
feed_prob = poly.ratio_to_prob(feed_ratio)

ter_smiles = "[H][*]"
EC_smiles = "O=C1OCCO1"
DEC_smiles = "CCOC(=O)OCC"
EMC_smiles = "CCOC(=O)OC"
Li_smiles = "[Li+]"
PF6_smiles = "F[P-](F)(F)(F)(F)F"
Na_smiles = "[Na+]"

graphite_nx = 8
graphite_ny = 6
graphite_layers = 3
graphite_edge_cap = "H"

cmc_chain_count = 2
cmc_dp = 50
cmc_pack_density_g_cm3 = 0.03
cmc_slab_z_nm = 4.5

electrolyte_slab_z_nm = 5.5
electrolyte_density_g_cm3 = 1.28
electrolyte_pack_density_g_cm3 = 0.88
solvent_mass_ratio = (3.0, 2.0, 5.0)
salt_molarity_M = 1.0
min_salt_pairs = 6

graphite_to_cmc_gap_ang = 4.0
cmc_to_electrolyte_gap_ang = 4.0

omp_psi4 = 8
mem_mb = 16000

BASE_DIR = Path(__file__).resolve().parent
work_dir = workdir(BASE_DIR / "work_dir", clean=not restart)


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    mol2_dir = work_dir / "00_molecules"
    graphite_dir = work_dir.child("01_graphite")
    polymer_rw_dir = work_dir.child("02_cmc_rw")
    polymer_term_dir = work_dir.child("03_cmc_term")
    cmc_slab_dir = work_dir.child("04_cmc_slab")
    electrolyte_slab_dir = work_dir.child("05_electrolyte_slab")
    stack_dir = work_dir.child("06_graphite_cmc_electrolyte")

    graphite = yp.build_graphite(
        nx=graphite_nx,
        ny=graphite_ny,
        n_layers=graphite_layers,
        orientation="basal",
        edge_cap=graphite_edge_cap,
        ff=ff,
        name="graphite_layer",
        top_padding_ang=15.0,
    )
    write_mol2(mol=graphite.layer_mol, out_dir=mol2_dir)

    monomers = [
        ff.ff_assign(ff.mol(smiles, require_ready=False, prefer_db=False), report=False)
        for smiles in (glucose_smiles, glucose_2_smiles, glucose_3_smiles, glucose_6_smiles)
    ]
    if any(mol is False for mol in monomers):
        raise RuntimeError("Can not assign force field parameters for the CMC monomer set.")

    ter = utils.mol_from_smiles(ter_smiles)
    qm.assign_charges(
        ter,
        charge="RESP",
        opt=True,
        work_dir=work_dir,
        omp=omp_psi4,
        memory=mem_mb,
        log_name=None,
    )
    CMC = poly.random_copolymerize_rw(
        monomers,
        cmc_dp,
        ratio=feed_prob,
        tacticity="atactic",
        name="CMC",
        work_dir=polymer_rw_dir,
    )
    CMC = poly.terminate_rw(CMC, ter, name="CMC", work_dir=polymer_term_dir)
    CMC = ff.ff_assign(CMC, report=False)
    if not CMC:
        raise RuntimeError("Can not assign force field parameters for CMC.")
    Na = ion_ff.ff_assign(ion_ff.mol(Na_smiles), report=False)
    if not Na:
        raise RuntimeError("Can not assign force field parameters for Na.")
    write_mol2(mol=CMC, out_dir=mol2_dir)
    write_mol2(mol=Na, out_dir=mol2_dir)

    solvents = {
        label: ff.ff_assign(ff.mol(smiles, require_ready=False, prefer_db=False), report=False)
        for label, smiles in (("EC", EC_smiles), ("DEC", DEC_smiles), ("EMC", EMC_smiles))
    }
    if any(mol is False for mol in solvents.values()):
        raise RuntimeError("Can not assign force field parameters for the carbonate solvents.")
    Li = ion_ff.ff_assign(ion_ff.mol(Li_smiles), report=False)
    if not Li:
        raise RuntimeError("Can not assign force field parameters for Li.")
    PF6 = ff.ff_assign(ff.mol(PF6_smiles, charge="RESP", require_ready=True, prefer_db=True), bonded="DRIH")
    if not PF6:
        raise RuntimeError("Can not assign force field parameters for MolDB-backed PF6.")
    for mol in list(solvents.values()) + [Li, PF6]:
        write_mol2(mol=mol, out_dir=mol2_dir)

    cmc_box_nm = (graphite.box_nm[0], graphite.box_nm[1], cmc_slab_z_nm)
    cmc_slab = poly.amorphous_cell(
        [CMC, Na],
        [cmc_chain_count, cmc_chain_count],
        cell=make_orthorhombic_pack_cell(cmc_box_nm),
        density=None,
        charge_scale=[1.0, 1.0],
        neutralize=False,
        work_dir=cmc_slab_dir,
        retry=20,
        retry_step=2000,
        threshold=1.55,
        dec_rate=0.75,
    )
    register_cell_species_metadata(cmc_slab, [CMC, Na], [cmc_chain_count, cmc_chain_count], pack_mode="cmc_slab")

    electrolyte_plan = plan_direct_electrolyte_counts(
        target_box_nm=(graphite.box_nm[0], graphite.box_nm[1], electrolyte_slab_z_nm),
        target_density_g_cm3=electrolyte_density_g_cm3,
        solvent_mol_weights=[
            molecular_weight(solvents["EC"], strict=True),
            molecular_weight(solvents["DEC"], strict=True),
            molecular_weight(solvents["EMC"], strict=True),
        ],
        solvent_mass_ratio=solvent_mass_ratio,
        salt_mol_weights=[molecular_weight(Li, strict=True), molecular_weight(PF6, strict=True)],
        salt_molarity_M=salt_molarity_M,
        min_salt_pairs=min_salt_pairs,
        solvent_species_names=["EC", "DEC", "EMC"],
        salt_species_names=["Li", "PF6"],
        min_solvent_counts=(2, 2, 2),
    )

    electrolyte_slab = poly.amorphous_cell(
        [solvents["EC"], solvents["DEC"], solvents["EMC"], Li, PF6],
        list(electrolyte_plan.target_counts),
        cell=make_orthorhombic_pack_cell((graphite.box_nm[0], graphite.box_nm[1], electrolyte_slab_z_nm)),
        density=None,
        charge_scale=[1.0, 1.0, 1.0, 0.8, 0.8],
        neutralize=False,
        work_dir=electrolyte_slab_dir,
        retry=30,
        retry_step=2400,
        threshold=1.55,
        dec_rate=0.70,
    )
    register_cell_species_metadata(
        electrolyte_slab,
        [solvents["EC"], solvents["DEC"], solvents["EMC"], Li, PF6],
        list(electrolyte_plan.target_counts),
        charge_scale=[1.0, 1.0, 1.0, 0.8, 0.8],
        pack_mode="electrolyte_slab",
    )

    stacked = stack_cell_blocks(
        [graphite.cell, cmc_slab, electrolyte_slab],
        z_gaps_ang=[graphite_to_cmc_gap_ang, cmc_to_electrolyte_gap_ang],
        top_padding_ang=12.0,
    )
    register_cell_species_metadata(
        stacked.cell,
        [graphite.layer_mol, CMC, Na, solvents["EC"], solvents["DEC"], solvents["EMC"], Li, PF6],
        [
            graphite.layer_count,
            cmc_chain_count,
            cmc_chain_count,
            electrolyte_plan.target_counts[0],
            electrolyte_plan.target_counts[1],
            electrolyte_plan.target_counts[2],
            electrolyte_plan.target_counts[3],
            electrolyte_plan.target_counts[4],
        ],
        charge_scale=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.8],
        pack_mode="graphite_cmc_electrolyte_stack",
    )

    out = export_system_from_cell_meta(
        cell_mol=stacked.cell,
        out_dir=stack_dir,
        ff_name=ff.name,
        charge_method="RESP",
        write_system_mol2=False,
    )

    print("graphite_box_nm =", tuple(round(v, 4) for v in graphite.box_nm))
    print("cmc_box_nm =", tuple(round(v, 4) for v in cmc_box_nm))
    print("electrolyte_box_nm =", tuple(round(v, 4) for v in (graphite.box_nm[0], graphite.box_nm[1], electrolyte_slab_z_nm)))
    print("final_box_nm =", tuple(round(v, 4) for v in stacked.box_nm))
    print("system_top =", out.system_top)
    print("system_gro =", out.system_gro)
