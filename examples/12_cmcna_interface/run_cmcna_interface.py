from __future__ import annotations

from pathlib import Path

from yadonpy.runtime import set_run_options
from yadonpy.core import molecular_weight, poly, utils, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2, MERZ
from yadonpy.interface import (
    InterfaceBuilder,
    InterfaceDynamics,
    equilibrate_bulk_with_eq21,
    format_cell_charge_audit,
    make_orthorhombic_pack_cell,
    plan_probe_polymer_matched_interface_preparation,
    plan_resized_polymer_matched_interface_from_probe,
    read_equilibrated_box_nm,
    recommend_polymer_diffusion_interface_recipe,
)
from yadonpy.io.mol2 import write_mol2
from yadonpy.sim import qm


restart = True
set_run_options(restart=restart)

ff = GAFF2()
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

temp = 300.0
press = 1.0
mpi = 1
omp = 14
gpu = 1
gpu_id = 0

omp_psi4 = 12
mem_mb = 20000

# This example is intentionally larger than the neutral interface examples.
# It builds a free bulk CMC phase first, learns the relaxed polymer XY footprint
# from that bulk, then matches the electrolyte to the polymer XY footprint
# before assembling a vacuum-buffered diffusion interface.
n_CMC = 6
dp = 150
polymer_initial_pack_density_g_cm3 = 0.015
polymer_pack_retry = 80
polymer_pack_retry_step = 6000
polymer_pack_threshold_ang = 1.65
polymer_pack_dec_rate = 0.75

solvent_mass_ratio = (3.0, 2.0, 5.0)
salt_molarity_M = 1.0
min_salt_pairs = 20
electrolyte_probe_density_g_cm3 = 1.00
electrolyte_probe_volume_scale = 2.30
electrolyte_probe_pack_density_g_cm3 = 0.48
electrolyte_resized_pack_density_g_cm3 = 0.32

electrolyte_pack_retry = 60
electrolyte_pack_retry_step = 2500
electrolyte_pack_threshold_ang = 1.45
electrolyte_pack_dec_rate = 0.75

interface_gap_nm = 0.80
interface_vacuum_nm = 14.0
interface_bottom_thickness_nm = 7.0
interface_top_thickness_nm = 8.0
interface_surface_shell_nm = 1.0
interface_core_guard_nm = 0.8

BASE_DIR = Path(__file__).resolve().parent
work_dir = workdir(BASE_DIR / "work_dir", clean=not restart)


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    mol2_dir = work_dir / "00_molecules"
    ac_CMC_dir = work_dir.child("ac_CMC")
    ac_CMC_build_dir = ac_CMC_dir.child("00_build_cell")
    probe_electrolyte_dir = work_dir.child("probe_electrolyte")
    probe_electrolyte_build_dir = probe_electrolyte_dir.child("00_build_cell")
    ac_electrolyte_dir = work_dir.child("ac_electrolyte")
    ac_electrolyte_build_dir = ac_electrolyte_dir.child("00_build_cell")
    interface_dir = work_dir.child("interface_route_b")
    interface_md_dir = work_dir.child("interface_route_b_md")

    glucose = ff.mol(glucose_smiles)
    glucose = ff.ff_assign(glucose)
    if not glucose:
        raise RuntimeError("Can not assign force field parameters for glucose.")
    write_mol2(mol=glucose, out_dir=mol2_dir)

    glucose_2 = ff.mol(glucose_2_smiles)
    glucose_2 = ff.ff_assign(glucose_2)
    if not glucose_2:
        raise RuntimeError("Can not assign force field parameters for glucose_2.")
    write_mol2(mol=glucose_2, out_dir=mol2_dir)

    glucose_3 = ff.mol(glucose_3_smiles)
    glucose_3 = ff.ff_assign(glucose_3)
    if not glucose_3:
        raise RuntimeError("Can not assign force field parameters for glucose_3.")
    write_mol2(mol=glucose_3, out_dir=mol2_dir)

    glucose_6 = ff.mol(glucose_6_smiles)
    glucose_6 = ff.ff_assign(glucose_6)
    if not glucose_6:
        raise RuntimeError("Can not assign force field parameters for glucose_6.")
    write_mol2(mol=glucose_6, out_dir=mol2_dir)

    ter1 = utils.mol_from_smiles(ter_smiles)
    qm.assign_charges(
        ter1,
        charge="RESP",
        opt=True,
        work_dir=work_dir,
        omp=omp_psi4,
        memory=mem_mb,
        log_name=None,
    )
    cmc_rw_dir = work_dir.child("CMC_rw")
    cmc_term_dir = work_dir.child("CMC_term")
    CMC = poly.random_copolymerize_rw(
        [glucose, glucose_2, glucose_3, glucose_6],
        dp,
        ratio=feed_prob,
        tacticity="atactic",
        name="CMC",
        work_dir=cmc_rw_dir,
    )
    CMC = poly.terminate_rw(CMC, ter1, name="CMC", work_dir=cmc_term_dir)
    CMC = ff.ff_assign(CMC)
    if not CMC:
        raise RuntimeError("Can not assign force field parameters for CMC.")
    write_mol2(mol=CMC, out_dir=mol2_dir)

    EC = ff.mol(EC_smiles)
    EC = ff.ff_assign(EC)
    if not EC:
        raise RuntimeError("Can not assign force field parameters for EC.")
    write_mol2(mol=EC, out_dir=mol2_dir)

    DEC = ff.mol(DEC_smiles)
    DEC = ff.ff_assign(DEC)
    if not DEC:
        raise RuntimeError("Can not assign force field parameters for DEC.")
    write_mol2(mol=DEC, out_dir=mol2_dir)

    EMC = ff.mol(EMC_smiles)
    EMC = ff.ff_assign(EMC)
    if not EMC:
        raise RuntimeError("Can not assign force field parameters for EMC.")
    write_mol2(mol=EMC, out_dir=mol2_dir)

    Li = ion_ff.mol(Li_smiles)
    Li = ion_ff.ff_assign(Li)
    if not Li:
        raise RuntimeError("Can not assign force field parameters for Li.")
    write_mol2(mol=Li, out_dir=mol2_dir)

    Na = ion_ff.mol(Na_smiles)
    Na = ion_ff.ff_assign(Na)
    if not Na:
        raise RuntimeError("Can not assign force field parameters for Na.")
    write_mol2(mol=Na, out_dir=mol2_dir)

    try:
        PF6 = ff.mol(PF6_smiles, charge="RESP", require_ready=True, prefer_db=True)
        PF6 = ff.ff_assign(PF6, bonded="DRIH")
    except Exception as exc:
        raise RuntimeError(
            "PF6 is expected to be precomputed in MolDB for Example 12. "
            "Please build it first with examples/01_Li_salt/run_pf6_to_moldb.py."
        ) from exc
    if not PF6:
        raise RuntimeError("Can not assign force field parameters for MolDB-backed PF6.")
    write_mol2(mol=PF6, out_dir=mol2_dir)

    q_poly = int(sum(int(atom.GetFormalCharge()) for atom in CMC.GetAtoms()))
    n_Na = int(abs(q_poly) * n_CMC) if q_poly != 0 else 0

    mw_CMC = molecular_weight(CMC, strict=True)
    mw_Na = molecular_weight(Na, strict=True)
    mw_EC = molecular_weight(EC, strict=True)
    mw_DEC = molecular_weight(DEC, strict=True)
    mw_EMC = molecular_weight(EMC, strict=True)
    mw_Li = molecular_weight(Li, strict=True)
    mw_PF6 = molecular_weight(PF6, strict=True)

    ac_CMC = poly.amorphous_cell(
        [CMC, Na],
        [n_CMC, n_Na],
        density=polymer_initial_pack_density_g_cm3,
        retry=polymer_pack_retry,
        retry_step=polymer_pack_retry_step,
        threshold=polymer_pack_threshold_ang,
        dec_rate=polymer_pack_dec_rate,
        charge_scale=[0.8, 0.8],
        neutralize=False,
        work_dir=ac_CMC_build_dir,
    )
    print(format_cell_charge_audit("CMC packed cell", ac_CMC))
    equilibrate_bulk_with_eq21(
        label="CMC",
        ac=ac_CMC,
        work_dir=ac_CMC_dir,
        temp=temp,
        press=press,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
    )
    cmc_box_nm = read_equilibrated_box_nm(work_dir=ac_CMC_dir)

    probe_interface_prep = plan_probe_polymer_matched_interface_preparation(
        reference_box_nm=cmc_box_nm,
        bottom_thickness_nm=interface_bottom_thickness_nm,
        top_thickness_nm=interface_top_thickness_nm,
        gap_nm=interface_gap_nm,
        surface_shell_nm=interface_surface_shell_nm,
        target_density_g_cm3=electrolyte_probe_density_g_cm3,
        solvent_mol_weights=[mw_EC, mw_DEC, mw_EMC],
        solvent_mass_ratio=solvent_mass_ratio,
        salt_mol_weights=[mw_Li, mw_PF6],
        salt_molarity_M=salt_molarity_M,
        min_salt_pairs=min_salt_pairs,
        solvent_species_names=["EC", "DEC", "EMC"],
        salt_species_names=["Li", "PF6"],
        min_solvent_counts=(1, 1, 1),
        probe_volume_scale=electrolyte_probe_volume_scale,
        initial_pack_density_g_cm3=electrolyte_probe_pack_density_g_cm3,
        z_padding_factor=1.20,
        is_polyelectrolyte=True,
        minimum_margin_nm=1.0,
        fixed_xy_npt_ns=5.0,
    )
    for note in probe_interface_prep.notes:
        print("[PLAN]", note)

    ac_electrolyte_probe = poly.amorphous_cell(
        [EC, DEC, EMC, Li, PF6],
        list(probe_interface_prep.probe_prep.direct_plan.target_counts),
        density=probe_interface_prep.probe_prep.build_density_g_cm3,
        retry=electrolyte_pack_retry,
        retry_step=electrolyte_pack_retry_step,
        threshold=electrolyte_pack_threshold_ang,
        dec_rate=electrolyte_pack_dec_rate,
        charge_scale=[1.0, 1.0, 1.0, 0.8, 0.8],
        neutralize=False,
        work_dir=probe_electrolyte_build_dir,
    )
    print(format_cell_charge_audit("Electrolyte probe packed cell", ac_electrolyte_probe))
    equilibrate_bulk_with_eq21(
        label="Electrolyte probe",
        ac=ac_electrolyte_probe,
        work_dir=probe_electrolyte_dir,
        temp=temp,
        press=press,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
    )

    resized_interface_prep = plan_resized_polymer_matched_interface_from_probe(
        reference_box_nm=cmc_box_nm,
        interface_plan=probe_interface_prep.interface_plan,
        probe_work_dir=probe_electrolyte_dir,
        probe_counts=probe_interface_prep.probe_prep.direct_plan.target_counts,
        mol_weights=[mw_EC, mw_DEC, mw_EMC, mw_Li, mw_PF6],
        species_names=["EC", "DEC", "EMC", "Li", "PF6"],
        solvent_indices=(0, 1, 2),
        salt_pair_indices=(3, 4),
        min_solvent_counts=(1, 1, 1),
        min_salt_pairs=min_salt_pairs,
        initial_pack_density_g_cm3=electrolyte_resized_pack_density_g_cm3,
        z_padding_factor=1.25,
        minimum_pack_z_factor=2.80,
        pressure_bar=press,
    )
    for note in resized_interface_prep.notes:
        print("[PLAN]", note)

    ac_electrolyte = poly.amorphous_cell(
        [EC, DEC, EMC, Li, PF6],
        list(resized_interface_prep.resized_prep.resize_plan.target_counts),
        cell=make_orthorhombic_pack_cell(resized_interface_prep.resized_prep.pack_plan.initial_pack_box_nm),
        density=None,
        retry=electrolyte_pack_retry,
        retry_step=electrolyte_pack_retry_step,
        threshold=electrolyte_pack_threshold_ang,
        dec_rate=electrolyte_pack_dec_rate,
        charge_scale=[1.0, 1.0, 1.0, 0.8, 0.8],
        neutralize=False,
        work_dir=ac_electrolyte_build_dir,
    )
    print(format_cell_charge_audit("Electrolyte packed cell", ac_electrolyte))
    equilibrate_bulk_with_eq21(
        label="Electrolyte",
        ac=ac_electrolyte,
        work_dir=ac_electrolyte_dir,
        temp=temp,
        press=press,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
        eq21_npt_mdp_overrides=resized_interface_prep.resized_prep.relax_mdp_overrides,
        additional_mdp_overrides=resized_interface_prep.resized_prep.relax_mdp_overrides,
        final_npt_ns=probe_interface_prep.interface_plan.electrolyte_alignment.fixed_xy_npt_ns,
        final_npt_mdp_overrides=resized_interface_prep.resized_prep.relax_mdp_overrides,
    )

    recipe = recommend_polymer_diffusion_interface_recipe(
        interface_plan=resized_interface_prep.interface_plan,
        temperature_k=temp,
        pressure_bar=press,
        vacuum_nm=interface_vacuum_nm,
        core_guard_nm=interface_core_guard_nm,
        top_lateral_shift_fraction=(0.31, 0.57),
        wall_atomtype="OW",
        pre_contact_ps=180.0,
        density_relax_ps=500.0,
        contact_ps=500.0,
        release_ps=500.0,
        exchange_ns=4.0,
        production_ns=8.0,
    )
    for note in recipe.notes:
        print("[ROUTE]", note)

    built = InterfaceBuilder(work_dir=interface_dir).build_from_bulk_workdirs(
        name="CMC_vs_LiPF6_electrolyte",
        bottom_name="ac_CMC",
        bottom_work_dir=ac_CMC_dir,
        top_name="ac_electrolyte",
        top_work_dir=ac_electrolyte_dir,
        route=recipe.route_spec,
    )
    final_interface_gro = InterfaceDynamics(built=built, work_dir=interface_md_dir).run(
        protocol=recipe.protocol,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
    )

    print("[INFO] Example 12 finished.")
    print("  CMC formal charge per chain:", q_poly)
    print("  Na counter ions:", n_Na)
    print("  CMC box (nm):", tuple(round(float(x), 4) for x in cmc_box_nm))
    print(
        "  interface XY (nm):",
        tuple(round(float(x), 4) for x in resized_interface_prep.interface_plan.interface_xy_nm),
    )
    print(
        "  probe electrolyte box (nm):",
        tuple(round(float(x), 4) for x in probe_interface_prep.probe_prep.probe_box_nm),
    )
    print(
        "  final electrolyte target box (nm):",
        tuple(round(float(x), 4) for x in resized_interface_prep.interface_plan.electrolyte_target_box_nm),
    )
    print(
        "  probe electrolyte counts:",
        dict(
            zip(
                probe_interface_prep.probe_prep.direct_plan.species_names,
                probe_interface_prep.probe_prep.direct_plan.target_counts,
            )
        ),
    )
    print(
        "  resized electrolyte counts:",
        dict(
            zip(
                resized_interface_prep.resized_prep.resize_plan.species_names,
                resized_interface_prep.resized_prep.resize_plan.target_counts,
            )
        ),
    )
    print("  route:", recipe.route_spec.route)
    print("  staged protocol:", [stage.name for stage in recipe.protocol.stages()])
    print("  GRO:", built.system_gro)
    print("  TOP:", built.system_top)
    print("  NDX:", built.system_ndx)
    print("  final interface GRO:", final_interface_gro)
