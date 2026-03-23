from __future__ import annotations

from pathlib import Path

from yadonpy.runtime import set_run_options
from yadonpy.core import molecular_weight, poly, utils, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2, MERZ
from yadonpy.interface import (
    AreaMismatchPolicy,
    InterfaceBuilder,
    InterfaceDynamics,
    InterfaceProtocol,
    InterfaceRouteSpec,
    equilibrate_bulk_with_eq21,
    fixed_xy_semiisotropic_npt_overrides,
    format_cell_charge_audit,
    make_orthorhombic_pack_cell,
    plan_fixed_xy_direct_pack_box,
    plan_probe_polymer_matched_interface_preparation,
    plan_resized_polymer_matched_interface_from_probe,
    read_equilibrated_box_nm,
)
from yadonpy.io.mol2 import write_mol2_from_rdkit
from yadonpy.sim import qm


def _named(mol, name: str):
    try:
        mol.SetProp("_Name", name)
    except Exception:
        pass
    return mol


def _resolved(mol):
    resolved = getattr(mol, "resolved_mol", None)
    return resolved if resolved is not None else mol


def prepare_template_species(ff_obj, smiles: str, name: str, *, mol2_dir: Path, bonded: str | None = None):
    mol = ff_obj.mol(smiles, name=name)
    kwargs = {} if bonded is None else {"bonded": bonded}
    ok = ff_obj.ff_assign(mol, **kwargs)
    if not ok:
        raise RuntimeError(f"Can not assign force field parameters for {name}.")
    mol = _named(_resolved(mol), name)
    write_mol2_from_rdkit(mol=mol, out_dir=mol2_dir)
    return mol


def build_cmc(ff_obj: GAFF2, *, work_dir, mol2_dir: Path):
    glucose = prepare_template_species(ff_obj, glucose_smiles, "glucose", mol2_dir=mol2_dir)
    glucose_2 = prepare_template_species(ff_obj, glucose_2_smiles, "glucose_2", mol2_dir=mol2_dir)
    glucose_3 = prepare_template_species(ff_obj, glucose_3_smiles, "glucose_3", mol2_dir=mol2_dir)
    glucose_6 = prepare_template_species(ff_obj, glucose_6_smiles, "glucose_6", mol2_dir=mol2_dir)

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
    ter1 = _named(ter1, "ter1")

    cmc_rw_dir = work_dir.child("CMC_rw")
    cmc_term_dir = work_dir.child("CMC_term")
    cmc = poly.random_copolymerize_rw(
        [glucose, glucose_2, glucose_3, glucose_6],
        dp,
        ratio=feed_prob,
        tacticity="atactic",
        name="CMC",
        work_dir=cmc_rw_dir,
    )
    cmc = poly.terminate_rw(cmc, ter1, name="CMC", work_dir=cmc_term_dir)
    if not ff_obj.ff_assign(cmc):
        raise RuntimeError("Can not assign force field parameters for CMC.")
    cmc = _named(cmc, "CMC")
    write_mol2_from_rdkit(mol=cmc, out_dir=mol2_dir)
    return cmc


def mol_formal_charge(mol) -> int:
    return int(sum(int(atom.GetFormalCharge()) for atom in mol.GetAtoms()))


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

n_CMC = 1
dp = 36
polymer_initial_pack_density_g_cm3 = 0.08
polymer_initial_lateral_nm = 5.2
polymer_initial_min_z_nm = 6.0
cmc_flatten_npt_ns = 2.5

solvent_mass_ratio = (3.0, 2.0, 5.0)
salt_molarity_M = 0.6
min_salt_pairs = 8
electrolyte_probe_density_g_cm3 = 0.95
electrolyte_probe_volume_scale = 1.75
electrolyte_probe_pack_density_g_cm3 = 0.60
electrolyte_resized_pack_density_g_cm3 = 0.45

electrolyte_pack_retry = 24
electrolyte_pack_retry_step = 1500
electrolyte_pack_threshold_ang = 1.50
electrolyte_pack_dec_rate = 0.65

interface_gap_nm = 0.60
interface_bottom_thickness_nm = 3.2
interface_top_thickness_nm = 3.4
interface_surface_shell_nm = 0.8
interface_core_guard_nm = 0.5

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
    interface_dir = work_dir.child("interface_route_a")
    interface_md_dir = work_dir.child("interface_route_a_md")

    CMC = build_cmc(ff, work_dir=work_dir, mol2_dir=mol2_dir)
    EC = prepare_template_species(ff, EC_smiles, "EC", mol2_dir=mol2_dir)
    DEC = prepare_template_species(ff, DEC_smiles, "DEC", mol2_dir=mol2_dir)
    EMC = prepare_template_species(ff, EMC_smiles, "EMC", mol2_dir=mol2_dir)
    Li = prepare_template_species(ion_ff, Li_smiles, "Li", mol2_dir=mol2_dir)
    Na = prepare_template_species(ion_ff, Na_smiles, "Na", mol2_dir=mol2_dir)
    try:
        PF6 = ff.mol(PF6_smiles, name="PF6", charge="RESP", require_ready=True, prefer_db=True)
        PF6 = ff.ff_assign(PF6, bonded="DRIH")
    except Exception as exc:
        raise RuntimeError(
            "PF6 is expected to be precomputed in MolDB for Example 12. "
            "Please build it first with examples/01_Li_salt/run_pf6_to_moldb.py."
        ) from exc
    if not PF6:
        raise RuntimeError("Can not assign force field parameters for MolDB-backed PF6.")
    PF6 = _named(_resolved(PF6), "PF6")
    write_mol2_from_rdkit(mol=PF6, out_dir=mol2_dir)

    q_poly = mol_formal_charge(CMC)
    n_Na = int(abs(q_poly) * n_CMC) if q_poly != 0 else 0

    mw_CMC = molecular_weight(CMC, strict=True)
    mw_Na = molecular_weight(Na, strict=True)
    mw_EC = molecular_weight(EC, strict=True)
    mw_DEC = molecular_weight(DEC, strict=True)
    mw_EMC = molecular_weight(EMC, strict=True)
    mw_Li = molecular_weight(Li, strict=True)
    mw_PF6 = molecular_weight(PF6, strict=True)

    cmc_pack_plan = plan_fixed_xy_direct_pack_box(
        reference_box_nm=(polymer_initial_lateral_nm, polymer_initial_lateral_nm, polymer_initial_min_z_nm),
        target_counts=(n_CMC, n_Na),
        mol_weights=(mw_CMC, mw_Na),
        species_names=("CMC", "Na"),
        initial_pack_density_g_cm3=polymer_initial_pack_density_g_cm3,
        z_padding_factor=1.10,
        minimum_z_nm=polymer_initial_min_z_nm,
    )
    ac_CMC = poly.amorphous_cell(
        [CMC, Na],
        [n_CMC, n_Na],
        cell=make_orthorhombic_pack_cell(cmc_pack_plan.initial_pack_box_nm),
        density=None,
        charge_scale=[0.8, 0.8],
        neutralize=False,
        work_dir=ac_CMC_build_dir,
    )
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
        final_npt_ns=float(cmc_flatten_npt_ns),
        final_npt_mdp_overrides=fixed_xy_semiisotropic_npt_overrides(pressure_bar=press),
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
        z_padding_factor=1.15,
        is_polyelectrolyte=True,
        minimum_margin_nm=0.8,
        fixed_xy_npt_ns=4.0,
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
        z_padding_factor=1.20,
        minimum_pack_z_factor=2.40,
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

    builder = InterfaceBuilder(work_dir=interface_dir)
    route = InterfaceRouteSpec.route_a(
        axis="Z",
        gap_nm=interface_gap_nm,
        bottom_thickness_nm=interface_bottom_thickness_nm,
        top_thickness_nm=interface_top_thickness_nm,
        surface_shell_nm=interface_surface_shell_nm,
        core_guard_nm=interface_core_guard_nm,
        area_policy=AreaMismatchPolicy(reference_side="bottom", max_lateral_strain=0.08),
    )
    built = builder.build_from_bulk_workdirs(
        name="CMC_vs_LiPF6_electrolyte",
        bottom_name="ac_CMC",
        bottom_work_dir=ac_CMC_dir,
        top_name="ac_electrolyte",
        top_work_dir=ac_electrolyte_dir,
        route=route,
    )
    protocol = InterfaceProtocol.route_a_diffusion(
        axis="Z",
        temperature_k=temp,
        pressure_bar=press,
        pre_contact_ps=120.0,
        density_relax_ps=300.0,
        contact_ps=300.0,
        release_ps=300.0,
        exchange_ns=2.0,
        production_ns=5.0,
    )
    final_interface_gro = InterfaceDynamics(built=built, work_dir=interface_md_dir).run(
        protocol=protocol,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
    )

    print("[INFO] Example 12 finished.")
    print("  CMC box (nm):", tuple(round(float(x), 4) for x in cmc_box_nm))
    print("  interface XY (nm):", tuple(round(float(x), 4) for x in probe_interface_prep.interface_plan.interface_xy_nm))
    print("  probe electrolyte box (nm):", tuple(round(float(x), 4) for x in probe_interface_prep.probe_prep.probe_box_nm))
    print("  probe electrolyte counts:", dict(zip(probe_interface_prep.probe_prep.direct_plan.species_names, probe_interface_prep.probe_prep.direct_plan.target_counts)))
    print("  resized electrolyte counts:", dict(zip(resized_interface_prep.resized_prep.resize_plan.species_names, resized_interface_prep.resized_prep.resize_plan.target_counts)))
    print("  staged protocol:", [stage.name for stage in protocol.stages()])
    print("  GRO:", built.system_gro)
    print("  TOP:", built.system_top)
    print("  NDX:", built.system_ndx)
    print("  final interface GRO:", final_interface_gro)
