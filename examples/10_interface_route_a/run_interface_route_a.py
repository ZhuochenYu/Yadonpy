from __future__ import annotations

from pathlib import Path

from yadonpy.runtime import set_run_options
from yadonpy.core import molecular_weight, poly, utils, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff import GAFF2_mod, MERZ
from yadonpy.interface import (
    AreaMismatchPolicy,
    InterfaceBuilder,
    InterfaceDynamics,
    InterfaceProtocol,
    InterfaceRouteSpec,
    equilibrate_bulk_with_eq21,
    make_orthorhombic_pack_cell,
    plan_direct_polymer_matched_interface_preparation,
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


def assign_template_species(ff_obj, smiles: str, name: str, *, mol2_dir: Path, bonded: str | None = None):
    mol = ff_obj.mol(smiles, name=name)
    kwargs = {} if bonded is None else {"bonded": bonded}
    ok = ff_obj.ff_assign(mol, **kwargs)
    if not ok:
        raise RuntimeError(f"Can not assign force field parameters for {name}.")
    mol = _named(_resolved(mol), name)
    write_mol2_from_rdkit(mol=mol, out_dir=mol2_dir)
    return mol


restart = True
set_run_options(restart=restart)

ff = GAFF2_mod()
ion_ff = MERZ()

polymer_smiles = r"*CC1=CC=C(CCC2=CC=C(C*)C=C2)C=C1"
ter_smiles = "[H][*]"
EC_smiles = "O=C1OCCO1"
DEC_smiles = "CCOC(=O)OCC"
EMC_smiles = "CCOC(=O)OC"
Li_smiles = "[Li+]"
PF6_smiles = "F[P-](F)(F)(F)(F)F"

temp = 300.0
press = 1.0
mpi = 1
omp = 14
gpu = 1
gpu_id = 0
omp_psi4 = 12
mem_mb = 20000

polymer_num_atoms = 900
polymer_chain_count = 6
polymer_pack_density_g_cm3 = 0.08

solvent_mass_ratio = (3.0, 2.0, 5.0)
salt_molarity_M = 1.0
min_salt_pairs = 12
electrolyte_target_density_g_cm3 = 1.32
electrolyte_pack_density_g_cm3 = 0.85

interface_gap_nm = 0.60
interface_bottom_thickness_nm = 4.5
interface_top_thickness_nm = 5.0
interface_surface_shell_nm = 0.8
interface_core_guard_nm = 0.5

BASE_DIR = Path(__file__).resolve().parent
work_dir = workdir(BASE_DIR / "work_dir", clean=not restart)


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()

    mol2_dir = work_dir / "00_molecules"
    poly_P_dir = work_dir.child("poly_P")
    poly_P_term_dir = work_dir.child("poly_P_term")
    ac_poly_dir = work_dir.child("ac_poly")
    ac_poly_build_dir = ac_poly_dir.child("00_build_cell")
    ac_electrolyte_dir = work_dir.child("ac_electrolyte")
    ac_electrolyte_build_dir = ac_electrolyte_dir.child("00_build_cell")
    interface_dir = work_dir.child("interface_route_a")
    interface_md_dir = work_dir.child("interface_route_a_md")

    monomer_P = assign_template_species(ff, polymer_smiles, "monomer_P", mol2_dir=mol2_dir)

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

    dp = max(1, int(poly.calc_n_from_num_atoms(monomer_P, polymer_num_atoms, terminal1=ter1)))
    poly_P = poly.polymerize_rw(monomer_P, dp, tacticity="atactic", work_dir=poly_P_dir)
    poly_P = poly.terminate_rw(poly_P, ter1, name="poly_P", work_dir=poly_P_term_dir)
    if not ff.ff_assign(poly_P):
        raise RuntimeError("Can not assign force field parameters for poly_P.")
    poly_P = _named(poly_P, "poly_P")
    write_mol2_from_rdkit(mol=poly_P, out_dir=mol2_dir)

    EC = assign_template_species(ff, EC_smiles, "EC", mol2_dir=mol2_dir)
    DEC = assign_template_species(ff, DEC_smiles, "DEC", mol2_dir=mol2_dir)
    EMC = assign_template_species(ff, EMC_smiles, "EMC", mol2_dir=mol2_dir)
    Li = assign_template_species(ion_ff, Li_smiles, "Li", mol2_dir=mol2_dir)
    try:
        PF6 = ff.mol(PF6_smiles, name="PF6", charge="RESP", require_ready=True, prefer_db=True)
        PF6 = ff.ff_assign(PF6, bonded="DRIH")
    except Exception as exc:
        raise RuntimeError(
            "PF6 is expected to be precomputed in MolDB for this example. "
            "Please build it first with examples/01_Li_salt/run_pf6_to_moldb.py."
        ) from exc
    if not PF6:
        raise RuntimeError("Can not assign force field parameters for MolDB-backed PF6.")
    PF6 = _named(_resolved(PF6), "PF6")
    write_mol2_from_rdkit(mol=PF6, out_dir=mol2_dir)

    ac_poly = poly.amorphous_cell(
        [poly_P],
        [polymer_chain_count],
        charge_scale=[1.0],
        density=polymer_pack_density_g_cm3,
        neutralize=False,
        work_dir=ac_poly_build_dir,
    )
    poly_eq = equilibrate_bulk_with_eq21(
        label="Polymer",
        ac=ac_poly,
        work_dir=ac_poly_dir,
        temp=temp,
        press=press,
        mpi=mpi,
        omp=omp,
        gpu=gpu,
        gpu_id=gpu_id,
    )
    poly_box_nm = read_equilibrated_box_nm(work_dir=ac_poly_dir)

    interface_prep = plan_direct_polymer_matched_interface_preparation(
        reference_box_nm=poly_box_nm,
        bottom_thickness_nm=interface_bottom_thickness_nm,
        top_thickness_nm=interface_top_thickness_nm,
        gap_nm=interface_gap_nm,
        surface_shell_nm=interface_surface_shell_nm,
        target_density_g_cm3=electrolyte_target_density_g_cm3,
        solvent_mol_weights=[molecular_weight(EC, strict=True), molecular_weight(DEC, strict=True), molecular_weight(EMC, strict=True)],
        solvent_mass_ratio=solvent_mass_ratio,
        salt_mol_weights=[molecular_weight(Li, strict=True), molecular_weight(PF6, strict=True)],
        salt_molarity_M=salt_molarity_M,
        min_salt_pairs=min_salt_pairs,
        solvent_species_names=["EC", "DEC", "EMC"],
        salt_species_names=["Li", "PF6"],
        min_solvent_counts=(1, 1, 1),
        initial_pack_density_g_cm3=electrolyte_pack_density_g_cm3,
        pressure_bar=press,
    )
    for note in interface_prep.notes:
        print("[PLAN]", note)

    ac_electrolyte = poly.amorphous_cell(
        [EC, DEC, EMC, Li, PF6],
        list(interface_prep.electrolyte_prep.direct_plan.target_counts),
        cell=make_orthorhombic_pack_cell(interface_prep.electrolyte_prep.pack_plan.initial_pack_box_nm),
        density=None,
        charge_scale=[1.0, 1.0, 1.0, 0.8, 0.8],
        neutralize=False,
        work_dir=ac_electrolyte_build_dir,
    )
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
        eq21_npt_mdp_overrides=interface_prep.electrolyte_prep.relax_mdp_overrides,
        additional_mdp_overrides=interface_prep.electrolyte_prep.relax_mdp_overrides,
        final_npt_ns=interface_prep.interface_plan.electrolyte_alignment.fixed_xy_npt_ns,
        final_npt_mdp_overrides=interface_prep.electrolyte_prep.relax_mdp_overrides,
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
        name="polymer_vs_electrolyte",
        bottom_name="ac_poly",
        bottom_work_dir=ac_poly_dir,
        top_name="ac_electrolyte",
        top_work_dir=ac_electrolyte_dir,
        route=route,
    )
    protocol = InterfaceProtocol.route_a_diffusion(
        axis="Z",
        temperature_k=temp,
        pressure_bar=press,
        pre_contact_ps=120.0,
        density_relax_ps=250.0,
        contact_ps=250.0,
        release_ps=250.0,
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

    print("[INFO] Example 10 finished.")
    print("  polymer box (nm):", tuple(round(float(x), 4) for x in poly_box_nm))
    print("  interface XY (nm):", tuple(round(float(x), 4) for x in interface_prep.interface_plan.interface_xy_nm))
    print("  electrolyte target box (nm):", tuple(round(float(x), 4) for x in interface_prep.interface_plan.electrolyte_target_box_nm))
    print("  staged protocol:", [stage.name for stage in protocol.stages()])
    print("  GRO:", built.system_gro)
    print("  TOP:", built.system_top)
    print("  NDX:", built.system_ndx)
    print("  final interface GRO:", final_interface_gro)
