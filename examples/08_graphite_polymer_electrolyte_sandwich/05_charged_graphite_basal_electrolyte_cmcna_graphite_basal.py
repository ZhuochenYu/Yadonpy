from __future__ import annotations

"""Example 08-05: fixed-charge graphite sweep for a four-layer stack.

The fixed-charge model is not constant potential.  It distributes a prescribed
surface charge density onto the interior graphite faces, leaving the outer
faces neutral.  The default sweep is small and intended for fast EDL sanity
checks before committing to larger production cells.
"""

from pathlib import Path

from yadonpy import clean_md_trajectory_files
from yadonpy.core import poly, utils, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.merz import MERZ
from yadonpy.interface import (
    ElectrodeChargeSpec,
    GraphiteLayerSpec,
    LayerStackRelaxationSpec,
    LayerStackSpec,
    MolecularLayerSpec,
    analyze_layer_stack_interface,
    build_layer_stack,
    run_layer_stack_nvt,
)
from yadonpy.runtime import set_run_options


# ---------------- user inputs ----------------
restart_status = True
set_run_options(restart=restart_status)

ff = GAFF2_mod()
ion_ff = MERZ()

temp = 318.15
mpi = 1
omp = 14
gpu = 1
gpu_id = 0
run_sampling = True
sample_ns = 2.0
surface_charge_sweep_uC_cm2 = (0.0, 2.0, -2.0, 5.0, -5.0)
analysis_profile = "interface_fast"
interface_bin_nm = 0.05
interface_region_width_nm = 0.75
graphite_adsorption_cutoff_nm = 0.50
penetration_species = ("EC", "EMC", "DEC", "PF6", "Li", "Na")
adsorption_species = ("EC", "EMC", "DEC")
clean_trajectories_after_analysis = False

glucose6_smiles = "*OC1OC(COCC(=O)[O-])C(*)C(O)C1O"
ter_smiles = "[H][*]"
EC_smiles = "O=C1OCCO1"
EMC_smiles = "CCOC(=O)OC"
DEC_smiles = "CCOC(=O)OCC"
Li_smiles = "[Li+]"
Na_smiles = "[Na+]"
PF6_smiles = "F[P-](F)(F)(F)(F)F"

cmc_dp = 5
cmc_chain_count = 1
solvent_counts = (8, 6, 14)
salt_pairs = 3
charge_scale = 0.7

BASE_DIR = Path(__file__).resolve().parent
work_dir = BASE_DIR / "work_dir" / "05_charged_graphite_basal_electrolyte_cmcna_graphite_basal"


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()
    work_dir = workdir(work_dir, restart=restart_status)

    glucose6 = ff.mol(glucose6_smiles, charge="RESP", prefer_db=True, require_ready=True, polyelectrolyte_mode=True)
    glucose6 = ff.ff_assign(glucose6, report=False)
    ter = utils.mol_from_smiles(ter_smiles)
    EC = ff.mol(EC_smiles, charge="RESP", prefer_db=True, require_ready=True)
    EC = ff.ff_assign(EC, report=False)
    EMC = ff.mol(EMC_smiles, charge="RESP", prefer_db=True, require_ready=True)
    EMC = ff.ff_assign(EMC, report=False)
    DEC = ff.mol(DEC_smiles, charge="RESP", prefer_db=True, require_ready=True)
    DEC = ff.ff_assign(DEC, report=False)
    PF6 = ff.mol(PF6_smiles, charge="RESP", prefer_db=True, require_ready=True)
    PF6 = ff.ff_assign(PF6, bonded="DRIH", report=False)
    Li = ion_ff.mol(Li_smiles)
    Li = ion_ff.ff_assign(Li, report=False)
    Na = ion_ff.mol(Na_smiles)
    Na = ion_ff.ff_assign(Na, report=False)
    if not all((glucose6, ter, EC, EMC, DEC, PF6, Li, Na)):
        raise RuntimeError("MolDB/FF assignment failed for one or more species.")

    for surface_charge in surface_charge_sweep_uC_cm2:
        case_name = f"charge_{surface_charge:+.1f}_uC_cm2".replace("+", "p").replace("-", "m").replace(".", "p")
        case_dir = work_dir.child(case_name)
        cmc_rw_dir = case_dir.child("00_cmc_rw")
        cmc_term_dir = case_dir.child("01_cmc_term")

        CMC = poly.random_copolymerize_rw([glucose6], cmc_dp, ratio=[1.0], tacticity="atactic", work_dir=cmc_rw_dir)
        CMC = poly.terminate_rw(CMC, ter, work_dir=cmc_term_dir)
        CMC = ff.ff_assign(CMC, report=False)
        if not CMC:
            raise RuntimeError("Cannot assign GAFF2_mod parameters to the CMC-Na chain.")

        graphite_bottom = GraphiteLayerSpec(
            name="GRAPHITE_BOTTOM",
            nx=6,
            ny=5,
            n_layers=3,
            orientation="basal",
            periodic_xy=True,
            electrode_charge=ElectrodeChargeSpec(
                mode="surface_charge_density",
                top_surface_charge_uC_cm2=surface_charge,
            ),
        )
        graphite_top = GraphiteLayerSpec(
            name="GRAPHITE_TOP",
            nx=6,
            ny=5,
            n_layers=3,
            orientation="basal",
            periodic_xy=True,
            electrode_charge=ElectrodeChargeSpec(
                mode="surface_charge_density",
                bottom_surface_charge_uC_cm2=-surface_charge,
            ),
        )
        electrolyte = MolecularLayerSpec(
            name="ELECTROLYTE",
            species=(EC, EMC, DEC, Li, PF6),
            counts=(*solvent_counts, salt_pairs, salt_pairs),
            thickness_nm=2.2,
            density_target_g_cm3=1.2,
            layer_kind="electrolyte",
            charge_scale=(1.0, 1.0, 1.0, charge_scale, charge_scale),
        )
        cmcna = MolecularLayerSpec(
            name="CMCNA",
            species=(CMC, Na),
            counts=(cmc_chain_count, cmc_chain_count * cmc_dp),
            thickness_nm=1.8,
            density_target_g_cm3=1.25,
            layer_kind="cmcna",
            charge_scale=(charge_scale, charge_scale),
            polyelectrolyte_mode=True,
        )
        stack = LayerStackSpec(
            layers=(graphite_bottom, electrolyte, cmcna, graphite_top),
            order="bottom_to_top",
            pbc_mode="xyz",
            name=f"charged_graphite_stack_{case_name}",
            default_gap_nm=0.35,
        )
        relaxation = LayerStackRelaxationSpec(temperature_K=temp, sample_ns=sample_ns)

        result = build_layer_stack(stack=stack, relaxation=relaxation, work_dir=case_dir, restart=restart_status)
        print(f"[{surface_charge:+.1f} uC/cm2] layer_stack_manifest = {result.manifest_path}")
        print(f"[{surface_charge:+.1f} uC/cm2] stack_gmx_dir = {result.system_gro.parent}")
        print(f"[{surface_charge:+.1f} uC/cm2] acceptance = {result.acceptance}")

        analyze_layer_stack_interface(
            work_dir=case_dir,
            manifest_path=result.manifest_path,
            analysis_profile=analysis_profile,
            bin_nm=interface_bin_nm,
            region_width_nm=interface_region_width_nm,
            surface_distance_nm=graphite_adsorption_cutoff_nm,
            penetration_species=penetration_species,
            adsorption_species=adsorption_species,
            compute_transport=False,
        )
        if run_sampling:
            nvt = run_layer_stack_nvt(
                result,
                time_ns=sample_ns,
                temp=temp,
                mpi=mpi,
                omp=omp,
                gpu=gpu,
                gpu_id=gpu_id,
                dt_ps=0.001,
                constraints="none",
                run_analysis=False,
                restart=restart_status,
            )
            print(f"[{surface_charge:+.1f} uC/cm2] nvt_summary = {nvt.summary_path}")
            analy = nvt.analyze()
            interface = analy.interface(
                manifest_path=result.manifest_path,
                analysis_profile=analysis_profile,
                bin_nm=interface_bin_nm,
                region_width_nm=interface_region_width_nm,
                surface_distance_nm=graphite_adsorption_cutoff_nm,
                penetration_species=penetration_species,
                adsorption_species=adsorption_species,
                split_electrodes=True,
                report_potential_drop=True,
            )
            health = interface.geometry_health()
            z_profile = interface.z_profiles()
            edl = interface.edl_profiles(
                split_electrodes=True,
                potential_reference="zero_mean",
                report_potential_drop=True,
            )
            penetration = interface.penetration(species=penetration_species)
            adsorption = interface.graphite_adsorption(species=adsorption_species)
            coordination = interface.coordination_by_region()
            transport = interface.region_transport()
            summary = interface.summary()
            print(f"[{surface_charge:+.1f} uC/cm2] interface_phase_order_ok = {health.get('phase_order_ok')}")
            print(f"[{surface_charge:+.1f} uC/cm2] interface_outputs = {summary.get('outputs', {}).get('interface_profile_summary_json')}")

        clean_md_trajectory_files(case_dir, enabled=clean_trajectories_after_analysis)
