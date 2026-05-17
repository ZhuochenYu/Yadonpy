from __future__ import annotations

"""Example 08-06: large flat fixed-charge graphite | electrolyte | CMC-Na stack.

This is the production-sized counterpart of Example 08-05.  The default system
uses a broad basal-graphite XY footprint, DP=20 CMC-Na, and eight CMC chains.
The initial molecular layers are deliberately loose in z so packing succeeds;
the fixed-XY compression anneal and z-NPT stages then compact the confined
region before the final NVT trajectory is sampled and analyzed.
"""

from pathlib import Path

from yadonpy import clean_md_trajectory_files
from yadonpy.core import poly, utils, workdir
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.merz import MERZ
from yadonpy.interface import (
    FixedChargeRegionSpec,
    GraphiteLayerSpec,
    LayerStackRelaxationSpec,
    LayerStackSpec,
    MolecularLayerSpec,
    ZCompressionAnnealSpec,
    analyze_layer_stack_interface,
    build_layer_stack,
    run_layer_stack_relaxation,
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

# This script is meant for a real large-cell production test.  For a quick
# build-only check, set run_sampling=False above.
sample_ns = 20.0

# Keep the first large flat run neutral.  Change this single value to run a
# fixed-charge basal-electrode case after the neutral structure looks healthy.
surface_charge_uC_cm2 = 0.0

# ---------------- post-processing controls ----------------
analysis_profile = "interface_fast"
interface_bin_nm = 0.05
interface_region_width_nm = 0.75
interface_surface_grid_nm = 0.50
graphite_adsorption_cutoff_nm = 0.50
penetration_threshold_nm = 0.20
adsorption_min_residence_ps = 10.0
potential_reference = "zero_mean"
split_electrodes_for_edl = True
report_potential_drop = True
compute_interface_transport = True
time_series_analysis = True
interface_time_series_sample_count = 10
interface_time_series_fps = 1.0
interface_time_series_rdf = True
interface_time_series_concentration = True
interface_time_series_angles = True
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

# Large flat-cell composition.
cmc_dp = 20
cmc_chain_count = 8
solvent_counts = (96, 72, 168)
salt_pairs = 36
charge_scale = 0.7

BASE_DIR = Path(__file__).resolve().parent
work_dir = BASE_DIR / "work_dir" / "06_large_flat_charged_graphite_basal_electrolyte_cmcna_graphite_basal"


if __name__ == "__main__":
    doctor(print_report=True)
    ensure_initialized()
    work_dir = workdir(work_dir, restart=restart_status)

    case_name = f"charge_{surface_charge_uC_cm2:+.1f}_uC_cm2".replace("+", "p").replace("-", "m").replace(".", "p")
    case_dir = work_dir.child(case_name)
    cmc_rw_dir = case_dir.child("00_cmc_rw")
    cmc_term_dir = case_dir.child("01_cmc_term")

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

    CMC = poly.random_copolymerize_rw([glucose6], cmc_dp, ratio=[1.0], tacticity="atactic", work_dir=cmc_rw_dir)
    CMC = poly.terminate_rw(CMC, ter, work_dir=cmc_term_dir)
    CMC = ff.ff_assign(CMC, report=False)
    if not CMC:
        raise RuntimeError("Cannot assign GAFF2_mod parameters to the DP=20 CMC-Na chain.")

    graphite_bottom = GraphiteLayerSpec(
        name="GRAPHITE_BOTTOM",
        nx=20,
        ny=18,
        n_layers=3,
        orientation="basal",
        periodic_xy=True,
    )
    graphite_top = GraphiteLayerSpec(
        name="GRAPHITE_TOP",
        nx=20,
        ny=18,
        n_layers=3,
        orientation="basal",
        periodic_xy=True,
    )
    electrolyte = MolecularLayerSpec(
        name="ELECTROLYTE",
        species=(EC, EMC, DEC, Li, PF6),
        counts=(*solvent_counts, salt_pairs, salt_pairs),
        thickness_nm=2.2,
        density_target_g_cm3=1.15,
        layer_kind="electrolyte",
        charge_scale=(1.0, 1.0, 1.0, charge_scale, charge_scale),
        large_system_mode="large",
    )
    cmcna = MolecularLayerSpec(
        name="CMCNA",
        species=(CMC, Na),
        counts=(cmc_chain_count, cmc_chain_count * cmc_dp),
        thickness_nm=2.6,
        # Loose insertion target: DP=20 x 8 chains should not be packed directly
        # at the bulk CMC-Na reference density.  Compression annealing and z-NPT
        # compact this layer after all molecules have been placed.
        density_target_g_cm3=0.75,
        layer_kind="cmcna",
        charge_scale=(charge_scale, charge_scale),
        polyelectrolyte_mode=True,
        large_system_mode="large",
        counterion_contact_mode="carboxylate",
    )
    stack = LayerStackSpec(
        layers=(graphite_bottom, electrolyte, cmcna, graphite_top),
        order="bottom_to_top",
        pbc_mode="xyz",
        name=f"large_flat_charged_graphite_stack_{case_name}",
        default_gap_nm=0.35,
        molecular_packing_expand="z",
        fixed_charge_regions=(
            FixedChargeRegionSpec(
                layer_name="GRAPHITE_BOTTOM",
                region="top",
                mode="surface_charge_density",
                surface_charge_uC_cm2=surface_charge_uC_cm2,
                elements=("C",),
                label="bottom_graphite_inner_face",
            ),
            FixedChargeRegionSpec(
                layer_name="GRAPHITE_TOP",
                region="bottom",
                mode="surface_charge_density",
                surface_charge_uC_cm2=-surface_charge_uC_cm2,
                elements=("C",),
                label="top_graphite_inner_face",
            ),
        ),
    )
    relaxation = LayerStackRelaxationSpec(temperature_K=temp, sample_ns=sample_ns)

    result = build_layer_stack(stack=stack, relaxation=relaxation, work_dir=case_dir, restart=restart_status)
    print(f"layer_stack_manifest = {result.manifest_path}")
    print(f"stack_gmx_dir = {result.system_gro.parent}")
    print(f"acceptance = {result.acceptance}")

    analyze_layer_stack_interface(
        work_dir=case_dir,
        manifest_path=result.manifest_path,
        analysis_profile=analysis_profile,
        bin_nm=interface_bin_nm,
        region_width_nm=interface_region_width_nm,
        surface_grid_nm=interface_surface_grid_nm,
        surface_distance_nm=graphite_adsorption_cutoff_nm,
        penetration_threshold_nm=penetration_threshold_nm,
        adsorption_min_residence_ps=adsorption_min_residence_ps,
        potential_reference=potential_reference,
        split_electrodes=split_electrodes_for_edl,
        report_potential_drop=report_potential_drop,
        penetration_species=penetration_species,
        adsorption_species=adsorption_species,
        compute_transport=False,
        time_series_analysis=False,
    )

    if run_sampling:
        relax = run_layer_stack_relaxation(
            result,
            time_ns=sample_ns,
            temp=temp,
            mpi=mpi,
            omp=omp,
            gpu=gpu,
            gpu_id=gpu_id,
            dt_ps=0.001,
            constraints="none",
            run_analysis=True,
            relax_z=True,
            pre_nvt_ns=0.10,
            z_npt_ns=1.00,
            compression_anneal=ZCompressionAnnealSpec(
                enabled=True,
                cycles=12,
                tmax_K=400.0,
                pmax_bar=4000.0,
                max_z_shrink_per_cycle=0.03,
                hot_nvt_ns=0.02,
                compression_npt_ns=0.08,
                cool_nvt_ns=0.03,
            ),
            restart=restart_status,
        )
        print(f"relaxation_summary = {relax.summary_path}")

        analy = relax.analyze()
        interface = analy.interface(
            manifest_path=result.manifest_path,
            analysis_profile=analysis_profile,
            bin_nm=interface_bin_nm,
            region_width_nm=interface_region_width_nm,
            surface_grid_nm=interface_surface_grid_nm,
            surface_distance_nm=graphite_adsorption_cutoff_nm,
            penetration_threshold_nm=penetration_threshold_nm,
            adsorption_min_residence_ps=adsorption_min_residence_ps,
            potential_reference=potential_reference,
            penetration_species=penetration_species,
            adsorption_species=adsorption_species,
            split_electrodes=split_electrodes_for_edl,
            report_potential_drop=report_potential_drop,
            compute_transport=compute_interface_transport,
            time_series_sample_count=interface_time_series_sample_count,
            time_series_fps=interface_time_series_fps,
            time_series_rdf=interface_time_series_rdf,
            time_series_concentration=interface_time_series_concentration,
            time_series_angles=interface_time_series_angles,
        )
        health = interface.geometry_health(time_series_analysis=time_series_analysis)
        z_profile = interface.z_profiles(time_series_analysis=time_series_analysis)
        edl = interface.edl_profiles(
            split_electrodes=split_electrodes_for_edl,
            potential_reference=potential_reference,
            report_potential_drop=report_potential_drop,
            time_series_analysis=time_series_analysis,
        )
        penetration = interface.penetration(species=penetration_species, time_series_analysis=time_series_analysis)
        adsorption = interface.graphite_adsorption(
            species=adsorption_species,
            time_series_analysis=time_series_analysis,
        )
        coordination = interface.coordination_by_region(time_series_analysis=time_series_analysis)
        transport = interface.region_transport(time_series_analysis=time_series_analysis)
        time_series = interface.time_series(time_series_analysis=time_series_analysis)
        summary = interface.summary(time_series_analysis=time_series_analysis)
        print(f"interface_phase_order_ok = {health.get('phase_order_ok')}")
        print(f"interface_outputs = {summary.get('outputs', {}).get('interface_profile_summary_json')}")
        print(f"interface_time_series = {time_series.get('outputs', {})}")

    clean_md_trajectory_files(case_dir, enabled=clean_trajectories_after_analysis)
