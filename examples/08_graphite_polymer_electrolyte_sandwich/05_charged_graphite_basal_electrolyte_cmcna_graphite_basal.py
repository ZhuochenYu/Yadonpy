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
sample_ns = 2.0
surface_charge_sweep_uC_cm2 = (0.0, 2.0, -2.0, 5.0, -5.0)

# ---------------- post-processing controls ----------------
# `analysis_profile="interface_fast"` keeps the interface-specific analyses
# focused on robust slab observables instead of bulk 3D transport defaults.
analysis_profile = "interface_fast"

# z-bin width for density, charge, EDL species, integrated charge, and
# electrostatic-potential profiles. Smaller bins give sharper interfaces but
# need more trajectory frames to reduce noise.
interface_bin_nm = 0.05

# Width of automatically named near-interface regions. For this four-layer
# stack, the manifest defines layer order, then z-quantiles and density overlap
# define graphite-near, CMC/electrolyte mixed, and core-like regions.
interface_region_width_nm = 0.75

# xy grid used for graphite adsorption occupancy maps on the basal planes.
interface_surface_grid_nm = 0.50

# A molecule is counted as graphite-near adsorbed when its mass-weighted COM is
# within this distance of the nearest graphite quantile surface. This is a
# residence/geometric diagnostic, not a binding free energy.
graphite_adsorption_cutoff_nm = 0.50

# Minimum COM depth inside a CMC/polymer-rich or mixed region before a frame is
# counted as penetration. This avoids counting molecules that merely touch the
# region boundary due to z-bin noise.
penetration_threshold_nm = 0.20

# Minimum cumulative adsorbed residence used for the `passes_min_residence`
# flag in `adsorption_summary.json`; the raw frame fractions are always written.
adsorption_min_residence_ps = 10.0

# Potential is obtained by one-dimensional integration of sampled fixed-charge
# density using vacuum permittivity. `zero_mean` removes the mean potential;
# `zero_start` pins the first z bin. This is not a constant-potential electrode
# solver and should be interpreted as a slab diagnostic.
potential_reference = "zero_mean"
split_electrodes_for_edl = True
report_potential_drop = True

# Compute anisotropic MSD summaries from the NVT trajectory. Dxy is the main
# in-plane interface mobility metric; Dz is confined-direction mobility.
compute_interface_transport = True

# Generate slow MP4 animations for interface time evolution. This is off by
# default in the API and must be passed explicitly to each post-processing call.
# The trajectory is split into ten equal time windows by default, so RDF/CN,
# molecule-COM z concentration, and adsorbed-angle distributions are sampled at
# roughly every one-tenth of the total NVT duration instead of producing a dense
# movie.
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
            # Initial packing target only: keep insertion looser than bulk
            # CMC-Na (~1.5 g/cm3), then densify via compression annealing/z-NPT.
            density_target_g_cm3=1.0,
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
            molecular_packing_expand="z",
        )
        relaxation = LayerStackRelaxationSpec(temperature_K=temp, sample_ns=sample_ns)

        result = build_layer_stack(stack=stack, relaxation=relaxation, work_dir=case_dir, restart=restart_status)
        print(f"[{surface_charge:+.1f} uC/cm2] layer_stack_manifest = {result.manifest_path}")
        print(f"[{surface_charge:+.1f} uC/cm2] stack_gmx_dir = {result.system_gro.parent}")
        print(f"[{surface_charge:+.1f} uC/cm2] acceptance = {result.acceptance}")

        # Static-stack post-processing: this reads the freshly built `system.gro`
        # plus `system.top`, `system.ndx`, and `layer_stack_manifest.json`. It is
        # a geometry/charge sanity pass before NVT sampling:
        #   - `manifest_path` preserves the intended bottom-to-top layer order,
        #     which is more reliable than raw z-quantiles when an xyz-periodic
        #     stack wraps around the z boundary.
        #   - `bin_nm` controls the z histogram resolution for mass density,
        #     charge density, EDL species layering, integrated charge, and the
        #     fixed-charge electrostatic-potential diagnostic.
        #   - `region_width_nm` controls the width of graphite-near, mixed, and
        #     core-like z regions used by enrichment, penetration, coordination,
        #     and region transport summaries.
        #   - `surface_distance_nm` and `surface_grid_nm` define graphite
        #     adsorption occupancy: molecule COM within the cutoff of a graphite
        #     surface is counted, and x/y locations are binned on this grid.
        #   - `penetration_threshold_nm` requires a molecule COM to sit at least
        #     this far inside a CMC/polymer-rich or mixed region before counting
        #     it as penetrated.
        #   - `potential_reference`, `split_electrodes`, and
        #     `report_potential_drop` annotate the fixed-charge EDL diagnostic.
        #     The potential is a 1D integral of sampled charge density, not a
        #     constant-potential or Poisson-Boltzmann electrode solution.
        #   - `time_series_analysis=False` is used here because the static stack
        #     has only one coordinate frame; time-series MP4s are generated from
        #     the sampled NVT trajectory below by explicitly passing
        #     `time_series_analysis=True` to the facade methods.
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
                compression_anneal=ZCompressionAnnealSpec(
                    enabled=True,
                    cycles=8,
                    tmax_K=380.0,
                    pmax_bar=3000.0,
                    max_z_shrink_per_cycle=0.04,
                ),
                restart=restart_status,
            )
            print(f"[{surface_charge:+.1f} uC/cm2] relaxation_summary = {relax.summary_path}")

            # Sampled-trajectory post-processing facade. `relax.analyze()` resolves
            # the final NVT `md.gro`, `md.tpr`, `md.edr`, topology, index file,
            # and coordinate stream (`md.xtc` by default; `md.trr` is used when
            # the trajectory policy requests full-precision coordinates).
            #
            # The facade keeps expensive work cached per parameter set and lets
            # the script ask physical questions explicitly:
            #   geometry_health(): intended vs sampled layer order, interphase
            #     distances, severe-overlap flags, and direct graphite/electrolyte
            #     contact checks.
            #   z_profiles(): phase/moltype mass density, charge density, number
            #     density, and phase z-quantiles.
            #   edl_profiles(): fixed-charge EDL species profiles, integrated
            #     charge, electric field, and reference-shifted potential.
            #   penetration(...): molecule COM residence in CMC/polymer-rich or
            #     mixed regions, filtered by `penetration_threshold_nm`.
            #   graphite_adsorption(...): graphite-near residence, surface
            #     occupancy map, and simple carbonyl/dipole orientation proxies.
            #   coordination_by_region(): cation donor-state partitioning by z
            #     region using fallback Li/Na-O/F cutoffs.
            #   region_transport(): anisotropic MSD summaries; use Dxy for
            #     in-plane interface mobility and Dz only as confined mobility.
            #   time_series(): slow MP4 animations and CSV data sampled by
            #     trajectory deciles. The RDF movie uses cation-centered
            #     cation-polymer O, cation-solvent O, and cation-anion F pairs
            #     when those sites exist, and always writes the paired CN(r)
            #     curves plus first-shell CN versus time.
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
            print(f"[{surface_charge:+.1f} uC/cm2] interface_phase_order_ok = {health.get('phase_order_ok')}")
            print(
                f"[{surface_charge:+.1f} uC/cm2] interface_outputs = "
                f"{summary.get('outputs', {}).get('interface_profile_summary_json')}"
            )
            print(f"[{surface_charge:+.1f} uC/cm2] interface_time_series = {time_series.get('outputs', {})}")

        clean_md_trajectory_files(case_dir, enabled=clean_trajectories_after_analysis)
