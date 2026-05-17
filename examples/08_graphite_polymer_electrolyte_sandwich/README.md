# Example 08: Generic Layer-Stack Interfaces

Example 08 now uses the generic `interface.build_layer_stack(...)` engine
instead of a hard-coded sandwich implementation.  You describe layers from
bottom to top, and YadonPy plans a shared XY footprint, packs/places each layer,
adds physically separated z gaps or vacuum spacers, writes `system.gro/top/ndx`,
and records `layer_stack_manifest.json`.

Five public scripts are kept in this folder:

- `01_electrolyte_graphite_basal.py`: basal graphite plus carbonate/LiPF6 electrolyte.
- `02_electrolyte_graphite_edge.py`: finite edge graphite plus electrolyte, with editable `edge_cap`.
- `03_electrolyte_cmcna_graphite_basal.py`: basal graphite, CMC-Na, and electrolyte.
- `04_graphite_basal_electrolyte_cmcna_graphite_basal.py`: two basal graphite layers around electrolyte and CMC-Na.
- `05_charged_graphite_basal_electrolyte_cmcna_graphite_basal.py`: the same four-layer stack with a fixed-charge sweep of `0, +2, -2, +5, -5 uC/cm2`.

All five scripts use the same script-first style as Examples 02/05/07.  The
default systems are compact and, when executed directly, run a fixed-XY
relaxation workflow: steep pre-minimization, short pre-NVT, z-only
semi-isotropic NPT, then a 2 ns final NVT sampling stage.

## MolDB Requirements

The examples intentionally fail fast if a species is missing from MolDB.  They
do not run new DFT/RESP calculations inside the interface workflow.

Required species:

- EC, EMC, DEC with adaptive RESP charges.
- PF6- with adaptive RESP and `bonded="DRIH"`.
- For CMC-Na: glucose_6 repeat unit.  The `[H][*]` terminator is a local
  polymer-linker placeholder, not a QM species, so it is built directly from
  SMILES without RESP/DFT.

## Layer-Stack Outputs

Each run writes under `examples/08_graphite_polymer_electrolyte_sandwich/work_dir/...`.
The important files are:

- `layer_stack_manifest.json`
- `02_system/system.gro`
- `02_system/system.top`
- `02_system/system.ndx`
- `03_relaxation_sampling/relaxation_followup_summary.json`
- `03_relaxation_sampling/05_relaxation_workflow/04_final_nvt/md.gro`
- `06_analysis/layer_stack_interface/interface_profile_summary.json`

`system.ndx` includes generic layer groups such as `LAYER_00_GRAPHITE`,
semantic aliases such as `GRAPHITE`, `ELECTROLYTE`, `CMCNA`, and `MOBILE`.

## Physics Notes

- Default stack PBC is `xyz` with explicit z gaps or vacuum layers.  No walls
  are inserted by default.
- Closed `xyz` stacks also get a periodic top-bottom closing gap by default, so
  the outer surfaces do not start overlapped through PBC.
- Graphite is not frozen by the builder.  If you want a rigid-electrode control
  simulation, freeze it explicitly in the MD stage.
- Basal graphite is XY-periodic by default. Edge graphite is a finite capped slab.
- The per-layer `density_target_g_cm3` values are initial packing targets.  They
  are not a guarantee that a fresh fixed-volume NVT run would immediately have a
  physical local density everywhere.
- The sampled workflow uses `run_layer_stack_relaxation(...)`: `01_pre_minimize`
  removes fresh contacts, `02_pre_nvt` releases local overlaps at fixed volume,
  `03_z_npt` keeps XY fixed while the z length responds to 1 bar pressure, and
  `04_final_nvt` produces the trajectory used for interface analysis.
- For high-chain-count CMC/electrolyte sandwiches, `compression_anneal` can add
  repeated small fixed-XY z-compression geometry moves plus hot/high-pressure
  z-only annealing before final z-NPT.  This keeps graphite XY periodic bonding
  intact while letting an expanded-Z packing collapse gradually.
- CMC-Na examples use `1.5 g/cm3` as the bulk-density reference for initial
  geometry.  The final confined layer need not equal bulk density exactly, but
  `relaxation_followup_summary.json` flags CMCNA rich-region density below
  `0.90 g/cm3` as a warning and below `0.75 g/cm3` as severe.
- The z-NPT stage is controlled by `relax_z`. Use `relax_z=True` for confined
  graphite/polymer/electrolyte stacks such as graphite | electrolyte | CMC-Na |
  graphite. Use `relax_z=False` for explicit vacuum | electrolyte | vacuum
  controls where the vacuum spacing is part of the model. With `relax_z="auto"`,
  explicit `VacuumLayerSpec` layers and `pbc_mode="xy"` skip z-NPT.
- The compact four-layer validation scripts use `constraints="none"` and `1 fs`
  throughout the relaxation workflow, which avoids an early constrained-settle
  minimization on a deliberately tight fresh interface. Larger production runs
  can switch back to `h-bonds + 2 fs` after additional relaxation.
- Constant-charge graphite is a fixed-charge model: surface charge is
  distributed over selected top/bottom surface atoms once at build time.  It is
  not a constant-potential electrode model.
- Interface analysis is intentionally different from bulk analysis. The scripts
  use `analy = relax.analyze()` followed by `analy.interface(...)`, then call
  readable methods such as `geometry_health()`, `z_profiles()`,
  `edl_profiles()`, `penetration(...)`, `graphite_adsorption(...)`,
  `coordination_by_region()`, `region_transport()`, and `time_series()`.
- Static stack post-processing uses `analyze_layer_stack_interface(...)` on
  `system.gro/top/ndx` and `layer_stack_manifest.json`; it is a geometry and
  charge sanity pass before relaxation.
- Sampled post-processing uses `relax.analyze().interface(...)`, which reads the
  final NVT coordinate stream after z-NPT relaxation (`md.xtc` by default, with
  `md.trr` available when the trajectory policy requests it) and writes
  `06_analysis/interface_profile/`.
- `manifest_path` supplies intended layer order, `bin_nm` controls z-profile
  resolution, `region_width_nm` controls near-interface/core regions,
  `surface_distance_nm` controls graphite-near COM adsorption, and
  `surface_grid_nm` controls xy adsorption maps.
- `penetration_threshold_nm` requires molecule COMs to sit inside a
  CMC/polymer-rich or mixed region by that depth before counting penetration.
  `adsorption_min_residence_ps` controls only the residence-pass flag.
- `potential_reference`, `split_electrodes`, and `report_potential_drop`
  describe the one-dimensional fixed-charge EDL potential diagnostic. This is
  not a constant-potential electrode model and does not change MD charges.
- `penetration_species` and `adsorption_species` filter moltype names; leaving
  them unset analyzes all non-graphite molecules.
- The interface report includes z density, charge density, integrated charge,
  electrostatic potential diagnostics for fixed-charge graphite, double-layer
  species profiles, small-molecule penetration into CMC/polymer-rich regions,
  graphite-near adsorption/residence and orientation proxies, cation
  coordination partitioning, and anisotropic MSD summaries when trajectories are
  present. Use `Dxy` as the in-plane interface mobility metric; treat `Dz` as
  confined-direction mobility.
- Time-series analysis is disabled by default.  Pass
  `time_series_analysis=True` explicitly to applicable facade calls such as
  `interface.z_profiles(...)`, `interface.graphite_adsorption(...)`,
  `interface.coordination_by_region(...)`, or `interface.time_series(...)` to
  write `time_series/` CSVs and slow MP4 overlays.  The default samples ten
  equal time windows, so RDF/CN, molecule-COM z concentration, and
  adsorbed-angle distributions can be inspected without creating a dense movie.
  MP4 output requires `ffmpeg` in the active conda environment; CSV artifacts
  are written even if the movie writer is unavailable.
- RDF time-series outputs are cation-centered when Li+/Na+ sites exist.  The
  same animation and CSV set includes CN(r) curves and first-shell CN values
  versus time, so RDF changes are interpreted together with coordination-number
  changes.
