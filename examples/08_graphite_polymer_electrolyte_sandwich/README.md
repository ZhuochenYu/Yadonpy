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
default systems are compact and run 2 ns NVT when executed directly.

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
- The NVT follow-up always starts with a no-constraints steep minimization before
  bridge/main NVT, which removes local contacts introduced by fresh stacking.
- The compact four-layer validation scripts use `constraints="none"` and `1 fs`
  for the first NVT observation, which avoids an early constrained-settle
  minimization on a deliberately tight fresh interface. Larger production runs
  can switch back to `h-bonds + 2 fs` after additional relaxation.
- Constant-charge graphite is a fixed-charge model: surface charge is
  distributed over selected top/bottom surface atoms once at build time.  It is
  not a constant-potential electrode model.
- Interface analysis reports z density, charge density, adjacent-interface
  spacing, graphite/electrolyte EDL charge diagnostics, Li/Na coordination
  partition where available, and anisotropic MSD summaries when trajectories are
  present.
