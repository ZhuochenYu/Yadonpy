# Example 08: Graphite | Polymer | Electrolyte Interfaces

Example 08 now exposes only two public workflows. Both follow the same
script-first style as Example 02: user knobs are near the top of the file, the
main block runs linearly, and MolDB-ready species are reused instead of running
new DFT inside the interface script.

## Scripts

- `01_peo_graphite_electrolyte.py`
  - Builds a `graphite | PEO | EC/EMC/DEC + LiPF6` three-phase stack.
  - Production defaults use `nx=6, ny=5, n_layers=3` graphite and MolDB-backed
    `*CCO*`, EC, EMC, DEC, and PF6.
  - Change `profile = "smoke"` near the top of the script for a smaller
    validation system.
- `02_cmcna_graphite_electrolyte.py`
  - Builds a `graphite | CMC-Na(glucose_6) | EC/EMC/DEC + LiPF6` stack.
  - Uses periodic-edge graphite and MolDB-backed glucose_6, EC, EMC, DEC, and
    PF6, with Na/Li handled by the MERZ ion force field.
  - Change `profile = "compact"` near the top of the script for a three-layer
    screening model designed to stay below roughly 20k atoms, or `"smoke"` for
    a syntax/build validation system.

## MolDB Requirements

These scripts fail fast when a required species is missing from MolDB. Refresh
the database with Example 01/07 before running production interface jobs. The
interface examples intentionally do not fall back to new QM/RESP work.

Required entries:

- PEO case: `*CCO*`, EC, EMC, DEC, PF6.
- CMC-Na case: glucose_6, EC, EMC, DEC, PF6.

## Outputs

Each script writes into `examples/08_graphite_polymer_electrolyte_sandwich/work_dir/...`.
To redirect a run, edit the `work_dir` value near the top of the script. The
most important outputs are:

- `00_interface_design/interface_design.json`
- `06_full_stack_release/interface_manifest.json`
- `06_full_stack_release/00_stack_attempt_*/system.gro`, `system.top`, `system.ndx`
- `06_full_stack_release/01_relax_*/02_stack_settle_nvt/md.gro`

The terminal summary prints `stack_gmx_dir` and `stack_relaxed_gro` directly.

## Optional 4 ns NVT Follow-Up

Set `run_nvt_after_stack = True` near the top of the script to run a short
fixed-volume observation after stack release:

```bash
python examples/08_graphite_polymer_electrolyte_sandwich/01_peo_graphite_electrolyte.py
```

The follow-up uses `run_sandwich_nvt_followup(...)`, starts from the accepted
stack `relaxed_gro`, performs a short 1 fs no-constraints / 1 fs constrained
NVT handoff, then runs the requested observation NVT and writes
`07_nvt_followup/nvt_followup_summary.json`. The CMC-Na script uses bonded
mobile graphite (`stack_freeze_group=None`) so dynamic stages can use full GPU
offload; pass `final_freeze_group="GRAPHITE"` only for a rigid-electrode
control run.

The CMC-Na script overrides the final follow-up step to `1 fs` because the
hydroxyl-rich, ring-rich CMC phase is more sensitive to hydrogen-bond constraint
instabilities than PEO.

## Interface Statistics

After stack release or the optional NVT follow-up, run:

```python
import yadonpy as yp

profile = yp.analyze_sandwich_interface(
    work_dir="examples/08_graphite_polymer_electrolyte_sandwich/work_dir/peo_graphite_electrolyte",
    analysis_profile="interface_fast",
)
```

This writes `06_analysis/interface_profile/` with geometry health, z density
profiles, graphite/polymer/electrolyte region definitions, interpenetration,
enrichment, Li coordination-by-region, and anisotropic MSD. Use `Dxy` as the
default interface transport metric; `Dz` is only a confined-direction mobility
diagnostic.

## Development Harness

The old remote autofix and matrix harness is no longer part of the public
example folder. Developer-only tools now live under `tools/example08_autofix/`.
