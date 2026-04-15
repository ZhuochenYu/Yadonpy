# Example 08: Graphite-Polymer-Electrolyte Sandwich Workflows

This merged example replaces the old Example 10/11/12/13 split with one
coherent workflow family built around the same staged sandwich workflow:

1. equilibrate true 3D polymer and electrolyte bulk phases independently;
2. keep those bulk phases only as calibration references for density, composition,
   and packing difficulty;
3. negotiate one graphite master footprint in `XY`;
4. rebuild each soft phase directly on that final `XY` footprint with explicit
   vacuum and repulsive-only Z walls;
5. stack `graphite -> polymer -> electrolyte` by direct Z translation;
6. relax the stacked system with graphite frozen so the soft phases can settle
   mainly along the surface normal;
7. record bulk-calibration summaries, walled-phase diagnostics, and final phase ordering.

The scripts are intentionally thin but no longer black-box. Each one now spells
out the same public stages used by the convenience builder:

1. `prepare_graphite_substrate(...)`
2. `calibrate_polymer_bulk_phase(...)`
3. `calibrate_electrolyte_bulk_phase(...)`
4. `build_graphite_*_interphase(...)`
5. `build_*_electrolyte_interphase(...)`
6. `release_graphite_*_electrolyte_stack(...)`

That keeps the top-level code close to Example 02: you can see the workflow at
a glance, but the expensive internals still live in the reusable library
functions so species preparation, graphite negotiation, and bulk calibration are
not repeated.

## Scripts

- `01_peo_smoke.py`
  - small remote-friendly smoke for `graphite + PEO + LiPF6 carbonate electrolyte`
- `02_peo_carbonate_full.py`
  - larger neutral polymer workflow on the same builder path
- `03_cmcna_smoke.py`
  - small charged-polymer smoke using CMC-Na monomers and LiPF6 carbonate electrolyte
- `04_cmcna_full.py`
  - fuller CMC-Na study on the same three-phase builder
- `05_cmcna_glucose6_periodic_case.py`
  - target case: `1 M LiPF6 in EC:EMC:DEC = 3:2:5` above 8 chains of `DP=50` CMC-Na built only from the `glucose_6` repeat unit, on top of a 4-layer graphite substrate using the new uncapped `edge_cap="periodic"` mode
  - uses the same explicit stage API sequence as the other scripts, but keeps
    the chemistry fixed to MolDB-ready `glucose_6`, `EC`, `EMC`, `DEC`, and
    `PF6`

## Routes And Profiles

- `route="screening"`
  - used by smoke-sized scripts
  - prioritizes quick validation and minimal repeat work
- `route="production"`
  - used by full-sized scripts
  - keeps the same stage layout but writes stage summaries and restart points
    under the production-sized work directory

`profile` in these examples now only controls system size and runtime settings.
The physical workflow is visible in the script itself instead of being hidden in
one giant wrapper.

## Automation

- `run_iteration_matrix.py`
  - `--mode observe`
  - runs the fast Example 08 matrix and writes `status.jsonl`,
    `latest_status.json`, and per-iteration summaries
- `run_autofix_loop.py`
  - thin entrypoint for the unattended autofix driver
  - clones a snapshot workspace, syncs it to the remote GPU node, runs one
    remote matrix round at a time, classifies failures, applies one whitelisted
    mutation recipe, runs local verification, and only then allows commit/push
- `autofix_config.json`
  - centralizes remote paths, time budget, stop thresholds, recipe enablement,
    and verification commands

The unattended loop is intentionally split this way: the persistent driver owns
the long-running state machine, while the matrix harness remains the reusable
single source of truth for launching and summarizing Example 08 cases.

## Notes

- These scripts expect `PF6-` to already exist in MolDB with the RESP + DRIH
  variant. Build it first with Example 01 or the merged Example 07 catalog.
- `05_cmcna_glucose6_periodic_case.py` is intentionally MolDB-only for the
  chemistry inputs it names: `glucose_6`, `EC`, `EMC`, `DEC`, and `PF6` must
  already be present and ready in MolDB, otherwise the script fails fast instead
  of silently falling back to new QM work.
- The PEO smoke is the quickest way to validate the whole sandwich path on a
  remote GPU node before switching to the more demanding CMC-Na case.
- The CMC scripts use the same grouped-polyelectrolyte RESP and counterion-aware
  packing path as Example 05, but now place that polymer into the same graphite
  sandwich builder used by the neutral PEO case.
- `05_cmcna_glucose6_periodic_case.py` also accepts `YADONPY_PROFILE=smoke` so
  the same chemistry can be validated with a much smaller remote-friendly system
  before launching the exact production-sized case.
- The manifest written under each work directory includes:
  - `01_graphite/graphite_preparation_summary.json`
  - `02_polymer_bulk_calibration/polymer_bulk_calibration_summary.json`
  - `03_electrolyte_bulk_calibration/electrolyte_bulk_calibration_summary.json`
  - `04_graphite_polymer_interphase/polymer_phase_confined_summary.json`
  - `05_polymer_electrolyte_interphase/electrolyte_phase_confined_summary.json`
  - `06_full_stack_release/interface_manifest.json`
  - polymer/electrolyte bulk calibration summaries
  - walled-phase build summaries for each soft phase
  - confined phase summaries for each soft phase
  - final `GRAPHITE -> POLYMER -> ELECTROLYTE` phase-order checks
  - explicit acceptance fields for density windows, wrapped-Z detection, and
    positive core gaps
  - the chosen stacked export paths and main relaxation outputs
- The printed terminal summary now exposes the main acceptance booleans and a
  `failed_checks` list directly, so remote monitoring does not require opening
  the full manifest just to see which acceptance gate is still failing.
