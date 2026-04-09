# Example 08: Graphite-Polymer-Electrolyte Sandwich Workflows

This merged example replaces the old Example 10/11/12/13 split with one
coherent workflow family built around the same high-level sandwich builder:

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

The scripts are intentionally short. Most of the workflow logic now lives in
`yadonpy.interface.sandwich` plus the small preset helpers in
`yadonpy.interface.sandwich_examples`, so the examples focus on study setup
through the public `import yadonpy as yp` API rather than repeating lower-level
imports or workflow plumbing. The old
route-A, route-B, charged CMC interface, and graphite stack cases are now
treated as parameter choices on one substrate-assisted sandwich path.

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
  - now uses the high-level `build_graphite_cmcna_glucose6_periodic_case(...)`
    shortcut so the script stays close to Example 02's linear setup style

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
