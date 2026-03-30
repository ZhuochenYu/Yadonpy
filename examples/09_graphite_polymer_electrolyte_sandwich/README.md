# Example 09: Graphite-Polymer-Electrolyte Sandwich Workflows

This merged example replaces the old Example 10/11/12/13 split with one
coherent workflow family built around the same high-level sandwich builder:

1. equilibrate the polymer slab with `XY` locked to the graphite footprint;
2. equilibrate the electrolyte slab against the same `XY`;
3. stack `graphite -> polymer -> electrolyte` explicitly;
4. relax the stacked system with graphite frozen so the soft phases can settle
   mainly along the surface normal;
5. record bulk densities and final phase ordering in `sandwich_manifest.json`.

The scripts are intentionally short. Most of the workflow logic now lives in
`yadonpy.interface.sandwich`, so the examples focus on study setup rather than
repeating the same packing and relaxation boilerplate.

## Scripts

- `01_peo_smoke.py`
  - small remote-friendly smoke for `graphite + PEO + LiPF6 carbonate electrolyte`
- `02_peo_carbonate_full.py`
  - larger neutral polymer workflow on the same builder path
- `03_cmcna_smoke.py`
  - small charged-polymer smoke using CMC-Na monomers and LiPF6 carbonate electrolyte
- `04_cmcna_full.py`
  - fuller CMC-Na study on the same three-phase builder

## Notes

- These scripts expect `PF6-` to already exist in MolDB with the RESP + DRIH
  variant. Build it first with Example 01 or the merged Example 07 catalog.
- The PEO smoke is the quickest way to validate the whole sandwich path on a
  remote GPU node before switching to the more demanding CMC-Na case.
- The manifest written under each work directory includes:
  - polymer/electrolyte bulk density reports
  - final `GRAPHITE -> POLYMER -> ELECTROLYTE` phase-order checks
  - the chosen stacked export paths and main relaxation outputs
