# Example 05: CMC-Na random copolymer + 1M LiPF6 in EC/EMC/DEC

## Overview
Build a **CMC-Na** random copolymer electrolyte system and run the standard `EQ21step` preset.

Targets (as coded):
- Polymer: CMC random copolymer from 4 glucose-based monomers
- Solvent: EC / EMC / DEC with equal mass contribution
- Salt: 1 M LiPF6 (with a minimum of `min_salt_pairs` ion pairs)
- Counter-ion: Na+ to neutralize polymer formal charge
- Charge scaling: polymer and all ions scaled by 0.8
- RDF center: Li+
- Analysis style: independent `rdf()`, `msd()`, and `sigma()` calls

## Run

```bash
cd examples/05_cmcna_electrolyte
python run_cmcna_random_copolymer.py
```

## Notes
- This system can be computationally heavy (often thousands of solvent molecules).
- PF6- is expected to be precomputed in MolDB and is then reused through `ff.mol(...)` plus `ff.ff_assign(..., bonded="DRIH")`. Run Example 01 first if your MolDB does not yet contain the PF6 RESP + DRIH record.
- `sigma_ne_upper_bound_S_m` is expected to overestimate the real conductivity in this concentrated charged-polymer system. Prefer `sigma_eh_total_S_m` and `haven_ratio` when available.
