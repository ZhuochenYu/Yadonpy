# Example 08: CMC-Na random copolymer + 1M LiPF6 in EC/EMC/DEC (1:1:1 mass ratio)

This example builds a **CMC-Na** random copolymer electrolyte system and runs the standard `EQ21step` preset.

Targets (as coded):
- Polymer: CMC random copolymer from 4 glucose-based monomers, **Mw ~ 10000 g/mol**, `n_CMC=4`
- Solvent: EC / EMC / DEC with **equal mass** contribution
- Salt: **1 M LiPF6** (with a **minimum** of `min_salt_pairs` ion pairs)
- Counter-ion: **Na+** to neutralize polymer formal charge (CMC-Na)
- Charge scaling: polymer and all ions scaled by **0.8**
- RDF center: **Li+**

Notes:
- The molecule counts are estimated from a density-based volume (`density_g_cm3`) and will often produce
  **thousands** of solvent molecules. This can be computationally heavy.
- Monomers and solvents use RESP charges from Psi4; PF6- uses GAFF2_mod parameters directly to keep the workflow light.
