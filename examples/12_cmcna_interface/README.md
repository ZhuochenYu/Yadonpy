# Example 12 - CMC-Na vs 1M LiPF6 electrolyte interface

This is the large charged-polymer interface example in the current release.

Current study shape:

- CMC degree of polymerization: `150`
- number of CMC chains: `6`
- electrolyte: `1 M` LiPF6
- solvent recipe: `EC:DEC:EMC = 3:2:5`

Workflow:

1. build and equilibrate the CMC-Na bulk first;
2. use the equilibrated CMC `XY` box as the authoritative interface footprint;
3. build and equilibrate an isotropic probe electrolyte bulk;
4. resize and rebuild the final electrolyte against the polymer-matched target box with extra initial `Z` slack;
5. let `recommend_polymer_diffusion_interface_recipe(...)` choose the route-B vacuum-buffered staged diffusion setup;
6. assemble the interface and run staged release before unrestricted exchange.

This example is intentionally larger and more conservative than Examples 10 and 11. It is meant as the reference workflow when interface robustness matters more than minimum system size.

PF6 is still treated as a MolDB-backed RESP + DRIH species. Run Example 01 first if the PF6 record is not already present.
