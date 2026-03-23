# Example 11 - Route B wall-ready polymer-matched interface workflow

This is the neutral route-B counterpart to Example 10.

Workflow:

1. build and equilibrate the polymer bulk;
2. match the standalone electrolyte build to the polymer `XY` footprint;
3. relax the standalone electrolyte first;
4. let `recommend_polymer_diffusion_interface_recipe(...)` choose the vacuum-buffered route-B staged protocol;
5. assemble the wall-ready interface and run staged diffusion.

Route B keeps wall settings in the MD protocol layer while the geometry builder only prepares the vacuum-buffered interface box.

PF6 follows the MolDB-backed RESP + DRIH reuse policy. Run Example 01 first if the PF6 record is not already present.
