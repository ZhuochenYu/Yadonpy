# Example 10 - Route A polymer-matched interface workflow

This is the neutral, fully periodic interface example.

Workflow:

1. build and equilibrate the polymer bulk;
2. use the equilibrated polymer `XY` box as the shared interface footprint;
3. build and relax the standalone electrolyte against that footprint;
4. let `recommend_polymer_diffusion_interface_recipe(...)` choose the route-A staged diffusion settings;
5. assemble and run the interface workflow.

The script is intentionally linear and follows the same style as Example 02.

PF6 follows the MolDB-backed RESP + DRIH reuse policy. Run Example 01 first if the PF6 record is not already present.
