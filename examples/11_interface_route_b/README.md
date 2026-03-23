# Example 11 - Route B wall-ready polymer-matched interface workflow

This example now follows the same polymer-first interface workflow as Example 10, but targets the wall-ready route-B geometry:

1. build and equilibrate the polymer bulk;
2. lock the electrolyte build to the equilibrated polymer `X,Y` footprint;
3. equilibrate the standalone electrolyte bulk before interface assembly;
4. assemble the route-B interface with both an internal vacuum gap and the outer vacuum padding required for later wall-driven MD;
5. run the staged route-B diffusion protocol with early slab support and delayed unrestricted exchange.

Route B still keeps wall forces inside the MD protocol layer. The geometry builder only prepares the vacuum-padded wall-ready box.

The script now uses `plan_direct_polymer_matched_interface_preparation(...)` plus the staged `InterfaceProtocol.route_b_wall_diffusion(...)` helper so the example expresses the intended workflow directly.

PF6 follows the same MolDB-backed reuse policy as Example 12. Run Example 01 first to precompute the RESP + DRIH PF6 entry if your MolDB does not already contain it.
