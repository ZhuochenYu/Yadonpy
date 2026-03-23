# Example 12 - CMC-Na vs 1M LiPF6 electrolyte interface

This example now follows the intended charged-polymer workflow explicitly:

1. build and equilibrate the CMC-Na bulk;
2. use the equilibrated CMC `X,Y` box as the interface reference footprint;
3. build an isotropic probe electrolyte bulk first, so the electrolyte composition is learned from a separately equilibrated liquid instead of being guessed directly in the final interface box;
4. resize that electrolyte against the polymer-matched target box and rebuild the final electrolyte bulk with extra initial `Z` slack;
5. assemble the route-A interface with the explicit vacuum gap;
6. run the staged interface diffusion protocol with early phase-wise density relaxation and gradual release before unrestricted exchange.

The example now uses:

- `plan_probe_polymer_matched_interface_preparation(...)` for the probe electrolyte path;
- `plan_resized_polymer_matched_interface_from_probe(...)` for the final polymer-matched electrolyte rebuild;
- `InterfaceProtocol.route_a_diffusion(...)` for the staged interface relaxation and exchange protocol.

PF6 is still treated as a MolDB-backed RESP + DRIH species. Run Example 01 first if your MolDB does not yet contain the required PF6 record.
