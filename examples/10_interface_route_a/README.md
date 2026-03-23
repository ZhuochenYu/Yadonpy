# Example 10 - Route A polymer-matched interface workflow

This example now follows the intended neutral polymer workflow more directly:

1. build and equilibrate the polymer bulk;
2. take the equilibrated polymer `X,Y` lengths as the interface reference footprint;
3. build a standalone electrolyte bulk directly against that polymer-matched footprint and relax it to density equilibrium before interface assembly;
4. assemble the route-A interface with an explicit vacuum gap between the two slabs;
5. run the new staged interface diffusion protocol:
   - gap-preserving EM/NVT,
   - phase-wise density relaxation with early slab-core support,
   - gentle contact,
   - staged release,
   - unrestricted exchange and production.

This script now uses the higher-level `plan_direct_polymer_matched_interface_preparation(...)` helper so the example describes the workflow itself instead of repeating the same box-planning details inline.

PF6 follows the same MolDB-backed reuse policy as Example 12. Run Example 01 first to precompute the RESP + DRIH PF6 entry if your MolDB does not already contain it.
