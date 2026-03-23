# Example 01: PF6- Build Once, Then Reuse from MolDB

This example is a focused PF6- workflow for the exact pattern you want to keep in later scripts.

It is also the prerequisite PF6 builder for Example 12.

It does two passes:

1. Build PF6- from scratch with OpenBabel 3D + QM OPT/RESP + `bonded="DRIH"`, then store the finished result into MolDB.
2. Re-load PF6- from MolDB via the normal script style

```python
PF6 = ff.mol(PF6_smiles)
PF6 = ff.ff_assign(PF6, bonded="DRIH")
```

and export the GROMACS-ready files from that MolDB-backed object.

If the optional `psi4` / `resp` stack is not installed, the example now falls back to `gasteiger` charges so the MolDB round-trip and direct `ff.mol(...)` / `ff.ff_assign(...)` workflow can still be executed end-to-end. In a fully provisioned QM environment, it still uses the original RESP route and the exact two-line PF6 reuse pattern shown above.

## Run

```bash
cd examples/01_Li_salt
python run_pf6_to_moldb.py
```

## Outputs

- `work_pf6_only/00_molecules/` – exported MOL2 files for the built PF6 and the MolDB-backed PF6
- `work_pf6_only/01_pf6_build_exports/` – `.gro/.itp/.top` written immediately after the first charge-assignment + DRIH build (`RESP` when `psi4/resp` is available, otherwise `gasteiger` fallback)
- `work_pf6_only/02_pf6_from_moldb_gmx/` – `.gro/.itp/.top` written after the MolDB-backed `ff.mol(...)` / `ff.ff_assign(...)` reload
- MolDB entry in `~/.yadonpy/moldb` (or `$YADONPY_MOLDB`)

## Reuse

After running this example, PF6- can be reused in later workflows exactly as:

```python
from yadonpy.ff.gaff2_mod import GAFF2_mod

ff = GAFF2_mod()
PF6_smiles = "F[P-](F)(F)(F)(F)F"
PF6 = ff.mol(PF6_smiles)
PF6 = ff.ff_assign(PF6, bonded="DRIH")
```

The important point is that the MolDB entry now carries not only the geometry and RESP charges, but also the extra DRIH bonded patch needed for this high-symmetry inorganic anion.

When the fallback route is used, replace the reuse line with `PF6 = ff.mol(PF6_smiles, charge="gasteiger")`; the second line stays the same.
