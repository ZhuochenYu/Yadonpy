# Example 08: OPLS-AA assignment for ethylene carbonate (EC)

This example focuses on **OPLS-AA atom typing and bonded assignment only**.

It demonstrates:

1. Build EC from SMILES
2. Add explicit hydrogens
3. Assign **OPLS-AA** atom types, bonded terms, and OPLS type charges
4. Print a compact typing summary

It intentionally stops after force-field assignment. It does **not** build an amorphous cell or run GROMACS.

## Run

```bash
cd examples/08_oplsaa_assign
python run_ec_oplsaa_assign.py
```

## Molecule

- Ethylene carbonate (EC)
- SMILES: `O=C1OCCO1`

## Notes

- These examples use the bundled `yadonpy.ff.OPLSAA` implementation.
- The example prints the assigned `ff_type`, `ff_btype`, and OPLS charges for each atom.
- No topology export or box build is performed here.

## Additional script

- `run_oplsaa_moldb_and_ion.py`
  - Loads a RESP-charged molecule from MolDB and assigns OPLS-AA **without** overwriting the external charges.
  - Demonstrates direct OPLS-AA assignment for `Na+` using the built-in ion parameters.
