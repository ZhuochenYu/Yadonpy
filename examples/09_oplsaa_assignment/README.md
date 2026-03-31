# Example 09: OPLS-AA assignment workflows

This example family isolates the compact OPLS-AA workflows from Example 07.
The goal here is not MolDB catalog building, but clean force-field assignment
scripts written in ordinary YadonPy style.

These scripts intentionally avoid explicit RDKit calls. They use the same
script-first pattern as the main workflow examples:

- `doctor()` and `ensure_initialized()` at entry,
- `set_run_options(...)` at the top,
- `yadonpy.get_ff(...)` and `yadonpy.mol_from_smiles(...)` for preparation,
- linear workflow steps without extra helper layers.

## Scripts

```bash
python 01_oplsaa_ec.py
python 02_oplsaa_moldb_and_ion.py
```

### `01_oplsaa_ec.py`

- builds ethylene carbonate from SMILES with YadonPy defaults,
- assigns OPLS-AA atom types, charges, and bonded terms,
- exports a compact GROMACS-ready result.

### `02_oplsaa_moldb_and_ion.py`

- loads RESP-charged EC from MolDB and preserves those external charges while
  assigning OPLS-AA atom types and bonded terms,
- shows direct OPLS-AA assignment for `Na+`,
- exports both prepared species.

## Output locations

- Example work directory: `examples/09_oplsaa_assignment/work_dir/`
- EC export: `examples/09_oplsaa_assignment/work_dir/01_ec_gmx/`
- MolDB-backed EC export: `examples/09_oplsaa_assignment/work_dir/02_ec_from_moldb_gmx/`
- Na export: `examples/09_oplsaa_assignment/work_dir/03_na_gmx/`
