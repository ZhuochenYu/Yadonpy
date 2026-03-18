# Example 10: Batch build Molecule DB from a CSV (initial 3D + charges)

## What this example does
Given a CSV with two columns:
1) `name`
2) `smiles` (can be SMILES or PSMILES containing `*`)

This script:
- builds an initial 3D geometry
- computes partial charges (e.g., RESP)
- stores results into a portable on-disk Molecule DB directory (`moldb/`)

## Files
- `template.csv`: 10 sample entries
- `run_batch_build_db.py`: batch builder
- `index.csv`: (generated) mapping MolDB `key` <-> `name` <-> input `smiles` so you can trace the DB IDs

## How to run
```bash
cd examples/10_moldb_batch_from_csv
python run_batch_build_db.py template.csv
```

If you omit the argument, it defaults to `template.csv`.

## Outputs
- `moldb/objects/<key>/manifest.json`
- `moldb/objects/<key>/best.mol2`
- `moldb/objects/<key>/charges.json`

In addition, the example writes `index.csv` in the example folder.

You can copy the whole `moldb/` folder to another project/machine to reuse computed charges.
