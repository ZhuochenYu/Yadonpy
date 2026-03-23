# Example 08: Build MolDB from a pasted text table

This example shows a **minimal** workflow:

1. Paste a table (CSV-like) into a Python script.
2. Convert it to a clean `template.csv` (lower-case header: `name,smiles,...`).
3. Run `MolDB.autocalculate(..., add_to_moldb=True)` to precompute 3D geometries + RESP charge variants into the global MolDB.

Run:

```bash
python 01_text_to_csv_and_build_moldb.py
```

Outputs:
- `template.csv` (generated)
- `work_dir/01_build_moldb/*` (QM scratch)
- MolDB entries stored under `~/.yadonpy/moldb/objects/`
