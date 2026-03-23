# Example 07: Precompute MolDB records and reuse them in a workflow

This example combines two steps that are commonly used in production scripts:

1. Precompute geometry and charge variants into the global MolDB.
2. Reuse those records in a polymer-electrolyte workflow.

By default the global MolDB lives under `~/.yadonpy/moldb/`.

## What MolDB stores

MolDB stores only the expensive molecular assets:

- the preferred 3D geometry (`mol2`)
- charge variants (`json`)

Topology files such as `.itp`, `.top`, and `.gro` are not cached in MolDB and
are regenerated on demand.

## Optional environment overrides

```bash
export YADONPY_HOME=/path/to/.yadonpy
export YADONPY_MOLDB=/path/to/moldb
```

## Step 1: build MolDB records from `template.csv`

Edit `template.csv`, then run:

```bash
python 01_build_moldb.py
```

The script reads `name,smiles` entries and, when requested, performs conformer
search, QM optimization, and charge generation before writing the results into
MolDB.

Recognized optional columns include:

- `opt`
- `confsearch`
- `charge_method`
- `basis_set`
- `method`

## Step 2: reuse MolDB records in a workflow

```bash
python 02_polymer_electrolyte_from_moldb.py
```

This script reuses precomputed monomers, terminal groups, and solvents directly
from MolDB. It also demonstrates the v0.7.13 handle workflow:

```python
monomer_A = ff.mol(smiles_A)
ff.ff_assign(monomer_A)
```

The handle remains valid for downstream calls such as polymer construction and
amorphous-cell packing.

PF6- is intentionally left out of the CSV. When you need PF6-, build it first
with Example 01 and then reuse it in later workflows through the same MolDB
pattern used in Example 12:

```python
PF6 = ff.mol("F[P-](F)(F)(F)(F)F", charge="RESP", require_ready=True, prefer_db=True)
PF6 = ff.ff_assign(PF6, bonded="DRIH")
```

## Output locations

- MolDB: `~/.yadonpy/moldb/objects/<key>/` (or the directory specified by `YADONPY_MOLDB`)
- Example working directory: `examples/07_moldb_precompute_and_reuse/work_dir/`
