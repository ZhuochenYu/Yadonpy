# Example 07: One-shot MolDB precompute for common electrolyte species

This example is now organized around one practical entry point:

```bash
python 01_build_moldb.py
```

It reads a single catalog file:

- `electrolyte_species.csv`

and writes a broad reusable species set into the active MolDB.

## What gets precomputed

The catalog is meant to cover the species that recur across the shipped
electrolyte examples, so you can precompute them once and then keep reusing
them through `ff.mol(..., require_ready=True, prefer_db=True)`.

Included categories:

- Example 02/06 monomers and hydrogen terminator:
  - `*CCO*`
  - `*COC*`
  - `[H][*]`
- Example 05/12/13 CMC monomer set:
  - `glucose_0`
  - `glucose_2`
  - `glucose_3`
  - `glucose_6`
  - `glucose_23`
  - `glucose_26`
  - `glucose_36`
  - `glucose_236`
- Example 10/11 polymer repeat unit:
  - `*CC1=CC=C(CCC2=CC=C(C*)C=C2)C=C1`
- Common solvents / diluents / additives:
  - `EC`, `EMC`, `DEC`, `DMC`, `PC`, `FEC`, `VC`, `DTD`
  - `DME`, `Diglyme`, `Triglyme`, `Tetraglyme`
  - `DOL`, `THF`, `Dioxane`, `CPME`, `TTE`
- Common ions / lithium-salt anions:
  - `Li+`, `Na+`
  - `PF6-`, `BF4-`, `ClO4-`, `AsF6-`, `FSI-`, `TFSI-`

## Special handling used by the builder

- monoatomic ions such as `Li+` and `Na+` use `MERZ`
- high-symmetry inorganic anions such as `PF6-`, `BF4-`, `ClO4-`, `AsF6-` use
  RESP plus `DRIH`
- `FSI-` and `TFSI-` stay on the standard RESP path
- charged polymer monomers are stored with `polyelectrolyte_mode=True`
- the hydrogen terminator `[H][*]` follows the stable placeholder shortcut path

## Reuse in a workflow

After the one-shot build finishes, `02_polymer_electrolyte_from_moldb.py`
directly reuses ready entries from MolDB instead of building a small temporary
CSV on the side:

```bash
python 02_polymer_electrolyte_from_moldb.py
```

The script now expects the required species to already exist in MolDB and will
raise a clear error if they do not.

## Legacy alias

`03_rebuild_reference_moldb_species.py` is kept only as a compatibility alias
and now redirects to `01_build_moldb.py`.

## Output locations

- MolDB: `~/.yadonpy/moldb/objects/` (or the directory specified by `YADONPY_MOLDB`)
- Example work directory: `examples/07_moldb_precompute_and_reuse/work_dir/`
- Build summary: `examples/07_moldb_precompute_and_reuse/work_dir/01_build_moldb/build_moldb_summary.json`
