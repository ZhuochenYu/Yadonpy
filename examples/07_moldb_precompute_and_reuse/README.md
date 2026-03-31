# Example 07: MolDB precompute and reuse

This example keeps the curated catalog path and the quick text-import path in
one place. It is focused on MolDB precomputation and MolDB-backed reuse.

The main entry point is:

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
- Additional polymer-electrolyte repeat units:
  - `PAA`, `PAAH`
  - `PVDF`, `PAN`
  - `PTMC`, `PCL`
  - `PVAc`, `PMMA`, `PS`
- Example 05 and the merged sandwich workflows CMC monomer set:
  - `glucose_0`
  - `glucose_2`
  - `glucose_3`
  - `glucose_6`
  - `glucose_23`
  - `glucose_26`
  - `glucose_36`
  - `glucose_236`
- Sandwich-workflow aromatic polymer repeat unit:
  - `*CC1=CC=C(CCC2=CC=C(C*)C=C2)C=C1`
- Common solvents / diluents / additives:
  - `EC`, `EMC`, `DEC`, `DMC`, `PC`, `FEC`, `VC`, `DTD`
  - `DME`, `Diglyme`, `Triglyme`, `Tetraglyme`
  - `DOL`, `THF`, `Dioxane`, `CPME`, `TTE`
- Common ions / lithium-salt anions:
  - `Li+`, `Na+`
  - `PF6-`, `BF4-`, `ClO4-`, `AsF6-`, `SbF6-`
  - `BOB-`, `DFOB-`, `NO3-`, `OTf-`, `FSI-`, `TFSI-`

## Special handling used by the builder

- monoatomic ions such as `Li+` and `Na+` use `MERZ`
- high-symmetry inorganic anions such as `PF6-`, `BF4-`, `ClO4-`, `AsF6-`, `SbF6-` use
  RESP plus `DRIH`
- `FSI-` and `TFSI-` stay on the standard RESP path
- charged polymer monomers are stored with `polyelectrolyte_mode=True`
- the hydrogen terminator `[H][*]` follows the stable placeholder shortcut path
- QM levels are chosen explicitly at build time:
  - neutral species use `wb97m-d3bj / def2-SVP -> def2-TZVP`
  - anions prefer `wb97m-d3bj / def2-SVPD -> def2-TZVPD`
  - if a diffuse def2 basis is unavailable for the actual element set, the
    builder falls back to the first Psi4-supported option in the built-in
    ladder and records the chosen levels in the build summary JSON
- `AsF6-` and `SbF6-` stay on the diffuse def2 route when the active Psi4 build
  supports those elements, rather than being hard-coded into a downgrade path

## Additional text-table import

For quick local expansion from a pasted CSV-like text block, use:

```bash
python 02_text_table_to_moldb.py
```

This script converts the inline table into `template.csv` and then feeds it
into MolDB autocalculation.

## Reuse in a workflow

After the one-shot build finishes, `04_polymer_electrolyte_from_moldb.py`
directly reuses ready entries from MolDB instead of building a small temporary
CSV on the side:

```bash
python 04_polymer_electrolyte_from_moldb.py
```

The script now expects the required species to already exist in MolDB and will
raise a clear error if they do not.

## OPLS-AA workflows

The compact OPLS-AA examples now live in:

```bash
examples/09_oplsaa_assignment/
```

This keeps Example 07 focused on MolDB itself and keeps the OPLS-AA scripts in
their own smaller example family.

## Legacy alias

`03_rebuild_reference_moldb_species.py` is kept only as a compatibility alias
and now redirects to `01_build_moldb.py`.

## Output locations

- MolDB: `~/.yadonpy/moldb/objects/` (or the directory specified by `YADONPY_MOLDB`)
- Example work directory: `examples/07_moldb_precompute_and_reuse/work_dir/`
- Build summary: `examples/07_moldb_precompute_and_reuse/work_dir/01_build_moldb/build_moldb_summary.json`
