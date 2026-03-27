# OPLS-AA 2024 Import Provenance

This release updates YadonPy's OPLS-AA data against the upstream `moltemplate` repository and its
`oplsaa2024` source set.

## Upstream source

- Repository: `https://github.com/jewettaij/moltemplate`
- Local snapshot used during import:
  - `D:\codex_project\yadonpy\_external\moltemplate`
- Primary upstream files:
  - `moltemplate/force_fields/oplsaa2024.lt`
  - `moltemplate/force_fields/oplsaa2024_original_format/Jorgensen_et_al-2024-The_Journal_of_Physical_Chemistry_B.sup-2.par`

## Local importer

- Import script:
  - `tools/oplsaa/import_moltemplate_oplsaa2024.py`

The importer performs two jobs:

1. Rebuilds the OPLS-AA particle table from the upstream 2024 `.par` source.
2. Merges the bonded coefficient tables from `oplsaa2024.lt` into YadonPy's `oplsaa.json`.

## Unit conversion

The upstream OPLS/BOSS style files are interpreted as follows:

- Lennard-Jones `sigma`: Angstrom -> nm via `0.1`
- Lennard-Jones `epsilon`: kcal/mol -> kJ/mol via `4.184`
- Bond `k`: upstream harmonic constant -> GROMACS harmonic form via `k * 4.184 * 400`
- Bond `r0`: Angstrom -> nm via `0.1`
- Angle `k`: upstream harmonic constant -> GROMACS harmonic form via `k * 4.184 * 4`
- Dihedral Fourier terms: `abs(Vn) * 4.184 / 2`, with phase derived from the sign and OPLS parity convention

## SMARTS extension scope

The SMARTS rule table was not available upstream. YadonPy therefore keeps the existing rule base and extends it
only for chemistries that can be assigned unambiguously from RDKit SMARTS:

- Updated monoatomic ion mappings to OPLS-AA 2024 ion types
- Silicon hydrides, silanols, silyl ethers, and disilanes
- Carbon dioxide
- Allenes and ketenes
- Epoxides
- Zinc ion

This means the parameter table is more complete than the SMARTS rule table. Newly imported parameter types that do
not yet have explicit SMARTS rules remain available for future rule expansion.
