# YadonPy

Current release: **v0.8.73**

YadonPy is a script-oriented molecular modeling and simulation workflow package for polymer, electrolyte, substrate, bulk-phase, and interface studies built around GROMACS. It accepts SMILES or PSMILES as the primary chemistry input, prepares reusable molecular assets, constructs packed systems, exports GROMACS-ready topologies, and runs staged workflows for equilibration and analysis.

## Release focus

Version `0.8.73` keeps **PsiRESP** as the RESP/ESP backend, preserves grouped-polyelectrolyte RESP variants in MolDB, and restructures post-processing so the default analysis outputs have explicit physical meaning.

This change was made because grouped charge constraints are required for rigorous polyelectrolyte workflows, and PsiRESP provides the necessary primitives:

- grouped charge-sum constraints;
- equivalence constraints;
- explicit two-stage RESP and ESP job objects;
- auditable constraint metadata that can be preserved in the workflow.

The current release keeps the grouped-polyelectrolyte RESP/scaling path and adds a structured post-processing model:

- `polyelectrolyte_mode=True` on RESP/ESP assignment;
- automatic charged-group detection by template first, graph fallback second;
- grouped charge constraints for charged motifs plus constrained neutral remainder;
- residue-preserving polymer export for `.gro` and `.itp`;
- `charge_groups.json`, `resp_constraints.json`, `residue_map.json`, and `charge_scaling_report.json` in exported systems;
- simulation-level local charge scaling of charged groups while preserving raw RESP templates;
- MolDB variant records that now distinguish grouped polyelectrolyte RESP variants from ordinary RESP variants and restore those tags on load.
- bundled-species rebuild tooling for the shipped MolDB archive, including additional battery anions such as `ClO4-`, `BF4-`, `AsF6-`, `FSI-`, and `TFSI-`.
- adaptive MSD outputs that distinguish atomic-ion, molecular COM, chain COM, residue COM, and charged-group COM motion;
- site-level RDF/CN as the default coordination analysis path, with strict center-species resolution and one shared first-shell detector for plots and JSON summaries;
- Nernst-Einstein conductivity for charged polymers computed from charged-group diffusion coefficients rather than whole-chain net charges.

## Scope

YadonPy is designed for workflows where the user wants the full study logic to remain visible in code.

The package covers:

- small molecules, ions, monomers, and polymers from SMILES or PSMILES;
- charge assignment (`gasteiger`, `RESP`, `ESP`, `Mulliken`, `Lowdin`, and selected quick-charge methods);
- GAFF, GAFF2, GAFF2_mod, OPLS-AA, DREIDING, MERZ, and water-model assignment;
- MolDB-backed reuse of expensive prepared molecules;
- amorphous bulk construction;
- polymer/electrolyte and substrate-assisted interface workflows;
- GROMACS export and staged MD preparation;
- analysis/report generation and restart-aware work directories.

## Installation

### Baseline environment

- Python `3.11`
- `numpy`, `scipy`, `pandas`, `matplotlib`, `packaging`
- `rdkit`
- `parmed`
- `mdtraj`

### Optional but commonly required

- `openbabel` for robust 3D recovery and inorganic handling
- `psi4` for QM geometry/ESP generation
- `psiresp` for RESP/ESP fitting
- `dftd3-python` when the chosen QM method requires it
- `gromacs` for MD execution

### Recommended conda environment

```bash
conda create -n yadonpy python=3.11
conda activate yadonpy

conda install -c conda-forge rdkit openbabel parmed mdtraj matplotlib pandas scipy packaging
conda install -c psi4 psi4 dftd3-python
conda install -c conda-forge psiresp

pip install -e .
pip install -e .[accel]
```

### Environment check

```bash
python -c "from yadonpy.diagnostics import doctor; doctor(print_report=True)"
```

The doctor report should show, at minimum:

- Python path and data root;
- `gmx` discovery status;
- `rdkit`, `psi4`, and `psiresp` module availability.

## Core workflow model

### 1. Prepare molecular species

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
EC = ff.mol("O=C1OCCO1")
ok = ff.ff_assign(EC)
```

For explicit RESP:

```python
from yadonpy.sim import qm

qm.assign_charges(
    EC,
    charge="RESP",
    work_dir="./work_ec",
)
ok = ff.ff_assign(EC)
```

For polyelectrolytes:

```python
qm.assign_charges(
    CMC_monomer,
    charge="RESP",
    work_dir="./work_cmc_monomer",
    polyelectrolyte_mode=True,
)
```

### 2. Reuse expensive molecular assets through MolDB

MolDB stores:

- converged geometry;
- charge variants;
- readiness flags;
- bonded-patch sidecars when required by a charge variant.

From `v0.8.72`, MolDB variant records can also carry:

- `polyelectrolyte_mode`
- `polyelectrolyte_detection`
- `constraint_signature`
- `charge_groups`
- `resp_constraints`
- `polyelectrolyte_summary`

Those fields are also part of variant resolution. A grouped-polyelectrolyte RESP fit is therefore not reused as if it were an unconstrained RESP fit.

MolDB does **not** treat old exported `.gro/.itp/.top` files as the primary persistent source.

### 3. Build the system

Typical bulk path:

1. define species;
2. assign charges and force fields;
3. pack an amorphous cell;
4. export the GROMACS system;
5. run EQ21 or another staged workflow;
6. inspect density, box convergence, and derived properties.

Typical interface path:

1. equilibrate polymer bulk first;
2. take polymer `XY` dimensions as the authoritative lateral footprint;
3. size electrolyte against that footprint;
4. equilibrate the electrolyte bulk independently;
5. assemble slabs or blocks;
6. run staged interface relaxation and release.

## Polyelectrolyte RESP and local charge scaling

The current implementation distinguishes three layers:

### Raw QM charge template

RESP or ESP is fit by PsiRESP and stored as the raw molecular charge template.

### Constraint model

With `polyelectrolyte_mode=True`, YadonPy:

- detects charged groups using built-in templates first;
- falls back to graph-based formal-charge neighborhoods when needed;
- applies one charge-sum constraint per charged group;
- applies one charge-sum constraint to the neutral remainder;
- adds conservative equivalence constraints on the neutral remainder only.

### Simulation-level scaling

`charge_scale` remains a simulation/export option. It does not overwrite the raw RESP template stored in the molecule or MolDB.

When `polyelectrolyte_mode=True` and charge-group metadata exists:

- charged groups are scaled locally;
- the neutral remainder is left unchanged;
- the export writes a `charge_scaling_report.json` file.

When metadata is missing or detection fails:

- the export records a fallback;
- the workflow reverts to whole-molecule scaling.

## Export artifacts added by v0.8.72

For polymeric or polyelectrolyte systems, exports now preserve residue-level metadata and write:

- `residue_map.json`
- `charge_groups.json`
- `resp_constraints.json`
- `charge_scaling_report.json`

These files are intended for:

- auditing the constraint model;
- postprocessing by residue or charged group;
- reproducing the exact scaling logic used in a given topology export.

## Analysis defaults added by v0.8.73

### MSD

`AnalyzeResult.msd()` no longer treats one moltype-wide atom selection as a generic diffusion coefficient by default.

The default metric now depends on species type:

- single-atom ions: `ion_atomic_msd`
- ordinary small molecules and salts: `molecule_com_msd`
- polymers: `chain_com_msd`

For polymers, the analysis also writes:

- `residue_com_msd`
- `charged_group_com_msd` when charged-group metadata exist

Each metric writes a raw CSV series, an explicit fit window, fit confidence, and a diffusion coefficient only when a credible diffusive regime is found.

### RDF and CN

`AnalyzeResult.rdf()` now defaults to `granularity="site"` rather than full atomtype enumeration.

Important consequences:

- the center species must resolve explicitly from exported metadata;
- unresolved centers now raise rather than falling back to `IONS`;
- first-shell JSON summaries and SVG annotations use the same shell detector and therefore report the same cutoff and CN.

Legacy atomtype-wide RDF remains available, but only as an explicit opt-in mode.

### Ionic conductivity for charged polymers

`AnalyzeResult.sigma()` now computes the Nernst-Einstein conductivity of charged polymers from **charged-group diffusion coefficients**.

This means:

- the contributing charge is the charged-group formal charge (`+1`, `-1`, `+2`, etc.), not the whole-chain net charge;
- cationic and anionic charged groups are reported separately;
- a charged polymer without `charged_group_com_msd` metadata is excluded from the Nernst-Einstein sum and recorded as ignored rather than being approximated as one large polyion.

## Rebuild the bundled MolDB species set

The repository ships a batch rebuild script:

`tools/moldb/rebuild_bundle_species.py`

It reads `yd_moldb.tar`, rebuilds the bundled species list, and extends it with the battery-relevant extras:

- `ClO4-`
- `BF4-`
- `AsF6-`
- `FSI-`
- `TFSI-`
- `Li+`

Example:

```bash
python tools/moldb/rebuild_bundle_species.py --db-dir ~/.yadonpy/moldb --work-root ./work_rebuild_bundle_species
```

`DRIH` is applied only to recognized high-symmetry polyhedral ions. `FSI-` and `TFSI-` stay on the standard RESP path.

## Examples

Recommended reading order:

1. `examples/07_moldb_precompute_and_reuse`
2. `examples/08_text_to_csv_and_build_moldb`
3. `examples/01_Li_salt`
4. `examples/02_polymer_electrolyte`
5. `examples/05_cmcna_electrolyte`
6. `examples/10_interface_route_a`
7. `examples/11_interface_route_b`
8. `examples/12_cmcna_interface`
9. `examples/13_graphite_cmc_electrolyte`

Relevant updates in this release:

- `examples/05` now uses `polyelectrolyte_mode=True` for CMC monomer RESP and charge-scaled cell construction;
- `examples/12` now uses the same grouped polyelectrolyte path and no longer requires manual charged-atom index handling in the script.
- `tools/moldb/rebuild_bundle_species.py` rebuilds bundled MolDB species and adds `ClO4-`, `BF4-`, `AsF6-`, `FSI-`, `TFSI-`, and `Li+`.

## Documentation

- Manual: `docs/Yadonpy_manul.md`
- User guide: `docs/Yaonpyd_user_guide.md`
- API reference: `docs/Yadonpy_API_v0.8.73.md`

Recommended reading order:

1. README for package scope and installation;
2. user guide for practical study assembly;
3. manual for architecture and persistence rules;
4. API reference for callable interfaces and important parameters.

## Development checks

```bash
python -m compileall src examples tests
PYTHONPATH=src pytest -q
```

If GROMACS is not installed, code-level and topology-level checks can still be run. MD execution remains an environment-dependent validation step.
