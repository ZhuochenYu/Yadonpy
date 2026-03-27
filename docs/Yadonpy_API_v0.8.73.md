# YadonPy API Reference (v0.8.73)

## 1. Scope

This document describes the primary script-facing APIs in YadonPy `v0.8.73`. It focuses on stable or intended-to-be-stable entry points that are relevant for study scripts and workflow assembly.

The package exposes two layers:

- **high-level convenience API** in `yadonpy.api`;
- **workflow and subsystem modules** in `yadonpy.core`, `yadonpy.sim`, `yadonpy.ff`, `yadonpy.interface`, `yadonpy.io`, and `yadonpy.gmx`.

## 2. Package root

Import pattern:

```python
import yadonpy as yp
```

Important package-level exports:

- `yadonpy.__version__`
- `yadonpy.assign_charges`
- `yadonpy.assign_forcefield`
- `yadonpy.build_graphite`
- `yadonpy.conformation_search`
- `yadonpy.get_ff`
- `yadonpy.list_charge_methods`
- `yadonpy.list_forcefields`
- `yadonpy.load_from_moldb`
- `yadonpy.mol_from_smiles`
- `yadonpy.parameterize_smiles`
- `yadonpy.build_interface`
- `yadonpy.build_interface_from_workdirs`
- `yadonpy.InterfaceBuilder`
- `yadonpy.InterfaceProtocol`
- `yadonpy.InterfaceDynamics`

## 3. High-level convenience API (`yadonpy.api`)

### `get_ff(ff_name: str, **kwargs)`

Return a force-field object by canonical name or alias.

Examples:

```python
ff = yp.get_ff("gaff2_mod")
ff = yp.get_ff("oplsaa")
```

### `list_forcefields() -> tuple[str, ...]`

Return the canonical names of supported force fields.

### `list_charge_methods() -> tuple[str, ...]`

Return common charge method tokens recognized by YadonPy, including:

- baseline non-QM methods;
- Psi4/PsiRESP methods;
- selected scaled quick-charge tokens.

### `mol_from_smiles(smiles: str, *, coord: bool = True, name: str | None = None)`

Build an RDKit molecule using YadonPy defaults.

Parameters:

- `smiles`: SMILES string
- `coord`: whether to generate coordinates
- `name`: optional molecule label

### `build_graphite(**kwargs)`

Thin wrapper around `yadonpy.core.graphite.build_graphite(...)`.

Current graphite builder supports:

- basal slabs;
- edge slabs;
- configurable expansion and layer count;
- edge saturation patterns;
- export-ready metadata for system stacking and GROMACS export.

### `conformation_search(mol, **kwargs)`

Thin wrapper around `yadonpy.sim.qm.conformation_search(...)`.

### `assign_charges(mol, *, charge: str = "RESP", **kwargs)`

Thin wrapper around `yadonpy.sim.qm.assign_charges(...)`.

Important keyword arguments include:

- `charge`
- `opt`
- `work_dir`
- `total_charge`
- `total_multiplicity`
- `method`
- `basis`
- `polyelectrolyte_mode`
- `polyelectrolyte_detection`

### `assign_forcefield(mol, *, ff_name: str = "gaff2_mod", charge: str | None = None, **kwargs)`

Instantiate a force field and call `ff.ff_assign(...)`.

Returns:

- `(ff_object, ok_bool)`

### `load_from_moldb(smiles, *, charge="RESP", basis_set=None, method=None, require_ready=True, return_record=False, polyelectrolyte_mode=None, polyelectrolyte_detection=None)`

Load a molecule from MolDB using the requested charge variant.

Parameters:

- `smiles`
- `charge`
- `basis_set`
- `method`
- `require_ready`
- `return_record`
- `polyelectrolyte_mode`
- `polyelectrolyte_detection`

Returns:

- molecule only, or
- `(molecule, MolRecord)` when `return_record=True`

If grouped-polyelectrolyte variant metadata exist in MolDB, these arguments allow explicit selection of that variant. If they are omitted, the loader uses the charge/basis/method triple first and then resolves among matching variants by stored metadata.

### `parameterize_smiles(smiles, *, ff_name="gaff2_mod", charge_method="RESP", work_dir="./", total_charge=None, total_multiplicity=None, name=None, allow_ff_without_requested_charges=False)`

High-level script helper: SMILES -> charges -> force field assignment.

Returns:

- `(mol, ok_bool)`

Use this helper for fast small-molecule prototyping. Use explicit `ff.mol(...)` plus `ff.ff_assign(...)` when you need finer control or MolDB reuse.

## 4. Force-field objects

The main pattern is:

```python
ff = yp.get_ff("gaff2_mod")
mol = ff.mol("CCO")
ok = ff.ff_assign(mol, charge="RESP")
```

Supported families include:

- `gaff`
- `gaff2`
- `gaff2_mod`
- `oplsaa`
- `dreiding`
- `merz`
- water-model helpers

Common force-field methods:

### `ff.mol(smiles_or_psmiles, ...)`

Create a lightweight molecule handle or prepared RDKit molecule, depending on the force-field backend and data source.

Important options may include:

- `prefer_db`
- `require_db`
- `require_ready`
- `charge`
- `name`

### `ff.ff_assign(mol, *, charge="RESP", bonded=None, report=True, **charge_kwargs)`

Assign the force field and, where needed, charges.

From `v0.8.71`, `charge_kwargs` can include:

- `polyelectrolyte_mode`
- `polyelectrolyte_detection`

Examples:

```python
ok = ff.ff_assign(CMC_monomer, charge="RESP", polyelectrolyte_mode=True)
ok = ff.ff_assign(PF6, bonded="DRIH")
```

## 5. QM and charge API (`yadonpy.sim.qm`)

### `assign_charges(mol, charge="RESP", opt=True, work_dir=None, ..., polyelectrolyte_mode=False, polyelectrolyte_detection="auto")`

Assign atomic charges to a molecule.

Relevant arguments:

- `charge`: `RESP`, `ESP`, `Mulliken`, `Lowdin`, `gasteiger`, `zero`, quick-charge variants
- `opt`: whether to optimize geometry before the QM charge stage
- `work_dir`
- `method`
- `basis`
- `total_charge`
- `total_multiplicity`
- `polyelectrolyte_mode`
- `polyelectrolyte_detection`

Behavior:

- RESP and ESP now use **PsiRESP**
- grouped constraint generation is enabled when `polyelectrolyte_mode=True`
- charge metadata are cached and validated against the chosen charge model and polyelectrolyte settings

### `conformation_search(...)`

Perform conformer search before QM preparation.

### `bond_angle_params_mseminario(...)`

Generate bond and angle parameters from QM Hessian information using the modified Seminario approach.

### `bond_angle_params_drih(...)`

Generate bonded parameters for supported high-symmetry inorganic ions. This is not a general replacement for grouped RESP or arbitrary bonded fitting.

## 6. Polyelectrolyte helper API (`yadonpy.core.polyelectrolyte`)

This is a new subsystem in `v0.8.71`.

### `detect_charged_groups(mol, *, detection="auto") -> dict`

Detect charged groups in a molecule.

Detection strategy:

- template first;
- graph fallback second.

### `annotate_polyelectrolyte_metadata(mol, *, detection="auto") -> dict`

Persist charge-group and RESP-constraint metadata onto the molecule.

Stored molecule properties:

- `_yadonpy_charge_groups_json`
- `_yadonpy_resp_constraints_json`
- `_yadonpy_polyelectrolyte_summary_json`

### `get_charge_groups(mol) -> list[dict]`

Return charged-group metadata, using cached properties when available.

### `get_resp_constraints(mol) -> dict`

Return grouped RESP constraint metadata.

### `get_polyelectrolyte_summary(mol) -> dict`

Return the summarized detection result.

### `build_residue_map(mol, *, mol_name=None) -> dict`

Build a residue-level atom map using atom-level residue metadata when present.

### `scale_charged_groups_inplace(mol, *, scale, charge_prop="AtomicCharge", groups=None) -> dict`

Apply local charge scaling to detected charged groups while leaving the neutral remainder unchanged.

## 7. Bulk construction API (`yadonpy.core.poly`)

### `amorphous_cell(mols, n, ..., density=..., charge_scale=..., polyelectrolyte_mode=False, work_dir=None, ...)`

Construct an amorphous cell.

Important behaviors:

- stores composition and export metadata in `_yadonpy_cell_meta`
- supports species-level charge scaling
- from `v0.8.71`, can preserve charged-group metadata for grouped export scaling

### `amorphous_mixture_cell(...)`

Mixture-oriented variant of amorphous packing with the same grouped charge-scaling support.

### `random_copolymerize_rw(...)`

Random-walk copolymerization helper.

### `terminate_rw(...)`

Random-walk termination helper for polymers.

## 8. GROMACS export API

### `yadonpy.io.gromacs_molecule.write_gro_from_rdkit(...)`

Write a `.gro` file for a single molecule. From `v0.8.71`, this function preserves residue metadata when available.

### `yadonpy.io.gromacs_molecule.write_gromacs_single_molecule_topology(...)`

Write a single-molecule GROMACS topology. From `v0.8.71`, `[ atoms ]` residue fields preserve polymer residue identity when present.

### `yadonpy.io.gromacs_system.export_system_from_cell_meta(...)`

Export a mixed system from `poly.amorphous_cell(...)` metadata.

Important arguments:

- `cell_mol`
- `out_dir`
- `ff_name`
- `charge_method`
- `charge_scale`
- `polyelectrolyte_mode`
- `source_molecules_dir`
- `system_gro_template`
- `system_ndx_template`

Key behaviors in `v0.8.71`:

- species-level artifact resolution and reuse;
- grouped local charge scaling when charge-group metadata exist;
- residue-preserving export for polymeric species;
- machine-readable reports:
  - `residue_map.json`
  - `charge_groups.json`
  - `resp_constraints.json`
  - `charge_scaling_report.json`

## 9. Interface API

Primary package-level exports:

- `InterfaceBuilder`
- `InterfaceProtocol`
- `InterfaceDynamics`
- `InterfaceRouteSpec`
- `build_interface(...)`
- `build_interface_from_workdirs(...)`

Use these when assembling slab interfaces, route-A or route-B workflows, and staged interface diffusion protocols.

## 10. Analysis API (`yadonpy.sim.analyzer.AnalyzeResult`)

`AnalyzeResult` is the structured post-processing entry point used by the workflow presets after MD output exists.

Primary methods relevant to `v0.8.73`:

### `AnalyzeResult.msd(mols=None, *, begin_ps=None, end_ps=None, policy="adaptive", include_legacy_atom_msd=False) -> dict`

Default behavior is **adaptive and metric-specific**.

Default metric by species class:

- monatomic ions: `ion_atomic_msd`
- ordinary small molecules and salts: `molecule_com_msd`
- polymers: `chain_com_msd`

Additional polymer metrics may be emitted:

- `residue_com_msd`
- `charged_group_com_msd`

Return schema, per species:

- `default_metric`
- `metrics`
- compatibility aliases `metric`, `D_nm2_ps`, `D_m2_s` for the default metric

Each metric may contain:

- `series_csv`
- `fit_t_start_ps`
- `fit_t_end_ps`
- `fit_r2`
- `fit_slope_nm2_ps`
- `alpha_mean`
- `alpha_std`
- `confidence`
- `status`
- `warning`
- `D_nm2_ps`
- `D_m2_s`

Interpretation rule:

- `chain_com_msd` is whole-chain translation, not local segmental motion;
- `residue_com_msd` is the default local polymer segmental metric;
- `charged_group_com_msd` is the group-level charged-site transport metric used downstream by charged-polymer conductivity.

### `AnalyzeResult.rdf(mol_or_mols, *, center_mol=None, include_h=False, bin_nm=0.002, granularity="site", exhaustive_atomtypes=False, strict_center=True, shell_policy="confidence") -> dict`

Default behavior is **site-level RDF/CN**.

Important parameters:

- `granularity="site"`: use chemical site classes as targets
- `exhaustive_atomtypes=False`: do not sweep every atomtype unless explicitly requested
- `strict_center=True`: fail closed if the center species cannot be resolved from exported metadata
- `shell_policy="confidence"`: only promote first-shell CN into the formal summary when shell detection confidence is acceptable

Outputs include:

- `site_map.json`
- per-site `rdf_*.csv`
- per-site `cn_*.csv`
- per-site `rdf_*.svg`
- summary overlay plot

Legacy atomtype-wide RDF remains available by setting `granularity="atomtype"` or `exhaustive_atomtypes=True`.

### `AnalyzeResult.sigma(*, temp_k=None, msd=None) -> dict`

Compute ionic conductivity using:

- Nernst-Einstein (`ne`)
- Einstein-Helfand (`eh`) when current/velocity data are available

`v0.8.73` behavior for charged polymers:

- polymer contributions are taken from `charged_group_com_msd` component metrics when available;
- each charged-group component carries its own `charge_sign`, `formal_charge_e`, and `n_groups`;
- the Nernst-Einstein equation therefore uses charged-group formal charges (`+1`, `-1`, `+2`, ...) rather than a whole-chain net charge;
- a charged polymer without charged-group MSD metadata is excluded from the conductivity sum and recorded under `ignored_components`.

This is the intended default. Whole-chain conductivity fallback for charged polymers is disabled.

## 11. Diagnostics API

### `yadonpy.diagnostics.doctor(print_report=True)`

Report environment status, including:

- data root;
- MolDB directory;
- executable discovery;
- Python module discovery.

For QM workflows, the expected modules are:

- `rdkit`
- `psi4`
- `psiresp`

## 12. Return-value conventions

Several APIs return `(object, ok)` or `ok` rather than raising on every nonideal state. The user should still treat failed parameterization or failed export as hard workflow blockers.

Common patterns:

- `mol, ok = parameterize_smiles(...)`
- `ff, ok = assign_forcefield(...)`
- `ok = ff.ff_assign(mol, ...)`

## 13. Version-specific notes

`v0.8.73` extends the earlier grouped-polyelectrolyte work by making post-processing semantics explicit:

- MSD defaults are metric-specific rather than generic moltype-wide atom selections;
- RDF/CN defaults are site-level and fail closed on unresolved centers;
- charged-polymer Nernst-Einstein conductivity uses charged-group diffusion rather than whole-chain net charge.

Earlier version-specific behavior remains in force:

- RESP/ESP are implemented through PsiRESP;
- grouped polyelectrolyte constraints are available in the main charge-assignment path;
- grouped local scaling is integrated into GROMACS system export;
- polymer residue identity is preserved through `.gro` and `[ atoms ]` export.
