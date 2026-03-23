# YadonPy API (v0.8.52)

This document describes the public, script-facing API that is visible in the current release. It focuses on stable entry points that are intended for end users, examples, and automation scripts.

Related documents:

- manual: `docs/Yadonpy_manul.md`
- user guide: `docs/Yaonpyd_user_guide.md`

## API scope

YadonPy exposes a compact top-level API and a broader set of package-level modules.

Use these layers as a rule:

- use `import yadonpy as yp` for ordinary scripts;
- use `yadonpy.interface` for explicit interface-building workflows;
- use `yadonpy.sim.qm` when you need lower-level QM or charge-control details;
- use `yadonpy.moldb.MolDB` when you need direct database operations;
- avoid depending on `yadonpy.core.*` internals unless you are extending the package itself.

## Package root exports

The package root currently exports the following names:

```python
import yadonpy as yp

yp.__version__
yp.get_ff
yp.list_forcefields
yp.list_charge_methods
yp.mol_from_smiles
yp.conformation_search
yp.assign_charges
yp.assign_forcefield
yp.parameterize_smiles
yp.load_from_moldb
yp.InterfaceBuilder
yp.InterfaceDynamics
yp.InterfaceProtocol
yp.InterfaceRouteSpec
yp.build_interface
yp.build_interface_from_workdirs
yp.get_run_options
yp.set_run_options
yp.run_options
yp.qm
```

## Core convenience functions

### `get_ff(ff_name, **kwargs)`

Return a force-field object created from the registry.

Typical use:

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
```

Use this when you want explicit control over the force-field object and later call methods such as `ff.mol(...)` and `ff.ff_assign(...)` yourself.

### `list_forcefields()`

Return the canonical force-field names currently registered in YadonPy.

### `list_charge_methods()`

Return the common charge method tokens recognized by the high-level API. This includes:

- baseline non-QM methods such as `zero` and `gasteiger`;
- Psi4-driven methods such as `RESP`, `ESP`, `Mulliken`, and `Lowdin`;
- lightweight scaled forms supported by the QM layer.

### `mol_from_smiles(smiles, coord=True, name=None)`

Build a YadonPy/RDKit molecule from a SMILES string using the package defaults.

Notes:

- `coord=True` requests 3D coordinates during molecule preparation;
- `name` is optional metadata and is useful for exported files and reports.

### `conformation_search(mol, **kwargs)`

Thin wrapper around `yadonpy.sim.qm.conformation_search`. Use it when you want the script-facing entry point without importing the lower-level module.

### `assign_charges(mol, charge="RESP", **kwargs)`

Thin wrapper around `yadonpy.sim.qm.assign_charges`.

Typical use:

```python
import yadonpy as yp

mol = yp.mol_from_smiles("O=C1OCCO1", name="EC")
yp.assign_charges(mol, charge="RESP", work_dir="./work_ec")
```

### `assign_forcefield(mol, ff_name="gaff2_mod", charge=None, **kwargs)`

Create a force field and call its `ff_assign(...)` method.

Return value:

- `(ff, ok)` where `ff` is the instantiated force-field object and `ok` is the success flag.

### `parameterize_smiles(...)`

High-level helper that runs a common script workflow:

1. create a molecule from SMILES;
2. assign charges;
3. run `ff_assign(...)`.

Current behavior is intentionally strict by default. If the requested charge generation fails, the helper raises unless you pass `allow_ff_without_requested_charges=True`.

Typical use:

```python
import yadonpy as yp

mol, ok = yp.parameterize_smiles(
    "CCO",
    ff_name="gaff2_mod",
    charge_method="RESP",
    work_dir="./work_ethanol",
)
```

### `load_from_moldb(...)`

Load a molecule from MolDB by SMILES or PSMILES key.

Important behavior in the current release:

- the requested charge variant is selected by `charge`, `basis_set`, and `method`;
- `require_ready=True` requires that the variant was marked ready in MolDB;
- `return_record=True` returns `(mol, record)`;
- if the selected MolDB variant stores a DRIH or mSeminario bonded patch, that bonded patch is restored automatically onto the returned molecule.

Typical use:

```python
import yadonpy as yp

pf6 = yp.load_from_moldb(
    "F[P-](F)(F)(F)(F)F",
    charge="RESP",
    require_ready=True,
)
```

## Force-field object workflow

The convenience API is intentionally small. Many examples use the explicit force-field workflow instead:

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
mol = ff.mol("CCO")
mol = ff.ff_assign(mol)
```

This style matters because `ff.mol(...)` may return a lightweight `MolSpec` handle rather than a fully realized molecule. The handle is then resolved during `ff.ff_assign(...)`.

That same pattern is used in the MolDB-backed PF6 example:

```python
PF6_smiles = "F[P-](F)(F)(F)(F)F"
PF6 = ff.mol(PF6_smiles)
PF6 = ff.ff_assign(PF6, bonded="DRIH")
```

If a matching MolDB entry already exists for the requested variant, YadonPy can reuse the stored geometry, charges, and bonded patch metadata through that same script style.

## Runtime options API

The runtime module provides context-local defaults for workflow behavior.

### `get_run_options()`

Return the current `RunOptions` object.

### `set_run_options(restart=None, strict_inputs=None)`

Update the current defaults and return the new `RunOptions`.

### `run_options(restart=None, strict_inputs=None)`

Context manager for temporary overrides.

Typical use:

```python
import yadonpy as yp

yp.set_run_options(restart=False)

with yp.run_options(restart=True):
    pass
```

Environment variables recognized at import time:

- `YADONPY_RESTART`
- `YADONPY_STRICT_INPUTS`

## Interface API

The `yadonpy.interface` package exposes the interface-building layer used by Examples 10, 11, and 12.

Primary classes and helpers:

- `InterfaceBuilder`
- `InterfaceDynamics`
- `InterfaceProtocol`
- `InterfaceRouteSpec`
- `build_interface(...)`
- `build_interface_from_workdirs(...)`

Planning and preparation helpers:

- `build_bulk_equilibrium_profile(...)`
- `plan_direct_electrolyte_counts(...)`
- `plan_rescaled_bulk_counts(...)`
- `plan_resized_electrolyte_counts(...)`
- `plan_fixed_xy_direct_pack_box(...)`
- `plan_fixed_xy_direct_electrolyte_preparation(...)`
- `make_orthorhombic_pack_cell(...)`
- `equilibrate_bulk_with_eq21(...)`
- `recommend_electrolyte_alignment(...)`
- `fixed_xy_semiisotropic_npt_overrides(...)`
- `read_equilibrated_box_nm(...)`

Post-processing helpers:

- `build_interface_group_catalog(...)`
- `export_interface_group_catalog(...)`
- `read_ndx_groups(...)`
- `summarize_cell_charge(...)`
- `summarize_charge_meta(...)`
- `format_cell_charge_audit(...)`
- `format_charge_meta_audit(...)`

Key route logic:

- Route A builds a periodic dual-interface geometry without a vacuum layer.
- Route B adds vacuum padding and is intended for wall-ready protocols.
- Example 12 uses the newer fixed-XY electrolyte preparation path and the reusable `interface.prep` helpers.

## MolDB API and persistence model

For ordinary scripts, use `load_from_moldb(...)`.

If you need direct database operations, import `MolDB` explicitly:

```python
from yadonpy.moldb import MolDB

db = MolDB()
```

The current persistence model is:

- MolDB stores expensive molecule-level preparation results;
- topology trees are generated on demand from those prepared molecules;
- bonded patch metadata can travel with the stored charge variant when relevant.

## Diagnostics and lower-level modules

Useful lower-level modules for advanced scripts:

- `yadonpy.diagnostics.doctor`
- `yadonpy.sim.qm`
- `yadonpy.io.gromacs_system`
- `yadonpy.moldb.MolDB`

Use these when the convenience layer is too small for the workflow you want, but prefer the public root API for ordinary scripts when possible.