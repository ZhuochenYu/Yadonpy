# YadonPy API (v0.8.66)

This document describes the public, script-facing API for the current release. It focuses on entry points that users are expected to call directly from study scripts.

Related documents:

- manual: `docs/Yadonpy_manul.md`
- user guide: `docs/Yaonpyd_user_guide.md`

## 1. API layers

YadonPy has three practical API layers.

### 1.1 Top-level convenience API

Use:

```python
import yadonpy as yp
```

This is the preferred layer for ordinary scripts.

### 1.2 Package-level workflow APIs

Use modules such as `yadonpy.interface`, `yadonpy.sim.qm`, and `yadonpy.moldb` when the top-level helpers are too small for the workflow you need.

### 1.3 Internal utility modules

Modules under `yadonpy.core.*`, `yadonpy.io.*`, and some `yadonpy.gmx.*` internals are not the first choice for ordinary scripts. They are useful when extending the package or debugging lower-level behavior, but they should not be the default dependency surface for user scripts.

## 2. Package root exports

The package root exports these main names:

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

## 3. Top-level convenience functions

### `get_ff(ff_name, **kwargs)`

Return a force-field object from the registry.

Typical use:

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
```

### `list_forcefields()`

Return the registered force-field names.

### `list_charge_methods()`

Return the common charge method tokens recognized by the convenience layer.

### `mol_from_smiles(smiles, coord=True, name=None)`

Create a molecule from SMILES with YadonPy defaults.

Use this when you want a concrete molecule immediately instead of a force-field handle.

### `conformation_search(mol, **kwargs)`

Thin wrapper around `yadonpy.sim.qm.conformation_search`.

### `assign_charges(mol, charge="RESP", **kwargs)`

Thin wrapper around `yadonpy.sim.qm.assign_charges`.

### `assign_forcefield(mol, ff_name="gaff2_mod", charge=None, **kwargs)`

Create a force field and run its `ff_assign(...)` method.

Return value:

- `(ff, ok)`

### `parameterize_smiles(...)`

High-level helper for:

1. molecule creation;
2. charge assignment;
3. force-field assignment.

This helper is intentionally strict by default. If the requested charge path fails, it raises unless you explicitly allow fallback behavior.

### `load_from_moldb(...)`

Load a molecule from MolDB by key.

Key behavior:

- selects the requested charge variant through `charge`, `basis_set`, and `method`;
- can require that the variant is already marked ready;
- can return the backing record with `return_record=True`;
- restores stored bonded-patch metadata when the selected variant carries it.

## 4. Explicit force-field object workflow

Most serious scripts use this style:

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
mol = ff.mol("O=C1OCCO1", name="EC")
ok = ff.ff_assign(mol)
```

This pattern matters because `ff.mol(...)` may return a lightweight handle instead of a fully materialized molecule. The same script shape also works when the handle resolves through MolDB.

Use this style when:

- the script should stay close to the shipped examples;
- the same code path should work with or without MolDB reuse;
- you need explicit bonded overrides.

## 5. Runtime options

### `get_run_options()`

Return the current runtime default object.

### `set_run_options(restart=None, strict_inputs=None)`

Set global defaults for the current context.

### `run_options(restart=None, strict_inputs=None)`

Context manager for temporary overrides.

Recognized environment variables:

- `YADONPY_RESTART`
- `YADONPY_STRICT_INPUTS`

## 6. Interface API

The interface layer is intended for explicit, script-visible interface studies.

Primary classes:

- `InterfaceBuilder`
- `InterfaceDynamics`
- `InterfaceProtocol`
- `InterfaceRouteSpec`

Convenience functions:

- `build_interface(...)`
- `build_interface_from_workdirs(...)`

### 6.1 Route selection

`InterfaceRouteSpec` exposes two route constructors:

- `InterfaceRouteSpec.route_a(...)`
- `InterfaceRouteSpec.route_b(...)`

Use route A for a periodic interfacial workflow and route B for a vacuum-buffered wall-ready workflow.

### 6.2 Protocol selection

`InterfaceProtocol` exposes:

- `InterfaceProtocol.route_a(...)`
- `InterfaceProtocol.route_a_diffusion(...)`
- `InterfaceProtocol.route_b_wall(...)`
- `InterfaceProtocol.route_b_wall_diffusion(...)`

The diffusion constructors currently also accept:

- `pre_contact_dt_ps`
- `freeze_cores_pre_contact`
- `use_region_thermostat_early`

These are used by the new interface recipe helper.

### 6.3 Interface planning helpers

Useful planning helpers under `yadonpy.interface`:

- `plan_direct_electrolyte_counts(...)`
- `plan_fixed_xy_direct_pack_box(...)`
- `plan_fixed_xy_direct_electrolyte_preparation(...)`
- `plan_polymer_anchored_interface_preparation(...)`
- `plan_direct_polymer_matched_interface_preparation(...)`
- `plan_probe_electrolyte_preparation(...)`
- `plan_probe_polymer_matched_interface_preparation(...)`
- `plan_resized_electrolyte_preparation_from_probe(...)`
- `plan_resized_polymer_matched_interface_from_probe(...)`
- `recommend_electrolyte_alignment(...)`
- `fixed_xy_semiisotropic_npt_overrides(...)`
- `equilibrate_bulk_with_eq21(...)`
- `make_orthorhombic_pack_cell(...)`
- `read_equilibrated_box_nm(...)`

### 6.4 New route/protocol recipe helper

`recommend_polymer_diffusion_interface_recipe(...)`

Purpose:

- consume an already planned `PolymerAnchoredInterfacePreparation`;
- choose route A or route B defaults for a polymer/electrolyte diffusion study;
- generate a matching staged `InterfaceProtocol`.

Return object:

- `PolymerDiffusionInterfaceRecipe`

Key attributes:

- `interface_plan`
- `route_spec`
- `protocol`
- `notes`

Typical use:

```python
from yadonpy.interface import (
    InterfaceBuilder,
    InterfaceDynamics,
    plan_direct_polymer_matched_interface_preparation,
    recommend_polymer_diffusion_interface_recipe,
)

prep = plan_direct_polymer_matched_interface_preparation(...)
recipe = recommend_polymer_diffusion_interface_recipe(
    interface_plan=prep.interface_plan,
    temperature_k=300.0,
    pressure_bar=1.0,
    prefer_vacuum=False,
)

built = InterfaceBuilder(work_dir="./work/interface").build_from_bulk_workdirs(
    name="demo",
    bottom_name="ac_poly",
    bottom_work_dir="./work/ac_poly",
    top_name="ac_electrolyte",
    top_work_dir="./work/ac_electrolyte",
    route=recipe.route_spec,
)
final_gro = InterfaceDynamics(built=built, work_dir="./work/interface_md").run(protocol=recipe.protocol)
```

## 7. MolDB API

For simple reuse, prefer `load_from_moldb(...)`.

For direct database operations:

```python
from yadonpy.moldb import MolDB

db = MolDB()
```

Common direct operations include:

- checking existing entries;
- generating workdir-local snippets;
- running batch precomputation.

## 8. Diagnostics and lower-level modules

Useful lower-level entry points:

- `yadonpy.diagnostics.doctor`
- `yadonpy.sim.qm`
- `yadonpy.moldb.MolDB`
- `yadonpy.io.gromacs_system`

These are appropriate when a study needs more control than the top-level layer provides.

## 9. Stability guidance

For user scripts, prefer depending on:

- top-level exports;
- `yadonpy.interface`;
- `yadonpy.moldb`;
- `yadonpy.sim.qm` when charge/QM control is required.

Avoid making ordinary scripts depend directly on deep internal helpers unless you are intentionally extending the package internals.
