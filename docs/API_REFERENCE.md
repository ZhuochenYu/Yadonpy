# YadonPy API Reference

This document covers the public, script-facing API intended for study scripts and
workflow assembly. It focuses on the top-level package API and the higher-level
workflow modules that users are expected to call directly.

For QM-derived charges, the supported environment is Psi4 plus `psiresp-base`.

## 1. Import Pattern

```python
import yadonpy as yp
```

Package root exports include:

- `yp.get_ff`
- `yp.list_forcefields`
- `yp.list_charge_methods`
- `yp.mol_from_smiles`
- `yp.build_graphite`
- `yp.conformation_search`
- `yp.assign_charges`
- `yp.assign_forcefield`
- `yp.load_from_moldb`
- `yp.parameterize_smiles`
- `yp.build_graphite_polymer_electrolyte_sandwich`
- `yp.build_graphite_peo_electrolyte_sandwich`
- `yp.build_graphite_cmcna_electrolyte_sandwich`
- `yp.resolve_prepared_system`
- `yp.run_tg_scan_gmx`
- `yp.run_elongation_gmx`
- `yp.print_mechanics_result_summary`
- `yp.InterfaceBuilder`
- `yp.InterfaceProtocol`
- `yp.InterfaceDynamics`
- `yp.build_interface`
- `yp.build_interface_from_workdirs`
- `yp.get_run_options`
- `yp.set_run_options`
- `yp.run_options`
- `yp.qm`

## 2. High-Level Convenience API

```python
yp.get_ff(ff_name: str, **kwargs)
yp.list_forcefields() -> tuple[str, ...]
yp.list_charge_methods() -> tuple[str, ...]
yp.mol_from_smiles(smiles: str, *, coord: bool = True, name: str | None = None)
yp.build_graphite(**kwargs)
yp.build_graphite_polymer_electrolyte_sandwich(**kwargs)
yp.build_graphite_peo_electrolyte_sandwich(**kwargs)
yp.build_graphite_cmcna_electrolyte_sandwich(**kwargs)
yp.resolve_prepared_system(
    *,
    gro: str | Path | None = None,
    top: str | Path | None = None,
    work_dir: str | Path | None = None,
    source_name: str | None = None,
)
yp.run_tg_scan_gmx(**kwargs)
yp.run_elongation_gmx(**kwargs)
yp.print_mechanics_result_summary(result)
yp.conformation_search(mol, **kwargs)
yp.assign_charges(mol, *, charge: str = "RESP", **kwargs)
yp.assign_forcefield(mol, *, ff_name: str = "gaff2_mod", charge: str | None = None, **kwargs)
yp.load_from_moldb(
    smiles: str,
    *,
    charge: str = "RESP",
    basis_set: str | None = None,
    method: str | None = None,
    require_ready: bool = True,
    return_record: bool = False,
    polyelectrolyte_mode: bool | None = None,
    polyelectrolyte_detection: str | None = None,
)
yp.parameterize_smiles(
    smiles: str,
    *,
    ff_name: str = "gaff2_mod",
    charge_method: str = "RESP",
    work_dir: str = "./",
    total_charge: int | None = None,
    total_multiplicity: int | None = None,
    name: str | None = None,
    allow_ff_without_requested_charges: bool = False,
    polyelectrolyte_mode: bool = False,
    polyelectrolyte_detection: str = "auto",
)
```

Key points:

- `get_ff(...)` returns a force-field object; this is the normal starting point for
  `ff.mol(...)` and `ff.ff_assign(...)`.
- `assign_forcefield(...)` is a convenience wrapper around `get_ff(...).ff_assign(...)`.
- `load_from_moldb(...)` restores prepared species and, when present, charge-variant
  metadata plus bonded patches.
- `parameterize_smiles(...)` is a concise one-shot helper for
  `SMILES -> charges -> ff_assign`.
- `resolve_prepared_system(...)` resolves a reusable `gro/top` pair from explicit
  paths or from a standard YadonPy equilibration `work_dir`.
- `run_tg_scan_gmx(...)` and `run_elongation_gmx(...)` are the preferred high-level
  study entry points for properties of already equilibrated systems.

Supported canonical force-field names:

- `gaff`
- `gaff2`
- `gaff2_mod`
- `merz`
- `oplsaa`
- `dreiding`

Common charge tokens returned by `list_charge_methods()` include:

- `zero`
- `gasteiger`
- `RESP`
- `ESP`
- `Mulliken`
- `Lowdin`
- the lightweight quick-charge aliases understood by `qm.assign_charges(...)`

## 3. Runtime Control

Module: `yadonpy.runtime`

```python
from yadonpy.runtime import (
    RecommendedResources,
    RunOptions,
    get_run_options,
    recommend_local_resources,
    run_options,
    set_run_options,
)
```

### Dataclasses

```python
RunOptions(
    restart: bool = True,
    strict_inputs: bool = True,
)

RecommendedResources(
    mpi: int = 1,
    omp: int = 1,
    gpu: int = 1,
    gpu_id: int | None = 0,
    omp_psi4: int = 1,
    cpu_total: int = 1,
    cpu_cap: int | None = None,
)
```

### Functions

```python
get_run_options() -> RunOptions
set_run_options(*, restart: bool | None = None, strict_inputs: bool | None = None) -> RunOptions
run_options(*, restart: bool | None = None, strict_inputs: bool | None = None) -> Iterator[RunOptions]
recommend_local_resources(
    *,
    cpu_cap: int | None = None,
    mpi_default: int = 1,
    gpu_default: int = 1,
    gpu_id_default: int | None = 0,
    omp_psi4_cap: int | None = 8,
) -> RecommendedResources
```

Use this layer when you want consistent restart defaults or a conservative
resource layout for local scripts.

## 4. MolDB API

Module: `yadonpy.moldb`

```python
from yadonpy.moldb import MolDB, MolRecord, canonical_key
```

### Canonicalization

```python
canonical_key(smiles_or_psmiles: str) -> tuple[str, str, str]
```

Returns:

- kind: `"smiles"` or `"psmiles"`
- canonical string
- content-addressed key

### MolRecord

```python
MolRecord(
    key: str,
    kind: str,
    canonical: str,
    name: str,
    charge_method: str | None = None,
    charge_unit: str = "e",
    ready: bool = False,
    variants: dict[str, dict[str, object]] = ...,
    connectors: list[dict[str, int]] | None = None,
)
```

### Main database methods

```python
MolDB(db_dir: Path | None = None)

MolDB.load_mol(
    smiles_or_psmiles: str,
    *,
    require_ready: bool = False,
    charge: str = "RESP",
    basis_set: str | None = None,
    method: str | None = None,
    polyelectrolyte_mode: bool | None = None,
    polyelectrolyte_detection: str | None = None,
) -> tuple[Chem.Mol, MolRecord]

MolDB.save_geometry(key: str, mol: Chem.Mol, *, name: str) -> Path

MolDB.save_charges(
    key: str,
    mol: Chem.Mol,
    *,
    charge: str = "RESP",
    basis_set: str | None = None,
    method: str | None = None,
    polyelectrolyte_mode: bool | None = None,
    polyelectrolyte_detection: str | None = None,
) -> str
```

Behavioral notes:

- MolDB stores expensive reusable molecular assets, not final exported topologies.
- Variant resolution distinguishes charge method, basis, method, and grouped-polyelectrolyte metadata.
- `load_from_moldb(...)` is the top-level convenience wrapper around this layer.

## 5. Diagnostics

Module: `yadonpy.diagnostics`

Primary public helper:

```python
doctor(*, print_report: bool = True) -> dict[str, object]
```

This reports:

- Python version,
- YadonPy data root,
- MolDB directory,
- GROMACS discovery,
- import status of `rdkit`, `psi4`, and `psiresp`.

## 6. QM and Charge Assignment

Module: `yadonpy.sim.qm`

### Charge assignment and conformers

```python
qm.assign_charges(
    mol,
    charge="RESP",
    confId=0,
    opt=True,
    work_dir=None,
    tmp_dir=None,
    log_name=None,
    qm_solver="psi4",
    opt_method="wb97m-d3bj",
    opt_basis="def2-SVP",
    opt_basis_gen={"Br": "def2-SVP", "I": "def2-SVP"},
    geom_iter=50,
    geom_conv="QCHEM",
    geom_algorithm="RFO",
    charge_method="wb97m-d3bj",
    charge_basis="def2-TZVP",
    charge_basis_gen={"Br": "def2-TZVP", "I": "def2-TZVP"},
    auto_level=True,
    bonded_params="auto",
    total_charge=None,
    total_multiplicity=None,
    polyelectrolyte_mode=False,
    polyelectrolyte_detection="auto",
    symmetrize=True,
    symmetrize_geometry=True,
    restart=None,
    **kwargs,
)

qm.conformation_search(
    mol,
    ff=None,
    nconf=1000,
    dft_nconf=4,
    etkdg_ver=2,
    rmsthresh=0.5,
    tfdthresh=0.02,
    clustering="TFD",
    qm_solver="psi4",
    opt_method="wb97m-d3bj",
    opt_basis="def2-SVP",
    opt_basis_gen={"Br": "def2-SVP", "I": "def2-SVP"},
    geom_iter=50,
    geom_conv="QCHEM",
    geom_algorithm="RFO",
    log_name=None,
    work_dir=None,
    tmp_dir=None,
    etkdg_omp=-1,
    psi4_omp=-1,
    psi4_mp=0,
    mm_mp=0,
    memory=1000,
    mm_solver="rdkit",
    gmx_refine_n=0,
    gmx_ntomp=None,
    gmx_ntmpi=None,
    gmx_gpu_id=None,
    total_charge=None,
    total_multiplicity=None,
    restart=None,
    **kwargs,
)
```

Important behavior:

- RESP and ESP use Psi4 for the actual QM calculation and `psiresp-base` for fitting.
- grouped-polyelectrolyte RESP is available through `polyelectrolyte_mode=True`.
- hydrogen terminator placeholders are handled specially so polymer workflows do not crash.

### Single-point and optical-property helpers

```python
qm.sp_prop(...)
qm.polarizability(...)
qm.refractive_index(...)
qm.abbe_number_cc2(...)
qm.polarizability_sos(...)
qm.refractive_index_sos(...)
qm.abbe_number_sos(...)
```

These routines provide higher-level wrappers around optimization, single-point, and
response-property workflows. Use them when the project needs optical properties in
addition to force-field preparation.

### Bonded-parameter derivation

```python
qm.bond_angle_params_mseminario(
    mol: Chem.Mol,
    *,
    confId: int = 0,
    opt: bool = True,
    work_dir: str | None = None,
    tmp_dir: str | None = None,
    log_name: str = "bond_angle",
    qm_solver: str = "psi4",
    opt_method: str = "wb97m-d3bj",
    opt_basis: str = "6-31G(d,p)",
    opt_basis_gen: dict[str, object] | None = None,
    geom_iter: int = 50,
    geom_conv: str = "QCHEM",
    geom_algorithm: str = "RFO",
    hess_method: str | None = None,
    hess_basis: str | None = None,
    hess_basis_gen: dict[str, object] | None = None,
    linear_angle_deg_cutoff: float = 175.0,
    projection_mode: str = "abs",
    keep_linear_angles: bool = True,
    symmetrize_equivalents: bool = True,
    total_charge: int | None = None,
    total_multiplicity: int | None = None,
    write_itp: bool = True,
    itp_name: str = "bond_angle_params.itp",
    json_name: str = "bond_angle_params.json",
    **kwargs,
)

qm.bond_angle_params_drih(
    mol: Chem.Mol,
    *,
    confId: int = 0,
    work_dir: str,
    log_name: str,
    smiles_hint: str = None,
    k_bond_kj_mol_nm2: float = 350000.0,
    k_angle_kj_mol_rad2: float = 2500.0,
    k_angle_linear_kj_mol_rad2: float = 6000.0,
    stiffness_scale: float = 1.0,
    write_itp: bool = True,
    itp_name: str = "bonded_drih_patch.itp",
    json_name: str = "bonded_drih_params.json",
) -> dict
```

Use these when standard force-field typing is not enough and a QM-derived bonded patch
should be attached to the prepared species.

## 7. Interface Builder API

Module: `yadonpy.interface`

### Core assembly calls

```python
InterfaceBuilder(*, work_dir: str | Path | WorkDir, restart: bool | None = None)
InterfaceDynamics(*, built: BuiltInterface, work_dir: str | Path | WorkDir, restart: bool | None = None)

build_interface(
    *,
    work_dir: str | Path | WorkDir,
    name: str,
    bottom: BulkSource,
    top: BulkSource,
    route: InterfaceRouteSpec,
    restart: bool | None = None,
) -> BuiltInterface

build_interface_from_workdirs(
    *,
    work_dir: str | Path | WorkDir,
    name: str,
    bottom_name: str,
    bottom_work_dir: str | Path,
    top_name: str,
    top_work_dir: str | Path,
    route: InterfaceRouteSpec,
    restart: bool | None = None,
) -> BuiltInterface
```

### Planning and bulk helpers

```python
equilibrate_bulk_with_eq21(
    *,
    label: str,
    ac,
    work_dir,
    temp: float,
    press: float,
    mpi: int,
    omp: int,
    gpu: int,
    gpu_id: int,
    additional_loops: int = 4,
    eq21_npt_mdp_overrides=None,
    additional_mdp_overrides=None,
    final_npt_ns: float = 0.0,
    final_npt_mdp_overrides=None,
    eq21_exec_kwargs: dict[str, object] | None = None,
) -> BulkEq21Outcome

build_bulk_equilibrium_profile(
    *,
    counts: Sequence[int],
    mol_weights: Sequence[float],
    species_names: Sequence[str] | None = None,
    work_dir: Path | None = None,
    gro_path: Path | None = None,
) -> BulkEquilibriumProfile

read_equilibrated_box_nm(*, work_dir: Path | None = None, gro_path: Path | None = None) -> tuple[float, float, float]

recommend_electrolyte_alignment(
    *,
    top_thickness_nm: float,
    gap_nm: float,
    surface_shell_nm: float,
    is_polyelectrolyte: bool = False,
    minimum_margin_nm: float = 1.0,
    fixed_xy_npt_ns: float | None = None,
) -> ElectrolyteAlignmentPlan

recommend_polymer_diffusion_interface_recipe(...)
```

### Configuration records

```python
AreaMismatchPolicy(
    max_lateral_strain: float = 0.03,
    prefer_larger_area: bool = True,
    max_lateral_replicas_xy: tuple[int, int] = (8, 8),
    reference_side: Literal["larger", "smaller", "bottom", "top"] = "larger",
)

BulkSource(...)

SlabBuildSpec(
    axis: Axis = "Z",
    target_thickness_nm: float | None = None,
    target_density_g_cm3: float | None = None,
    gap_nm: float = 0.6,
    vacuum_nm: float = 0.0,
    surface_shell_nm: float = 0.8,
    core_guard_nm: float = 0.5,
    min_replicas_xy: tuple[int, int] = (1, 1),
    prefer_densest_window: bool = True,
    lateral_recentering: bool = True,
)

InterfaceRouteSpec(
    route: Route,
    axis: Axis = "Z",
    area_policy: AreaMismatchPolicy = ...,
    bottom: SlabBuildSpec = ...,
    top: SlabBuildSpec = ...,
    top_lateral_shift_fraction: tuple[float, float] = (0.5, 0.5),
)

InterfaceRouteSpec.route_a(...)
InterfaceRouteSpec.route_b(...)
```

Additional exported planning/result records for advanced workflows:

- `BulkEq21Outcome`
- `BulkEquilibriumProfile`
- `BulkRescalePlan`
- `DirectElectrolytePlan`
- `FixedXYDirectPackPlan`
- `FixedXYElectrolytePreparation`
- `PolymerAnchoredInterfacePreparation`
- `DirectPolymerMatchedInterfacePreparation`
- `ProbePolymerMatchedInterfacePreparation`
- `ResizedPolymerMatchedInterfacePreparation`
- `ProbeElectrolytePreparation`
- `ResizedElectrolytePreparation`
- `PolymerDiffusionInterfaceRecipe`
- `BuiltInterface`
- `PreparedSlab`

`InterfaceBuilder` is the stateful object-oriented entry point for explicit interface assembly.
`InterfaceDynamics` is the stateful object-oriented entry point for running staged interface dynamics
on a previously built interface.

## 8. Interface Dynamics

```python
InterfaceStageSpec(name: str, kind: str, description: str, mdp: MdpSpec)

InterfaceProtocol(
    route: str,
    axis: str = "Z",
    stage_mode: str = "simple",
    pre_contact_ps: float = 100.0,
    pre_contact_dt_ps: float = 0.001,
    density_relax_ps: float = 200.0,
    contact_ps: float = 200.0,
    release_ps: float = 200.0,
    exchange_ns: float = 2.0,
    production_ns: float = 5.0,
    temperature_k: float = 300.0,
    pressure_bar: float = 1.0,
    semiisotropic: bool = True,
    wall_mode: str | None = None,
    wall_atomtype: str | None = None,
    wall_r_linpot_nm: float | None = 0.05,
    wall_density_nm3: float | None = None,
    freeze_cores_pre_contact: bool = True,
    use_region_thermostat_early: bool = True,
    density_relax_barostat: str = "Berendsen",
    contact_barostat: str = "Berendsen",
    release_barostat: str = "Berendsen",
    exchange_barostat: str = "Berendsen",
    production_barostat: str = "C-rescale",
    density_relax_tau_p: float = 16.0,
    contact_tau_p: float = 12.0,
    release_tau_p: float = 10.0,
    exchange_tau_p: float = 8.0,
    production_tau_p: float = 5.0,
    berendsen_compressibility_scale: float = 0.5,
)
```

Use `InterfaceProtocol` when the default route builder is not enough and you want direct
control over staged interface relaxation.

## 9. Graphite-Polymer-Electrolyte Sandwich API

This is the highest-level interface workflow exposed by YadonPy.

### Specification records

```python
MoleculeSpec(
    name: str,
    smiles: str,
    charge_method: str = "RESP",
    bonded: str | None = None,
    prefer_db: bool = False,
    require_ready: bool = False,
    use_ion_ff: bool = False,
    charge_scale: float = 1.0,
)

GraphiteSubstrateSpec(
    nx: int = 4,
    ny: int = 4,
    n_layers: int = 3,
    edge_cap: str = "H",
    orientation: str = "basal",
    name: str = "GRAPH",
    top_padding_ang: float = 15.0,
)

PolymerSlabSpec(
    name: str = "PEO",
    monomer_smiles: str = "*CCO*",
    monomers: tuple[MoleculeSpec, ...] = (),
    monomer_ratio: tuple[float, ...] = (1.0,),
    terminal_smiles: str = "[H][*]",
    terminal: MoleculeSpec | None = None,
    chain_target_atoms: int = 280,
    dp: int | None = None,
    chain_count: int | None = None,
    counterion: MoleculeSpec | None = None,
    target_density_g_cm3: float = 1.1,
    slab_z_nm: float = 3.6,
    min_chain_count: int = 2,
    tacticity: str = "atactic",
    charge_scale: float = 1.0,
    initial_pack_z_scale: float = 1.18,
    pack_retry: int = 30,
    pack_retry_step: int = 2400,
    pack_threshold_ang: float = 1.55,
    pack_dec_rate: float = 0.72,
)

ElectrolyteSlabSpec(
    solvents: tuple[MoleculeSpec, ...] = ...,
    salt_cation: MoleculeSpec = ...,
    salt_anion: MoleculeSpec = ...,
    solvent_mass_ratio: tuple[float, ...] = (1.0,),
    target_density_g_cm3: float = 1.18,
    slab_z_nm: float = 4.0,
    salt_molarity_M: float = 1.0,
    min_salt_pairs: int = 3,
    initial_pack_density_g_cm3: float | None = None,
    pack_retry: int = 30,
    pack_retry_step: int = 2400,
    pack_threshold_ang: float = 1.55,
    pack_dec_rate: float = 0.72,
)

SandwichRelaxationSpec(
    temperature_k: float = 300.0,
    pressure_bar: float = 1.0,
    mpi: int = 1,
    omp: int = 8,
    gpu: int = 1,
    gpu_id: int | None = 0,
    psi4_omp: int = 8,
    psi4_memory_mb: int = 16000,
    bulk_eq21_final_ns: float = 0.1,
    bulk_additional_loops: int = 1,
    bulk_eq21_exec_kwargs: dict[str, float] = ...,
    graphite_to_polymer_gap_ang: float = 3.8,
    polymer_to_electrolyte_gap_ang: float = 4.2,
    top_padding_ang: float = 12.0,
    stacked_pre_nvt_ps: float = 20.0,
    stacked_z_relax_ps: float = 80.0,
    stacked_exchange_ps: float = 120.0,
)

SandwichPhaseReport(...)
GraphitePolymerElectrolyteSandwichResult(...)
```

### Factory helpers

```python
default_peo_polymer_spec(**kwargs) -> PolymerSlabSpec
default_peo_electrolyte_spec(**kwargs) -> ElectrolyteSlabSpec
default_cmcna_polymer_spec(**kwargs) -> PolymerSlabSpec
default_carbonate_lipf6_electrolyte_spec(**kwargs) -> ElectrolyteSlabSpec
```

### Main builders

```python
build_graphite_polymer_electrolyte_sandwich(**kwargs)
build_graphite_peo_electrolyte_sandwich(**kwargs)
build_graphite_cmcna_electrolyte_sandwich(**kwargs)
```

Use the PEO entry point for compact smoke tests and the generic or CMC-Na entry points
when you want explicit control over polymer composition and charged groups.

## 10. Internal Modules

Modules under `yadonpy.core.*`, lower-level `yadonpy.gmx.*`, and utility-heavy helpers
inside `yadonpy.interface.builder` are intentionally not the first dependency surface for
ordinary user scripts. They are useful for extending the package, but the recommended public
surface is the set described above.
