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
- `yp.build_layer_stack`
- `yp.run_layer_stack_nvt`
- `yp.analyze_layer_stack_interface`
- `yp.conformation_search`
- `yp.assign_charges`
- `yp.assign_forcefield`
- `yp.load_from_moldb`
- `yp.parameterize_smiles`
- `yp.resolve_prepared_system`
- `yp.run_tg_scan_gmx`
- `yp.run_elongation_gmx`
- `yp.AnalyzeResult`
- `yp.IOAnalysisPolicy`
- `yp.resolve_io_analysis_policy`
- `yp.LayerStackSpec`
- `yp.GraphiteLayerSpec`
- `yp.MolecularLayerSpec`
- `yp.VacuumLayerSpec`
- `yp.ElectrodeChargeSpec`
- `yp.LayerStackNvtResult`
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
yp.build_layer_stack(stack=LayerStackSpec(...), work_dir="./work_layer_stack", ...)
yp.run_layer_stack_nvt(result, *, time_ns=2.0, temp=318.15, ...)
yp.analyze_layer_stack_interface(*, work_dir="./work_layer_stack", analysis_profile="interface_fast", ...)
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

## 2A. Analysis Result API

Analysis is normally driven from an `AnalyzeResult` returned by a workflow stage:

```python
analy = production.analyze()
rdf = analy.rdf(center_mol=li_mol)
msd = analy.msd()
sigma = analy.sigma(msd=msd, temp_k=300.0)
migration = analy.migration(center_mol=li_mol)
```

Recommended public methods:

```python
AnalyzeResult.rdf(
    mol_or_mols=None,
    *,
    center_mol=None,
    analysis_profile: str | None = None,
    site_filter=None,
    r_max_nm: float | None = None,
    frame_stride: int = 1,
    resume: bool = False,
    region: str = "auto",
    ...
)

AnalyzeResult.msd(
    *,
    analysis_profile: str | None = None,
    resume: bool = False,
    geometry: str = "auto",
    unwrap: str = "auto",
    drift: str = "auto",
    selection_mode: str = "default",
    ...
)

AnalyzeResult.get_all_prop(
    *,
    temp: float,
    press: float,
    include_polymer_metrics: bool = True,
    analysis_profile: str | None = None,
    ...
)

AnalyzeResult.sigma(
    *,
    temp_k: float | None = None,
    msd: dict | None = None,
    geometry: str = "auto",
    unwrap: str = "auto",
    drift: str = "auto",
    eh_mode: str = "auto",
)

AnalyzeResult.dielectric(
    *,
    temp_k: float | None = None,
    group: str = "System",
    dt_ps: float | None = None,
    resume: bool = False,
    ...
)

AnalyzeResult.migration(
    center_mol,
    *,
    polymer_mols=None,
    solvent_mols=None,
    anion_mols=None,
    cation_mols=None,
    stride: int | str = "auto",
    rdf_stride: int = 10,
    lag_ps: str | float | int = "auto",
    state_basis: str = "dual",
    residence: bool = True,
    markov: bool = True,
    expert_mode: bool = False,
    out_dir: str | Path | None = None,
)

AnalyzeResult.migration_markov(center_mol, **kwargs)
AnalyzeResult.migration_residence(center_mol, **kwargs)
AnalyzeResult.interface_profile(
    *,
    bin_nm: float = 0.05,
    frame_stride: int | str = "auto",
    region_width_nm: float = 0.75,
    analysis_profile: str = "interface_fast",
    compute_transport: bool = True,
    resume: bool = False,
)
```

Transport semantics:

- `RDF` remains an independent analysis because it is the only routine method
  that requires a center species.
- `analysis_profile="transport_fast"` filters RDF to the main Li coordination
  sites, uses coarser RDF defaults, can resume cached RDF/MSD JSON, and is meant
  for screening many cases.
- `analysis_profile="auto"` reads the production performance policy when present
  and resolves to `transport_fast` or `minimal` for large/long runs.
- In non-`full` profiles, the analyzer also estimates trajectory frame count and
  automatically increases the read-time frame stride or GROMACS `-dt` interval
  for dense legacy `.xtc`/`.trr`/`.edr` streams. This covers RDF, MSD, cell,
  thermo, number-density profiles, Rg/polymer metrics, dielectric dipoles,
  interface profiles, and migration analyses. The resolved policy is written to
  `06_analysis/analysis_runtime_policy.json`; caps can be tuned with
  global `MAX_ANALYSIS_FRAMES` or section-specific `MAX_RDF_FRAMES`,
  `MAX_MSD_FRAMES`, `MAX_CELL_FRAMES`, `MAX_THERMO_FRAMES`,
  `MAX_DENSITY_DISTRIBUTION_FRAMES`, `MAX_DIELECTRIC_FRAMES`,
  `MAX_MIGRATION_FRAMES`, and `MAX_POLYMER_METRIC_FRAMES`.
- MSD uses a hybrid backend rather than a plain atom-index average. Metrics
  marked `gmx_msd_mol_equivalent=true` are delegated to
  `gmx msd -n system.ndx -mol` so GROMACS splits the ndx selection into topology
  molecules and computes COM MSD. YadonPy keeps metadata selection, cache keys,
  local diagnostics, and adaptive log-log fit selection.
- Polymer diffusion uses each independent polymer molecule's chain COM MSD
  (`chain_com_msd`) by default, not atom/residue MSD.  The preferred GROMACS
  backend uses topology molecules from the ndx selection; the Python fallback
  reconstructs whole chains from the bonded graph before unwrapping.
- `residue_com_msd` and `charged_group_com_msd` are intentionally kept as local
  polymer mobility diagnostics. They should not be used as the polymer chain
  self-diffusion coefficient.
- Diffusion fits are selected from the log-log slope trace, but candidate
  windows must also satisfy minimum point and duration requirements
  (`min_fit_points`, `min_fit_duration_ps`).  Short slope-islands are reported
  as apparent diffusion diagnostics rather than promoted to formal `D_m2_s`.
- `analysis_profile="minimal"` is the most aggressive screening mode: necessary
  transport species only, coarser RDF, and default MSD metrics only.
- `analysis_profile="interface_fast"` is for graphite/polymer/electrolyte stacks.
  It writes `06_analysis/interface_profile/` with z density profiles, region
  summaries, Li coordination partitioning, enrichment, and anisotropic `Dxy/Dz`
  MSD diagnostics.
- `analysis_profile="full"` keeps the historical all-site RDF/MSD behavior.
- `get_all_prop(..., include_polymer_metrics=False)` skips expensive Rg,
  end-to-end, and persistence-length post-processing while preserving thermo and
  cell summaries.
- `dielectric()` wraps `gmx dipoles`; set `YADONPY_GMX_CMD` when the default
  `gmx` binary cannot read the production `.tpr`.
- bulk systems default to drift-corrected `3D` diffusion.
- slab and layer-stack interface systems default to drift-corrected `xy` diffusion.
- `sigma_ne_upper_bound_S_m` is reported explicitly as an upper bound.
- `sigma_eh_total_S_m` is the preferred total conductivity when a stable
  Einstein-Helfand fit is available.
- `eh_mode="gmx_current_only"` disables positions-based EH fallback and is the
  recommended mode for benchmark / literature-comparison workflows that must use
  only `gmx current -dsp`.
- `haven_ratio` is reported whenever both conductivities are available.
- charged-polymer self terms are retained as
  `polymer_charged_group_self_ne_contribution_S_m` and component diagnostics;
  they are not labeled as total polymer ionic conductivity.

Production presets accept adaptive output cadence:

```python
eq.NPT(...).exec(
    temp=300.0,
    press=1.0,
    time=300.0,
    traj_ps="auto",
    energy_ps="auto",
    performance_profile="auto",
)
```

`performance_profile="auto"` estimates trajectory and atom-frame cost from the
production length and system size, then records the resolved cadence and analysis
policy in `05_*_production/summary.json`. Explicit numeric `traj_ps`,
`energy_ps`, `log_ps`, `trr_ps`, or `velocity_ps` always override the policy.

The default coordinate stream is compressed XTC only. This keeps large screening
runs manageable because full-precision TRR is much larger and slower to analyze.
Set `trajectory_format="trr"` or `TRAJECTORY_FORMAT=trr` when you deliberately
want TRR-only coordinates; `AnalyzeResult` will use `md.trr` if `md.xtc` is
absent. Set `trajectory_format="xtc_trr"` only for short diagnostics or
explicitly coarse `trr_ps` values.

- `migration()` is the preferred high-level migration workflow for:
  - pure electrolytes,
  - polymer-electrolyte composites,
  - solid polymer electrolytes.
- `migration()` writes default outputs under `06_analysis/migration/` including
  residence summaries, role/site Markov summaries, transition matrices,
  predicted event counts, and static SVG plots.
- `migration()` uses a dual-state basis by default:
  - role-level Markov states: `polymer / solvent / anion / none`
  - site-level Markov states: specific donor anchors plus role-scoped `OTHER`
    buckets when states become too sparse.
- `migration_markov()` exposes the role/site transition summaries directly, while
  `migration_residence()` exposes just the residence layer.
- `AnalyzeResult.from_work_dir(...)` is available for post-hoc analysis of an
  already completed YadonPy work directory.

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

### Segment-first polymer helpers

```python
from yadonpy.core import poly

poly.seg_gen(
    units,
    *,
    name=None,
    label=1,
    cap_head=None,
    cap_tail=None,
    work_dir=None,
    restart=None,
    **rw_kwargs,
)

poly.block_segment_rw(
    segments,
    block_lengths,
    *,
    name=None,
    label=1,
    work_dir=None,
    restart=None,
    **rw_kwargs,
)

poly.branch_segment_rw(
    base,
    branches,
    *,
    position=2,
    ds=None,
    exact_map=None,
    branch_terminator="[H][*]",
    mode="post",
    work_dir=None,
    restart=None,
    **rw_kwargs,
)
```

Use `*` or `[1*]` for main-chain segment ends. Use `[2*]`, `[3*]`, ... for branch
sites. `seg_gen` consumes only the main-chain label and preserves branch labels.
`branch_segment_rw` supports both pre-branching a reusable segment and post-branching
an already-grown chain. Existing per-atom charge properties are preserved; QM/RESP is
only rerun when the user explicitly calls `qm.assign_charges()`.

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

## 9. Generic Layer-Stack Interface API

The preferred public interface builder is now `build_layer_stack(...)`.  It
accepts any ordered sequence of graphite, molecular, and vacuum layers and
writes a GROMACS-ready stacked system plus `layer_stack_manifest.json`.

```python
LayerStackSpec(
    layers: tuple[GraphiteLayerSpec | MolecularLayerSpec | VacuumLayerSpec, ...],
    order: str = "bottom_to_top",
    pbc_mode: str = "auto",
    name: str = "layer_stack",
    default_gap_nm: float = 0.35,
    bottom_padding_nm: float = 0.0,
    top_padding_nm: float = 0.0,
    auto_expand_graphite: bool = True,
)

GraphiteLayerSpec(
    name: str = "GRAPHITE",
    nx: int = 6,
    ny: int = 5,
    n_layers: int = 3,
    orientation: str = "basal",
    edge_cap: str | Sequence[str] = "H",
    periodic_xy: bool | None = None,
    electrode_charge: ElectrodeChargeSpec | None = None,
    ff_name: str = "gaff2_mod",
)

MolecularLayerSpec(
    name: str,
    species: Sequence[rdkit.Chem.Mol],
    counts: Sequence[int],
    thickness_nm: float,
    density_target_g_cm3: float | None = None,
    layer_kind: str = "generic",
    charge_scale: float | Sequence[float] | dict | None = None,
    polyelectrolyte_mode: bool | None = None,
)

VacuumLayerSpec(thickness_nm: float, name: str = "VACUUM")

ElectrodeChargeSpec(
    mode: str = "total_charge",
    top_charge_e: float | None = None,
    bottom_charge_e: float | None = None,
    surface_charge_uC_cm2: float | None = None,
    top_surface_charge_uC_cm2: float | None = None,
    bottom_surface_charge_uC_cm2: float | None = None,
)
```

Main calls:

```python
build_layer_stack(stack=LayerStackSpec(...), work_dir="./work")
run_layer_stack_nvt(result, time_ns=2.0, temp=318.15, omp=14, gpu_id=0)
analyze_layer_stack_interface(work_dir="./work", analysis_profile="interface_fast")
```

Notes:

- Basal graphite defaults to `periodic_xy=True` and uses a periodic basal-plane
  construction. Edge graphite defaults to `periodic_xy=False` and is capped as a
  finite bonded slab.
- `edge_cap` supports `H`, `OH`, `O`/carbonyl, `CHO`, `COOH`, and random mixtures.
- Fixed graphite electrode charge is assigned once to surface atoms.  It is not
  a constant-potential model.
- For two-electrode stacks, use `top_surface_charge_uC_cm2` on the lower
  graphite and `bottom_surface_charge_uC_cm2` on the upper graphite to charge
  only the interior surfaces.
- `pbc_mode="auto"` currently resolves to `xyz`; vacuum layers are explicit
  empty z regions, not implicit GROMACS walls.
- In `xyz` stacks, the top-bottom periodic boundary is treated as a closing
  interface.  The builder adds enough closing spacer to reach `default_gap_nm`
  unless explicit top/bottom padding or vacuum already does so; the value is
  recorded as `acceptance.pbc_closing_gap_nm`.
- `run_layer_stack_nvt(...)` starts with a no-constraints steep minimization,
  then a short bridge NVT, then the requested NVT.  This protects freshly
  stacked CMC/electrolyte/graphite models from local-contact explosions at
  step 0.
- The generated `system.ndx` contains `LAYER_XX_NAME`, semantic phase groups
  such as `GRAPHITE`, `ELECTROLYTE`, `CMCNA`, and `MOBILE`.

## 10. Retired Sandwich API

The old sandwich-specific public API has been removed.  Use the generic
layer-stack API above for graphite/electrolyte, graphite/CMC-Na/electrolyte,
graphite/electrolyte/graphite, vacuum stacks, and fixed-charge electrode
studies.  This hard cut avoids keeping two independent interface builders with
different geometry assumptions.

## 11. Internal Modules

Modules under `yadonpy.core.*`, lower-level `yadonpy.gmx.*`, and utility-heavy helpers
inside `yadonpy.interface.builder` are intentionally not the first dependency surface for
ordinary user scripts. They are useful for extending the package, but the recommended public
surface is the set described above.
