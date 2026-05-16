# YadonPy User Guide

This guide is written for day-to-day use. It focuses on the practical decisions
you need to make when preparing molecules, building systems, and choosing between
local runs and heavier remote calculations.

## 1. Install a Working Environment

Recommended setup:

```bash
conda create -n yadonpy python=3.11
conda activate yadonpy

conda install -c conda-forge rdkit openbabel parmed mdtraj matplotlib pandas scipy packaging psi4=1.10 dftd3-python psiresp-base
python -m pip install "pydantic==1.10.26"
python -m pip install -e .
```

The repository also ships a default `moldb/` folder beside `examples/`. On the
first YadonPy import or `ensure_initialized()` call, that reference catalog is
seeded into `~/.yadonpy/moldb`.

Then check the environment:

```bash
python -c "from yadonpy.diagnostics import doctor; doctor(print_report=True)"
```

What matters most:

- `rdkit` is needed for SMILES parsing, embedding, and polymer construction.
- `psi4` plus `psiresp-base` are needed for RESP and ESP charge fitting.
- `gmx` is needed for export verification and MD workflows.

## 2. Choose the Right Entry Point

YadonPy works best when you pick the smallest API layer that still matches the job.

Use the top-level package API when you want concise scripts:

```python
import yadonpy as yp
```

Use `yadonpy.sim.qm` when you need explicit control of:

- conformer search,
- QM optimization,
- basis and functional choices,
- RESP or ESP charge generation,
- modified Seminario or DRIH bonded patches.

Use `yadonpy.interface` when the work is about:

- generic layer-stack interface construction,
- graphite basal/edge surfaces,
- fixed-charge electrode sweeps,
- layer-aware interface analysis.

## 3. Prepare Molecules from SMILES

### Minimal force-field assignment

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
mol = ff.mol("O=C1OCCO1")
ok = ff.ff_assign(mol)
```

### QM-derived charges

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
mol = ff.mol("O=C1OCCO1")

yp.assign_charges(
    mol,
    charge="RESP",
    work_dir="./work_ec",
)
ok = ff.ff_assign(mol)
```

### Polymer repeat units and terminal groups

For polymers, start from PSMILES-style monomers such as `*CCO*`.
Terminal groups are part of the workflow, not an afterthought. This matters for:

- repeat-unit geometry,
- force-field typing at the chain ends,
- RESP fitting for real terminal groups,
- robust handling of the special hydrogen terminator placeholder `[H][*]`.

If the terminal group is chemically meaningful, keep it explicit.
If it is the internal hydrogen placeholder used by YadonPy polymer builders,
YadonPy treats it as a builder marker instead of sending it through fragile QM
paths.

## 4. Reuse Prepared Assets with MolDB

MolDB is the persistent store for expensive molecular preparation.

It keeps:

- canonical SMILES or PSMILES,
- prepared geometry,
- charge variants,
- grouped-polyelectrolyte RESP metadata,
- optional bonded patches such as DRIH or modified Seminario outputs.

It does not treat exported `.top`, `.itp`, or `.gro` files as the canonical persistent asset.

Typical lookup:

```python
import yadonpy as yp

pf6 = yp.load_from_moldb(
    "F[P-](F)(F)(F)(F)F",
    charge="RESP",
    require_ready=True,
)
```

Use Example 07 when you want to precompute a broader catalog of reusable species first.
Use Example 09 when the task is specifically OPLS-AA typing or OPLS-AA export rather than MolDB catalog construction.
Use Example 11 for segment-first long-block and branched-polymer construction.

## 5. Build Polymer-Electrolyte Systems

The most reliable bulk workflow is:

1. prepare monomers, solvents, and ions;
2. assign or restore charges;
3. assign force fields;
4. construct the polymer chains;
5. pack the amorphous cell;
6. export and equilibrate;
7. check density before trusting production properties.

For charged polymers, use grouped RESP when appropriate:

```python
from yadonpy.sim import qm

qm.assign_charges(
    monomer,
    charge="RESP",
    work_dir="./work_monomer",
    polyelectrolyte_mode=True,
)
```

This keeps later scaling and analysis tied to explicit charged-group metadata.

### Segment-first and branched polymers

For branchable repeat units or large pre-optimized repeat motifs, use segment-first
construction instead of forcing every atom into one monomer definition:

```python
from yadonpy.core import poly

segment_a = poly.seg_gen([monomer_A, monomer_A, monomer_B])
segment_b = poly.seg_gen([branchable_unit, branchable_unit, monomer_A])
side_segment = poly.seg_gen([side_unit], cap_tail="[H][*]")

prebranched_segment = poly.branch_segment_rw(
    segment_b,
    [side_segment],
    mode="pre",
    position=2,
    exact_map={"position": 2, "site_index": 0, "branch": 0},
)
polymer = poly.block_segment_rw([segment_a, prebranched_segment], [3, 2])
polymer = poly.branch_segment_rw(polymer, [side_segment], mode="post", position=2, ds=[1.0])
```

Connection labels are explicit:

- `*` or `[1*]` is the main-chain head/tail label consumed by `seg_gen` and segment polymerization.
- `[2*]`, `[3*]`, ... are preserved branch sites consumed by `branch_segment_rw`.
- Existing per-atom charges are preserved and merged across consumed linker atoms; segment building does not automatically rerun QM/RESP.

## 5A. Analyze Transport Carefully

For routine post-processing, keep `rdf()`, `msd()`, and `sigma()` separate:

```python
analy = production.analyze()
rdf = analy.rdf(center_mol=li_mol)
msd = analy.msd()
sigma = analy.sigma(msd=msd, temp_k=300.0)
dielectric = analy.dielectric(temp_k=300.0)
migration = analy.migration(center_mol=li_mol)
```

For screening many electrolyte cases, switch to the fast transport profile:

```python
prop = analy.get_all_prop(
    temp=300.0,
    press=1.0,
    include_polymer_metrics=False,
    analysis_profile="transport_fast",
)
rdf = analy.rdf(center_mol=li_mol, analysis_profile="transport_fast", resume=True)
msd = analy.msd(analysis_profile="transport_fast", resume=True)
```

This profile computes the coordination sites needed for Li transport diagnosis
with coarser RDF settings and skips expensive full-chain conformation metrics.
Use `analysis_profile="full"` for publication-style all-site RDF and polymer
metric reports.

For long production runs or large boxes, let YadonPy choose output and analysis
resolution automatically:

```bash
PERFORMANCE_PROFILE=auto ANALYSIS_PROFILE=auto PROD_NS=300 python benchmark_peo_litfsi_jpcb2020.py
```

The auto policy keeps short/small jobs close to 2 ps trajectory output, but moves
large or long jobs to coarser 10-50 ps output and matching RDF/MSD settings. Use
`PERFORMANCE_PROFILE=full` or explicit `TRAJ_PS`, `ENERGY_PS`, and `LOG_PS`
when you need dense trajectories for short-time dynamics or final publication
audits.

Production presets write adaptive TRR-only coordinates by default.  This keeps
the trajectory directly usable by `gmx current` for Einstein-Helfand
conductivity while still avoiding dense output on long or large runs.  If you
prefer smaller screening trajectories, set `TRAJECTORY_FORMAT=xtc`; the analyzer
will read either `md.trr` or `md.xtc`. Use `TRAJECTORY_FORMAT=xtc_trr` only for
short diagnostics or explicitly coarse `TRR_PS` output.

For disk cleanup after analysis, place the trajectory cleanup helper near the
end of the workflow:

```python
from yadonpy import clean_md_trajectory_files

clean_md_trajectory_files(work_dir, enabled=True)
```

It removes trajectory streams (`.xtc`, `.trr`, `.trj`, `.tng`) but keeps final
coordinates, topology, energy files, summary JSON, CSV tables, and plots.

When an existing run already contains an overly dense trajectory, the analyzer
protects post-processing time by increasing the read-time frame stride instead
of rewriting the `.xtc`. The resolved runtime policy is saved as
`06_analysis/analysis_runtime_policy.json`. The main controls are
global `MAX_ANALYSIS_FRAMES` plus section-specific `MAX_RDF_FRAMES`,
`MAX_MSD_FRAMES`, `MAX_CELL_FRAMES`, `MAX_THERMO_FRAMES`,
`MAX_DENSITY_DISTRIBUTION_FRAMES`, `MAX_DIELECTRIC_FRAMES`,
`MAX_MIGRATION_FRAMES`, and `MAX_POLYMER_METRIC_FRAMES`;
`analysis_profile="full"` disables this automatic runtime thinning.

Dielectric constants are available through `dielectric()`, which calls
`gmx dipoles` and reads the final static dielectric estimate from
`06_analysis/dielectric.json`. On remote machines with more than one GROMACS,
set `YADONPY_GMX_CMD` to the same major version used for production; otherwise
post-processing can fail on newer `.tpr` files.

This keeps the defaults physically aligned:

- bulk systems use drift-corrected `3D` diffusion by default;
- layer-stack and slab systems use drift-corrected `xy` diffusion by default;
- wrapped trajectories are normalized before transport analysis;
- MSD uses a topology-molecule strategy by default: single-atom ions use atom
  trajectories, molecular ions and solvents use molecule COM trajectories, and
  polymers use independent chain COM trajectories;
- `gmx msd -n system.ndx -mol` is used automatically for molecule/chain COM
  metrics that are exactly equivalent to GROMACS topology molecules; Python
  remains the backend for ions, charged-group/residue diagnostics, interface
  metadata, caching, and adaptive-fit orchestration;
- polymer diffusion is reported from each chain center-of-mass MSD by default;
  the preferred backend lets GROMACS split the selected ndx group into topology
  molecules, while the Python fallback uses bonded-graph whole-chain
  reconstruction before unwrapping;
- adaptive diffusion fits require a sufficiently long time window as well as a
  near-one log-log MSD slope, so short accidental linear islands are not treated
  as reliable diffusion coefficients;
- `sigma_ne_upper_bound_S_m` is treated explicitly as an upper bound;
- `sigma_eh_total_S_m` is preferred when a stable EH fit exists;
- `haven_ratio` is reported whenever both conductivities are available.

Practical interpretation:

- A large `sigma_ne_upper_bound_S_m` with a much smaller `sigma_eh_total_S_m`
  usually means ion correlations are strong.
- For polymers, treat `chain_com_msd` as the self-diffusion observable. Use
  `residue_com_msd` and charged-site MSDs to discuss local segmental or
  functional-group mobility, not whole-chain transport.
- If `D_m2_s` is missing but `apparent_D_m2_s` is present, the trajectory did
  not contain a sufficiently long formal diffusion window. This is common for
  slow polymers or short screening trajectories and is safer than fitting an
  early subdiffusive regime.
- `RDF` stays independent because it is the only routine analysis that needs a
  center species and coordination-shell semantics.
- In interface systems, do not interpret the default diffusion coefficient as a
  free `3D` value unless you explicitly request `geometry="3d"`.
- Charged-polymer results are kept as
  `polymer_charged_group_self_ne_contribution_S_m`; they are not equivalent to
  a rigorously separated polymer ionic conductivity.
- Migration analysis is available through `AnalyzeResult` methods rather than a
  standalone monolithic script. The default path reports:
  - coordination roles,
  - residence times for polymer / solvent / anion donors,
  - role-level and site-level Markov models,
  - transition-matrix-based event flux predictions,
  - migration summary plots under `06_analysis/migration/`.

## 6. Build Generic Layer-Stack Interface Systems

Use `build_layer_stack(...)` for graphite, polymer, electrolyte, and vacuum
stacks.  Instead of hard-coding `graphite | polymer | electrolyte`, provide any
ordered list of layers: `electrolyte | graphite`,
`graphite | electrolyte | graphite`, `graphite | CMC-Na | electrolyte`, or
`vacuum | layer | layer | vacuum`.

The builder:

- plans one master XY footprint from graphite and molecular density targets,
- packs molecular layers under that shared XY footprint,
- keeps CMC-Na polymer and its Na+ counterions in one layer group,
- stacks layers by z quantiles plus adaptive gaps,
- supports explicit fixed graphite surface charge,
- writes `layer_stack_manifest.json`, `system.gro`, `system.top`, and layer-aware `system.ndx`.
- treats basal graphite as XY-periodic by default and edge graphite as a finite capped slab.

Smoke-scale example:

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
ion_ff = yp.get_ff("merz")
EC = ff.mol("O=C1OCCO1", charge="RESP", prefer_db=True, require_ready=True)
ff.ff_assign(EC)
Li = ion_ff.mol("[Li+]")
ion_ff.ff_assign(Li)

stack = yp.LayerStackSpec(
    layers=(
        yp.GraphiteLayerSpec(name="GRAPHITE", nx=6, ny=5, n_layers=3),
        yp.MolecularLayerSpec(
            name="ELECTROLYTE",
            species=(EC, Li),
            counts=(100, 10),
            thickness_nm=4.0,
            density_target_g_cm3=1.2,
            layer_kind="electrolyte",
        ),
        yp.VacuumLayerSpec(thickness_nm=2.0),
    )
)

result = yp.build_layer_stack(stack=stack, work_dir="./work_layer_stack")
profile = yp.analyze_layer_stack_interface(
    work_dir="./work_layer_stack",
    analysis_profile="interface_fast",
)
```

Use `ElectrodeChargeSpec` for fixed-charge graphite electrodes.  This is a
static charge assignment to surface atoms, not a constant-potential model.  In
two-electrode stacks, use `top_surface_charge_uC_cm2` on the lower electrode and
`bottom_surface_charge_uC_cm2` on the upper electrode to charge only the interior
faces.

Use `run_layer_stack_nvt(...)` to append a short NVT observation run from the
exported stack artifact without rebuilding the layers:

```python
nvt = yp.run_layer_stack_nvt(result, time_ns=2.0, temp=318.15, omp=14, gpu_id=0)
```

For all MD helpers, `gpu=0` is an explicit CPU-mode switch.  If `gpu_id` is
also present, it is ignored and no GROMACS `-gpu_id` flag is emitted, so you can
leave `gpu_id = ...` in scripts while toggling CPU/GPU behavior with `gpu`.

For `pbc_mode="auto"`/`xyz`, the builder also checks the periodic top-bottom
closing interface and reserves the same `default_gap_nm` spacer unless explicit
vacuum or padding already supplies it.  The NVT follow-up begins with a
no-constraints steep minimization, so freshly stacked CMC/electrolyte/graphite
cells can relax local contacts before GPU MD starts.

Interface analysis is not bulk analysis.  For a sampled layer stack, prefer the
eg02-style facade:

```python
analy = nvt.analyze()
interface = analy.interface(
    manifest_path=result.manifest_path,
    analysis_profile="interface_fast",
    bin_nm=0.05,
    region_width_nm=0.75,
    surface_distance_nm=0.50,
    surface_grid_nm=0.50,
    penetration_threshold_nm=0.20,
    adsorption_min_residence_ps=10.0,
    potential_reference="zero_mean",
    split_electrodes=True,
    report_potential_drop=True,
)
health = interface.geometry_health()
z_profile = interface.z_profiles()
edl = interface.edl_profiles(potential_reference="zero_mean", report_potential_drop=True)
penetration = interface.penetration(species=("EC", "EMC", "DEC", "PF6"))
adsorption = interface.graphite_adsorption(species=("EC", "EMC", "DEC"))
coordination = interface.coordination_by_region()
transport = interface.region_transport()
summary = interface.summary()
```

The helper writes `06_analysis/layer_stack_interface/` with z density and charge
profiles, electrostatic potential diagnostics for fixed-charge stacks, EDL
species layering, penetration/residence diagnostics, graphite-near adsorption
statistics, Li/Na coordination partitioning, and anisotropic MSD summaries.
Treat `Dxy` as the main interface transport metric; `Dz` is confined-direction
mobility, not a bulk diffusion coefficient. Bulk-style 3D diffusion,
conductivity, and dielectric analysis should be used only as explicit controls.

Parameter meanings:

- `manifest_path` keeps the intended layer names and bottom-to-top order. Use it
  for closed `xyz` stacks because raw z coordinates can wrap across the periodic
  boundary.
- `bin_nm` is the z-bin width for density, charge, integrated-charge, EDL
  species, and electrostatic-potential profiles.
- `region_width_nm` controls the graphite-near, mixed, and core-like z regions
  used for enrichment, penetration, coordination, and region transport.
- `surface_distance_nm` is the molecule-COM cutoff for graphite-near adsorption;
  `surface_grid_nm` is the xy grid used for graphite surface occupancy maps.
- `penetration_threshold_nm` is the minimum COM depth inside a CMC/polymer-rich
  or mixed region before a frame is counted as penetration.
- `adsorption_min_residence_ps` controls the `passes_min_residence` flag only;
  raw residence counts and fractions are still written.
- `potential_reference` chooses the reference shift for the 1D fixed-charge
  potential diagnostic: `zero_mean` subtracts the mean, and `zero_start` pins the
  first z bin. This is not a constant-potential electrode calculation.
- `split_electrodes` and `report_potential_drop` request two-electrode EDL
  reporting metadata and potential-drop diagnostics; they do not change the MD
  charge model.
- `penetration_species` and `adsorption_species` filter moltype names by exact
  or substring match.
- `compute_transport=True` adds anisotropic MSD summaries when a trajectory is
  available. Use `compute_transport=False` for static build-only sanity checks.

Method meanings:

- `geometry_health()` checks layer order, phase z-quantiles, interphase
  distances, severe overlaps, and direct graphite-electrolyte contact.
- `z_profiles()` returns density/charge profile outputs.
- `edl_profiles()` returns fixed-charge EDL species, integrated charge, electric
  field, and potential diagnostics.
- `penetration(...)` reports molecule COM residence in CMC/polymer-rich or mixed
  regions.
- `graphite_adsorption(...)` reports graphite-near residence, surface occupancy,
  and simple orientation proxies.
- `coordination_by_region()` reports cation donor-state partitioning by z region.
- `region_transport()` reports `Dxy`/`Dz` anisotropic MSD diagnostics.

Sandwich-specific interface builders are not part of the public workflow
surface.  Use layer-stack specs for both simple two-layer contacts and
multi-layer graphite/polymer/electrolyte systems.

### Example script standard

New examples should follow the Example 02 pattern: use explicit imports, put
user-editable settings at the top instead of driving ordinary runs through many
environment variables, keep `if __name__ == "__main__"` linear, write all
artifacts under a restartable `work_dir`, prefer MolDB-ready molecules with
`prefer_db=True` and `require_ready=True`, and avoid hiding the scientific
workflow behind local framework helpers. Developer automation belongs under
`tools/`, not inside public example folders.

## 7. Literature Benchmarks

Example 02 includes a dedicated PEO-LiTFSI charge-scaling reproduction entry for
Gudla/Zhang/Brandell, J. Phys. Chem. B 2020 (`10.1021/acs.jpcb.0c05108`):

```bash
cd examples/02_polymer_electrolyte
DRY_RUN=1 python benchmark_peo_litfsi_jpcb2020.py
```

The workflow uses GAFF2 + MERZ + RESP, keeps the paper's EO:Li ratio
(`12.5:1`), and stores the selected charge-scaling case, Tg, target
normalized inverse temperature, and production-time provenance in the benchmark
metadata.  Use `JPCB_CASES=P1.00S1.00,P1.00S0.75,P1.20S0.75` to choose cases,
`PAPER_SIZE=1` for the original 200-chain system, and `PROD_NS=...` for
long-production validation.

## 8. Restart, Resume, and Work Directories

Restart behavior is a first-class feature.

Keep these habits:

- give each workflow a dedicated `work_dir`;
- do not manually delete intermediate files inside a partially completed run;
- prefer rerunning the same script with the same `work_dir` when you want restart behavior;
- use a new `work_dir` when you deliberately change the chemistry or the protocol.

Global defaults can be controlled with:

```python
from yadonpy.runtime import set_run_options

set_run_options(restart=True, strict_inputs=True)
```

Or temporarily:

```python
from yadonpy.runtime import run_options

with run_options(restart=False):
    ...
```

## 9. Local vs Remote Execution

Not every task belongs on the remote GPU node.

Prefer local execution for:

- quick force-field checks,
- API exploration,
- small regression tests,
- documentation and packaging work,
- short RESP tests on small molecules.

Prefer the remote machine for:

- long GROMACS workflows,
- larger layer-stack interface systems,
- repeated bulk equilibrations,
- heavier Psi4 jobs,
- anything that needs both sustained CPU and GPU time.

## 10. Thermomechanical Studies

For `Tg` and elongation, the recommended pattern is:

1. equilibrate the system first,
2. resolve the prepared `gro/top`,
3. run a high-level study wrapper,
4. inspect the summary before opening the raw CSV or XVG files.

Example:

```python
import yadonpy as yp

prepared = yp.resolve_prepared_system(
    work_dir="./examples/02_polymer_electrolyte/work_dir",
)

tg_result = yp.run_tg_scan_gmx(
    prepared=prepared,
    out_dir="./work_tg",
    profile="default",
)
yp.print_mechanics_result_summary(tg_result)
```

`run_tg_scan_gmx(...)` writes:

- `summary.json`
- `density_vs_T.csv`
- a global `tg_density_vs_T.svg`
- one stage folder per temperature

`run_elongation_gmx(...)` writes:

- `summary.json`
- `stress_strain.csv`
- `stress_strain.svg`
- material summary fields such as Young's modulus and peak stress

## 11. Common Problems

### `doctor()` says `psiresp` is missing or broken

Install the supported package set:

```bash
conda install -c conda-forge rdkit openbabel parmed mdtraj matplotlib pandas scipy packaging psi4=1.10 dftd3-python psiresp-base
```

If the import error mentions `PydanticUserError`, `jobname = 'optimization'`,
or missing type annotations, your environment likely has `pydantic>=2`, which
is incompatible with the supported `psiresp-base` package set. The verified
working fix is:

```bash
python -m pip install "pydantic==1.10.26"
```

### RESP works for ordinary terminal groups but hydrogen termination is awkward

That is expected. The internal hydrogen terminator placeholder is special because it is
used as a polymer-building marker. Real terminal groups such as methyl or hydroxyl do not
share that limitation and can go through ordinary QM and RESP paths.

### Polymer random walk keeps retrying

This usually means one of three things:

- the chain is too dense for the requested initial box;
- the chain flexibility assumptions are too optimistic;
- the retry budget is too small for the requested degree of polymerization.

YadonPy scales retry budgets for longer chains, but density and chain
specification still matter.

### A phase density looks unreasonable in an interface workflow

Do not trust the final stacked system until the phase-level bulk densities are already close
to expectation. Bulk-first equilibration is the safest starting point.

## 12. Where to Go Next

- Read [API Reference](API_REFERENCE.md) for the full public API surface.
- Read [Architecture](ARCHITECTURE.md) for the design rules behind MolDB, restart behavior,
  and interface assembly.
- Read [Technical Notes](TECHNICAL_NOTES.md) for packaged force-field provenance notes.
