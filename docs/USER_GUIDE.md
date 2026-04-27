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

- bulk equilibration before interface assembly,
- slab extraction from equilibrated phases,
- interface route planning,
- graphite-polymer-electrolyte sandwich construction.

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
the code will now avoid sending that placeholder through fragile QM paths.

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

Dielectric constants are available through `dielectric()`, which calls
`gmx dipoles` and reads the final static dielectric estimate from
`06_analysis/dielectric.json`. On remote machines with more than one GROMACS,
set `YADONPY_GMX_CMD` to the same major version used for production; otherwise
post-processing can fail on newer `.tpr` files.

This keeps the defaults physically aligned:

- bulk systems use drift-corrected `3D` diffusion by default;
- sandwich and slab systems use drift-corrected `xy` diffusion by default;
- wrapped trajectories are normalized before transport analysis;
- `sigma_ne_upper_bound_S_m` is treated explicitly as an upper bound;
- `sigma_eh_total_S_m` is preferred when a stable EH fit exists;
- `haven_ratio` is reported whenever both conductivities are available.

Practical interpretation:

- A large `sigma_ne_upper_bound_S_m` with a much smaller `sigma_eh_total_S_m`
  usually means ion correlations are strong.
- `RDF` stays independent because it is the only routine analysis that needs a
  center species and coordination-shell semantics.
- In interface systems, do not interpret the default diffusion coefficient as a
  free `3D` value unless you explicitly request `geometry="3d"`.
- Charged-polymer results are kept as
  `polymer_charged_group_self_ne_contribution_S_m`; they are not equivalent to
  a rigorously separated polymer ionic conductivity.
- Migration analysis is now a first-class `AnalyzeResult` method instead of a
  standalone monolithic script. The default path reports:
  - coordination roles,
  - residence times for polymer / solvent / anion donors,
  - role-level and site-level Markov models,
  - transition-matrix-based event flux predictions,
  - migration summary plots under `06_analysis/migration/`.

## 6. Build Graphite-Polymer-Electrolyte Sandwich Systems

YadonPy exposes a one-shot interface builder for the common
`graphite -> CMC-Na -> electrolyte` stack, plus the older staged API for
debugging individual steps. For routine CMC-Na/graphite/electrolyte work, prefer
the one-shot path:

- equilibrate each phase independently,
- treat those bulk runs as calibration for density, chain count, solvent counts, and packing backoff,
- preserve the graphite footprint as the one lateral reference,
- rebuild each soft phase directly on that shared XY footprint with repulsive-only Z walls and explicit vacuum,
- assemble the stack by direct Z translation instead of relying on cut-slab periodic healing,
- relax the combined system with a natural-contact protocol and acceptance gate.

CMC-Na smoke-scale example:

```python
import yadonpy as yp
graphite = yp.GraphiteSubstrateSpec(nx=4, ny=4, n_layers=2)
polymer = yp.default_cmcna_polymer_spec(dp=20)
electrolyte = yp.default_carbonate_lipf6_electrolyte_spec()
relax = yp.SandwichRelaxationSpec(omp=8, gpu=1, psi4_omp=8)
policy = yp.InterfaceBuildPolicy(
    phase_preparation="final_xy_walled",
    stack_relaxation="natural_contact",
    retry_profile="conservative",
)

result = yp.build_cmcna_graphite_electrolyte_stack(
    work_dir="./work_cmcna_sandwich",
    ff=yp.get_ff("gaff2_mod"),
    ion_ff=yp.get_ff("merz"),
    graphite=graphite,
    polymer=polymer,
    electrolyte=electrolyte,
    relax=relax,
    policy=policy,
)
```

The builder writes `00_interface_design/interface_design.json` before expensive
phase rebuilding, then records stack attempts, charge balance, minimum-distance
checks, phase order, density windows, and acceptance status in
`06_full_stack_release/interface_manifest.json`. The staged functions
`prepare_graphite_substrate`, `calibrate_*_bulk_phase`, `build_*_interphase`,
and `release_graphite_*_electrolyte_stack` remain available when you need to
inspect or resume a specific step manually.

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
- larger sandwich systems,
- repeated bulk equilibrations,
- heavier Psi4 jobs,
- anything that needs both sustained CPU and GPU time.

## 10. Thermomechanical Studies

For `Tg` and elongation, the recommended pattern is now:

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
breaks current `psiresp-base`. The verified working fix is:

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

YadonPy now scales retry budgets more sensibly for longer chains, but density and chain
specification still matter.

### A phase density looks unreasonable in an interface workflow

Do not trust the final stacked system until the phase-level bulk densities are already close
to expectation. Bulk-first equilibration is the safest starting point.

## 12. Where to Go Next

- Read [API Reference](API_REFERENCE.md) for the full public API surface.
- Read [Architecture](ARCHITECTURE.md) for the design rules behind MolDB, restart behavior,
  and interface assembly.
- Read [Technical Notes](TECHNICAL_NOTES.md) for packaged force-field provenance notes.
