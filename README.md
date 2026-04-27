# YadonPy

YadonPy is a script-first molecular modeling and simulation toolkit for polymers,
electrolytes, graphite-supported interfaces, and GROMACS-based workflows.
It keeps the scientific procedure visible in ordinary Python scripts while still
providing reusable building blocks for charge assignment, force-field preparation,
bulk packing, interface assembly, equilibration, and analysis.

## What YadonPy Does

- Builds molecules directly from SMILES and polymers from PSMILES.
- Assigns force fields with GAFF, GAFF2, GAFF2_mod, OPLS-AA, DREIDING, and MERZ.
- Supports QM-derived charges with Psi4 plus `psiresp-base`, including RESP and ESP.
- Stores expensive prepared molecular assets in MolDB for later reuse.
- Exports GROMACS-ready systems and runs staged MD workflows.
- Builds bulk polymer-electrolyte systems and graphite-polymer-electrolyte sandwich structures.
- Preserves restart-aware work directories and auditable metadata such as charge-group manifests,
  export manifests, and interface build records.

## Installation

The recommended environment uses Python 3.11.

```bash
conda create -n yadonpy python=3.11
conda activate yadonpy

conda install -c conda-forge rdkit openbabel parmed mdtraj matplotlib pandas scipy packaging psi4=1.10 dftd3-python psiresp-base
python -m pip install "pydantic==1.10.26"
python -m pip install -e .
```

This exact sequence was re-tested on the remote Linux compute node and is the
current minimal working setup for Example 01 and RESP fitting. `openbabel`
already provides the Open Babel Python bindings used by YadonPy, so no separate
`pybel` package is required.

The source tree now ships a default `moldb/` folder beside `examples/`. On the
first YadonPy import or `ensure_initialized()` call, that catalog is seeded into
`~/.yadonpy/moldb` so a fresh editable install starts with the reference MolDB.

Check the environment after installation:

```bash
python -c "from yadonpy.diagnostics import doctor; doctor(print_report=True)"
```

For RESP and ESP workflows, `doctor()` should report both `psi4` and `psiresp` as available.
If `doctor()` reports `psiresp: BROKEN` with a `PydanticUserError`, pin the
verified working Pydantic version:

```bash
python -m pip install "pydantic==1.10.26"
```

## Quick Start

### Prepare a small molecule

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
ec = ff.mol("O=C1OCCO1")
yp.assign_charges(ec, charge="RESP", work_dir="./work_ec")
ok = ff.ff_assign(ec)
```

### Reuse a prepared species from MolDB

```python
import yadonpy as yp

pf6 = yp.load_from_moldb(
    "F[P-](F)(F)(F)(F)F",
    charge="RESP",
    require_ready=True,
)
```

### Build a graphite-polymer-electrolyte sandwich

```python
import yadonpy as yp

graphite = yp.GraphiteSubstrateSpec(nx=4, ny=4, n_layers=2)
polymer = yp.default_cmcna_polymer_spec(dp=20)
electrolyte = yp.default_carbonate_lipf6_electrolyte_spec()
relax = yp.SandwichRelaxationSpec(omp=8, gpu=1, psi4_omp=8)
policy = yp.InterfaceBuildPolicy(stack_relaxation="natural_contact")

result = yp.build_cmcna_graphite_electrolyte_stack(
    work_dir="./work_sandwich",
    ff=yp.get_ff("gaff2_mod"),
    ion_ff=yp.get_ff("merz"),
    graphite=graphite,
    polymer=polymer,
    electrolyte=electrolyte,
    relax=relax,
    policy=policy,
)
yp.print_interface_result_summary(result)
```

## Workflow Areas

### Molecular preparation

Use the top-level API when you want a short script:

- `mol_from_smiles(...)`
- `assign_charges(...)`
- `assign_forcefield(...)`
- `parameterize_smiles(...)`
- `load_from_moldb(...)`

Use `yadonpy.sim.qm` directly when you need explicit control of conformer search,
QM levels, basis selection, or bonded-parameter derivation.

### Bulk systems

YadonPy can build and equilibrate polymer-electrolyte systems starting from monomers,
solvents, ions, and salts. A typical workflow is:

1. prepare species and assign charges;
2. assign force fields;
3. pack an amorphous cell;
4. export the GROMACS system;
5. run staged equilibration;
6. analyze density, transport, and coordination behavior.

For post-processing, the recommended pattern is now:

```python
analy = production.analyze()
rdf = analy.rdf(center_mol=li_mol)
msd = analy.msd()
sigma = analy.sigma(msd=msd, temp_k=300.0)
dielectric = analy.dielectric(temp_k=300.0)
migration = analy.migration(center_mol=li_mol)
```

For high-throughput electrolyte screening, use the lighter transport profile:

```python
prop = analy.get_all_prop(temp=300.0, press=1.0, include_polymer_metrics=False, analysis_profile="transport_fast")
rdf = analy.rdf(center_mol=li_mol, analysis_profile="transport_fast", resume=True)
msd = analy.msd(analysis_profile="transport_fast", resume=True)
```

`transport_fast` keeps Li coordination and diffusion diagnostics, but skips full
site RDF and expensive polymer conformation metrics. Use `analysis_profile="full"`
when you want the complete report.

Production output cadence is adaptive by default in the production presets and
benchmark scripts. `PERFORMANCE_PROFILE=auto` keeps short/small runs near 2 ps
trajectory output, but switches long or large systems to coarser 10-50 ps output
and matching fast-analysis defaults. Set `PERFORMANCE_PROFILE=full` or explicit
`TRAJ_PS` / `ENERGY_PS` / `LOG_PS` values when you need dense trajectories for
short-time dynamics.

`dielectric()` wraps `gmx dipoles` and estimates the static dielectric constant
from total dipole fluctuations. Use the same GROMACS major version that produced
the `.tpr` file, e.g. set `YADONPY_GMX_CMD=/path/to/gmx` on clusters with
multiple GROMACS installations.

This keeps the defaults physically aligned:

- bulk systems use drift-corrected `3D` diffusion by default;
- sandwich and slab systems use drift-corrected `xy` diffusion by default;
- `sigma_ne_upper_bound_S_m` is reported explicitly as an upper bound;
- `sigma_eh_total_S_m` is the preferred total conductivity when a stable EH fit exists;
- `haven_ratio` is written whenever both values are available.
- `migration()` now defaults to a dual Markov model:
  - role states: `polymer / solvent / anion / none`
  - site states: specific donor anchors with sparse states lumped into `OTHER`

### Segment-first branched polymers

For large repeat units, long block architectures, or branchable polymers, build reusable
segments first and then polymerize those segments:

```python
from yadonpy.core import poly

segment1 = poly.seg_gen([monomer_A, monomer_A, monomer_B])
segment2 = poly.seg_gen([branchable_unit, branchable_unit, monomer_A])
side = poly.seg_gen([side_unit], cap_tail="[H][*]")

prebranched = poly.branch_segment_rw(
    segment2,
    [side],
    mode="pre",
    position=2,
    exact_map={"position": 2, "site_index": 0, "branch": 0},
)
block = poly.block_segment_rw([segment1, prebranched], [3, 2])
branched = poly.branch_segment_rw(block, [side], mode="post", position=2, ds=[1.0])
```

Use `*` or `[1*]` for the main-chain head/tail and `[2*]`, `[3*]`, ... for branch
attachment sites. Segment generation preserves existing atom charges; it does not
automatically rerun QM/RESP.

### Interface systems

For interface work, the recommended pattern is:

1. equilibrate each phase independently;
2. use those bulk runs only as calibration for density, composition, and packing difficulty;
3. negotiate one graphite master footprint in `XY`;
4. rebuild each soft phase directly on that final `XY` footprint with repulsive-only Z walls and explicit vacuum;
5. assemble the final interface or sandwich structure by Z translation;
6. run staged relaxation with restrained early dynamics and a later release stage.

This is the model used by the high-level graphite-polymer-electrolyte sandwich builder.

### Restart and metadata

Work directories are part of the product behavior. YadonPy writes explicit manifests,
export metadata, charge-group records, and restart markers so interrupted studies can be
resumed or audited without guessing which intermediate files are authoritative.

## Included Examples

- `examples/01_Li_salt`: prepare and store a reference salt species in MolDB.
- `examples/02_polymer_electrolyte`: end-to-end polymer-electrolyte workflow from PSMILES, including PEO-LiTFSI charge-scaling benchmarks.
- `examples/03_tg_gmx`: high-level `Tg` scan workflow for an equilibrated system.
- `examples/04_elongation_gmx`: high-level elongation and stress-strain workflow for an equilibrated system.
- `examples/05_cmcna_electrolyte`: CMC-Na polymer-electrolyte construction.
- `examples/06_polymer_electrolyte_nvt`: polymer-electrolyte workflow with NVT-focused staging.
- `examples/07_moldb_precompute_and_reuse`: one-shot MolDB catalog build and MolDB-backed reuse scripts.
- `examples/08_graphite_polymer_electrolyte_sandwich`: graphite-polymer-electrolyte sandwich workflows for PEO and CMC-Na.
- `examples/09_oplsaa_assignment`: compact OPLS-AA assignment workflows written in the same script-first style as the main examples.
- `examples/11_segment_branch_polymer`: segment-first long-block and branched-polymer construction.

## Documentation

- [User Guide](docs/USER_GUIDE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Technical Notes](docs/TECHNICAL_NOTES.md)

## Practical Notes

- Use local runs for fast API checks, packaging work, and short unit tests.
- Use the remote GPU node for long GROMACS jobs, larger sandwich systems, and heavy QM workflows.
- For charged polymers, prefer `polyelectrolyte_mode=True` so RESP constraints and later charge scaling remain auditable.
- MolDB is intended for reusable molecular assets such as geometry and charge variants, not as a topology cache.

## Thermomechanical Studies

For `Tg` and uniaxial elongation, YadonPy now provides high-level study wrappers
that work on an already prepared `gro/top` pair or on a standard equilibration
`work_dir`.

- `run_tg_scan_gmx(...)`: staged temperature scan plus `Tg` fit summary.
- `run_elongation_gmx(...)`: `deform`-based stress-strain workflow plus material summary.
- `print_mechanics_result_summary(...)`: compact terminal summary for either study.

The shipped examples for these studies now resolve the prepared system
automatically instead of manually wiring `workflow.steps` calls.

## Transport Analysis Notes

YadonPy now treats transport analysis as a physically opinionated set of
independent analyses rather than a loose collection of plots.

- `RDF` remains an independent analysis because it is the only one that needs a
  center species.
- `MSD` defaults are geometry-aware:
  - bulk: drift-corrected `3D`
  - sandwich/slab: drift-corrected `xy`
- `Nernst-Einstein` conductivity is reported as
  `sigma_ne_upper_bound_S_m`, not as the default true conductivity.
- `Einstein-Helfand` conductivity is reported as `sigma_eh_total_S_m` when a
  stable positive-slope regime is found.
- Charged-polymer self terms are retained only as
  `polymer_charged_group_self_ne_contribution_S_m` and component diagnostics;
  they are not labeled as total polymer ionic conductivity.
