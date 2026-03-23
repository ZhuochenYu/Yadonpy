# YadonPy User Guide (v0.8.53)

This guide is for users who want to run YadonPy productively without reading the whole codebase. It is written as a practical companion to the examples, with enough detail to help you choose the right workflow style and avoid the most common mistakes.

Python requirement: Python 3.11+

Related documents:

- API reference: `docs/Yadonpy_API_v0.8.53.md`
- manual: `docs/Yadonpy_manul.md`

## 1. Who this guide is for

Read this guide if you want to answer questions like these quickly:

- how do I get a working environment with the right external tools;
- should I use one-shot helpers, explicit force-field objects, or MolDB reuse;
- which example should I start from for my system;
- what should go into MolDB and what should stay in a study work directory;
- how should I think about dense packing, restart, and interface preparation.

If you need formal function signatures, use the API reference. If you need architecture or design rationale, read the manual. If you want to get productive with the fewest wrong turns, start here.

## 2. Installation

### 2.1 Recommended environment

- Python 3.11+
- RDKit
- Open Babel for robust 3D generation in difficult ionic or inorganic cases
- GROMACS for MD workflows
- Psi4 plus Python `resp` when RESP charges are required

### 2.2 Example environment

```bash
conda create -n yadonpy python=3.11
conda activate yadonpy
conda install -c conda-forge rdkit openbabel parmed mdtraj matplotlib pandas scipy
conda install -c psi4 psi4 resp dftd3-python
pip install -e .
pip install -e .[accel]
```

### 2.3 First environment check

Run this before debugging anything by hand:

```bash
python -c "from yadonpy.diagnostics import doctor; doctor(print_report=True)"
```

That single command is usually the fastest way to detect missing backends, missing Python packages, or an environment that cannot support the workflow you are trying to run.

### 2.4 Minimal installation strategy

You do not need every optional dependency to use every part of YadonPy.

Typical layering is:

- basic molecule creation and many tests: Python + RDKit;
- robust 3D generation for awkward molecules: add Open Babel;
- RESP workflows: add Psi4 and `resp`;
- production MD workflows: add GROMACS;
- large interface optimization acceleration: optionally install `.[accel]`.

## 3. The shortest useful script

The shortest practical script is usually:

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
mol = yp.mol_from_smiles("O=C1OCCO1", name="EC")
yp.assign_charges(mol, charge="RESP", work_dir="./work_ec")
ok = ff.ff_assign(mol)
```

Use this when you want a prepared small molecule and do not need direct MolDB reuse yet.

What this script is doing:

1. choose a force-field family;
2. create a named molecule with 3D-capable defaults;
3. assign charges through the QM layer;
4. assign force-field terms.

If this basic path does not work in your environment, there is no value in jumping ahead to larger polymer or interface examples.

## 4. Choose the right workflow style

YadonPy supports several ways of writing a script. Picking the right one early saves time.

### 4.1 One-shot convenience path

Use `parameterize_smiles(...)` when you want a compact script:

```python
import yadonpy as yp

mol, ok = yp.parameterize_smiles(
    "CCO",
    ff_name="gaff2_mod",
    charge_method="RESP",
    work_dir="./work_ethanol",
)
```

This is a good fit when:

- you are prototyping a small molecule flow;
- you want minimal code;
- you do not need unusual bonded settings;
- you are still exploring whether the chemistry setup is viable.

### 4.2 Explicit force-field path

Use this when the workflow must stay close to the example style:

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
mol = ff.mol("CCO")
mol = ff.ff_assign(mol)
```

This is also the right style when you need bonded options such as `bonded="DRIH"`.

Use this path when:

- you want direct control over the force-field object;
- you want scripts that resemble the shipped examples;
- you may later swap in MolDB-backed reuse with the same flow;
- you need to keep the assignment step explicit.

### 4.3 MolDB reuse path

Use MolDB when the expensive part is molecule preparation rather than system export.

```python
import yadonpy as yp

pf6 = yp.load_from_moldb("F[P-](F)(F)(F)(F)F", charge="RESP")
```

If you need the explicit example style that goes through the force-field handle, Example 01 demonstrates the intended pattern.

### 4.4 Practical rule of thumb

Use this decision rule:

- one-shot helper for quick trials;
- explicit force-field path for serious reusable scripts;
- MolDB lookup when the same expensive species will appear in multiple studies.

## 5. Understand work directories early

One of the easiest ways to misuse YadonPy is to treat the work directory as an opaque temporary folder. It is not temporary. It is the record of one study run.

Typical content includes:

- prepared molecule files;
- packed-cell build folders;
- exported system files;
- equilibration stage outputs;
- analysis JSON files;
- interface assembly artifacts.

Recommended pattern:

```python
from pathlib import Path
from yadonpy.core import workdir
from yadonpy.runtime import set_run_options

restart = True
set_run_options(restart=restart)
BASE_DIR = Path(__file__).resolve().parent
work_dir = workdir(BASE_DIR / "work_dir", clean=not restart)
```

Then create logical child folders under that root rather than scattering files manually.

## 6. MolDB and what belongs there

MolDB is not a general project cache. It is a molecule-preparation cache.

Good candidates for MolDB:

- expensive RESP-prepared salts and solvents;
- reusable monomers or charged fragments;
- species that required special bonded patch handling.

Bad candidates for MolDB:

- random packed system snapshots;
- whole study trajectory outputs;
- one-off exported topology trees.

### 6.1 Best practices

- store molecules that are expensive to optimize or charge;
- do not treat MolDB as a replacement for study work directories;
- expect exported `.itp`, `.top`, and `.gro` files to be regenerated from the stored molecular state;
- remember that the current release can restore bonded patch metadata for DRIH and mSeminario style variants.

### 6.2 Why this separation matters

This design keeps the expensive chemistry reusable while allowing topology exports to stay consistent with the current code. If YadonPy changes export logic, you usually want regenerated files, not stale ones from an older run.

## 7. Recommended example order

For a new user, the examples make the most sense in this order:

1. `examples/07_moldb_precompute_and_reuse`
2. `examples/08_text_to_csv_and_build_moldb`
3. `examples/01_Li_salt`
4. `examples/02_polymer_electrolyte`
5. `examples/05_cmcna_electrolyte`
6. `examples/03_tg_gmx`
7. `examples/04_elongation_gmx`
8. `examples/06_polymer_electrolyte_nvt`
9. `examples/09_oplsaa_assign`
10. `examples/10_interface_route_a`
11. `examples/11_interface_route_b`
12. `examples/12_cmcna_interface`

This order is intentional. It starts with reusable molecule-preparation logic, then moves into bulk systems, and only then reaches interfaces.

## 8. What each key example teaches

### 8.1 Example 01: PF6 build and reuse

Purpose:

- compute PF6 from scratch;
- export structure and GROMACS files;
- store the finished result in MolDB;
- immediately show the later MolDB-backed reuse style.

This is the example to read if you care about DRIH bonded patches on high-symmetry inorganic ions.

### 8.2 Example 02: baseline polymer-electrolyte build

Purpose:

- understand the normal molecule to bulk-system path;
- see how species are prepared and then packed together;
- understand how equilibration is attached to the build process.

### 8.3 Example 05: charged polymer electrolyte

Purpose:

- work with CMC-like charged polymer logic;
- see counter-ion handling;
- understand why low initial packing density can be the correct choice.

### 8.4 Example 07: MolDB precompute and reuse

Purpose:

- build MolDB records from structured input;
- reuse those records in a later workflow;
- keep expensive QM work out of repeated study scripts.

### 8.5 Example 10 and Example 11: interface routes

Purpose:

- Example 10 demonstrates Route A, the fully periodic dual-interface path;
- Example 11 demonstrates Route B, the vacuum-padded wall-ready path.

Read these before Example 12 if you want to understand interface mechanics without charged-polymer complexity.

### 8.6 Example 12: CMC interface workflow

Purpose:

- build and equilibrate the CMC phase;
- plan electrolyte counts from the equilibrated CMC footprint;
- use a fixed-XY pack-and-relax strategy;
- assemble the final route-A interface.

Read this one when dense interfacial electrolyte packing is the main challenge.

## 9. Bulk-system workflow checklist

For many studies, the productive path is:

1. define species and names;
2. assign charges;
3. assign force fields;
4. store expensive prepared species in MolDB if they will be reused;
5. build a packed cell with realistic but not over-aggressive starting conditions;
6. export and equilibrate;
7. inspect the analysis outputs before using the final configuration as a new starting point.

The most common mistake is trying to make the first packed box too dense. If the system is large, charged, or structurally awkward, the initial packed state should prioritize physical plausibility over closeness to final density.

## 10. Interface workflow checklist

The interface examples introduce a stricter geometry requirement: the two bulk sides must agree on a lateral footprint.

Current best practice is:

1. relax the polymer side first;
2. use the equilibrated polymer XY dimensions as the authoritative interface footprint;
3. plan the top-side electrolyte only for the slab volume that will actually be used;
4. pack the electrolyte into an XY-locked but Z-looser initial box;
5. relax with fixed-XY semiisotropic pressure control;
6. only then assemble slabs into the final interface.

This strategy is especially important in Example 12 style systems, where forcing the final target box too early can exhaust packing retries before MD has a chance to stabilize the density.

## 11. Restart and rerun behavior

### 11.1 Global defaults

Global defaults can be set once:

```python
import yadonpy as yp

yp.set_run_options(restart=False)
```

Temporary override:

```python
import yadonpy as yp

with yp.run_options(restart=True):
    pass
```

Environment variables:

- `YADONPY_RESTART`
- `YADONPY_STRICT_INPUTS`

### 11.2 Practical meaning of restart

`restart=True` does not mean “trust all old files unconditionally.” It means “reuse valid expensive outputs where the current workflow still considers them compatible.”

When workflows or internal schemas change, rebuilds are the correct behavior.

## 12. What outputs to inspect

Bulk and production workflows often write merged analysis results under `work_dir/06_analysis/`.

Typical files include:

- `thermo_summary.json`
- `basic_properties.json`
- `cell_summary.json`
- `polymer_radius_of_gyration.json`
- `polymer_end_to_end_distance.json`
- `polymer_persistence_length.json`
- `polymer_metrics.json`
- `summary.json`

Do not treat these as optional extras. They are often the fastest way to see whether the workflow actually produced a physically reasonable result.

## 13. Common mistakes to avoid

- trying to run the most complex example first;
- skipping the diagnostics report when a backend is missing;
- storing study-level exports in MolDB instead of the prepared molecules;
- forcing dense final-box packing when a staged pack-and-relax workflow is available;
- ignoring force-field assignment reports and only discovering chemistry issues after MD fails;
- treating `restart=True` as a substitute for understanding which steps changed.

## 14. Troubleshooting

### 14.1 Environment problems

If the environment is suspect, run the diagnostics report first. This should be the default first move, not the last resort.

### 14.2 MolDB mismatch

If a script reuses MolDB but not the result you expected, check whether the requested charge variant was marked ready and whether the relevant bonded patch was stored.

### 14.3 Dense packing failure

If dense final-box packing fails, use the newer planning helpers or the example paths that derive a pack-friendly intermediate box instead of forcing the final box directly.

In practice, that usually means one or more of these changes:

- lower the initial packing density;
- allow a taller initial Z;
- preserve the final XY footprint but relax Z separately;
- reduce the default system size for early debugging runs.

### 14.4 Viewer versus engine mismatch

If a topology viewer warns about omitted pair terms while GROMACS itself is fine, remember that some downstream viewers are stricter than the actual simulation engine.

### 14.5 Interface instability

If an interface build gets through geometry preparation but becomes unstable during early MD, split the problem:

1. confirm the polymer bulk is well equilibrated;
2. confirm the electrolyte side was not over-constrained during packing;
3. confirm the assembled interface geometry is sensible before contact dynamics.

### 14.6 When to simplify the system

If a new workflow is failing repeatedly, do not start by increasing complexity. Shrink it.

Good first simplifications are:

- fewer polymer chains;
- lower degree of polymerization;
- fewer salt pairs;
- thinner slab targets;
- looser initial packing density.

## 15. Minimal release sanity checks

For a source tree, the useful lightweight checks are:

```bash
python -m pyflakes src examples tests
python -m compileall src examples tests
PYTHONPATH=src pytest -q
```

## 16. Final guidance

YadonPy is easiest to use when you accept its working model:

- keep study logic explicit in scripts;
- cache expensive molecular preparation cleanly in MolDB;
- rebuild exported topologies when the workflow changes;
- stabilize difficult systems with better staging rather than with blind retry inflation.

If you follow those rules, the examples, the public API, and the work-directory structure reinforce each other instead of fighting each other.
