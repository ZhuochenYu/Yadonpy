# YadonPy User Guide

This guide is written for day-to-day use. It focuses on the practical decisions
you need to make when preparing molecules, building systems, and choosing between
local runs and heavier remote calculations.

## 1. Install a Working Environment

Recommended setup:

```bash
conda create -n yadonpy python=3.11
conda activate yadonpy

conda install rdkit openbabel parmed mdtraj matplotlib pandas scipy packaging psi4 dftd3-python psiresp-base
pip install -e .
```

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

## 6. Build Graphite-Polymer-Electrolyte Sandwich Systems

YadonPy now exposes a high-level sandwich builder that packages the recommended logic:

- equilibrate each phase independently,
- derive slabs from equilibrated bulk instead of trusting only initial packed cells,
- preserve the graphite footprint as the lateral reference,
- assemble the stack with explicit gaps,
- relax the combined system in stages.

PEO-based smoke-scale example:

```python
import yadonpy as yp
from yadonpy.interface import (
    SandwichRelaxationSpec,
    default_carbonate_lipf6_electrolyte_spec,
    default_peo_polymer_spec,
)

result = yp.build_graphite_peo_electrolyte_sandwich(
    work_dir="./work_peo_sandwich",
    polymer=default_peo_polymer_spec(dp=20),
    electrolyte=default_carbonate_lipf6_electrolyte_spec(),
    relax=SandwichRelaxationSpec(omp=8, gpu=1, psi4_omp=8),
)
```

CMC-Na uses the same workflow family, but its polymer specification is more constrained
because the charged groups and counterions matter chemically.

## 7. Restart, Resume, and Work Directories

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

## 8. Local vs Remote Execution

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

## 9. Common Problems

### `doctor()` says `psiresp` is missing or broken

Install the supported package:

```bash
conda install psiresp-base
```

If the import error mentions `PydanticUserError`, `jobname = 'optimization'`,
or missing type annotations, your environment likely has `pydantic>=2`, which
breaks current `psiresp-base`. Fix it with:

```bash
conda install -c conda-forge "pydantic<2" "psiresp-base"
```

If conda cannot solve that environment cleanly, this also works in practice:

```bash
pip install "pydantic<2"
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

## 10. Where to Go Next

- Read [API Reference](API_REFERENCE.md) for the full public API surface.
- Read [Architecture](ARCHITECTURE.md) for the design rules behind MolDB, restart behavior,
  and interface assembly.
- Read [Technical Notes](TECHNICAL_NOTES.md) for packaged force-field provenance notes.
