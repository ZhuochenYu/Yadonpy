# YadonPy

Current release: **v0.8.60**

YadonPy is a Python package for building polymer, solvent, salt, bulk, and interface workflows directly from SMILES or PSMILES. It is designed for script-driven molecular simulation studies where the user wants to keep the real workflow visible in code instead of hiding it behind a monolithic project file.

## What YadonPy is for

YadonPy focuses on the full preparation chain:

- molecule construction from SMILES or PSMILES;
- conformer search and charge assignment;
- force-field assignment with GAFF, GAFF2, GAFF2_mod, OPLS-AA, MERZ, and DREIDING families;
- MolDB-backed reuse of expensive molecular preparation;
- bulk-cell and interface-system construction;
- GROMACS export, staged equilibration, and post-processing.

The package is built around two stable ideas:

- **script first**: the study logic should remain understandable from the user script;
- **MolDB first**: reusable expensive assets are molecular geometry, charge variants, and bonded-patch metadata, not old `.top/.gro/.itp` exports.

## What changed in v0.8.60

This release finishes the local GROMACS wiring for Example 12 and fixes the route-B interface workflow against real runtime failures observed on a `12`-core Windows machine with CUDA and a local GPU.

- `examples/12_cmcna_interface/run_eg12_local_cuda.bat` and `examples/12_cmcna_interface/run_eg12_remote_cuda.sh` are now the supported one-click entry points for EG12. The local wrapper defaults to `mpi=1`, `omp=12`, `gpu=1`, `gpu_id=0`, which matches the current workstation assumption.
- The EG12 smoke profile now shortens the bulk EQ21 tail and the fixed-XY electrolyte relax stage so the full local debug path can be rerun in practical time while still exercising the complete interface workflow.
- Route-B interface MDP generation now renders the actual intended settings: `pbc = xy`, `periodic-molecules = yes`, `ewald-geometry = 3dc`, and a topology-resolved wall atomtype instead of the old hard-coded water-model `OW`.
- Route-B assembly now pads the interface away from the `z` walls and adds `wall-r-linpot`, which fixes the previous failure where atoms placed exactly on the wall caused `01_pre_contact_em` to abort before real relaxation began.
- Slab and interface topologies now preserve the real fragment sequence in `[ molecules ]` instead of aggregating counts purely by species. This fixes a deeper ordering bug where charge-rebalancing could append `Li` or `PF6` fragments to the end of the slab, leaving `system.gro` and `system.top` out of sync and later causing impossible excluded-atom distances during `grompp`.
- This release was validated by actually running `eg12 --profile smoke` on native Windows GROMACS through all seven interface stages up to `07_production`.

## Installation

### Baseline requirements

- Python 3.11+
- RDKit
- ParmEd
- NumPy, SciPy, pandas, matplotlib

Optional but commonly needed:

- Open Babel for more robust 3D generation on awkward ions and inorganic species
- Psi4 plus Python `resp` for RESP or ESP charge workflows
- GROMACS for MD workflows
- `numba` for optional acceleration in some interface-build paths

### Example conda environment

```bash
conda create -n yadonpy python=3.11
conda activate yadonpy

conda install -c conda-forge rdkit openbabel parmed mdtraj matplotlib pandas scipy packaging
conda install -c psi4 psi4 resp dftd3-python

pip install -e .
pip install -e .[accel]
```

### First environment check

```bash
python -c "from yadonpy.diagnostics import doctor; doctor(print_report=True)"
```

Run that before debugging missing backends by hand.

## Workflow model

### 1. Prepare reusable species

Typical patterns are:

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
mol = ff.mol("O=C1OCCO1")
ok = ff.ff_assign(mol)
```

or:

```python
import yadonpy as yp

mol, ok = yp.parameterize_smiles(
    "CCO",
    ff_name="gaff2_mod",
    charge_method="RESP",
    work_dir="./work_ethanol",
)
```

Use the explicit `ff.mol(...)` plus `ff.ff_assign(...)` form for serious scripts, especially when the species may later come from MolDB or require bonded overrides such as `bonded="DRIH"`.

### 2. Reuse expensive chemistry with MolDB

MolDB stores:

- best available geometry;
- charge variants;
- readiness metadata;
- bonded-patch sidecar data when a charge variant depends on it.

MolDB does **not** store project-level topology trees as the long-term source of truth. Those are regenerated from the molecular state when needed.

### 3. Build a bulk cell

The bulk workflow is normally:

1. prepare species;
2. pack an amorphous cell;
3. export the system;
4. run EQ21 and optional follow-up relaxation;
5. read the equilibrated box and analysis outputs.

### 4. Build an interface

The current interface logic follows a polymer-first workflow:

1. equilibrate the polymer bulk first;
2. treat the equilibrated polymer `XY` lengths as the authoritative footprint;
3. build or resize the electrolyte against that footprint, but allow extra `Z` slack initially;
4. equilibrate the standalone electrolyte bulk first;
5. assemble slabs only after both sides are individually reasonable;
6. run staged interface diffusion dynamics.

For route selection:

- `route_a`: fully periodic interface workflow;
- `route_b`: vacuum-buffered, wall-ready interface workflow under `pbc = xy`.

Examples 10 and 11 show the neutral polymer route-A and route-B variants. Example 12 uses the more conservative route-B diffusion path for the large CMC system and now starts from a lower-density free-bulk CMC pack before reading the polymer `XY` footprint.

## Examples

Recommended reading order:

1. `examples/07_moldb_precompute_and_reuse`
2. `examples/08_text_to_csv_and_build_moldb`
3. `examples/01_Li_salt`
4. `examples/02_polymer_electrolyte`
5. `examples/05_cmcna_electrolyte`
6. `examples/10_interface_route_a`
7. `examples/11_interface_route_b`
8. `examples/12_cmcna_interface`

What the interface examples now demonstrate:

- **Example 10**: periodic polymer/electrolyte diffusion interface with a linear script and library-selected staged protocol.
- **Example 11**: vacuum-buffered route-B counterpart with the same linear script style.
- **Example 12**: larger CMC interface study using `6` chains, `DP = 150`, a low-density free-bulk CMC pack, polymer-first XY anchoring, probe electrolyte equilibration, resized final electrolyte rebuild, and staged diffusion release.

## Documentation map

- API reference: `docs/Yadonpy_API_v0.8.60.md`
- Manual: `docs/Yadonpy_manul.md`
- User guide: `docs/Yaonpyd_user_guide.md`

Use them in this order:

- README: package scope and quickest start;
- User guide: how to run a study productively;
- Manual: architecture, persistence model, restart model, and workflow constraints;
- API reference: callable entry points and script-facing objects.

## Working directories and restart

The recommended script pattern is:

```python
from pathlib import Path
from yadonpy.core import workdir
from yadonpy.runtime import set_run_options

restart = True
set_run_options(restart=restart)
BASE_DIR = Path(__file__).resolve().parent
work_dir = workdir(BASE_DIR / "work_dir", clean=not restart)
```

`restart=True` means "reuse valid finished work when the current workflow still agrees with it". It does not mean "trust all old artifacts blindly".

## Release hygiene

Release trees should not keep generated clutter such as:

- `__pycache__`
- `.pytest_cache`
- `.yadonpy_cache`
- `src/yadonpy.egg-info`

Current release packaging excludes those artifacts, and the maintenance rules require cleaning them during each version update.

## Development checks

For a source tree:

```bash
python -m compileall src examples tests
PYTHONPATH=src pytest -q
```

When GROMACS is not available, code-level and topology-level tests can still be run. Full MD execution is only required for real simulation validation.
