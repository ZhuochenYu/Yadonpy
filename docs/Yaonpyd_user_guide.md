# YadonPy User Guide (v0.8.56)

This guide is for users who want to run YadonPy productively without reading the whole implementation first.

Python requirement: Python 3.11+

Related documents:

- API reference: `docs/Yadonpy_API_v0.8.56.md`
- manual: `docs/Yadonpy_manul.md`

## 1. Start with the right mindset

YadonPy works best when you treat it as a workflow library, not as a black box.

Practical consequences:

- your script is part of the study record;
- expensive molecular preparation belongs in MolDB;
- exported GROMACS trees are rebuildable products;
- difficult systems should be stabilized by better staging rather than by guesswork.

## 2. Environment setup

### 2.1 Minimum useful environment

- Python 3.11+
- RDKit
- NumPy, SciPy, pandas, matplotlib
- ParmEd

### 2.2 Common optional tools

- Open Babel: better 3D recovery for awkward ions and inorganic species
- Psi4 plus `resp`: RESP or ESP charge workflows
- GROMACS: real MD workflows
- `numba`: optional acceleration in some interface-build paths

### 2.3 Example conda setup

```bash
conda create -n yadonpy python=3.11
conda activate yadonpy

conda install -c conda-forge rdkit openbabel parmed mdtraj matplotlib pandas scipy packaging
conda install -c psi4 psi4 resp dftd3-python

pip install -e .
pip install -e .[accel]
```

### 2.4 First check

```bash
python -c "from yadonpy.diagnostics import doctor; doctor(print_report=True)"
```

Run that before troubleshooting by hand.

## 3. Pick the right script style

### 3.1 Fastest convenience path

```python
import yadonpy as yp

mol, ok = yp.parameterize_smiles(
    "CCO",
    ff_name="gaff2_mod",
    charge_method="RESP",
    work_dir="./work_ethanol",
)
```

Use this when you are prototyping or parameterizing a small molecule quickly.

### 3.2 Explicit workflow path

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
mol = ff.mol("O=C1OCCO1", name="EC")
ok = ff.ff_assign(mol)
```

Use this when:

- the script should stay close to the examples;
- the same code should later work with MolDB reuse;
- you need bonded overrides or more control.

This is the preferred style for serious study scripts.

## 4. Learn the package in the right order

Recommended example order:

1. `examples/07_moldb_precompute_and_reuse`
2. `examples/08_text_to_csv_and_build_moldb`
3. `examples/01_Li_salt`
4. `examples/02_polymer_electrolyte`
5. `examples/05_cmcna_electrolyte`
6. `examples/10_interface_route_a`
7. `examples/11_interface_route_b`
8. `examples/12_cmcna_interface`

That order is intentional. It moves from reusable chemistry to bulk systems and only then to interface workflows.

## 5. Work directories

Use one explicit study root:

```python
from pathlib import Path
from yadonpy.core import workdir
from yadonpy.runtime import set_run_options

restart = True
set_run_options(restart=restart)

BASE_DIR = Path(__file__).resolve().parent
work_dir = workdir(BASE_DIR / "work_dir", clean=not restart)
```

Then create child folders such as:

- `work_dir.child("ac_poly")`
- `work_dir.child("ac_electrolyte")`
- `work_dir.child("interface_route_b")`

This is the expected structure for restartable studies.

## 6. MolDB: what to store and what not to store

Store in MolDB:

- expensive RESP-prepared ions;
- reusable solvents;
- monomers or fragments that appear in multiple studies;
- variants with bonded-patch sidecars.

Do not treat MolDB as a project dump for:

- system trajectories;
- packed bulk boxes;
- old exported topology trees.

## 7. Bulk workflow checklist

A normal bulk workflow is:

1. define the molecules;
2. assign charges;
3. assign force fields;
4. optionally store expensive species in MolDB;
5. build an amorphous cell;
6. run EQ21 and follow-up relaxation if needed;
7. inspect analysis outputs before continuing.

The most common mistake is making the first box too dense.

For awkward systems, start from:

- lower initial density;
- more `Z` room;
- better staging.

## 8. Interface workflow checklist

The current interface strategy is polymer first.

For the best success rate:

1. equilibrate the polymer bulk first;
2. use the equilibrated polymer `XY` box as the lateral reference;
3. plan the electrolyte only for the slab volume that will actually be used;
4. if the system is difficult, build an isotropic probe electrolyte first;
5. resize the final electrolyte from the equilibrated probe response;
6. rebuild the final electrolyte in a fixed-XY box with more `Z` slack;
7. assemble the final interface only after both sides are individually reasonable;
8. use staged diffusion release instead of immediate unrestricted contact.

## 9. Route A versus Route B

### Route A

Use route A when you want a periodic interface workflow and do not need an explicit vacuum buffer.

### Route B

Use route B when you want:

- a vacuum-buffered interface setup;
- `pbc = xy`;
- wall-ready dynamics;
- a one-sided interface workflow that should not immediately collapse into a fully periodic `Z` interpretation.

Examples 11 and 12 are the main references for route B.

## 10. New interface recipe helper in v0.8.54

The interface examples now use:

`recommend_polymer_diffusion_interface_recipe(...)`

Use it after the polymer-matched geometry is already planned. It returns:

- the recommended route object;
- the matching staged interface protocol;
- notes explaining the choice.

Why this helper exists:

- scripts stay linear and easy to imitate;
- route and protocol heuristics live in library code instead of being re-copied into every example;
- neutral and polyelectrolyte workflows can use different defaults without making scripts messy.

## 11. What Example 12 now demonstrates

Example 12 is intentionally large and conservative.

It now does this:

1. build a larger CMC bulk with `DP = 150` and `6` chains;
2. neutralize the polymer phase with explicit Na counter-ions;
3. equilibrate the polymer phase first;
4. build a `1 M` LiPF6 electrolyte using the same solvent ratio as before;
5. use a probe electrolyte bulk to learn the density response;
6. resize and rebuild the final electrolyte to the equilibrated polymer footprint;
7. assemble a vacuum-buffered route-B interface;
8. run a staged diffusion protocol with gap hold, density relax, contact, release, exchange, and production.

This is the main example to follow when interface robustness matters more than minimum system size.

## 12. Restart behavior

`restart=True` means:

- reuse completed compatible outputs when possible;
- skip finished expensive steps when their inputs still match;
- rebuild when schema or input expectations say an artifact is stale.

It does not mean "trust everything already on disk".

## 13. What to inspect after runs

Useful output categories:

- force-field assignment reports;
- `system_meta.json` and interface manifests;
- `06_analysis/*.json` summaries;
- final box dimensions and density summaries.

For interface workflows, also inspect:

- slab thickness and target box notes;
- route choice and stage list;
- assembled interface metadata.

## 14. Common failure patterns

### 14.1 Charge or QM failures

Check the diagnostics report first, then check backend availability.

### 14.2 Dense packing failures

Do not immediately increase retry counts. First ask whether the initial box is structurally too tight.

Usually the better fix is:

- lower initial density;
- increase initial `Z`;
- use a probe-and-resize flow;
- keep `XY` fixed only when it must be fixed.

### 14.3 Interface instability

Split the problem:

1. is the polymer bulk itself well relaxed;
2. is the standalone electrolyte bulk reasonable;
3. is the assembled interface geometry sensible before MD;
4. is the early-stage protocol too aggressive.

## 15. Lightweight validation when GROMACS is unavailable

You can still do code-level validation:

```bash
python -m compileall src examples tests
PYTHONPATH=src pytest -q
```

This is useful on machines that do not provide GROMACS locally.

## 16. Final advice

If a workflow is unstable, make the staging better before making the script more complicated.

That rule is the key to using YadonPy effectively.
