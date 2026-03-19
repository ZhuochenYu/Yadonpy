# yadonpy

`yadonpy` is a RadonPy-inspired workflow toolkit built by YZC. The intended
simulation backend is **GROMACS** (MD engine), while molecule/force-field
handling reuses RadonPy-style building blocks.

Current release: **v0.4.20**.

## Naming & artifact filenames

YadonPy uses a *molecule name* to label artifacts (`.itp/.gro/.top/.mol2`) and QM folders.

Priority order:
1) explicit name you set (e.g. `utils.mol_from_smiles(..., name="PF6")` or `utils.named(mol, "copoly")`)
2) existing RDKit props (`_yadonpy_name`, `name`, `_yadonpy_resname`, `_Name`)
3) best-effort inference from Python variable names when you pass molecules into workflows
   (e.g. `poly.amorphous_cell([copoly, solvent_A, solvent_B], ...)`).

As a result, in typical scripts you no longer need to manually call `utils.set_name_from_var()`.

## Installation

Recommended (conda + pip):

```bash
conda install -c conda-forge rdkit openbabel
pip install pybel
pip install .
```

Notes:
- OpenBabel/pybel is optional for neutral molecules. It is **strongly recommended** for
  explicitly charged (p)SMILES, where yadonpy will generate an initial 3D geometry
  with OpenBabel + UFF relaxation before Psi4 OPT+RESP.

## GROMACS workflow presets

Yadonpy is **GROMACS-only**. Starting from v0.1.4, it also provides RadonPy-style
workflow presets implemented with GROMACS:

- Multi-stage equilibration: `yadonpy.gmx.workflows.EquilibrationJob`
- Tg scan + auto piecewise-linear split: `yadonpy.gmx.workflows.TgJob`
- Uniaxial elongation (deform): `yadonpy.gmx.workflows.ElongationJob`
- Quick relax (minim + short NVT): `yadonpy.gmx.workflows.QuickRelaxJob`

## What's new in v0.1.40

- Faster GROMACS GPU runs by default: `mdrun` now uses
  `-nb gpu -bonded gpu -update gpu -pme gpu -pmefft gpu -pin auto` (when supported).
- More robust multi-stage restart: if a stage contains `md.tpr + md.cpt`, YadonPy continues with
  `mdrun -cpi md.cpt -append` instead of restarting from scratch.
- More predictable executable auto-detection: prefer `gmx` (thread-MPI) by default.
  Override with `export YADONPY_GMX_CMD=...`.

## Data directory initialization

On first use, yadonpy initializes a user-writable data directory (by default:
`~/.local/share/yadonpy/`). You can override it via:

```bash
export YADONPY_DATA_DIR=/path/to/yadonpy_data
```

It will create:

```
<DATA_ROOT>/
  ff/ff_dat/           # vendored RadonPy force-field json files
  ff/library.json      # yadonpy molecule library (SMILES-indexed)
  basic_top/           # cached molecule artifacts
```

## Auto-register non-polymer molecules

If a molecule SMILES **does not contain `*`**, it is treated as a non-polymer
molecule. After the first successful parameterization, yadonpy will add it to
the default library (`ff/library.json`) and cache artifacts under
`basic_top/<ff>/<mol_id>/`.

Newly created entries are tagged with:
- `is_original_from_lib: false`
- `is_polymer_monomer: false`

## GAFF scaling defaults

When exporting GROMACS topologies for GAFF/AMBER, yadonpy uses the default
scaling:
- `fudgeLJ = 0.5`
- `fudgeQQ = 0.8333333333`


## GPU execution

By default, YadonPy enables GROMACS GPU offload when available. The workflow adds `-v` to `gmx mdrun` (if supported) and streams progress to the terminal. Use `gpu=0` in presets or set `RunResources(use_gpu=False)` to force CPU.


### GROMACS execution defaults
- All `gmx mdrun` calls include `-v` for live progress output.
- Energy minimization stages use a minimal command line by default: `gmx mdrun -deffnm <name> -ntmpi 1 -ntomp <N> -nb gpu -gpu_id <id>`.
