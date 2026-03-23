# YadonPy

Current release: **v0.8.53**

YadonPy is a **SMILES / PSMILES-driven workflow toolkit** for polymer, solvent, and salt systems. It combines molecule construction, QM charge generation, force-field assignment, GROMACS export, equilibration workflows, and post-processing into a script-friendly Python package.

## What YadonPy does

YadonPy is designed to help you:

- build polymers, solvents, salts, and mixed systems from SMILES / PSMILES;
- run conformer search and QM-based charge assignment (for example RESP with Psi4);
- assign force fields such as GAFF / GAFF2 / GAFF2_mod / OPLS-AA / MERZ / DREIDING;
- export single molecules and packed systems to **GROMACS**;
- run equilibration, Tg, and elongation workflows;
- collect density, RDF, diffusion, conductivity, and polymer-metric outputs as JSON / SVG.

Since **v0.6.6**, YadonPy follows a **MolDB-first** design:

- persistent cache stores only the expensive parts: **best 3D geometry + charge variants**;
- legacy prebuilt topology cache (`basic_top/`) is removed;
- `.itp`, `.top`, and `.gro` files are generated on demand.

---

## Installation

### Recommended environment

- Linux
- Python 3.11+
- RDKit (`>= 2025.03.1` recommended)
- Open Babel for robust 3D generation in tricky inorganic / ionic cases
- GROMACS for MD workflows
- Psi4 + Python `resp` when RESP charges are needed

### Example conda environment

```bash
conda create -n yadonpy python=3.11
conda activate yadonpy

conda install -c conda-forge rdkit openbabel parmed mdtraj matplotlib pandas scipy
conda install -c psi4 psi4 resp dftd3-python

pip install -e .

# optional JIT acceleration for large interface builds
pip install -e .[accel]
```

### Quick self-check

```bash
python -c "from yadonpy.diagnostics import doctor; doctor(print_report=True)"
```

---


## Interface workflow (v0.8.20)

YadonPy v0.8.20 keeps the dedicated interface-build layer, retains the explicit script-style Examples 10, 11, and 12, restores the RDKit/MolSpec compatibility hotfix through public `yadonpy.core.as_rdkit_mol()` and `yadonpy.core.molecular_weight()` helpers, fixes liquid-only EQ/interface assembly edge cases found in Example 12, simplifies restart control in the interface examples while allowing `amorphous_cell(...)` to reuse completed packed-cell builds, improves the interface builder itself through denser slab-window selection plus lateral phase decorrelation during assembly, uses a more conservative semiisotropic `Berendsen` pressure-coupling policy during the early interface densification stages, now enables `periodic-molecules = yes` across the staged interface MDP defaults, rewrites stage handoff `md.gro` files through a topology-guided whole-molecule canonicalization pass before the next `grompp`, writes a richer interface `system.ndx` with region-aware moltype / atomtype / per-instance groups for downstream analysis, neutralizes tiny exported bulk-box charge residuals in `02_system`, keeps interface lateral phase shifts from tearing bonded molecules across the periodic box, labels packed-cell restart reuse explicitly while keeping a stable fallback cache for `amorphous_cell(...)` restores, further accelerates slab charge rebalancing through prefiltered vectorized candidate scoring, hardens molecular-weight estimation for monoatomic ions and unsanitized hypervalent species, keeps random-walk retry-budget estimation independent of fragile descriptor imports, restores random-walk `tqdm` progress bars by default, accepts chained `ff.ff_assign(ff.mol(...))` / `MERZ().ff_assign(MERZ().mol(...))` scripting style, and still avoids rebuilding the same large low-density bulk box twice when EQ21 needs both raw and charge-scaled exports.

### Random-walk acceleration

The v0.8.10 random-walk and interface changes target general-purpose speedups plus more robust charge-balanced exports without assuming the system is flexible or that multiprocessing is available:

- old-chain clash checking is localized to atoms near the newly attached monomer before the full proximity test runs;
- retry budgets are capped from an estimated monomer rigidity score so flexible systems stop wasting time on oversized retry windows while semi-rigid and rigid monomers still keep a larger fallback budget;
- default random-walk budgets are now `retry=60`, `rollback=3`, `retry_step=80`, and `retry_opt_step=4`.

Key rules in this release:

- use `work_dir = workdir(BASE_DIR / "work_dir", clean=not restart)` as the study root after setting the script-level `restart` switch;
- create child folders directly from the root workdir, for example `work_dir.child("ac_poly")`, `work_dir.child("ac_electrolyte")`, and `work_dir.child("interface_route_a")`;
- interface geometry construction writes numbered stage folders directly under the chosen interface child directory;
- representative bulk snapshots use the equilibrium-window midpoint when available, otherwise the latest equilibrated structure is used with that fallback recorded in metadata;
- slab preparation keeps whole molecules / whole polymer chains, searches for the densest target-thickness slab window, and then recenters the retained slab laterally before export;
- the top slab is phase-shifted laterally during assembly by default to avoid directly registering rough surface features from the two independent bulk snapshots;
- interface Examples 10, 11, and 12 now all use the same five-step probe-and-resize bulk workflow: equilibrate polymer, equilibrate probe electrolyte, resize composition to the polymer XY footprint, run a fixed-XY semiisotropic NPT on the rebuilt electrolyte, then assemble the final interface;
- `recommend_electrolyte_alignment(...)` now supplies the example defaults for resized-electrolyte target Z and fixed-XY semiisotropic NPT duration, using `max(1.0 nm, surface_shell + gap/2)` as the alignment margin and a longer default relaxation time for the CMC/polyelectrolyte case;
- the interface MD protocol now uses semiisotropic `Berendsen` pressure coupling for the contact and exchange stages by default, then switches to the requested production barostat (`C-rescale` by default) only for the final production stage;
- route A builds a fully periodic dual-interface slab, while route B builds a wall-ready geometry with a vacuum buffer;
- wall parameters belong only to the interface MD protocol, not to the geometry builder.

### Interface ndx naming

The interface builder now writes both `system.ndx` and `system_ndx_groups.json` under `03_interface/`.

The `system.ndx` file contains both coarse spatial labels and finer post-processing groups:

- region labels: `BOTTOM`, `TOP`, `BOTTOM_CORE`, `BOTTOM_SURFACE`, `TOP_CORE`, `TOP_SURFACE`, `INTERFACE_ZONE`
- region moltypes: `BOTTOM_MOL_<MOLTYPE>`, `TOP_MOL_<MOLTYPE>`
- region atomtypes: `BOTTOM_ATYPE_<ATOMTYPE>`, `TOP_ATYPE_<ATOMTYPE>`
- region moltype-atomtype groups: `BOTTOM_TYPE_<MOLTYPE>_<ATOMTYPE>`, `TOP_TYPE_<MOLTYPE>_<ATOMTYPE>`
- per-instance molecule groups: `BOTTOM_INST_<MOLTYPE>_0001`, `TOP_INST_<MOLTYPE>_0001`, ...
- representative atoms: `BOTTOM_REP_<MOLTYPE>`, `TOP_REP_<MOLTYPE>`, and `REP_<MOLTYPE>`

For scripted post-processing you can use the helper API:

```python
from yadonpy.interface import build_interface_group_catalog

catalog = build_interface_group_catalog("./work_dir/interface_route_a/03_interface/system.ndx")
print(catalog["categories"]["region_moltypes"]["BOTTOM"])
print(catalog["categories"]["region_instances"]["TOP"])
```

If the interface was built by YadonPy, the same information is also written automatically to `03_interface/system_ndx_groups.json`.

Typical usage:

```python
from pathlib import Path
from yadonpy.core import workdir
from yadonpy.interface import InterfaceBuilder, InterfaceDynamics, InterfaceProtocol, InterfaceRouteSpec

restart = True
work_dir = workdir(Path('./work_dir'), clean=not restart)
ac_poly_dir = work_dir.child('ac_poly')
ac_electrolyte_dir = work_dir.child('ac_electrolyte')
interface_dir = work_dir.child('interface_route_a')

builder = InterfaceBuilder(work_dir=interface_dir)
built = builder.build_from_bulk_workdirs(
    name='polymer_vs_electrolyte',
    bottom_name='ac_poly',
    bottom_work_dir=ac_poly_dir,
    top_name='ac_electrolyte',
    top_work_dir=ac_electrolyte_dir,
  route=InterfaceRouteSpec.route_a(axis='Z', gap_nm=0.60, bottom_thickness_nm=4.5, top_thickness_nm=5.0, top_lateral_shift_fraction=(0.5, 0.5)),
)

protocol = InterfaceProtocol.route_a(temperature_k=300.0, pressure_bar=1.0)
final_interface_gro = InterfaceDynamics(built=built, work_dir=work_dir.child('interface_route_a_md')).run(protocol=protocol)
```

See the new standalone examples under `examples/10_interface_route_a/`, `examples/11_interface_route_b/`, and `examples/12_cmcna_interface/`.
Those examples now build molecules, amorphous cells, equilibration runs, and interface geometry directly in the script so their control flow matches the earlier example style more closely.

## MolDB workflow

MolDB is the recommended cache layer for expensive QM results. It stores:

- best 3D geometry (`mol2`)
- atomic charges (`json`, multiple variants such as RESP / ESP / CM5)

It does **not** store `.itp/.top/.gro` artifacts.

Default location:

- `~/.yadonpy/moldb/`

Environment overrides:

- `YADONPY_MOLDB=/path/to/moldb`
- `YADONPY_HOME=/path/to/.yadonpy` (MolDB will be placed under `$YADONPY_HOME/moldb`)

### Handle-based MolDB usage

```python
from yadonpy.ff.gaff2_mod import GAFF2_mod

ff = GAFF2_mod()

# Build a lightweight lookup handle.
EC = ff.mol("O=C1OCCO1")

# The handle resolves during ff_assign().
ok = ff.ff_assign(EC)
if not ok:
    raise RuntimeError("GAFF2_mod assignment failed for EC")
```

With an explicit bonded override:

```python
ok = ff.ff_assign(ff.mol("O=C1OCCO1"), bonded="mseminario")
if not ok:
    raise RuntimeError("mSeminario assignment failed")
```

### Export a computed molecule into MolDB

If a script has already produced a geometry and charges via `qm.conformation_search()` and `qm.assign_charges()`, you can export it as a reusable MolDB entry:

```python
from yadonpy.moldb import MolDB

db = MolDB()
rec, out_dir = db.mol_gen(solvent_B, work_dir=work_dir, add_to_moldb=False)
```

By default this writes a copyable MolDB snippet under the working directory and **does not** modify the global MolDB. To write directly into the global MolDB, pass `add_to_moldb=True`.

### Batch pre-computation (`autocalculate`)

```python
from yadonpy.moldb import MolDB

db = MolDB()
db.read_calc_temp = "./template.csv"
db.autocalculate(work_dir="./work_dir/01_build", omp=64, mem=20000)
```

Default behavior is safe: results are written under a fresh
`moldb_generated_YYYYMMDD_HHMMSS/` directory inside `work_dir`, not directly into `~/.yadonpy/moldb`.

To write into the global MolDB explicitly:

```python
db.autocalculate(
    work_dir="./work_dir/01_build",
    omp=64,
    mem=20000,
    add_to_moldb=True,
)
```

### Inspect MolDB quickly

```python
from yadonpy.moldb import MolDB

db = MolDB()
db.check()
```

This prints canonical keys, names, canonical SMILES / PSMILES, and available charge variants.

---


## Working directories and restart behavior

YadonPy v0.7.17 introduces a path-compatible workflow directory helper:

```python
from pathlib import Path
from yadonpy.core import workdir

work_dir = workdir(Path('./work_dir'), restart=True)
```

`work_dir` behaves like a normal path object, so existing code such as
`work_dir / '00_molecules'` continues to work. Unlike older example snippets,
creating a `WorkDir` does not delete an existing directory unless `clean=True`
is requested explicitly.

The helper records lightweight metadata in `work_dir/.yadonpy/workdir.json` and
exposes the resolved restart flag to downstream helpers.

## MolSpec handles and backward compatibility

`ff.mol(...)` returns a lightweight MolDB-backed handle. In v0.7.13, once
`ff.ff_assign(handle)` succeeds, the handle caches the resolved RDKit molecule.
Core polymer-building helpers now accept that handle directly, so legacy script
patterns remain valid:

```python
mol_P = ff.mol(smiles_P)
ok = ff.ff_assign(mol_P)
if not ok:
    raise RuntimeError('force-field assignment failed')

dp = poly.calc_n_from_num_atoms(mol_P, 1000, terminal1=ter1)
poly_rw = poly.polymerize_rw(mol_P, dp, tacticity='atactic', work_dir=work_dir)
```

## QM defaults

YadonPy infers **net charge** and **spin multiplicity** from SMILES / RDKit whenever possible.

Default QM level policy:

- **Anions** (`net charge < 0`)
  - optimization: `wb97m-d3bj/def2-SVPD`
  - RESP / ESP single point: `wb97m-d3bj/def2-TZVPD`
- **Neutral molecules and cations**
  - optimization: `wb97m-d3bj/def2-SVP`
  - RESP / ESP single point: `wb97m-d3bj/def2-TZVP`

Compatibility note:

- if a CSV or script uses `wb97m`, YadonPy normalizes it to `wb97m-d3bj` for Psi4 compatibility.

Override the defaults with explicit `opt_method`, `opt_basis`, `charge_method`, `charge_basis`, or `auto_level=False`.

### Lightweight charge models

In addition to `RESP / ESP / Mulliken / Lowdin`, YadonPy also supports:

- `CM1A`
- `1.14*CM1A`
- `<scale>*CM1A`
- `CM5`
- `1.2*CM5`
- `<scale>*CM5`

Examples:

```python
qm.assign_charges(mol, charge="CM5", opt=False, work_dir=work_dir)
qm.assign_charges(mol, charge="1.2*CM5", opt=False, work_dir=work_dir)
qm.assign_charges(mol, charge="CM1A", opt=False, work_dir=work_dir)
qm.assign_charges(mol, charge="1.14*CM1A", opt=False, work_dir=work_dir)
```

Backend notes:

- `CM5` uses xTB / GFN1;
- `CM1A` uses LigParGen / BOSS;
- missing executables raise explicit errors instead of failing silently.

---

## Public API highlights

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
mol = yp.mol_from_smiles("O=C1OCCO1", name="EC")
yp.assign_charges(mol, charge="RESP", work_dir="./work")
ok = ff.ff_assign(mol)
```

Common high-level helpers:

- `yp.get_ff(ff_name)`
- `yp.list_forcefields()`
- `yp.list_charge_methods()`
- `yp.mol_from_smiles(smiles, coord=True, name=None)`
- `yp.conformation_search(mol, **kwargs)`
- `yp.assign_charges(mol, charge="RESP", **kwargs)`
- `yp.assign_forcefield(mol, ff_name="gaff2_mod", **kwargs)`
- `yp.parameterize_smiles(smiles, ff_name="gaff2_mod", charge_method="RESP", ...)`
  - strict by default: requested charge-assignment failures now raise unless you pass `allow_ff_without_requested_charges=True`.
- `yp.load_from_moldb(smiles, charge='RESP', basis_set=None, method=None, require_ready=True, return_record=False)`

`load_from_moldb()` returns the molecule by default. Use `return_record=True` when you also need the underlying `MolRecord`.

The detailed API reference is in **`docs/Yadonpy_API_v0.8.53.md`**.

---

## `ff_assign()` reporting

After a successful force-field assignment, YadonPy prints a formatted report by default. The report includes:

- atom index
- element
- `ff_type`
- `ff_btype` (when present)
- charge
- `sigma`
- `epsilon`
- force-field type description

Disable the automatic report with:

```python
ff.ff_assign(mol, report=False)
```

---

## Example order

Recommended reading / execution order:

- `examples/07_moldb_precompute_and_reuse/`
  - `01_build_moldb.py`
  - `02_polymer_electrolyte_from_moldb.py`
- `examples/08_text_to_csv_and_build_moldb/`
- `examples/01_Li_salt/`
- `examples/02_polymer_electrolyte/`
- `examples/05_cmcna_electrolyte/`
- `examples/03_tg_gmx/`
- `examples/04_elongation_gmx/`
- `examples/06_polymer_electrolyte_nvt/`
- `examples/09_oplsaa_assign/`

---

## Analysis outputs

`analy.get_all_prop(temp=..., press=..., save=True)` writes merged outputs under `work_dir/06_analysis/`, including:

- `thermo_summary.json`
- `basic_properties.json`
- `cell_summary.json`
- `polymer_radius_of_gyration.json`
- `polymer_end_to_end_distance.json`
- `polymer_persistence_length.json`
- `polymer_metrics.json`
- `summary.json`

---

## EQ21 preset (robust GROMACS mode)

`yadonpy.sim.preset.eq.EQ21step` writes a dedicated `03_EQ21/` layout:

- `03_EQ21/01_em`
- `03_EQ21/02_preNVT`
- `03_EQ21/03_EQ21/step_01` ... `step_21`

Before the simulation starts, YadonPy writes:

- `03_EQ21/eq21_parameters.json`
- `03_EQ21/eq21_schedule.csv`
- `03_EQ21/eq21_schedule.json`
- `03_EQ21/eq21_schedule.md`
- `03_EQ21/eq21_overview.svg`

Default ladder:

- `Tmax = 1000 K`
- `Tanal = temp`
- `Pmax = 50000 bar`
- `Panal = press`
- `dt = 1 fs`
- `eq21_robust = True`

In robust mode, YadonPy keeps the formal 21-step temperature / pressure schedule but makes the **GROMACS implementation** safer than a direct LAMMPS-style translation:

- only the pre-NVT stage generates new velocities by default;
- later stages continue the incoming velocity field instead of random re-sampling;
- hot / high-pressure stages use a smaller time step;
- intermediate densification NPT stages use a damped `Berendsen` barostat with larger `tau_p` and reduced compressibility;
- all EQ21 NPT stages are lengthened by default (`eq21_npt_time_scale = 2.0`) so the box has more time to collapse cavities and converge density without making pressure coupling more aggressive;
- the final ambient-pressure stage keeps the requested production barostat (`C-rescale` by default).

Important EQ21 knobs:
- density plateau gate now uses a relaxed slope limit of `5e-2 / ps` with `rel_std <= 1%`, which is less brittle for polymer-electrolyte NPT tails;

- `eq21_tmax`
- `eq21_tanal`
- `eq21_pmax`
- `eq21_panal`
- `eq21_dt_ps`
- `eq21_pre_nvt_ps`
- `eq21_tau_t_ps`
- `eq21_tau_p_ps`
- `eq21_barostat`
- `eq21_compressibility`
- `eq21_robust`
- `eq21_stage_reseed`
- `eq21_npt_time_scale`

---

## Conductivity note for large macro-ions

For large, highly charged polyions, a naive Nernst-Einstein sum can explode to unphysical values. Since v0.6.6, the NE workflow ignores multi-atom macro-ions with large total charge by default and records the decision explicitly in `summary.json` / `sigma.json`.

---

## Running examples from a source bundle

When examples are executed directly from the source bundle, the scripts prefer `./src` on `sys.path` so they do not accidentally import an older site-packages build.

```bash
python -c "import yadonpy; print(yadonpy.__version__); print(yadonpy.__file__)"
```

---

## Development self-check

For source-tree validation, YadonPy now ships a lightweight pytest smoke suite.

```bash
python -m pyflakes src examples tests
python -m compileall src examples tests
PYTHONPATH=src pytest
```

The smoke tests cover:

- public API return semantics such as `load_from_moldb(..., return_record=...)`;
- EQ21 schedule generation and robust/legacy control policies;
- a fake-runner GROMACS equilibration smoke test with restart-skip behavior.

---

## Troubleshooting

- ParmEd may warn about missing `[ pairs ]` when reading a topology that relies on `[ defaults ]` with `gen-pairs = yes`. This does **not** affect GROMACS runs; YadonPy suppresses the warning during optional `system.mol2` export where possible.
- If a MolDB lookup succeeds but topology export later differs from expectation, verify whether the intended bonded override (`plain`, `DRIH`, or `mSeminario`) is present in the cached metadata.
- If RESP or other QM steps fail, run `yadonpy.diagnostics.doctor(print_report=True)` first to confirm backend availability.

---

## Documentation

- API reference: `docs/Yadonpy_API_v0.8.53.md`
- User manual: `docs/Yadonpy_manul.md`
- User guide: `docs/Yaonpyd_user_guide.md`

### Release sanity check

```bash
PYTHONPATH=src pytest -q
python -m compileall src examples tests
```
