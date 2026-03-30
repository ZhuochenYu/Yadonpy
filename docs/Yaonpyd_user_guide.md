# YadonPy User Guide (v0.8.76)

This guide explains how to use YadonPy effectively in day-to-day study scripts.

Release baseline: Python 3.11+.

Related documents:

- README: package scope and installation
- manual: `docs/Yadonpy_manul.md`
- API reference: `docs/Yadonpy_API_v0.8.76.md`

## 1. Build the right environment

Recommended environment:

```bash
conda create -n yadonpy python=3.11
conda activate yadonpy

conda install rdkit openbabel parmed mdtraj matplotlib pandas scipy packaging psi4 dftd3-python psiresp-base
pip install pybel

pip install -e .
```

Check the environment:

```bash
python -c "from yadonpy.diagnostics import doctor; doctor(print_report=True)"
```

For RESP/ESP workflows, `doctor()` should report both `psi4` and `psiresp`.

If the study requires only force-field assignment and export, `psi4` and `psiresp` remain optional. If the study requires RESP or ESP, both are required.

From `v0.8.74`, two operational defaults changed:

- restart-aware workflows default to strict input checking instead of permissive reuse;
- ion packs created by `core.poly.ion(...)` must be passed explicitly via `ions=[...]`.

## 1.1 Rebuild the reference MolDB species set

YadonPy no longer ships `yd_moldb.tar`. The reference species list now lives as plain CSV files in Example 07. To rebuild the merged reference species set into a fresh MolDB and add the supported battery-anion extensions, run:

```bash
python examples/07_moldb_precompute_and_reuse/03_rebuild_reference_moldb_species.py
```

The script rebuilds the merged Example 07 species list and additionally includes:

- `ClO4-`
- `BF4-`
- `AsF6-`
- `FSI-`
- `TFSI-`
- `Li+`

`DRIH` is applied only to recognized high-symmetry polyhedral ions. `FSI-` and `TFSI-` remain on the standard RESP path.

## 2. Use an explicit script structure

Recommended study skeleton:

```python
from pathlib import Path

import yadonpy as yp
from yadonpy.core import workdir
from yadonpy.runtime import set_run_options

restart = True
set_run_options(restart=restart)

BASE_DIR = Path(__file__).resolve().parent
work_dir = workdir(BASE_DIR / "work_dir", clean=not restart)
ff = yp.get_ff("gaff2_mod")
```

This keeps restart logic, output paths, and chemistry preparation explicit.

## 3. Prepare species

### 3.1 Small neutral molecules

```python
EC = ff.mol("O=C1OCCO1")
ok = ff.ff_assign(EC)
```

### 3.2 Explicit QM charge assignment

```python
from yadonpy.sim import qm

qm.assign_charges(
    EC,
    charge="RESP",
    work_dir=work_dir / "01_ec_resp",
)
ok = ff.ff_assign(EC)
```

### 3.3 Polyelectrolyte monomers

```python
qm.assign_charges(
    glucose_6,
    charge="RESP",
    work_dir=work_dir / "01_glucose6_resp",
    polyelectrolyte_mode=True,
)
ok = ff.ff_assign(glucose_6)
```

Use `polyelectrolyte_mode=True` whenever the repeat unit contains persistent charged motifs and you want grouped RESP constraints and grouped export metadata.

## 4. Store expensive species in MolDB

Good MolDB candidates:

- PF6 and similar ions;
- reusable carbonate solvents;
- frequently reused monomers;
- species with expensive QM geometry or bonded-patch preparation.

Do not treat MolDB as a project dump for trajectories or full system exports.

## 5. Build a bulk cell

Typical pattern:

```python
from yadonpy.core import poly
from yadonpy.io.gmx import write_gmx

ac = poly.amorphous_cell(
    [polymer, EC, DEC, EMC, Li, PF6],
    [1, 50, 30, 20, 6, 6],
    density=0.9,
    work_dir=work_dir / "02_build_cell",
    charge_scale=0.8,
)
```

For polyelectrolyte systems:

```python
ac = poly.amorphous_cell(
    [CMC, Na],
    [2, 40],
    density=0.4,
    charge_scale=0.8,
    polyelectrolyte_mode=True,
    work_dir=work_dir / "02_build_cmc_cell",
)
```

This activates grouped local scaling if charged-group metadata are present on the species.

## 6. Understand charge scaling

### 6.1 Raw charges

Raw RESP charges are produced during molecular preparation and should be regarded as the authoritative QM-derived template.

### 6.2 Simulation scaling

`charge_scale` is a simulation-level model choice. It is applied during cell construction/export and does not overwrite the original RESP template.

### 6.3 Polyelectrolyte-aware scaling

With `polyelectrolyte_mode=True`:

- charged groups are scaled locally;
- the neutral backbone is left unchanged;
- the export records the result in `charge_scaling_report.json`.

If charged-group detection is unavailable, the workflow records a fallback and reverts to whole-molecule scaling.

## 7. Export and inspect the system

Modern workflows typically export via the internal system-export bridge used by the preset workflows. The important point for the user is that exports now preserve more metadata.

For polymeric/polyelectrolyte systems, inspect:

- `system.top`
- `system.gro`
- `residue_map.json`
- `charge_groups.json`
- `resp_constraints.json`
- `charge_scaling_report.json`

These files are now part of the standard audit path for grouped RESP and grouped scaling.

## 8. Run equilibration

Typical preset usage:

```python
from yadonpy.sim.preset import eq

job = eq.EQ21step(
    ac,
    work_dir=work_dir,
    omp=8,
    gpu=1,
)
result = job.exec()
```

For NPT-capable workflows, the run now also writes an NPT convergence plot that overlays density, volume, and box lengths.

## 8.1 Read post-processing outputs with the new defaults

### MSD

`AnalyzeResult.msd()` now writes metric-specific outputs.

Interpret them as follows:

- `ion_atomic_msd`: single-atom ion motion
- `molecule_com_msd`: whole-molecule center-of-mass diffusion
- `chain_com_msd`: polymer chain translation
- `residue_com_msd`: polymer segmental motion at the residue/monomer level
- `charged_group_com_msd`: motion of charged polymer groups only

Do not interpret `chain_com_msd` as local polymer segmental motion. For polymers, segmental mobility is represented by `residue_com_msd`.

### RDF/CN

`AnalyzeResult.rdf()` now defaults to site-level coordination analysis.

In practice:

- the center species must exist explicitly in `system_meta.json`;
- the output directory is `06_analysis/rdf_site/`;
- `site_map.json` shows exactly which atoms were grouped into each target site class;
- first-shell CN is only treated as formal when the shell detector reports an acceptable confidence level.

### Ionic conductivity of charged polymers

`AnalyzeResult.sigma()` now uses charged-group diffusion coefficients for charged polymers.

This is intentional. The conductivity contribution is computed from:

- the number of charged groups;
- the charged-group formal charge (`+1`, `-1`, `+2`, etc.);
- the charged-group diffusion coefficient from `charged_group_com_msd`.

If a charged polymer has no charged-group MSD metadata, the code does not fall back to a whole-chain net charge approximation. The ignored contribution is recorded in `sigma.json`.

## 9. Build interfaces

Use the polymer-first strategy for polymer/electrolyte studies:

1. equilibrate polymer bulk;
2. use polymer `XY` as the lateral reference;
3. prepare electrolyte bulk separately;
4. assemble the interface;
5. run staged release rather than immediate unrestricted contact.

Use route selection helpers rather than in-script ad hoc route logic where possible.

## 10. Use the examples in the right order

Recommended order:

1. `examples/01_Li_salt`
2. `examples/02_polymer_electrolyte`
3. `examples/05_cmcna_electrolyte`
4. `examples/07_moldb_precompute_and_reuse`
5. `examples/08_graphite_polymer_electrolyte_sandwich`

Specific to the current merged layout:

- `examples/05` and `examples/08` are the reference scripts for grouped polyelectrolyte RESP, local scaling, and charged-group-aware post-processing.

## 11. Common failure modes

### RESP stack missing

Symptoms:

- import errors for `psi4` or `psiresp`
- `doctor()` reports missing modules

Action:

- install `psi4`, `dftd3-python`, and `psiresp` through conda.

### Packed system too dense

Symptoms:

- non-finite forces during EM
- repeated packing retries
- severe density overshoot

Action:

- start from a lower density;
- give the system more `Z` slack;
- equilibrate the components separately before interface assembly.

### Polyelectrolyte fallback triggered

Symptoms:

- `charge_scaling_report.json` reports `whole_molecule_scale`

Action:

- inspect `charge_groups.json` and `resp_constraints.json`;
- verify the monomer chemistry and formal-charge pattern;
- extend the template library if the chemistry is a legitimate recurring motif.

## 12. Practical rule

If a system is expensive, difficult, or reused across studies, make its preparation auditable:

- store the prepared molecular state in MolDB;
- preserve charge metadata;
- keep the work directory explicit;
- inspect the export metadata before launching long MD runs.
