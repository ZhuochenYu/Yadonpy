# Example 05: CMC-Na random copolymer + 1M LiPF6 in EC/EMC/DEC

## Overview
Build a **CMC-Na** random copolymer electrolyte system and run the standard `EQ21step` preset.

Targets (as coded):
- Polymer: CMC random copolymer from 4 glucose-based monomers
- Solvent: EC / EMC / DEC with equal mass contribution
- Salt: 1 M LiPF6 (with a minimum of `min_salt_pairs` ion pairs)
- Counter-ion: Na+ to neutralize polymer formal charge
- Charge scaling: polymer and all ions scaled by 0.8
- RDF center: Li+
- Analysis style: independent `rdf()`, `msd()`, and `sigma()` calls

## Run

```bash
cd examples/05_cmcna_electrolyte
python run_cmcna_random_copolymer.py
```

## Notes
- This system can be computationally heavy (often thousands of solvent molecules).
- PF6- is expected to be precomputed in MolDB and is then reused through `ff.mol(...)` plus `ff.ff_assign(..., bonded="DRIH")`. Run Example 01 first if your MolDB does not yet contain the PF6 RESP + DRIH record.
- `sigma_ne_upper_bound_S_m` is expected to overestimate the real conductivity in this concentrated charged-polymer system. Prefer `sigma_eh_total_S_m` and `haven_ratio` when available.

## Remote Mixed-System Test Ladder
For mixed CMC systems, prefer short, isolated runs before launching a long production job.

```bash
# 1) topology/export only
YADONPY_WORK_DIR=work_dir_export YADONPY_EXPORT_ONLY=1 \
python run_cmcna_random_copolymer_dtd_oplsaa_from_moldb.py

# 2) EM + preNVT only
YADONPY_WORK_DIR=work_dir_stage2 YADONPY_EQ21_STAGE_CAP=2 \
python run_cmcna_random_copolymer_dtd_oplsaa_from_moldb.py

# 3) short production on GPU
YADONPY_WORK_DIR=work_dir_gpu YADONPY_PROD_NS=2 \
python run_cmcna_random_copolymer_dtd_from_moldb.py

# 4) matched short production on CPU-only for failure classification
YADONPY_WORK_DIR=work_dir_cpu YADONPY_GPU=0 YADONPY_PROD_NS=2 \
python run_cmcna_random_copolymer_dtd_from_moldb.py
```

- Always use a unique `YADONPY_WORK_DIR` for each debug run.
- `run_cmcna_random_copolymer_dtd_from_moldb.py` now supports additive selection with `YADONPY_ADDITIVE=DTD` or `YADONPY_ADDITIVE=VC`.
- For a fair `DTD` vs `VC` comparison, point both runs at the same `YADONPY_SHARED_POLYMER_ROOT` so they reuse the exact same cached CMC random-walk polymer instead of rebuilding different chains.
- If a production stage fails, the script writes `failure_diagnostics.json` into the selected work directory.

## Leaner Production Output
The production presets now support lower write frequency without changing the post-processing formulas.

```bash
YADONPY_PROD_TRAJ_PS=2 \
YADONPY_PROD_ENERGY_PS=2 \
YADONPY_PROD_LOG_PS=2 \
YADONPY_PROD_TRR_PS= \
YADONPY_PROD_VELOCITY_PS= \
YADONPY_PROD_CPT_MIN=5 \
python run_cmcna_random_copolymer_dtd_from_moldb.py
```

- `YADONPY_PROD_TRAJ_PS` controls compressed trajectory spacing.
- `YADONPY_PROD_ENERGY_PS` / `YADONPY_PROD_LOG_PS` control thermo output.
- Leave `YADONPY_PROD_TRR_PS` and `YADONPY_PROD_VELOCITY_PS` empty to disable production `trr`/velocity output.
- `YADONPY_PROD_CPT_MIN` forces earlier production checkpoints so LINCS fallback can resume from `md.cpt` instead of dying on missing checkpoint files.
