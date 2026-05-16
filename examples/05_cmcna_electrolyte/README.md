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
- MSD semantics: ions use atomic MSD, small molecules use molecular COM MSD,
  and CMC uses independent chain COM MSD rather than atom/residue MSD. Molecule
  and chain COM metrics use `gmx msd -n system.ndx -mol` when the ndx group maps
  cleanly to topology molecules; local diagnostics stay on the Python backend.

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
- `run_cmcna_random_copolymer_dtd_from_moldb.py` selects the additive with `YADONPY_ADDITIVE=DTD` or `YADONPY_ADDITIVE=VC`.
- For a fair `DTD` vs `VC` comparison, point both runs at the same `YADONPY_SHARED_POLYMER_ROOT` so they reuse the exact same cached CMC random-walk polymer instead of rebuilding different chains.
- If a production stage fails, the script writes `failure_diagnostics.json` into the selected work directory.

## Leaner Production Output
The benchmark scripts support automatic output and analysis thinning without
changing the MD physics. For long CMC runs, prefer:

```bash
PERFORMANCE_PROFILE=auto ANALYSIS_PROFILE=auto \
python benchmark_cmcna_carbonate_lipf6_bulk.py
```

- The resolved output cadence is written to production `summary.json`.
- Pre-existing dense trajectories are downsampled at read time for RDF/MSD/cell metrics;
  the effective stride is written to `06_analysis/analysis_runtime_policy.json`.
- Explicit `TRAJ_PS`, `ENERGY_PS`, `LOG_PS`, or `ANALYSIS_PROFILE=full` still
  override auto when dense final-analysis output is needed.
- Production coordinate output is adaptive TRR-only by default so `gmx current`
  conductivity can run directly. Set `TRAJECTORY_FORMAT=xtc` for smaller
  screening trajectories; avoid `TRAJECTORY_FORMAT=xtc_trr` on 100 ns CMC
  systems unless `TRR_PS` is very coarse.
- `YADONPY_PROD_CPT_MIN` forces earlier production checkpoints so LINCS fallback can resume from `md.cpt` instead of dying on missing checkpoint files.

## 100 ns CMC-Na Bulk Benchmark
The script `benchmark_cmcna_carbonate_lipf6_bulk.py` is the recommended
script-first entry for comparing CMC-Na swollen by 1 M LiPF6 in
EC:EMC:DEC = 3:2:5 by mass at 318.15 K.

```bash
# GAFF2 + MERZ ions + RESP, 0.7 charge scaling
YADONPY_FORCEFIELD=gaff2 \
YADONPY_CHARGE_SCALE=0.7 \
PERFORMANCE_PROFILE=auto ANALYSIS_PROFILE=auto \
python benchmark_cmcna_carbonate_lipf6_bulk.py

# OPLS-AA assignment diagnostics, with explicit refine profile
YADONPY_FORCEFIELD=oplsaa \
YADONPY_OPLSAA_PROFILE=refine \
YADONPY_CHARGE_SCALE=0.7 \
PERFORMANCE_PROFILE=auto ANALYSIS_PROFILE=auto \
python benchmark_cmcna_carbonate_lipf6_bulk.py
```

The transport table reports `Li`, `Na`, `PF6`, EC/EMC/DEC, and CMC rows. For
CMC, use `chain_com_msd` for whole-chain self diffusion. `residue_com_msd` and
`charged_group_com_msd` are local mobility diagnostics.

OPLS-AA for this mixed polyelectrolyte/electrolyte benchmark uses a stricter
stability policy than the GAFF2 path: the default OPLS production timestep is
`1 fs` with stronger LINCS settings and `conservative` GPU offload (`nb/pme` on
GPU, `bonded/update` on CPU). CMC diagnostics indicate that `balanced` and
`full` GPU offload can trigger CUDA illegal-address failures for refine-profile
CMC assignments. Keep this default unless a parameter set has passed a short
preflight. Override `YADONPY_PROD_DT_PS=0.002` or
`YADONPY_GPU_OFFLOAD_MODE=balanced/full` only for explicit stability tests.
