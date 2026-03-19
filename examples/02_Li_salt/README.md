# Example 02: Simple Li-salt (ion pair) parametrization + quick MD smoke test

## What this example does
- Build small ions (e.g., Li+ and an anion) from SMILES.
- Assign charges (if configured) and generate a minimal topology.
- Pack a tiny box and run a short GROMACS workflow to validate that `grompp/mdrun` works.

## How to run
```bash
cd examples/02_Li_salt
python run_full_workflow.py
```

## Outputs
- `work_dir/00_molecules/`: exported MOL2 files
- `work_dir/02_system/`: `system.gro/system.top`
- `work_dir/03_eq/`: equilibration stages
- `work_dir/06_analysis/`: basic thermo plots and summaries

## Analysis
Open `work_dir/06_analysis/` for:
- `thermo.xvg` and plots (temperature/pressure/density etc.)
- any equilibrium checks reported by the script
