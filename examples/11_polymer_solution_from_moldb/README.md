# Example 11: Build polymer solution from Molecule DB and run a workflow

## What this example does
- Loads monomers/solvents from a Molecule DB (created by Example 10)
- Builds a polymer (random walk), terminates it, assigns force field
- Packs an amorphous cell with solvent
- Runs an equilibration/production workflow and analysis

## How to run
1) Build the Molecule DB first:
```bash
cd examples/10_moldb_batch_from_csv
python run_batch_build_db.py template.csv
```

2) Run this example:
```bash
cd examples/11_polymer_solution_from_moldb
python run_full_workflow_from_db.py
```

## Outputs
- `work_dir/00_molecules/`: exported MOL2 (names follow your variable names when possible)
- `work_dir/02_system/`: GROMACS system files
- `work_dir/03_eq/`: equilibration stages
- `work_dir/06_analysis/`: thermo, Rg-gate convergence plots, MSD, etc.

## Analysis
Check `work_dir/06_analysis/`:
- `equilibrium.json` (overall equilibrium decision + Rg gate details for polymer systems)
- `plots/rg_convergence.svg` (diagnostic)
- `thermo.xvg` and derived property summary
