# Example 04: Uniaxial elongation (GROMACS deform)

## Overview
Perform uniaxial deformation with `deform` in GROMACS and extract a stress–strain curve.

## Prerequisites
Run Example 02 first. This script consumes:

- `../01_polymer_solution/work_dir/04_eq_*/.../md.gro`
- `../01_polymer_solution/work_dir/02_system/system.top`

## Run

```bash
cd examples/04_elongation_gmx
python run_elong.py
```

## Outputs
Written to `work_dir/` (stress–strain CSV/JSON + plots).
