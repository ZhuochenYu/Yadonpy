# Example 03: Tg scan (GROMACS)

## Overview
Run a temperature scan under NPT and fit Tg using an automatic piecewise-linear split.

## Prerequisites
Run Example 02 first. This script consumes:

- `../01_polymer_solution/work_dir/04_eq_*/.../md.gro`
- `../01_polymer_solution/work_dir/02_system/system.top`

## Run

```bash
cd examples/03_tg_gmx
python run_tg.py
```

## Outputs
Written to `work_dir/` (scan data + fitted Tg + plots).
