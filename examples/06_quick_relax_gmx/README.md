# Example 04: Quick relax (minim + short NVT)

This example replaces RadonPy's LAMMPS-based quick-min/quick-md with a **GROMACS-only** workflow.

## Prerequisites

Put your input files in this folder:

- `system.gro`
- `system.top`

## Run

```bash
python run_quick.py
```

Outputs will be written to `work_quick/`.