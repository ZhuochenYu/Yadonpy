# Example 02: Tg scan (GROMACS)

This example runs a temperature scan under NPT and fits Tg using **automatic piecewise linear split**.

## Prerequisites

Put your input files in this folder:

- `system.gro`
- `system.top`

## Run

```bash
python run_tg.py
```

Outputs will be written to `work_tg/`.