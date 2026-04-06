# Example 03: Tg scan (GROMACS)

## Overview

Run a temperature-dependent NPT scan on an already equilibrated system and fit
`Tg` with the high-level YadonPy study wrapper.

## Prerequisites

Run Example 02 first. This example resolves the prepared system automatically
from:

- `../02_polymer_electrolyte/work_dir/00_system/system.top`
- the latest equilibrated `md.gro` found under `../02_polymer_electrolyte/work_dir`

## Run

```bash
cd examples/03_tg_gmx
python run_tg.py
```

## Outputs

Written to `work_dir/`:

- `summary.json`
- `density_vs_T.csv`
- `plots/tg_density_vs_T.svg`
- per-temperature stage folders with their own `summary.json` and thermo plots
