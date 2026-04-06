# Example 04: Uniaxial elongation (GROMACS deform)

## Overview

Run a stress-strain study on an already equilibrated system with the high-level
YadonPy mechanics wrapper.

## Prerequisites

Run Example 02 first. This example resolves the prepared system automatically
from:

- `../02_polymer_electrolyte/work_dir/00_system/system.top`
- the latest equilibrated `md.gro` found under `../02_polymer_electrolyte/work_dir`

## Run

```bash
cd examples/04_elongation_gmx
python run_elong.py
```

## Outputs

Written to `work_dir/`:

- `summary.json`
- `stress_strain.csv`
- `stress_strain.svg`
- material summary fields such as `youngs_modulus_gpa`, `max_stress_gpa`, and `strain_at_max_stress`
