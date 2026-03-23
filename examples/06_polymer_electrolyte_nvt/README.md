# Example 06: Polymer electrolyte workflow (NVT production)

## Overview
This example mirrors Example 02, but runs **NVT production** after equilibration.

YadonPy will fix the NVT box size to an **equilibrium-average density target** (plateau window preferred; fallback: last 30% average), then run NVT production and analysis.

## Run

```bash
cd examples/06_polymer_electrolyte_nvt
python run_full_workflow.py
```

## Outputs
Same `work_dir/` layout as Example 02.

## Notes
- PF6 is expected to be precomputed in MolDB for this example. Reuse it through `ff.mol(...)` plus `ff.ff_assign(..., bonded="DRIH")`, and run Example 01 first if your MolDB does not yet contain the PF6 RESP + DRIH record.
- If you are comparing NPT vs NVT properties, ensure the production length and trajectory output frequency are comparable.
- For conductivity (E-H), make sure your run writes velocities in a trajectory format that supports them (e.g., `.trr`).


## Restart handling

The examples use `yadonpy.core.workdir()` to prepare `work_dir` without deleting
existing results. Restart behavior is controlled by the global restart flag.
