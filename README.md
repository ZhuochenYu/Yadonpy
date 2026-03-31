# YadonPy

YadonPy is a script-first molecular modeling and simulation toolkit for polymers,
electrolytes, graphite-supported interfaces, and GROMACS-based workflows.
It keeps the scientific procedure visible in ordinary Python scripts while still
providing reusable building blocks for charge assignment, force-field preparation,
bulk packing, interface assembly, equilibration, and analysis.

## What YadonPy Does

- Builds molecules directly from SMILES and polymers from PSMILES.
- Assigns force fields with GAFF, GAFF2, GAFF2_mod, OPLS-AA, DREIDING, and MERZ.
- Supports QM-derived charges with Psi4 plus `psiresp-base`, including RESP and ESP.
- Stores expensive prepared molecular assets in MolDB for later reuse.
- Exports GROMACS-ready systems and runs staged MD workflows.
- Builds bulk polymer-electrolyte systems and graphite-polymer-electrolyte sandwich structures.
- Preserves restart-aware work directories and auditable metadata such as charge-group manifests,
  export manifests, and interface build records.

## Installation

The recommended environment uses Python 3.11.

```bash
conda create -n yadonpy python=3.11
conda activate yadonpy

conda install rdkit openbabel parmed mdtraj matplotlib pandas scipy packaging psi4 dftd3-python "pydantic<2" psiresp-base
pip install -e .
```

`psiresp-base` is the supported RESP fitting package, and current supported setups
should keep `pydantic<2` for PsiRESP compatibility. `openbabel` already provides the
Open Babel Python bindings used by YadonPy, so no separate `pybel` package is required.

Check the environment after installation:

```bash
python -c "from yadonpy.diagnostics import doctor; doctor(print_report=True)"
```

For RESP and ESP workflows, `doctor()` should report both `psi4` and `psiresp` as available.
If `doctor()` reports `psiresp: BROKEN` with a `PydanticUserError`, install a
compatible Pydantic:

```bash
conda install -c conda-forge "pydantic<2" "psiresp-base"
```

## Quick Start

### Prepare a small molecule

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
ec = ff.mol("O=C1OCCO1")
yp.assign_charges(ec, charge="RESP", work_dir="./work_ec")
ok = ff.ff_assign(ec)
```

### Reuse a prepared species from MolDB

```python
import yadonpy as yp

pf6 = yp.load_from_moldb(
    "F[P-](F)(F)(F)(F)F",
    charge="RESP",
    require_ready=True,
)
```

### Build a graphite-polymer-electrolyte sandwich

```python
import yadonpy as yp
from yadonpy.interface import (
    SandwichRelaxationSpec,
    default_carbonate_lipf6_electrolyte_spec,
    default_peo_polymer_spec,
)

result = yp.build_graphite_peo_electrolyte_sandwich(
    work_dir="./work_sandwich",
    polymer=default_peo_polymer_spec(dp=20),
    electrolyte=default_carbonate_lipf6_electrolyte_spec(),
    relax=SandwichRelaxationSpec(omp=8, gpu=1, psi4_omp=8),
)
print(result.relaxed_gro)
```

## Workflow Areas

### Molecular preparation

Use the top-level API when you want a short script:

- `mol_from_smiles(...)`
- `assign_charges(...)`
- `assign_forcefield(...)`
- `parameterize_smiles(...)`
- `load_from_moldb(...)`

Use `yadonpy.sim.qm` directly when you need explicit control of conformer search,
QM levels, basis selection, or bonded-parameter derivation.

### Bulk systems

YadonPy can build and equilibrate polymer-electrolyte systems starting from monomers,
solvents, ions, and salts. A typical workflow is:

1. prepare species and assign charges;
2. assign force fields;
3. pack an amorphous cell;
4. export the GROMACS system;
5. run staged equilibration;
6. analyze density, transport, and coordination behavior.

### Interface systems

For interface work, the recommended pattern is:

1. equilibrate each phase independently;
2. derive consistent lateral dimensions from equilibrated bulk boxes;
3. cut slabs from the equilibrated phases;
4. assemble the final interface or sandwich structure;
5. run staged relaxation with restrained early dynamics and a later release stage.

This is the model used by the high-level graphite-polymer-electrolyte sandwich builder.

### Restart and metadata

Work directories are part of the product behavior. YadonPy writes explicit manifests,
export metadata, charge-group records, and restart markers so interrupted studies can be
resumed or audited without guessing which intermediate files are authoritative.

## Included Examples

- `examples/01_Li_salt`: prepare and store a reference salt species in MolDB.
- `examples/02_polymer_electrolyte`: end-to-end polymer-electrolyte workflow from PSMILES.
- `examples/03_tg_gmx`: staged temperature-dependent GROMACS workflow.
- `examples/04_elongation_gmx`: elongation and deformation workflow.
- `examples/05_cmcna_electrolyte`: CMC-Na polymer-electrolyte construction.
- `examples/06_polymer_electrolyte_nvt`: polymer-electrolyte workflow with NVT-focused staging.
- `examples/07_moldb_precompute_and_reuse`: one-shot MolDB catalog build and MolDB-backed reuse scripts.
- `examples/08_graphite_polymer_electrolyte_sandwich`: graphite-polymer-electrolyte sandwich workflows for PEO and CMC-Na.
- `examples/09_oplsaa_assignment`: compact OPLS-AA assignment workflows written in the same script-first style as the main examples.

## Documentation

- [User Guide](docs/USER_GUIDE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Technical Notes](docs/TECHNICAL_NOTES.md)

## Practical Notes

- Use local runs for fast API checks, packaging work, and short unit tests.
- Use the remote GPU node for long GROMACS jobs, larger sandwich systems, and heavy QM workflows.
- For charged polymers, prefer `polyelectrolyte_mode=True` so RESP constraints and later charge scaling remain auditable.
- MolDB is intended for reusable molecular assets such as geometry and charge variants, not as a topology cache.
