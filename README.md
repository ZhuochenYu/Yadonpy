# YadonPy

YadonPy is a script-first molecular modeling workflow toolkit for polymers,
electrolytes, graphite interfaces, and GROMACS-based molecular dynamics.  It is
designed for research workflows where the scientific procedure should remain
visible in ordinary Python scripts while molecular assets, force-field choices,
simulation settings, and analysis outputs stay reproducible.

## Scope

YadonPy provides:

- SMILES/PSMILES-based molecule, polymer, and segment construction.
- Force-field assignment with GAFF, GAFF2, GAFF2_mod, OPLS-AA, DREIDING, MERZ,
  and TIP-style ion/water helpers.
- Psi4/PsiRESP charge workflows, including adaptive RESP equivalence and
  localized resonance constraints.
- MolDB reuse for precomputed molecular geometries, charges, bonded patches, and
  metadata.
- GROMACS export, equilibration, production, restart, and resume workflows.
- Bulk polymer-electrolyte, liquid electrolyte, and generic layer-stack
  graphite/polymer/electrolyte interface builders.
- Analysis workflows for density, RDF, MSD, conductivity, dielectric response,
  ion coordination, migration states, thermomechanics, and interface profiles.

The project intentionally favors transparent, auditable scripts over opaque
configuration-only pipelines.

## Installation

The recommended development environment uses Python 3.11.

```bash
conda create -n yadonpy python=3.11
conda activate yadonpy

conda install -c conda-forge rdkit openbabel parmed mdtraj matplotlib pandas scipy packaging psi4=1.10 dftd3-python psiresp-base
python -m pip install "pydantic==1.10.26"
python -m pip install -e .
```

Initialize the bundled MolDB catalog and check the environment:

```bash
python - <<'PY'
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.diagnostics import doctor

ensure_initialized()
doctor(print_report=True)
PY
```

The bundled catalog is seeded into `~/.yadonpy/moldb`, which is the default
location for reusable molecular assets.

For RESP/ESP workflows, `doctor()` should report both `psi4` and `psiresp` as
available.

## Minimal Usage

Prepare a molecule:

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
ec = ff.mol("O=C1OCCO1")
yp.assign_charges(ec, charge="RESP", work_dir="./work_ec")
ff.ff_assign(ec)
```

Reuse a MolDB species:

```python
import yadonpy as yp

pf6 = yp.load_from_moldb(
    "F[P-](F)(F)(F)(F)F",
    charge="RESP",
    require_ready=True,
)
```

Build a graphite/electrolyte layer stack:

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
ion_ff = yp.get_ff("merz")

ec = ff.mol("O=C1OCCO1", charge="RESP", prefer_db=True, require_ready=True)
ff.ff_assign(ec)
li = ion_ff.mol("[Li+]")
ion_ff.ff_assign(li)

stack = yp.LayerStackSpec(
    layers=(
        yp.GraphiteLayerSpec(name="GRAPHITE", nx=6, ny=5, n_layers=3),
        yp.MolecularLayerSpec(
            name="ELECTROLYTE",
            species=(ec, li),
            counts=(100, 10),
            thickness_nm=4.0,
            density_target_g_cm3=1.2,
            layer_kind="electrolyte",
        ),
    ),
)

result = yp.build_layer_stack(stack=stack, work_dir="./work_layer_stack")
profile = yp.analyze_layer_stack_interface(work_dir="./work_layer_stack")
```

For sampled interfaces, use the analysis facade rather than bulk-only 3D
transport assumptions:

```python
analy = nvt.analyze()
interface = analy.interface(manifest_path=result.manifest_path, bin_nm=0.05)
edl = interface.edl_profiles()
adsorption = interface.graphite_adsorption(species=("EC", "EMC", "DEC"))
```

## Simulation And Analysis Notes

Production presets use adaptive output cadence.  By default, production writes
TRR coordinate frames so `gmx current` can be used for collective conductivity
analysis.  For storage-first screening, request XTC explicitly with
`trajectory_format="xtc"` or `TRAJECTORY_FORMAT=xtc`.

After analysis, large trajectory streams can be removed while keeping scientific
artifacts:

```python
yp.clean_md_trajectory_files(work_dir, enabled=True)
```

This removes `.xtc`, `.trr`, `.trj`, and `.tng` files recursively, while keeping
final structures, topology, energy files, JSON summaries, CSV tables, and plots.

For long or large simulations, the analyzer also applies runtime frame thinning
in non-`full` profiles.  The effective policy is written to
`06_analysis/analysis_runtime_policy.json`.

## Examples

- `examples/01_Li_salt`: salt species preparation and MolDB storage.
- `examples/02_polymer_electrolyte`: polymer-electrolyte workflow and benchmark scripts.
- `examples/03_tg_gmx`: glass-transition workflow.
- `examples/04_elongation_gmx`: uniaxial elongation workflow.
- `examples/05_cmcna_electrolyte`: CMC-Na electrolyte workflows.
- `examples/06_polymer_electrolyte_nvt`: NVT-focused polymer electrolyte workflow.
- `examples/07_moldb_precompute_and_reuse`: MolDB precomputation and reuse.
- `examples/08_graphite_polymer_electrolyte_sandwich`: layer-stack interface workflows.
- `examples/09_oplsaa_assignment`: OPLS-AA assignment validation.
- `examples/11_segment_branch_polymer`: segment-first branched polymer construction.

## Documentation

- [User Guide](docs/USER_GUIDE.md): workflow-level usage and recommended practice.
- [API Reference](docs/API_REFERENCE.md): public functions, classes, and analysis semantics.
- [Architecture](docs/ARCHITECTURE.md): package structure and design boundaries.
- [Technical Notes](docs/TECHNICAL_NOTES.md): force-field and implementation details.

## Practical Guidance

- Use MolDB for reusable molecular assets, not as a topology cache.
- Prefer `polyelectrolyte_mode=True` for charged polymer RESP workflows.
- Set `gpu=0` to force CPU MD; any simultaneous `gpu_id` value is ignored.
- Use `analysis_profile="transport_fast"` for screening and
  `analysis_profile="full"` for final detailed analysis.
- Use `run_tg_scan_gmx`, `run_elongation_gmx`, and
  `print_mechanics_result_summary` for high-level thermomechanical studies.
