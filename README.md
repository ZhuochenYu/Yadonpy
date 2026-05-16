# YadonPy

YadonPy is a script-first molecular-modeling workflow toolkit for the
polymer-electrolyte and graphite-interface workflows shipped in `examples/01`
through `examples/08`.  It is designed for research scripts where the chemistry,
force-field choices, MD settings, restart policy, and analysis calls remain
visible in ordinary Python rather than hidden behind a large configuration file.

The current public workflow surface is example-driven.  The examples are not
decorative demos; they define the supported usage patterns for preparing reusable
molecular assets, building polymer/electrolyte systems, running GROMACS
equilibration and production, and post-processing the resulting trajectories.

## Supported Workflows

YadonPy currently focuses on these concrete workflows.

1. **Reusable salt and molecule preparation**

   Example 01 builds PF6- once, assigns RESP charges when the QM stack is
   available, applies the DRIH bonded patch, stores the result in MolDB, and then
   reloads the same species in the normal script style:

   ```python
   from yadonpy.ff.gaff2_mod import GAFF2_mod

   ff = GAFF2_mod()
   PF6_smiles = "F[P-](F)(F)(F)(F)F"
   PF6 = ff.mol(PF6_smiles, charge="RESP", prefer_db=True, require_ready=True)
   PF6 = ff.ff_assign(PF6, bonded="DRIH")
   ```

   This is the intended path for high-symmetry anions reused by later
   electrolyte and interface examples.

2. **Bulk polymer-electrolyte MD**

   Example 02 is the main end-to-end polymer-electrolyte workflow.  It prepares
   monomers, solvents, and salts from SMILES or MolDB, polymerizes and terminates
   chains, packs an amorphous polymer/solvent/salt cell, exports a GROMACS
   topology, runs the EQ21 equilibration preset, then calls RDF, MSD, and
   conductivity analyses explicitly.  The example folder also contains benchmark
   scripts for PEO/LiTFSI and carbonate/LiPF6 comparison studies.

3. **Tg scan from a prepared polymer-electrolyte system**

   Example 03 reuses the prepared system from Example 02, runs a
   temperature-dependent NPT density scan, and fits Tg from the density versus
   temperature trend.  The script resolves the prepared system, calls
   `run_tg_scan_gmx`, and prints the result with
   `print_mechanics_result_summary`.

4. **Uniaxial elongation from a prepared polymer-electrolyte system**

   Example 04 also reuses the prepared Example 02 system, then runs a uniaxial
   GROMACS deform workflow and reports stress-strain quantities such as Young's
   modulus, maximum stress, and strain at maximum stress.  The script uses the
   same prepared-system resolver, calls `run_elongation_gmx`, and prints the
   mechanics summary with `print_mechanics_result_summary`.

5. **CMC-Na carbonate electrolytes**

   Example 05 builds CMC-Na random copolymer systems swollen by
   EC/EMC/DEC + LiPF6, with Na+ counterions and configurable charge scaling.  It
   includes mixed-system debug ladders, GAFF2/MERZ and OPLS-AA-oriented benchmark
   scripts, and analysis semantics for ions, carbonate molecules, and CMC chain
   center-of-mass MSD.

6. **NVT production after density equilibration**

   Example 06 mirrors the polymer-electrolyte workflow but switches the
   production stage to NVT.  The NVT volume is chosen from an equilibrium-average
   density target, preferring a plateau window and falling back to the last part
   of the NPT trajectory when needed.

7. **MolDB precomputation and audit**

   Example 07 precomputes reusable molecular species from a curated electrolyte
   catalog.  It separates charge preparation from force-field assignment,
   supports serial and CPU-parallel RESP builds, refreshes adaptive RESP records,
   checks force-field assignment, and audits the active user MolDB against the
   bundled default catalog.

8. **Graphite/polymer/electrolyte layer-stack interfaces**

   Example 08 builds generic layer stacks rather than a hard-coded sandwich.
   Scripts cover basal graphite/electrolyte, edge graphite/electrolyte,
   graphite + CMC-Na + electrolyte, two-graphite stacks, and a fixed-charge
   graphite sweep.  The interface workflow writes layer-aware
   `system.gro/top/ndx` files plus `layer_stack_manifest.json`, then provides
   interface-specific post-processing for z profiles, fixed-charge EDL
   diagnostics, penetration, adsorption, cation coordination, and anisotropic
   in-plane transport.

## What The Toolkit Provides

The examples above are built from a small set of reusable components:

- SMILES and PSMILES molecule construction with RDKit/OpenBabel helpers.
- Random-walk polymerization, chain termination, and amorphous packing for the
  shipped polymer-electrolyte systems.
- Force-field assignment routes used by these workflows: GAFF2/GAFF2_mod for
  organic molecules and polymers, MERZ ion parameters for monatomic ions,
  DRIH bonded patching for selected high-symmetry anions, and OPLS-AA-oriented
  comparison routes where the example explicitly selects them.
- Psi4/PsiRESP charge workflows for RESP/ESP preparation, including adaptive
  equivalence handling and polyelectrolyte-specific grouping.
- MolDB storage and reuse for molecule-level geometries, charges, bonded
  patches, and metadata. MolDB is not a complete system-topology cache.
- GROMACS export, EQ21 equilibration, NPT/NVT production, restart/resume
  handling, and adaptive output cadence.
- Analysis routines used by the examples: density/thermo summaries, RDF, MSD,
  Nernst-Einstein and Einstein-Helfand conductivity, dielectric response,
  polymer metrics, Tg fitting, uniaxial mechanics, and layer-stack interface
  profiles.

YadonPy does not claim to be a universal force-field generator or a general
constant-potential electrode engine.  In particular, the charged-graphite
workflow in Example 08 is a fixed-charge surface model; the post-processing
potential is a one-dimensional diagnostic from sampled charge density, not a
constant-potential solution.

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

The default reusable molecular catalog is copied into `~/.yadonpy/moldb`.
Set `YADONPY_MOLDB` only when you intentionally want to use another MolDB root.

For RESP/ESP workflows, `doctor()` should report both `psi4` and `psiresp` as
available.  Workflows that use GROMACS also require a compatible `gmx` binary.
If several GROMACS versions are installed, set `YADONPY_GMX_CMD=/path/to/gmx`
before production or post-processing.

## Script Style

YadonPy examples use direct imports and explicit variables.  A typical script
starts by selecting force fields, loading MolDB-backed species, building the
system, then calling simulation and analysis methods in order.

```python
from yadonpy.core.data_dir import ensure_initialized
from yadonpy.ff.gaff2_mod import GAFF2_mod
from yadonpy.ff.merz import MERZ

ensure_initialized()

ff = GAFF2_mod()
ion_ff = MERZ()

EC = ff.mol("O=C1OCCO1", charge="RESP", prefer_db=True, require_ready=True)
EC = ff.ff_assign(EC, report=False)

PF6 = ff.mol("F[P-](F)(F)(F)(F)F", charge="RESP", prefer_db=True, require_ready=True)
PF6 = ff.ff_assign(PF6, bonded="DRIH", report=False)

Li = ion_ff.mol("[Li+]")
Li = ion_ff.ff_assign(Li, report=False)
```

Layer-stack scripts use the same style:

```python
from yadonpy.interface import (
    GraphiteLayerSpec,
    LayerStackSpec,
    MolecularLayerSpec,
    analyze_layer_stack_interface,
    build_layer_stack,
)

stack = LayerStackSpec(
    layers=(
        GraphiteLayerSpec(name="GRAPHITE", nx=6, ny=5, n_layers=3),
        MolecularLayerSpec(
            name="ELECTROLYTE",
            species=(EC, Li, PF6),
            counts=(100, 10, 10),
            thickness_nm=4.0,
            density_target_g_cm3=1.2,
            layer_kind="electrolyte",
        ),
    ),
    order="bottom_to_top",
    pbc_mode="xyz",
)

result = build_layer_stack(stack=stack, work_dir="./work_layer_stack")
profile = analyze_layer_stack_interface(
    work_dir="./work_layer_stack",
    manifest_path=result.manifest_path,
    analysis_profile="interface_fast",
    compute_transport=False,
)
```

For sampled interfaces, use the interface analysis facade rather than bulk 3D
transport assumptions:

```python
analy = nvt.analyze()
interface = analy.interface(
    manifest_path=result.manifest_path,
    analysis_profile="interface_fast",
    bin_nm=0.05,
    region_width_nm=0.75,
    surface_distance_nm=0.50,
)
health = interface.geometry_health()
edl = interface.edl_profiles()
adsorption = interface.graphite_adsorption(species=("EC", "EMC", "DEC"))
summary = interface.summary()
```

## Simulation And Analysis Notes

- Production presets use adaptive output cadence.  Production writes TRR
  coordinate frames by default so `gmx current` can be used for collective
  conductivity analysis.  For storage-first screening, request XTC explicitly
  with `trajectory_format="xtc"` or `TRAJECTORY_FORMAT=xtc`.
- Analyzer profiles are part of the workflow contract.  Use
  `analysis_profile="transport_fast"` for large screening runs and
  `analysis_profile="full"` when dense, all-site post-processing is required.
  Use `analysis_profile="interface_fast"` for Example 08 layer stacks.
- Long or dense trajectories are thinned at read time in non-`full` profiles.
  The effective policy is written to
  `06_analysis/analysis_runtime_policy.json`.
- Diffusion semantics are molecule-aware.  Ions use atomic MSD, small molecules
  use molecule COM MSD, and polymers use independent chain COM MSD by default.
  Local residue or charged-group MSDs are diagnostics, not whole-chain
  self-diffusion.
- `sigma_ne_upper_bound_S_m` is reported as an upper bound.  When stable,
  `sigma_eh_total_S_m` is the preferred total conductivity estimate.
- For Example 08, `Dxy` is the interface mobility metric.  `Dz` is confined
  z-direction mobility and should not be interpreted as a bulk diffusion
  coefficient.

After all required analyses have completed, large trajectory streams can be
removed while keeping auditable outputs:

```python
from yadonpy import clean_md_trajectory_files

cleanup = clean_md_trajectory_files(work_dir, enabled=True)
print(cleanup.removed_files)
```

The cleanup helper removes `.xtc`, `.trr`, `.trj`, and `.tng` recursively.  It
keeps final coordinate snapshots, topology files, energy files, JSON summaries,
CSV tables, and plots.

## Examples

- `examples/01_Li_salt`: PF6- build, DRIH patching, and MolDB reuse.
- `examples/02_polymer_electrolyte`: bulk polymer-electrolyte construction,
  EQ21/NPT production, transport analysis, and benchmark scripts.
- `examples/03_tg_gmx`: Tg scan from a prepared Example 02 system.
- `examples/04_elongation_gmx`: uniaxial GROMACS deform study from a prepared
  Example 02 system.
- `examples/05_cmcna_electrolyte`: CMC-Na carbonate/LiPF6 electrolytes,
  mixed-system benchmarks, and CMC-aware transport analysis.
- `examples/06_polymer_electrolyte_nvt`: NVT production from an equilibrated
  density target.
- `examples/07_moldb_precompute_and_reuse`: curated MolDB build, adaptive RESP
  refresh, force-field assignment check, and bundled-catalog audit.
- `examples/08_graphite_polymer_electrolyte_sandwich`: generic graphite,
  electrolyte, and CMC-Na layer stacks with interface-specific post-processing.

Additional example folders may exist for narrower diagnostics or newer feature
work, but the list above is the stable introduction path.

## Documentation

- [User Guide](docs/USER_GUIDE.md): workflow-level usage and recommended practice.
- [API Reference](docs/API_REFERENCE.md): public functions, classes, and analysis semantics.
- [Architecture](docs/ARCHITECTURE.md): package structure and design boundaries.
- [Technical Notes](docs/TECHNICAL_NOTES.md): force-field and implementation details.

## Practical Guidance

- Run Example 01 or Example 07 before workflows that require ready MolDB species
  such as PF6-, carbonate solvents, or CMC repeat units.
- Use MolDB for reusable molecule-level assets; do not treat it as a complete
  packed-system or topology cache.
- Use `polyelectrolyte_mode=True` for charged polymer RESP workflows where
  equivalent charged groups should be handled consistently.
- Set `gpu=0` to force CPU MD.  If `gpu_id` is also present, it is ignored in
  CPU mode.
- Keep debugging runs small and isolated with separate `work_dir` values before
  launching long production simulations.
