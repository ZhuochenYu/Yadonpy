# YadonPy

YadonPy is a script-first molecular-modeling workflow toolkit for the
polymer-electrolyte, graphite-interface, and enhanced-sampling workflows shipped
in `examples/01` through `examples/12`.  It is designed for research scripts where the chemistry,
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
   graphite sweep.  Production templates include both graphite-footprint-first
   and prepared CMC/electrolyte-slab-footprint-first construction routes.  The interface workflow writes layer-aware
   `system.gro/top/ndx` files plus `layer_stack_manifest.json`, relaxes compact
   stacks with pre-minimization, short pre-NVT, fixed-XY z-NPT, and final NVT,
   then provides interface-specific post-processing for z profiles,
   fixed-charge EDL diagnostics, penetration-depth distributions, graphite-EDL
   adsorption orientation distributions, membrane permeation fractions, cation
   coordination, and anisotropic in-plane transport.

9. **Enhanced sampling at interfaces**

   Example 12 adds a GROMACS + PLUMED umbrella-sampling workflow for a solvated
   Li+ crossing from electrolyte into CMC-Na.  GROMACS pull-code supplies the
   umbrella bias and `gmx wham` PMF reconstruction, while PLUMED records
   solvent/CMC/anion coordination CVs for mechanistic interpretation.

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
  handling, wall-confined `pbc=xy` slab EQ21 for stack-ready polymers,
  fixed-XY z-NPT layer-stack relaxation, and adaptive output cadence.
- Analysis routines used by the examples: density/thermo summaries, RDF, MSD,
  Nernst-Einstein and Einstein-Helfand conductivity, dielectric response,
  polymer metrics, Tg fitting, uniaxial mechanics, and layer-stack interface
  profiles.
- Enhanced-sampling preparation and umbrella-PMF post-processing for the
  CMC/electrolyte Li+ transfer example.

YadonPy does not claim to be a universal force-field generator or a general
constant-potential electrode engine.  In particular, the charged-graphite
workflow in Example 08 is a fixed-charge surface model; the post-processing
potential is a one-dimensional diagnostic from sampled charge density, not a
constant-potential solution.

## 0. Installation of GROMACS

YadonPy writes GROMACS inputs and calls an external `gmx` executable.  For
workflows that use PLUMED, modern GROMACS releases do not require a separate
`plumed patch` step: the GROMACS source distribution includes the PLUMED
interface, CMake enables it with `-DGMX_USE_PLUMED=ON`, and the actual PLUMED
kernel is loaded at runtime through `PLUMED_KERNEL`.

The dependency stack below is conda-managed: compilers, CUDA 12.5 build/runtime
libraries including `nvcc`, FFTW, hwloc, OpenBLAS/LAPACK, CMake/Ninja, and the
PLUMED runtime kernel all come from conda-forge.  A GROMACS source tree is still
required when building GROMACS yourself; conda can manage the build environment,
but it does not replace the exact upstream source tree for a custom source
build.  On the GPU node used for the YadonPy examples this source tree is
`~/gromacs-2026.1`.

```bash
source ~/anaconda3/etc/profile.d/conda.sh

conda create -y --solver=libmamba -n gmx-2026-plumed-conda -c conda-forge \
  python=3.11 cmake ninja make pkg-config git \
  gcc_linux-64=12 gxx_linux-64=12 \
  cuda-version=12.5 cuda-compiler=12.5 cuda-libraries-dev=12.5 \
  fftw libhwloc openblas "libblas=*=*openblas" "liblapack=*=*openblas" \
  zlib libxml2 "plumed=2.9.2=*nompi*"

conda activate gmx-2026-plumed-conda
export PATH="$CONDA_PREFIX/bin:$PATH"

export GMX_SRC="$HOME/gromacs-2026.1"
export GMX_BUILD="$HOME/build-gromacs-2026.1-plumed-conda"
export GMX_PREFIX="$HOME/GROMACS-2026.1-PLUMED-CONDA"
export GMX_BUILD_JOBS="${GMX_BUILD_JOBS:-16}"

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export PLUMED_KERNEL="$CONDA_PREFIX/lib/libplumedKernel.so"

cmake -S "$GMX_SRC" -B "$GMX_BUILD" -G Ninja \
  -DCMAKE_INSTALL_PREFIX="$GMX_PREFIX" \
  -DCMAKE_C_COMPILER="$CC" \
  -DCMAKE_CXX_COMPILER="$CXX" \
  -DCMAKE_CUDA_HOST_COMPILER="$CUDAHOSTCXX" \
  -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" \
  -DCUDAToolkit_ROOT="$CONDA_PREFIX" \
  -DGMX_GPU=CUDA \
  -DGMX_USE_PLUMED=ON \
  -DGMX_MPI=OFF \
  -DGMX_THREAD_MPI=OFF \
  -DGMX_OPENMP=ON \
  -DGMX_HWLOC=ON \
  -DGMX_BUILD_OWN_FFTW=OFF \
  -DGMX_FFT_LIBRARY=fftw3 \
  -DGMX_BUILD_UNITTESTS=OFF \
  -DBUILD_TESTING=OFF \
  -DGMXAPI=OFF \
  -DGMX_INSTALL_NBLIB_API=OFF

cmake --build "$GMX_BUILD" -j "$GMX_BUILD_JOBS"
cmake --install "$GMX_BUILD"
```

The `CMAKE_CUDA_ARCHITECTURES=89` value targets the RTX 4080 SUPER GPUs on the
current GPU node.  Use the architecture code for the target GPU when compiling
on different hardware.

Activate the compiled binary together with the same conda environment:

```bash
conda activate gmx-2026-plumed-conda
export PATH="$CONDA_PREFIX/bin:$PATH"
source "$HOME/GROMACS-2026.1-PLUMED-CONDA/bin/GMXRC"
export PLUMED_KERNEL="$CONDA_PREFIX/lib/libplumedKernel.so"
export YADONPY_GMX_CMD="$HOME/GROMACS-2026.1-PLUMED-CONDA/bin/gmx"

"$CONDA_PREFIX/bin/nvcc" --version
gmx --version
gmx mdrun -h | grep -i plumed
test -f "$PLUMED_KERNEL"
```

For a PLUMED-driven simulation, pass the input file explicitly to `mdrun`, for
example `gmx mdrun -deffnm md -plumed plumed.dat`.  The non-MPI build above is
chosen deliberately because the current GROMACS PLUMED interface does not support
multi-rank thread-MPI use.  If you use a prebuilt conda-forge `gromacs` package
instead of compiling, first verify both CUDA driver compatibility and
`gmx mdrun -h | grep -i plumed`.

References: the
[GROMACS installation guide](https://manual.gromacs.org/documentation/current/install-guide/index.html)
documents `GMX_USE_PLUMED` and CUDA build options, the
[GROMACS PLUMED page](https://manual.gromacs.org/documentation/current/reference-manual/special/plumed.html)
documents `-plumed` and `PLUMED_KERNEL`, and the
[PLUMED installation guide](https://www.plumed.org/doc-v2.10/user-doc/html/_installation.html)
documents the conda-forge PLUMED kernel.

## 1. Installation of YadonPy

The recommended development environment uses Python 3.11.

```bash
conda create -n yadonpy python=3.11
conda activate yadonpy

conda install -c conda-forge rdkit openbabel parmed mdtraj matplotlib pandas scipy packaging psi4=1.10 dftd3-python psiresp-base
python -m pip install "pydantic==1.10.26"
python -m pip install -e .
```

`python -m pip install -e .` installs `imageio-ffmpeg`, so YadonPy's MP4
post-processing can use the bundled ffmpeg executable without requiring a
separate system `ffmpeg` package.

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
    FixedChargeRegionSpec,
    GraphiteLayerSpec,
    GraphiteRestraintSpec,
    InterdiffusionStartSpec,
    LayerStackSpec,
    MolecularLayerSpec,
    ZCompressionAnnealSpec,
    analyze_layer_stack_interface,
    build_layer_stack,
    run_layer_stack_relaxation,
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
    molecular_packing_expand="z",
)

result = build_layer_stack(stack=stack, work_dir="./work_layer_stack")
profile = analyze_layer_stack_interface(
    work_dir="./work_layer_stack",
    manifest_path=result.manifest_path,
    analysis_profile="interface_fast",
    compute_transport=False,
)
```

The layer density targets above define the initial geometry.  For sampled
interfaces, let the stack relax in z before analysis: XY stays fixed by the
graphite footprint, z is pressure-coupled, and the reported interface analysis
uses the final NVT trajectory after that z-NPT density relaxation.
For dense graphite/polymer/electrolyte sandwiches, `compression_anneal` adds
small fixed-XY z-compression moves followed by hot/high-pressure z-only
annealing before the final z-NPT.  In `auto` mode it skips explicit vacuum or
open-z controls and enables the loop for closed graphite sandwich stacks.
For large slab or high-DP CMC systems, also make the final z-NPT conservative
with `z_compressibility_bar_inv=4.5e-6` and `z_npt_tau_p_ps=20.0`; this keeps
the last density-relaxation step from applying a sudden large z scaling after
the gentle compression cycles.
For large CMC-Na layers, the preferred preparation route is an independent
wall-confined slab before the final stack is assembled.  Use
`prepare_cmcna_xy_bulk_slab(...)` to build a dilute fixed-XY CMC-Na amorphous
cell at `0.05 g/cm3`, run `periodicity="xy"` EQ21 with z walls, and let
fixed-XY/z-only wall-NPT compress the z-open slab naturally.  This avoids the
`xyz -> unwrap -> slab` route, where CMC chain segments can cross the z periodic
image before a clean slab boundary exists.  The observed density is reported as
`CMC-Na mass / (fixed XY area * active z extent)`, not the total GROMACS box
density, because wall padding would otherwise dilute the value.  Active density
is a convergence diagnostic and not a hard target; Rg convergence is checked at
the same time.  The exported `prepared_slab.gro` is deliberately
`wrapped_xy_z_open`: x/y coordinates are wrapped back into the primary periodic
image, while z coordinates keep the wall-confined open-slab boundary.  The
whole-molecule handoff file, `prepared_slab_whole.gro`, is kept for diagnostics
only and should not be used for layer-stack assembly.  Pass `prepared_slab_gro` to
`MolecularLayerSpec(prepared_slab_gro=...)` only after
`cmcna_slab_convergence.json` reports `ready_for_layer_stack=True`.  CMC-Na
layers that are still packed directly can initialize Na+ as local carboxylate
counterions with `counterion_contact_mode="carboxylate"`; prepared slabs
preserve the pre-equilibrated Na+/COO- contacts.  The relaxation summary reports
CMCNA phase density and total mass density inside CMC-rich regions after
stacking.
When the observable is electrolyte/CMCNA interdiffusion, use
`InterdiffusionStartSpec`: pre-release minimization/NVT/z-NPT applies temporary
z-only phase gates to the two soft phases, and the gates are removed only for
final NVT.  The summary records `diffusion_t0_stage="final_nvt"`, so the final
trajectory is the only trajectory used for diffusion statistics.
Graphite flatness is controlled separately by `GraphiteRestraintSpec`, which
uses z-only position restraints to suppress electrode wrinkling without
freezing in-plane graphite motion.

Constant-charge interface studies should use `FixedChargeRegionSpec` on
`LayerStackSpec.fixed_charge_regions`.  This is a fixed-charge approximation,
not a constant-potential electrode model: the selected atoms are assigned charge
once when `system.top` is generated, and the same topology is used through
compression annealing, final z-NPT, and final NVT sampling.  The selector is
layer based, so basal graphite can charge `region="top"` or `"bottom"` faces,
edge graphite can target a named edge slab, and amorphous layers can use
`region="z_range"` or `thickness_nm` with optional `elements` filters.

```python
relax = run_layer_stack_relaxation(
    result,
    time_ns=2.0,
    pre_nvt_ns=0.05,
    z_npt_ns=0.50,
    relax_z=True,
    z_compressibility_bar_inv=4.5e-6,
    z_npt_tau_p_ps=20.0,
    graphite_restraint=GraphiteRestraintSpec(
        enabled="auto",
        k_pre_kj_mol_nm2=5000.0,
        k_final_kj_mol_nm2=1000.0,
    ),
    interdiffusion_start=InterdiffusionStartSpec(
        enabled=True,
        hold_interphase=True,
        release_at_final_nvt=True,
        phase_gate_k_kj_mol_nm2=1500.0,
    ),
    compression_anneal=ZCompressionAnnealSpec(
        enabled="auto",
        cycles=6,
        tmax_K=380.0,
        pmax_bar=2000.0,
        max_z_shrink_per_cycle=0.04,
        compression_tau_p_ps=20.0,
        compression_z_compressibility_bar_inv=4.5e-6,
        geometry_clash_check=True,
    ),
    dt_ps=0.001,
    constraints="none",
)
analy = relax.analyze()
interface = analy.interface(
    manifest_path=result.manifest_path,
    analysis_profile="interface_fast",
    bin_nm=0.05,
    region_width_nm=0.75,
    surface_distance_nm=0.50,
    time_series_sample_count=10,
    time_series_fps=1.0,
    time_series_charge_potential=True,
)
health = interface.geometry_health()
z_profiles = interface.z_profiles(time_series_analysis=True)
edl = interface.edl_profiles(time_series_analysis=True)
penetration = interface.penetration(
    species=("EC", "EMC", "DEC", "PF6", "Li"),
    time_series_analysis=True,
)
membrane = interface.membrane_permeation(
    species=("EC", "EMC", "DEC", "PF6", "Li"),
    time_series_analysis=True,
)
adsorption = interface.graphite_adsorption(
    species=("EC", "EMC", "DEC"),
    time_series_analysis=True,
)
coordination = interface.coordination_by_region(time_series_analysis=True)
time_series = interface.time_series(time_series_analysis=True)
summary = interface.summary(time_series_analysis=True)
```

For charge sweeps or repeated Eg08 runs, use process-level post-processing
parallelism instead of launching one analyzer after another.  Each worker owns a
separate case directory and YadonPy caps BLAS/OpenMP helper threads inside the
worker, which avoids the common "many Python jobs times many BLAS threads"
oversubscription problem:

```python
from yadonpy import InterfaceAnalysisTask, run_interface_analyses_parallel

tasks = [
    InterfaceAnalysisTask(
        name="-9 uC/cm2",
        work_dir="work_dir/charge_m9_uC_cm2/03_relaxation_sampling",
        manifest_path="work_dir/charge_m9_uC_cm2/02_system/layer_stack_manifest.json",
        penetration_species=("EC", "EMC", "DEC", "PF6", "Li"),
        adsorption_species=("EC", "EMC", "DEC"),
        split_electrodes=True,
        report_potential_drop=True,
        time_series_analysis=True,
    ),
]
batch = run_interface_analyses_parallel(tasks, workers="auto", thread_limit=1)
```

Future enhanced-sampling setup can start from the relaxed layer-stack artifacts.
For a CMC-Na interface, `prepare_solvated_ion_pull(...)` selects a Li+ whose
initial solvent-oxygen coordination is closest to four, writes a PLUMED pulling
file that moves that ion toward the `CMCNA` layer COM, and prints coordination
CVs for solvent O, CMC O, and anion F ligands:

```python
from yadonpy.interface import SolvatedIonPullSpec, prepare_solvated_ion_pull

pull_plan = prepare_solvated_ion_pull(
    system_dir=result.system_gro.parent,
    spec=SolvatedIonPullSpec(
        target_group="CMCNA",
        target_coordination_number=4,
        step1=500_000,
        kappa1_kj_mol_nm2=1000.0,
        print_stride=100,
    ),
)
# pass pull_plan.mdrun_extra_args to a biased GROMACS segment:
# ("-plumed", ".../plumed.dat")
```

For PMF production, use the umbrella workflow instead of hand-editing windows:

```python
from yadonpy import SolvatedIonUmbrellaSpec, prepare_solvated_ion_umbrella

umbrella_plan = prepare_solvated_ion_umbrella(
    system_dir=relaxed.work_dir / "02_system",
    gro_path=relaxed.final_gro,
    spec=SolvatedIonUmbrellaSpec(
        target_group="CMCNA",
        target_coordination_number=4,
        window_count=31,
        window_production_ns=1.0,
    ),
)
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
- Example 08 layer density settings are initial packing targets, not a claim
  that the immediate NVT local density is already physical.  The default sampled
  workflow uses fixed-XY, semi-isotropic z-NPT to remove vacuum-like gaps or
  over-compressed regions before the final NVT trajectory is analyzed.  Pass
  `relax_z=False`, or leave `relax_z="auto"` on stacks with explicit
  `VacuumLayerSpec` or `pbc_mode="xy"`, when the constructed vacuum spacing is
  part of the physical model and should not be barostat-compressed.
- Example 08-07 is the preferred template when a pre-relaxed CMC-Na membrane
  slab should define the lateral footprint.  It builds the CMC-Na slab with
  `periodicity="xy"` walls first, reads the relaxed slab XY box, selects matching
  basal-graphite repeat counts, prepares an electrolyte `periodicity="xy"` slab
  at the same XY and near-bulk density, and assembles both slabs through
  `MolecularLayerSpec(prepared_slab_gro=...)`.  Prepared slabs must match the
  final stack XY within `0.02 nm`; the manifest records XY deltas and lateral
  occupancy diagnostics so side-channel voids are caught before production.
  The stack-facing prepared slabs are wrapped in XY and open in z; chain pieces
  may appear split at the box edge in a single image, which is the correct
  periodic-film representation rather than a broken CMC chain.  CMCNA prepared
  slabs also have a hard lateral-occupancy gate (`>=0.85` total cells and
  `>=0.80` edge cells at the default grid), so a sparse wrapped slab fails fast
  instead of being silently treated as a dense membrane.
  Temporary electrolyte/CMC phase gates remain active during pre-release
  relaxation and are removed only for final NVT, so final NVT frame 0 is the
  interdiffusion `t=0`.
- Example 08 interface time-series animations are disabled unless an applicable
  post-processing call explicitly receives `time_series_analysis=True`.  When
  enabled, they sample up to ten equal trajectory windows by default and write
  slow MP4 overlays for molecule-COM z concentration, cation-centered RDF/CN,
  graphite-EDL RDF/CN, graphite/EDL charge-potential profiles, and adsorbed
  orientation-angle distributions when the required trajectory and species are
  available.  The charge-potential movie plots total/phase charge density,
  integrated charge per area, and the one-dimensional reference-shifted
  electrostatic potential derived from the sampled fixed charges.  The EDL RDF/CN movie uses
  only centers whose molecule COM lies inside the graphite EDL cutoff, chooses
  one strongest opposite-charge site per target molecule, draws CN as dashed
  curves on a 0-6 right axis, and labels the first RDF peak position.  The same
  plotting frames are written as PNG images under `time_series/frames/` before
  MP4 encoding.  MP4 writing uses the pip-installed `imageio-ffmpeg` executable
  when available; CSV and PNG artifacts are still written if the movie writer is
  unavailable.

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
  electrolyte, and CMC-Na layer stacks with interface-specific post-processing,
  including large flat DP=20 CMC-Na graphite sandwich templates and a CMC-first
  xy-slab route where graphite and pre-relaxed electrolyte lateral dimensions
  are matched to the relaxed CMC-Na slab.
- `examples/09_oplsaa_assignment`: OPLS-AA assignment diagnostics.
- `examples/10_migration_analysis`: migration-analysis workflow entry point.
- `examples/11_segment_branch_polymer`: segment-first long-block and branched
  polymer construction.
- `examples/12_enhanced_sampling`: umbrella PMF for Li+ transfer from
  electrolyte into CMC-Na, with metadynamics and Blue Moon placeholders.

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
