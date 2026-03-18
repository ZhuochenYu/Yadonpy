## 0.4.35
- Fix: Example 11 script corrected (removed stray `ac = system`, uses `EQ21step`, correct `gpu` flag).
- Enhancement: Example 10 accepts an optional CSV path argument (defaults to `template.csv`), README aligned.
- Robustness: `poly.set_linker_flag` supports 1-linker terminal units but fails fast on >2 linkers to avoid silent wrong polymerization.

## 0.4.34
- Fix: restore GAFF/GAFF2 forcefield modules (remove accidental indentation corruption that broke classmethods and typing rules).
- Fix: MolDB now persists and restores PSMILES connector isotopes (linker H with isotope>=3) via `manifest.json` field `connectors`, preventing polymerization/analysis loops after DB reload.
- Enhancement: Example 10 now writes `index.csv` mapping MolDB key <-> name <-> smiles for traceability during batch DB builds.
- Maintenance: tighten FF assignment patterns in examples to consistently check `result = ff.ff_assign(...)`.

## 0.4.30
- Fix: re-export `qm` from package root so `from yadonpy import qm` works (examples).

## 0.4.27
- Introduce shared molecule database (MolDB) that stores canonical SMILES/PSMILES, initial geometry (mol2), and charges.
  Force-field assignment remains on-demand (fast) and is intentionally not stored in the DB.
- Add GAFF2.mol(...) and GAFF2.chg_assign(...) convenience classmethods that use the shared MolDB (GAFF2_mod inherits them).
- Add examples for batch DB population from CSV and for building/running a polymer solution workflow using precomputed DB entries.

## 0.4.29
- Fix: GAFF2.mol / shared MolDB API availability.
- Update: Example 10 uses Example 01 conformer search + RESP charge assignment, then stores to MolDB.
- Update: Polymer packing densities in examples adjusted (low initial packing density).


## 0.4.28
- Fix: example 08 assigns polymer force field via `result = ff.ff_assign(CMC)` (and checks result).
- Fix: treat RDKit default name "polymer" as a placeholder so exported MOL2 names follow caller variable names (e.g. `copoly.mol2`).
- Docs: add/refresh README for all examples (purpose, run, outputs, analysis).

## v0.4.26 (2026-02-05)

### Features
- Polymer equilibrium checking now always applies an Rg convergence gate (ported from yzc-gmx-gen), even when Density is available.
  - Added `yadonpy.gmx.analysis.rg_convergence` with robust plateau detection + diagnostics SVG plotting.
  - `06_analysis/plots/rg_convergence.svg` is generated for polymer systems (best-effort), and `equilibrium.json` now includes an `rg_gate` summary.

## v0.4.25 (2026-02-05)

### Fixes
- Fix `gmx mdrun -v` console progress: preserve carriage returns so progress updates in-place on a single line.
- Fix additional equilibration analyze artifact resolution to avoid false non-convergence loops.
- Fix `AnalyzeResult.get_all_prop()` calling convention for `stats_from_xvg` (keyword-only `col`).

### Features
- Add `eq.NVT` production preset with optional density control:
  - Compute mean density over the last fraction of the previous equilibrium (default 30%).
  - Scale starting box + coordinates using `gmx editconf -scale` to match the target density.
- Add new example: `examples/09_polymer_electrolyte_nvt` (NVT production variant of example 03).

## 0.4.23 (2026-02-04)

- Fix (critical): make equilibrium/convergence checks robust to missing or relocated artifacts.
  - `AnalyzeResult` now resolves missing `.tpr/.xtc/.edr/.trr` by searching under `work_dir` (prefers newest match), avoiding false "cannot check equilibrium" outcomes that caused repeated additional rounds.
- Fix: `gmx energy` term probing now captures combined stdout/stderr to reliably parse term lists across GROMACS versions/environments.

## 0.4.22 (2026-02-04)

- Fix: stream `gmx` live output in binary mode to preserve carriage-return (`\r`) progress updates; `gmx mdrun -v` now updates in-place on a single line instead of producing extra lines.
- Change: additional equilibration rounds now skip rebuild stages when a prior equilibrated structure exists.
  - Added a判定标签 `_skip_rebuild` in `eq.Additional.exec()`; when true, only the final equilibration stage (`04_md`) is executed.
- Fix/Change: equilibrium checking now matches thermo terms by substring to handle GROMACS term-name variations (e.g., units in column headers).
- Change: convergence plots are generated even when the system does not reach equilibrium (best-effort thermo/Rg plots).

## 0.4.21 (2026-02-04)

- Change: `write_mol2_from_rdkit()` now supports `name=...` to control the output filename stem.
  - Default behavior: when `name` is omitted, the filename is inferred from the caller's Python variable name (best-effort), e.g. `mol=copoly` -> `copoly.mol2`.
  - `mol_name` keeps its role as the in-file MOL2 residue/molecule name; if omitted, it defaults to the resolved `name`.
- Fix: `yadonpy.io.artifacts` now calls `write_mol2_from_rdkit()` with keyword arguments (function is keyword-only).
- Change: all bundled examples now default to `mpi = 1` and `omp = 16` (previously `mpi = 16`, `omp = 1`).

## 0.4.20 (2026-02-04)

- Fix: resolve `IndentationError: unexpected indent` in `yadonpy/gmx/engine.py` (broken `list_energy_terms()` indentation).
- Change: all EM-related `gmx mdrun` runs now request `-nb gpu` by default (including workflows that otherwise keep bonded/PME/update on CPU).
  - Added `nb="gpu"|"cpu"` override parameter to `GromacsRunner.mdrun()` to control nonbonded placement independently.

## 0.4.19 (2026-02-04)

- Fix: always add `-v` to `gmx mdrun` commands (help-text option detection was unreliable on some clusters).

## 0.4.18 (2026-02-04)

- Fix: use unique temp filename for `gmx energy` term listing and robust cleanup.

## 0.4.17 (2026-02-04)

- Change default GROMACS mdrun pinning from `-pin on` to `-pin auto`.

## 0.4.16 (2026-02-04)

- Fix: SyntaxError in `yadonpy.__init__` caused by missing newline between `__version__` and `__all__`.


## 0.4.15 (2026-02-04)

- Change: all `gmx mdrun` pinning flags switched from `-pin on` to `-pin auto`.


## 0.4.14 (2026-02-04)

- Fix: `write_mol2_from_rdkit` now infers names from caller variable names correctly (prevents `mol.mol2` overwrites in 00_molecules).


## 0.4.11 (2026-02-04)
- Restored GPU-offload defaults for all equilibration stages (including minimization) when GPU is enabled.
- `gmx mdrun` now always adds `-v` when supported and streams progress output to the console.

## 0.4.13 (2026-02-04)
- EQ workflow: run EM stages with minimal `gmx mdrun` args (`-deffnm <name> -ntmpi 1 -ntomp <N> -nb gpu -gpu_id <id>`), and always include `-v`.
- All `gmx mdrun` invocations now include `-v` for live console progress output.

## 0.4.10

- Add `poly.ratio_to_prob()` helper used by Example 08 to convert feed ratios to probabilities.

## 0.4.9

- Auto-correct per-atom partial charges on export so the molecule net charge matches the intended integer formal charge (or user-provided total_charge).
- Conductivity: hide the whole sigma block for systems without ionic species; keep sigma.json note for reproducibility.
- Summary outputs: prune raw artifact paths (xvg/gro/edr/tpr/xtc/trr) in summary.json, keep only plot basenames.

## 0.4.8

- Fix: gmx current PBC temp rewrite now checks output exists before replacing (prevents md.gro.pbc_tmp rename error).
- Fix: 02_system no longer generates charged_mol2 (use 00_molecules cache instead).
- Refactor: export_system_from_cell_meta writes per-species artifacts directly under 02_system/molecules (no from_cell_meta duplication).
- Fix: EH conductivity falls back to parsing gmx current stdout when -dsp has too few points.

## 0.4.7 (2026-02-04)
- Fix molecule cache artifact naming to avoid 'mol.*' overwrites when RDKit default names are present.
- MSD plots: highlight diffusion-fit window with a light band.
- Add Rg convergence plot output.
- Plot styling: prefer Helvetica/Arial-like sans-serif fonts.
## 0.4.6 (2026-02-04)

- Fix unnamed-molecule artifact overwrite: when variable-name inference fails (e.g., molecules created in loops/lists), assign a deterministic auto-name (formula/hash) instead of the generic fallback, preventing `mol.mol2` / `mol.itp` collisions.
- Improve MOL2 interoperability: write Tripos atom types as `Element.label` (e.g., `C.c3`, `O.o`) to help third-party importers parse elements correctly.

## 0.4.5 (2026-02-04)

- Fix molecule artifact naming regression: ignore RDKit default name "mol" and infer stable names when building cells to avoid mol2/itp overwrite.
- Add optional `name` parameter to polymer builder helpers (`random_copolymerize_rw`, `terminate_rw`) and persist explicit names in `mol_from_smiles(name=...)`.

## 0.4.4 (2026-02-04)

### Fixed
- Fix `export_system_from_cell_meta()` failing with "Packed cell coordinates mismatch" when RDKit fragment ordering differs from the requested species order.
  The exporter now matches packed fragments to species using an order-independent molecule signature (atom count + atomic-number composition), with a clear diagnostic error if matching fails.

## 0.4.3 (2026-02-04)

### Fixed
- Fix `NameError: os is not defined` in `core.system.cpu_count()` by importing `os`.

## 0.4.0 (2026-02-04)

## 0.4.2 (2026-02-04)

### Fixed
- Fix `NameError: const is not defined` in `core.logging_utils.yadon_print()` by importing `core.const` and `YadonPyError`.

## 0.4.1 (2026-02-04)

### Fixed
- Keep GAFF2 and GAFF2_mod as two distinct force fields with separate parameter DBs (gaff2.json vs gaff2_mod.json); removed stale per-module `__version__` strings to avoid version confusion.

### Docs
- Example 08 notes how to switch between GAFF2 (classic) and GAFF2_mod (modified) force fields.


- Examples: add **Example 08** (CMC-Na random copolymer + 1M LiPF6 in EC/EMC/DEC with 1:1:1 mass ratio), including:
  - Random copolymerization from 4 glucose-based monomers (anionic substituents) + Na+ counter-ions for charge neutrality
  - 1 M LiPF6 with minimum ion pairs safeguard and density-based volume estimate
  - Species-wise charge scaling (polymer + ions scaled by 0.8) and Li+ centered RDF/MSD/sigma analysis stub
- Fix: `core.molops` now imports required RDKit/const symbols (restores importability after the utils refactor).

## 0.3.14 (2026-02-04)

- Refactor: split the legacy `core.utils` "mega-module" into focused submodules:
  - `core.exceptions`, `core.logging_utils`, `core.system`, `core.topology`, `core.molops`,
    `core.serialization`, `core.chem_utils`, `core.rdkit_io`, `core.naming`.
- Compatibility: `core.utils` remains as a thin facade re-exporting public symbols to avoid breaking imports.
- Maintainability: clearer dependency boundaries reduce import weight and improve testability.

## 0.3.13 (2026-02-04)

- Fix: unify version source across the package (use `yadonpy.__version__`).
- Security: `utils.pickle_load()` documents pickle risks and supports `allow_unsafe=` guard (default keeps legacy behavior).
- Robustness: minor type checks (`isinstance`) and path handling improvements.

## 0.3.12 (2026-02-03)

- Feature: consistent, user-friendly naming for molecules and exported artifacts.
  - New helpers `utils.get_name()` / `utils.ensure_name()` / `utils.named()`.
  - `poly.amorphous_cell()` now infers default molecule names from Python variable names (copoly, solvent_A, ...) when no explicit name is set.
  - `qm.assign_charges()` now defaults its log folder/name to the molecule name (variable-name inferred), matching `conformation_search`.
- Improve: `io.write_mol2_from_rdkit()` now supports `out_dir=` and will default the MOL2 file name to `<name>.mol2`.

## 0.3.9 (2026-02-03)
- Fix: avoid missing [ angles ]/[ dihedrals ] in Example 01 by caching per-molecule artifacts before packing and copying them during system export.
- Fix: indentation error in io/artifacts.py.

## 0.3.11 (2026-02-03)

- Fix: ensure polymers generated by random_copolymerize_rw are assigned a distinct molecule name (POLYMER) instead of inheriting monomer names, preventing system export from producing monomer_A.itp for polymers.
- Improve: amorphous_cell early caching now prefers yadonpy naming props for artifact caching.


## 0.3.8 (2026-02-03)

- Fix: Example 01 mixed-system export could produce bonds-only `.itp` for cell fragments (missing `[ angles ]` / `[ dihedrals ]`) because `Chem.GetMolFrags(..., asMols=True)` drops Python-level bonded containers. We now rebuild bonded terms for fragment representatives before writing molecule artifacts.
- Refactor: centralize bonded-term reconstruction in `io.artifacts` and remove duplicated regeneration logic from the low-level `.itp` writer.

## 0.3.7 (2026-02-03)
### Fixed
- **Root-cause fix for missing `[ angles ]` / `[ dihedrals ]` in Example 01**: `core.utils.deepcopy_mol()` no longer relies on RDKit's pickle/deepcopy semantics (which drop arbitrary Python attributes). It now clones the RDKit molecule via `Chem.Mol(mol)` and explicitly preserves YadonPy's Python-side bonded-term attributes (`angles/dihedrals/impropers/cmaps`) and `cell` when present.

### Improved
- `io.artifacts.write_molecule_artifacts()` now tags molecules with `_yadonpy_artifact_dir` (best effort) to simplify downstream workflows that want to reuse cached per-molecule artifacts without re-deriving paths.


## 0.3.6 (2026-02-03)
- Fix: robust regeneration of angles/dihedrals when RDKit Mol loses Python-level attributes during cell/meta workflows (Example 01).

### Fixed
- **Severe topology bug**: added robust validation + forced regeneration when cached/basic_top `.itp` is missing `[ angles ]` / `[ dihedrals ]` (expected for >=3 / >=4 atom molecules).
  - Validation is performed both when loading from library and when exporting a multi-species system.
- Added a short **FF assignment summary** print after `ff_assign`:
  `FF assign summary (...): atoms=... bonds=... angles=... dihedrals=... impropers=...`
  to make missing bonded terms immediately visible.
- Fix: `write_gromacs_single_molecule_topology()` bonded-patch stage could crash with a Python `SyntaxError` due to a missing `except/finally` block.
  - This was triggered during `export_system` in the polymer-electrolyte example (Example 03).

### Behavior
- All `gmx mdrun` invocations run in verbose mode (`-v`) by default.

## 0.3.0 (2026-02-03)

### Added
- Auto-plotting (SVG-first, yzc-gmx-gen style): workflows and analyzer now generate `plots/` folders automatically.
  - QuickRelax / Equilibration: `thermo.svg` + split plots
  - Tg scan: per-T density traces + global Tg curve plot
  - Elongation: `stress_strain.svg`
  - Analyzer: MSD plots + overlay, RDF/CN plots, number-density profile plots, EH-fit plots

### Fixed
- `workflow.steps.elongation_gmx` parameter mapping updated to match `ElongationJob` (keeps `pressure_bar` for backward compatibility).

## 0.2.25 (2026-02-03)

### Fixed
- Fix: Inorganic anionic polyions (PF6⁻/BF4⁻/ClO4⁻/AsF6⁻, etc.): modified-Seminario Hessian now **follows the RESP/ESP single-point basis** (`charge_basis/charge_basis_gen`) by default. (Previously referenced an undefined variable and could crash.)

## 0.2.17 (2026-02-02)

### Fixed
- Analysis: `gmx msd` can fail on trajectories with frame spacing > 10 ps due to the
  default `-trestart 10` in recent GROMACS. YadonPy now passes a conservative
  large `-trestart` value by default to avoid unnecessary failures.

## 0.2.16 (2026-01-31)

### Performance
- Prefer GPU offload for production stages by default (NVT/NPT/MD): `-nb gpu -bonded gpu -pme gpu -pmefft gpu -update gpu`.
- Automatically fall back to `-update cpu` and retry once when GROMACS reports GPU-update incompatibility (constraint/update-group restrictions).
- Energy-term probing is now cached to avoid redundant `gmx energy` list calls.

### Examples
- Fixed indentation / syntax issues in example scripts (all examples now pass `py_compile`).

## 0.2.13 (2026-01-31)

### Bug fixes
- Fix Python 3.9 compatibility: remove `X | None` annotations in favor of `Optional[...]`.
- Robust basis fallback for inorganic ions: if `ma-def2-*` basis sets are unavailable in the local Psi4 build, automatically fall back to `def2-*` diffuse variants (SVPD/TZVPD/TZVPPD) without aborting.
- Do not retry Psi4 optimization with `engine=geometric` for basis-not-found errors; also skip geomeTRIC retry when the `geometric` Python package is not installed.


## 0.2.11 (2026-01-31)

### QM / RESP defaults

- RESP default level updated: default **no longer uses HF**.
  - Default RESP/ESP single point: `wB97X-D3BJ / def2-TZVP`.
  - For anions, prefer diffuse def2/ma-def2 basis (fallback ladder).
- Psi4 stage logs now print **name/charge/multiplicity/method/basis/smiles** for OPT/RESP/Hessian.

### Inorganic polyatomic ions (PF6-, etc.)

- When `auto_level=True` (default), PF6- / BF4- / ClO4- style ions use:
  - OPT: prefer `wB97X-D3BJ / ma-def2-SVPD`
  - RESP single point: prefer `wB97X-D3BJ / ma-def2-TZVPPD`
  with robust basis fallback.
- When `bonded_params='auto'` (default), generate **bond+angle** via modified Seminario (mseminario)
  from Psi4 Hessian, and **patch** the resulting constants into the exported `.itp`.

### Docs / Examples

- Manual updated with the new RESP defaults and inorganic-ion strategy.
- Add `examples/02_pf6_only/run_pf6_basic_top.py` as a standalone PF6- test.

## 0.2.10 (2026-01-30)

### Fixes

- Robust EM for packed systems (yzc-gmx-gen style): minimization now runs as
  **steep (constraints=none)** → **steep (constraints=h-bonds)** → **cg** (optional).
  This prevents early-stage LINCS/constraint failures that caused `cg` to abort.
- EM CG template now supports `emstep` (default smaller step) for improved stability.

### GROMACS hygiene

- Post-stage PBC cleanup remains enabled: `gmx trjconv -pbc mol -center -ur compact`
  is applied best-effort to stage `.gro` and `.xtc` in-place to avoid broken-molecule artifacts
  in visualization / mol2 conversions.

## 0.2.6 (2026-01-29)

### QM / RESP

- Explicitly charged (p)SMILES route:
  - Build initial 3D geometry using **OpenBabel + UFF** (when available)
  - Run **Psi4 geometry optimization + RESP** (opt=True)
  - Keep the neutral-molecule route unchanged (MM conformer search + optional DFT, then RESP).

### Docs / Examples

- Installation hints now include:
  - `conda install openbabel`
  - `pip install pybel`
- Example 01 README documents the charged-SMILES OPT+RESP behavior.

## 0.1.40 (2026-01-29)

### Performance

- GROMACS GPU acceleration defaults (mdrun): when GPU is enabled, YadonPy now calls:
  `-nb gpu -bonded gpu -update gpu -pme gpu -pmefft gpu -pin auto` (when supported by your GROMACS build).
  This matches the proven defaults used in yzc_gmx_gen.

### Fixes

- Robust stage restart: multi-stage workflows (EQ21, Tg, elongation, quick relax) can now continue from
  `md.tpr + md.cpt` via `mdrun -cpi md.cpt -append` (instead of silently restarting a stage from scratch).
- More predictable GROMACS executable auto-detection: prefer `gmx` (thread-MPI) over `gmx_mpi` by default.
  Override with `export YADONPY_GMX_CMD=...`.

## 0.1.37 (2026-01-29)

### Fixes

- PF6-/BF4-/ClO4- and similar small inorganic ions:
  - Prefer OpenBabel (pybel) 3D builder + UFF local optimization when available; fallback to RDKit embedding, then geometric templates.
  - Skip MM/DFT conformer search and DFT geometry optimization (OptKing internal-coordinate failures on highly symmetric ions).
- QM stage logging: `qm.conformation_search()` and `qm.assign_charges()` now print the current purpose (`log_name`) and the originating SMILES (from `_yadonpy_input_smiles` when available).

### Fixes

- Fix: `qm.conformation_search()` now keeps only the lowest-energy conformer (ID 0) to avoid RDKit `CombineMols` warnings/errors during random-walk polymerization when monomers have different numbers of conformers.
- Fix: Example 01 now calls `write_mol2_from_rdkit()` using keyword arguments (the function is keyword-only).

### Examples

- Cleanup: remove unused `run_eq.py` and `run_full_workflow_expanded.py` from `examples/01_full_workflow_smiles/`.

## 0.1.33 (2026-01-29)

### Fixes

- Fix: keep GAFF2 and GAFF2_mod as two distinct force fields (no aliasing/override in examples). GAFF2 now always refers to `ff/gaff2.json`, while GAFF2_mod refers to `ff/gaff2_mod.json`.

### Examples

- Update: Example 01 imports `GAFF2` from `yadonpy.ff.gaff2` and `GAFF2_mod` from `yadonpy.ff.gaff2_mod` explicitly.

## 0.1.32 (2026-01-28)

### Fixes

- Fix: `workflow.steps.resp_from_smiles()` now always re-applies force-field typing (`ff_assign`) on the returned molecule (prevents export-time re-parameterization).
- Fix: `api.parameterize_smiles()` auto-detects monoatomic ions and falls back to MERZ ion parameters when GAFF/GAFF2 is requested; QM/RESP is skipped and the formal charge is used.

### Features

- Feature: `workflow.steps.build_copolymer()` additionally exports a charged MOL2 (`out_mol2_name`, default `polymer.mol2`) for the generated polymer chain.

### Examples

- Update: Example 01 now assigns RESP + FF for solvents/ions via `steps.resp_from_smiles()`, and uses `MERZ()` for the cation force field.

# Changelog


## 0.4.12 (2026-02-04)

- Restore CPU-only execution for EM/minim stages in equilibration presets (GPU remains default for other stages).

## 0.3.10 (2026-02-03)

- Fix: system export now rewrites cached molecule artifacts to use species moltype names (filenames + [ moleculetype ] + residue names), preventing grompp errors like 'No such moleculetype monomer_A'.
- Fix: update example imports to use explicit FF classes (e.g., from yadonpy.ff.gaff2 import GAFF2).


## 0.1.38
- Inorganic ions (e.g., PF6-, BF4-) now keep OpenBabel-built 3D geometry and still run DFT optimization/RESP.
- Psi4 optimization: added robust retries (OptKing -> OPT_COORDINATES=CARTESIAN -> engine=geometric -> SP fallback when ignore_conv_error=True).

## v0.1.30 (2026-01-28)

### Fixes

- Extend inorganic/electrolyte anion 3D template support in `utils.mol_from_smiles()` for older RDKit builds:
  PF6-, BF4-, ClO4-, TFSI-, FSI-, and DFOB- (difluoro-oxalatoborate).
- Prefer rigid polyhedral ion templates (PF6-/BF4-/ClO4-) before ETKDG embedding to yield stable canonical geometries.

## v0.1.29 (2026-01-28)

### Fixes

- Make `utils.mol_from_smiles()` robust for inorganic ions / highly charged species on older RDKit builds.
  In particular, embedding PF6- (SMILES like `F[P-](F)(F)(F)(F)F`) could fail and crash basic-top generation.
  Added multiple fallbacks: higher-attempt random ETKDG, template geometries (PF6-, monoatomic ions), and a
  final spaced-out coordinate fallback to keep the pipeline running.

## v0.1.28 (2026-01-28)

### Features

- Introduce high-level workflow step APIs in `yadonpy.workflow.steps` to keep user scripts clean.
  All step functions accept `restart=<bool>` and use a unified restart state monitor.
- Unify GPU control semantics across GROMACS workflows:
  - `gpu` (0/1) switches GPU on/off
  - `gpu_id` selects which GPU card to use when GPU is enabled.

### Examples

- Rewrite examples 01–05 to the new workflow style (single `restart_status` + clean step calls).
  Example 01 provides both a concise script and an expanded script.

## v0.1.26 (2026-01-28)

### Fixes

- Fix `NameError: _safe_set_conf_positions is not defined` in `core.calc.conformation_search()`.
  Added a local safe coordinate setter to robustly write optimized coordinates back into RDKit conformers.

## v0.1.25 (2026-01-28)

### Fixes

- Restore Python 3.9 compatibility by removing PEP 604 union type hints (e.g. `int | None`, `Path | str`).
  Replaced with `typing.Optional` / `typing.Union` across the codebase.
- Remove accidentally bundled `__pycache__` / `*.pyc` artifacts from the source distribution.

## v0.1.21 (2026-01-28)

### Fixes

- Fix RDKit/MMFF coordinate extraction robustness in `core.calc.conformation_search()`.
  Some RDKit builds return forcefield positions as 1D object arrays or flat vectors;
  this could crash with: `IndexError: too many indices for array`.
  - Added coercion of MMFF/UFF positions into a strict `(natoms, 3)` float array.
  - Added a safe coordinate setter that never aborts the whole workflow if a
    conformer's coordinates cannot be coerced.

## v0.1.19 (2026-01-28)

### Features

- Add resumable charge cache for QM/RESP: `sim.qm.assign_charges()` now also writes
  `charged_mol2_qm/<name>.charges.json` (per-atom charges). Provide helper
  `yadonpy.sim.qm.load_atomic_charges_json()` to reload charges into an RDKit Mol.

### Examples

- `examples/01_full_workflow_smiles/run_full_workflow.py` becomes resumable:
  - If `work_dir/amorphous_cell.sdf` exists, the script loads it and skips upstream steps.
  - If charge cache (JSON or legacy MOL2) exists, it skips the corresponding RESP step.
  - PF6- basic_top forcefield changed to `gaff2_mod`.

### Docs

- Manual updated to v0.1.19.

## v0.1.18 (2026-01-28)

### Fixes

- Fix RDKit MM pre-optimization crash in `core.calc.conformation_search()` when MMFF forcefield is unavailable:
  `AttributeError: 'NoneType' object has no attribute 'Minimize'`.
  - Add robust fallback chain: **MMFF → UFF → no-op (energy=0, keep geometry)**.
  - Also apply the same logic to the parallel worker (`_conf_search_rdkit_worker`).

### Robustness

- For very small molecules (single-atom or no-bond terminators), automatically set `nconf=1` and disable DFT stage
  (`dft_nconf=0`) to avoid pointless large conformer generation.

### Docs

- Manual updated to v0.1.18.

## v0.1.17 (2026-01-28)

### Fixes

- Fix QM conformation search crashing with: `TypeError: QMw() got multiple values for keyword argument 'omp'`.
  - Sanitize accidental MD runtime kwargs (`omp/mpi/gpu/...`) before calling `QMw/Psi4w`.
  - Treat legacy `omp` as `psi4_omp` if `psi4_omp` is not provided.

### Examples

- Update `examples/01_full_workflow_smiles/run_full_workflow.py`:
  - Set the first monomer pSMILES to `*CCO*`.
  - Do not pass MD `mpi/omp` parameters into `qm.conformation_search()` (use `psi4_omp` / `psi4_mp` instead).

### Docs

- Manual updated to v0.1.17.


## v0.1.16 (2026-01-28)

### Fixes

- Fix `yadonpy.diagnostics.doctor()` crash: `DataLayout` now provides `gmx_forcefields_dir`.
  - Default: `$YADONPY_DATA_DIR/ff/gmx_forcefields`
  - `ensure_initialized()` creates the directory if missing.

## v0.1.15 (2026-01-28)

## v0.1.27 (2026-01-28)

### Fixes

- Fix polymerization restart path: do not reload polymerizable monomers from MOL2 because MOL2 does not reliably preserve isotope markers ("*" -> [3H]) in RDKit 2020.x.
  - Example 01 now recreates monomers from the original pSMILES and reloads charges from `charged_mol2_qm/*.charges.json`, preserving head/tail linker atoms.
- Make `poly.check_3d_proximity()` robust to empty distance matrices to avoid NumPy `zero-size array` reduction crashes during rare failed connection attempts.

### Fixes

- Fix Python 3.9 `dataclasses` field ordering error in `yadonpy.sim.analyzer.AnalyzeResult` (example import crash).

### Robustness & diagnostics

- Molecule artifact generation (`.itp/.gro/.top`) is now **fail-fast** instead of silently swallowing exceptions.
  - On failure, yadonpy writes `gromacs_error.txt` (with traceback) into the artifact directory.
  - The same error is recorded as `gromacs_error` in `meta.json`.
- Charge assignment now has a safety fallback: if a QM/RESP-style charge method is requested but fails (missing deps or runtime error), yadonpy falls back to `gasteiger` and prints a clear warning.
- `system.ndx` naming is more compatible: generate both `<moltype>` and `MOL_<moltype>` groups.
- System export metadata unifies a `moltype` field and improves error messages when reconstructing species from `cell_meta` fails.

### Docs

- Manual updated to v0.1.15 and includes the above behavior changes.