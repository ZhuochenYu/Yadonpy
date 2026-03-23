## 0.8.24 (2026-03-20)

- Interface assembly now performs an explicit post-canonicalization geometry validation pass and stores the result in `03_interface/system_meta.json`, checking that atoms stay inside the primary box and that the assembled slab gap is not strongly negative before MD.
- Focused route-A regressions now assert that the new geometry validation remains clean for both normal assembly and the large lateral-shift whole-molecule case.

## 0.8.23 (2026-03-20)

- Interface assembly now wraps laterally shifted top-slab molecules back into the XY box as intact molecular blocks after the phase shift, which directly targets remaining post-splice boundary risk without reintroducing the old per-atom wrapping bug.
- Slab preparation now always re-wraps lateral fragment coordinates after XY fitting, even when lateral recentering is disabled, so strained/replicated slabs stay inside their target lateral box before being written.
- Charge-audit formatting has been promoted into reusable interface helpers and Example 12 now consumes those shared helpers instead of carrying script-local audit code.

## 0.8.22 (2026-03-20)

- Example 12 defaults are now intentionally smaller: fewer CMC chains, a denser initial CMC pack, a smaller electrolyte probe, fewer minimum salt pairs, and thinner default route-A slabs. The example should stop over-building the starting bulk boxes for interface studies that only need the CMC footprint plus a matched electrolyte layer.
- Example 12 now applies a fixed-XY semiisotropic flattening NPT to the relaxed CMC bulk before using its XY footprint to resize the electrolyte bulk, which better matches the intended workflow of "CMC bulk first, then electrolyte aligned to CMC XY".
- Charge auditing is now printed throughout Example 12 for packed cells, equilibrated cells, exports, slabs, and the assembled interface, and the script explicitly warns users not to add a second manual pre-splice `-pbc mol` on top of InterfaceBuilder's own snapshot whole-molecule handling.

## 0.8.21 (2026-03-20)

- Export performance: EQ21 raw/scaled `02_system` exports no longer spend time writing a full-box `system.mol2`, which removes another avoidable stall for large sparse bulk systems while keeping later stage-level MOL2 generation intact.
- Charge diagnostics: packed-cell net-charge detection now recognizes RESP-only and other non-`AtomicCharge` atom properties, which prevents false large-charge warnings when ions or other species were parameterized through alternate charge fields.
- Interface robustness: representative bulk snapshots and the assembled interface `system.gro` now get a topology-guided whole-molecule canonicalization pass before interface MD, and the interface-build schema was bumped so restart rebuilds pre-fix geometries.

## 0.8.20 (2026-03-20)

- Interface robustness: stage-to-stage `md.gro` handoff now gets a topology-guided whole-molecule canonicalization pass, so wrapped polymers/ions are normalized before the next `grompp` step reads them.
- Interface MDP defaults: `periodic-molecules = yes` is now enabled for the staged interface protocol, which improves robustness for large route-A style interfaces whose molecules span periodic boundaries during pre-contact relaxation.
- Release/tests: added focused regressions for both behaviors and bumped package/docs metadata to `0.8.20`.

## 0.8.19 (2026-03-20)

- Progress display: random-walk `tqdm` bars are visible again by default; the old hard-coded global disable flag has been replaced with an explicit environment override `YADONPY_DISABLE_TQDM=1`.
- Script compatibility: common force-field `ff_assign(...)` calls now return the assigned molecule object on success and `False` on failure, which makes chained usage such as `ff.ff_assign(ff.mol(...))` compatible with existing boolean-style callers.
- Release/tests: added focused regressions for both behaviors and bumped package/docs metadata to `0.8.19`.

## 0.8.18 (2026-03-20)

- Progress display: fixed a regression where `core.const.tqdm_disable` was hard-coded to `True`, which suppressed the random-walk construction progress bars even in normal terminal runs. `tqdm` is now on by default again.
- Runtime control: if a silent run is desired, progress bars can now still be disabled explicitly through `YADONPY_DISABLE_TQDM=1`.
- Release/tests: added a focused regression for the default/override behavior and bumped package/docs metadata to `0.8.18`.

## 0.8.17 (2026-03-20)

- Export performance: non-interface EQ21 setup no longer pays for two full `export_system_from_cell_meta(...)` passes when both raw and charge-scaled systems are needed. The raw export is now treated as the canonical build, and the scaled export reuses its molecules, coordinates, and system index while regenerating only the charge-scaled topology files.
- Regression coverage: added a focused test that ensures the fast path reuses raw artifacts instead of re-entering FF/MolDB generation, and verified nearby naming/charge-correction export regressions still pass.
- Release/process: bumped package/docs metadata to `0.8.17` and keep the “every fix ships as a new version” rule.

## 0.8.16 (2026-03-20)

- Core/poly: random-walk retry-budget estimation no longer depends on the removed `Descriptors` import; the rigidity heuristic now uses direct molecule APIs plus `rdMolDescriptors`, which fixes the early `random_copolymerize_rw(...)` crash seen in Example 12.
- Release/process: this fix is published as a new `0.8.16` release rather than replacing `0.8.15`, so versioned archives stay aligned with the actual packaged code.
- Release/tests: revalidated the retry-budget and molecular-weight hardening regressions and bumped package/docs metadata to `0.8.16`.

## 0.8.15 (2026-03-20)

- Core/mass handling: molecular-weight estimation is now routed through a robust helper that first tries RDKit descriptors and then falls back to a property-cache-aware atom-wise sum. This removes the `getNumImplicitHs()` precondition crash from monoatomic Merz ions and other partially sanitized species.
- Examples/interface/runtime: Examples 05, 10, 11, and 12 plus internal box/mass statistics code no longer call `Descriptors.MolWt(...)` directly on runtime-built molecules, and random-walk rigidity estimation no longer depends on the removed `Descriptors` import. Li/Na/PF6 preparation and early random-walk retry-budget selection are no longer sensitive to RDKit descriptor cache state.
- Release/tests: added focused regressions for Li+/Na+ and unsanitized PF6-like mass estimation and bumped package/docs metadata to `0.8.15`.

## 0.8.14 (2026-03-20)

- Interface/bulk resize: a new electrolyte-aware planner now rescales grouped solvent species together and preserves coupled salt pairs, instead of independently rounding every species after volume scaling. This is a better default for electrolyte interface campaigns because it better preserves the intended solvent recipe and salt concentration while still shrinking the box to the polymer-matched XY footprint.
- Interface/bulk resize: the same planner now also supports grouped solvent families plus multiple salt-pair groups, and the new `recommend_electrolyte_alignment(...)` helper derives resized-electrolyte target Z plus fixed-XY semiisotropic NPT duration from route geometry instead of using repeated hard-coded margins in each example script.
- Examples/interface: Examples 10, 11, and 12 now all use the probe-and-resize electrolyte workflow plus a fixed-XY semiisotropic follow-up NPT, so the lower-particle-count interface construction strategy is applied consistently across the bundled interface routes.
- Release/tests: added grouped electrolyte resize regressions, mixed-solvent and multi-salt edge-case coverage, and refreshed the README/manual/API walkthrough so the three interface examples are documented as one shared workflow with route-specific differences.

## 0.8.13 (2026-03-20)

- Interface/bulk resize: interface workflows can now build a probe electrolyte bulk first, recover its equilibrated box size and effective density, and rescale the electrolyte composition to a smaller target box matched to the polymer XY footprint before rebuilding the final electrolyte bulk. This addresses the oversized-electrolyte problem in large interface campaigns by reducing particle count without inventing a target density from scratch.
- Sim/preset/eq: `eq.NPT(...).exec()` now accepts MDP overrides, and `_find_latest_equilibrated_gro(...)` prefers `05_npt_production/01_npt/md.gro`, which makes it practical to run a final fixed-XY semiisotropic NPT stage and have later interface steps automatically consume that post-alignment box.
- Examples/tests/release: Example 12 now uses the probe-and-resize electrolyte workflow with a fixed-XY semiisotropic follow-up NPT, and regression coverage was added for the new bulk planning helpers and NPT override path. Package/docs metadata were bumped to `0.8.13`.

## 0.8.12 (2026-03-19)

- Interface/builder: large prepared slabs and assembled interfaces can now exceed the 5-digit GRO display field safely. The writer wraps displayed residue/atom indices to fixed-width GRO columns, and the reader reconstructs atom ids from line order with a fallback coordinate parser for legacy overflowed lines, which fixes the `ValueError: could not convert string to float` failure seen when re-reading very large slab `.gro` files during interface assembly.
- Interface/builder: optional `numba.jit(nopython=True)` kernels now accelerate both dense-window slab scoring and charge-rebalance candidate picking when `numba` is installed, while keeping the previous NumPy/Python path as an automatic fallback; slab logs also report whether JIT acceleration is active, and the builder schema was bumped so restart does not silently reuse pre-JIT slab artifacts.
- Interface/builder: charge rebalance no longer iterates over a shrinking Python `remaining` list and recomputes per-fragment priorities one object at a time; it now precomputes scaled charges, COMs, and axial bounds, filters to only sign-opposed candidates that can actually improve neutrality, and ranks them with vectorized array operations before appending the best nearby counter-fragment. This cuts another large chunk out of slab-preparation time for big interface systems.
- Release/tests: added scaled-charge rebalance and large-index `.gro` safety regressions, exposed an optional `accel` extra for `numba`, and bumped package/docs metadata to `0.8.12`.

## 0.8.11 (2026-03-19)

- Interface/builder: slab preparation now does the Z-window selection and charge rebalance on the original bulk fragments before XY replication, then replicates only the fragments that actually belong to the slab; combined with the new sorted-array/prefix-sum dense-window scoring, this removes the worst object explosion during `Prepare bottom slab` for `3x3` / `4x4` lateral plans. Slab preparation also emits per-substep counts and elapsed timings so interface-build bottlenecks are visible directly in the console.
- Core/poly: `amorphous_cell(...)` restart now writes and reuses a stable workdir-local packed-cell cache in addition to the hashed random-walk cache, so completed AC box builds can be restored more reliably on continuation runs.
- Core/poly: packed-cell logs now distinguish restart cache misses, restart cache hits, and real packing work with explicit `[RESTART]`, `[SKIP]`, and `[PACK]` labels instead of making a resumed run look like a fresh placement loop.
- Examples: bundled example scripts no longer force their adjacent `src/` directory to the front of `sys.path`; editable installs and normal installed packages now win, with local-source fallback only when `yadonpy` cannot be imported from the environment.
- Release/tests: added regression coverage for restart skip labels and stable-cache fallback, and bumped package/docs metadata to `0.8.11`.

## 0.8.10 (2026-03-19)

- IO/GROMACS system: standalone bulk exports now absorb tiny residual system charge directly into the copied `02_system` ITPs after charge scaling, so cached molecule artifacts no longer leave `-0.00x e` net charge in nominally neutral boxes.
- Interface/builder: assembled interface topologies now derive `[ molecules ]` counts from the prepared slab topologies instead of slab metadata, which prevents stale metadata from turning a near-neutral interface into a `-24 e` topology.
- Interface/builder: top-slab lateral phase shifts no longer wrap coordinates atom-by-atom during final assembly, keeping bonded molecules whole and eliminating the downstream `largest distance between excluded atoms` `grompp` failure seen after pre-contact EM.
- Interface/builder: lateral box sizing now uses a joint replica/strain search instead of blindly taking the larger box length and `ceil`-replicating against it, so slabs that differ only slightly can stay at `1x1` with a small balanced rescale instead of exploding to `2x2`; bonded fragments are also unwrapped against topology before slab preparation so long polymers carry a consistent whole-molecule geometry into interface assembly.
- Interface/builder: slab preparation now rebalances charge-sensitive cuts by re-adding nearby counter-fragments, and the interface build schema was bumped again so stale pre-fix `03_interface` artifacts are rebuilt instead of silently reusing the old charged slab selection.
- Interface/protocol + GROMACS workflow: `01_pre_contact_em` is now explicitly unconstrained, unconstrained minimization stages skip the `steep_hbonds` bridge entirely, and interface preflight aborts early if the assembled interface still carries a large net charge before MD.
- Release/tests: added regression coverage for the optimized lateral sizing, topology-aware fragment unwrapping, slab-charge rebalancing, and unconstrained pre-contact minimization path, and kept package/docs metadata at `0.8.10`.

## 0.8.9 (2026-03-19)

- Core/poly: random-walk growth now does a local old-chain candidate filter before proximity checks, which reduces late-chain cost without relying on multiprocessing.
- Core/poly: retry defaults were tightened and now pass through a rigidity-aware budget cap so flexible chains avoid very large retry windows by default while rigid monomers still keep more fallback room.
- Interface/builder: the interface assembler now merges the two slab parameter includes into a single `molecules/ff_parameters.itp` and invalidates stale interface builds with a schema bump, so duplicate `atomtypes` parameter files are no longer emitted side-by-side.
- Interface/builder: mixed molecule ITPs that embed parameter sections before `[ moleculetype ]` are now normalized automatically, with those parameter blocks extracted into the merged `ff_parameters.itp` so hidden `[ atomtypes ]` directives cannot reappear after molecule includes.
- Interface/builder: interface `system.ndx` now also exports layered aliases such as `BOTTOM1_<MOLTYPE>` / `TOP1_<MOLTYPE>` and matching `TYPE`/`ATYPE`/`INST` groups while keeping the existing `BOTTOM_*` and `TOP_*` names for compatibility.
- Interface/builder: the long-running interface build step now prints formatted internal progress for target-box resolution, bottom slab prep, top slab prep, and final assembly.
- Interface/protocol: interface MD now validates both `system.top` and `system.ndx` before launching GROMACS, and if slab core groups are missing or empty it drops the pre-contact freeze block automatically instead of letting `grompp` fail.
- Interface/protocol + GROMACS workflow: pre-contact interface relaxation is now more tolerant of rough starting geometries by keeping EM helper runs CPU-only, falling back from failed `steep_hbonds` / CG constraint-sensitive minimizations to unconstrained steep minimization, and using a smaller-step no-constraints pre-contact NVT stage.
- Tests/docs/release: added focused regression coverage for the new random-walk acceleration path and bumped package/docs metadata to `0.8.9`.

## 0.8.7 (2026-03-19)

- Interface/builder: interface `system.ndx` files now include richer region-aware grouping, including per-region moltypes, per-region/per-moltype atomtypes, per-instance molecule groups, and representative-atom groups for downstream analysis.
- Interface/postprocess: interface builds now also write `system_ndx_groups.json`, and `yadonpy.interface` exports helpers for reading raw `.ndx` groups and building a structured catalog for scripted post-processing.
- Interface/builder: the resume signature for interface builds now includes the 0.8.7 schema so older interface artifacts are rebuilt automatically instead of silently reusing a smaller pre-fix `system.ndx`.
- Release: bumped package/docs metadata to `0.8.7`.

## 0.8.6 (2026-03-19)

- Interface/protocol: `03_contact` and `04_exchange` now default to semiisotropic `Berendsen` pressure coupling with gentler compressibility for more robust interfacial densification.
- Interface/protocol: `05_production` remains configurable and stays on `C-rescale` by default.
- Core/poly: cached random-walk polymers reloaded from older SDF restart artifacts now regain missing atom residue metadata automatically, and terminal-capping helpers no longer assume `GetPDBResidueInfo()` is always present.
- Interface/builder: assembled interface `system.top` files now place parameter-only includes such as `ff_parameters.itp` ahead of all molecule-type includes, eliminating the `Invalid order for directive atomtypes` startup failure in `gmx grompp`.
- Release: bumped package/docs metadata to `0.8.6`.

## 0.8.5 (2026-03-19)

- Interface/builder: switched slab selection from a fixed box-center cut to a denser-window search along the interface axis.
- Interface/builder: recenters slab phases laterally and adds a configurable top-slab lateral phase shift during interface assembly.
- Release: bumped package/docs metadata to `0.8.5`.

## 0.8.4 (2026-03-19)

- Examples/interface: replaced the tri-state restart pattern in Examples 10-12 with one explicit script-level switch and `WorkDir.clean` semantics.
- Core/poly: `amorphous_cell(...)` now restores cached packed cells together with their box geometry during restart.
- Interface/protocol: Examples 10-12 now execute `InterfaceDynamics.run(...)`; the staged interface workflow now forwards the interface index file into `grompp` and freezes `BOTTOM_CORE` / `TOP_CORE` during the early pre-contact stages.
- Release: bumped package/docs metadata to `0.8.4`.

## 0.8.3 (2026-03-19)

- Analyzer: non-polymer systems now skip polymer metrics/Rg work and use liquid-style density-only equilibrium criteria.
- InterfaceBuilder: same-named but different include files are renamed during assembly instead of raising a hard conflict.
- Tests/docs: added regressions for both fixes and bumped release markers to `0.8.3`.

## 0.8.2 (2026-03-18)

- Merge: integrated the external `rdkit_molspec` hotfix into the `v0.8.1` code line instead of reverting to the older package wholesale.
- Core: restored `yadonpy.core.as_rdkit_mol()` and used it at RDKit descriptor boundaries.
- Examples/tests/docs: updated MolWt call sites, added MolSpec regression coverage, and bumped release markers to `0.8.2`.

## 0.8.1 (2026-03-18)

- Examples: rewrote `examples/10_interface_route_a/`, `examples/11_interface_route_b/`, and `examples/12_cmcna_interface/` to use explicit script logic instead of `yadonpy.workflow.steps as wf`.
- Example 12: now follows the requested direct style for MolDB-backed template molecules, explicit RESP for `ter1` and `PF6`, explicit `poly.amorphous_cell(...)` calls for the CMC and electrolyte AC boxes, and then interface construction.
- Versioning/docs: bumped package and release markers to `0.8.1`.

## 0.8.0 (2026-03-18)

- add a dedicated `yadonpy.interface` layer for interface geometry build and staged interface MD;
- add `InterfaceBuilder`, `InterfaceRouteSpec`, `InterfaceProtocol`, and `InterfaceDynamics`;
- keep interface geometry separate from wall-model settings, with route A and route B APIs;
- add `WorkDir.child(...)` / `workunit(...)` so multi-box studies can live under one root `work_dir`;
- write interface build artifacts directly under `work_dir/<child>/01_snapshots`, `02_slabs`, and `03_interface`;
- add standalone examples for route A, route B, and CMC-Na vs 1M LiPF6 interface studies.

## 0.7.17 (2026-03-18)

- Release floor: raised the minimum supported Python version to 3.11 and synchronized README / API / Manual / package metadata.
- API safety: `parameterize_smiles()` is now strict by default for the requested charge workflow; callers must opt in explicitly to continue after a charge-assignment failure.
- Restart semantics: unified `core.poly` cell-builder restart resolution across explicit `restart`, legacy `restart_flag`, `WorkDir.restart`, and runtime defaults.
- Engineering cleanup: fixed the `verbose=False` streaming bug in `gmx.engine`, added shared in-file helpers for cell-builder setup, and added a manifest to keep release tarballs free of caches / temp workdirs.

## 0.7.12 (2026-03-16)

- release housekeeping: bump version metadata to v0.7.12;
- add smoke tests for runtime option parsing / context overrides;
- add release-consistency tests for README / API / Manual / pyproject version sync;
- make the PDF manual builder create the output directory automatically.
- EQ21: lengthen every NPT stage by default (`eq21_npt_time_scale = 2.0`) so GROMACS has more time to collapse voids and converge density without making barostat coupling harsher.

## 0.7.11 (2026-03-16)

- Engineering cleanup: normalized import order in touched modules, added a `tests/` smoke suite, and documented the maintainer self-check commands in README/API/Manual.
- Public API fix: `yadonpy.api.load_from_moldb()` now matches the documentation again — it returns the molecule by default and only returns `(mol, record)` when `return_record=True` is requested.
- Testability: `pyproject.toml` now declares an optional `test` extra and built-in pytest discovery settings for source-tree validation.
- Maintenance cleanup: removed stale imports/locals across src/examples, restored explicit re-export `__all__` lists, and fixed example path drift for the GROMACS Tg/elongation examples.
- Static checks: `pyflakes src examples docs/build_manual_pdf.py` is now clean (0 warnings), `python -m compileall` passes, and key module import smoke tests pass.
- Fixed a major packed-system export regression where explicit bonded overrides such as PF6 with `bonded="DRIH"` could be lost after `poly.amorphous_cell(...)`.
- `amorphous_cell` now records each species' cached artifact directory / molecule id / bonded signature into cell metadata.
- `io.gromacs_system` now prefers the original cached per-species artifacts recorded during packing, instead of rebuilding a representative fragment from the packed cell when possible. This preserves DRIH / mSeminario single-molecule ITPs in system exports.

# 0.7.9 (2026-03-15)

- Fix import-time regression in `io/gromacs_system.py` caused by a misplaced `@dataclass`.
- For anionic mSeminario Hessians, default to `def2-TZVPD` (with matching Br/I diffuse basis) unless explicitly overridden.
- Route `GAFF.ff_assign(..., bonded="mseminario")` through the automatic anion Hessian-basis selection.

## 0.7.8 (2026-03-15)

- Bonded override hardening: explicit `ff.ff_assign(..., bonded="DRIH"|"mseminario")` requests are now treated as strict overrides rather than best-effort hints. Export will fail loudly if the requested bonded patch fragment is missing instead of silently falling back to plain GAFF-family bonded terms.
- Cache safety: per-molecule artifact cache keys now include the bonded strategy (`plain` vs `drih` vs `mseminario`), preventing old plain-GAFF artifacts from shadowing later rigid-ion parameterizations for the same SMILES/force field. Cached metadata now records bonded override state as well.
- System export continuity: `poly.amorphous_cell()` now stores per-species bonded override metadata in `_yadonpy_cell_meta`, and `io.gromacs_system.export_system_from_cell_meta()` replays that bonded request when it has to rebuild a species from MolDB.
- DRIH algorithm refresh: common AX4/AX6 ions now use species-aware presets (e.g. PF6-/BF4-/ClO4-/AsF6-/SbF6-) plus mild bond-length scaling, while keeping exact symmetry-enforced bond/angle manifolds. PF6- now gets separate cis/trans stiffness with explicit 12-cis/3-trans bookkeeping in the JSON patch metadata.
- Modified Seminario refresh: near-linear angles are no longer dropped silently; they are handled with a dedicated two-plane fallback. Equivalent bonds/angles are symmetrized by symmetry rank + coarse geometry class, which is especially important for high-symmetry ions such as PF6-.
- Export consistency: if angle/dihedral containers must be rebuilt from typed bonded terms before writing ITPs, any attached bonded JSON patch is re-applied afterwards so the final topology still reflects the requested DRIH / mSeminario override.

## 0.7.7 (2026-03-15)

- Fixed a bonded-parameter propagation bug for DRIH / mSeminario workflows.
  Auto-generated bonded patch fragments from `qm.assign_charges(..., bonded_params="auto")`
  are now reused on restart and applied during GROMACS export whenever a valid
  patch fragment is attached to the molecule. This restores rigid-ion behavior
  for species such as PF6-.
- Preserved generic bonded patch metadata (`_yadonpy_bonded_*`) inside charge-cache
  JSON files so `restart=True` workflows can reattach DRIH / mSeminario patches
  without recomputing QM charges.
- `apply_mseminario_params_to_mol()` now correctly accepts angle entries stored as
  `{i, j, k, ...}` in addition to the older `{a, b, c, ...}` form, so in-memory
  angle overrides are no longer silently skipped.

## 0.7.6 (2026-03-14)
- Console UX: unified restart/QM/polymer-build/cell-packing/equilibration/analysis logs with consistent section banners, step labels, item summaries, restart-skip notices, and elapsed-time reporting.
- QM UX: `sim.qm.conformation_search()` and `sim.qm.assign_charges()` now print clearer purpose/smiles/level summaries and explicitly report cache-hit reuse on restart.
- Workflow UX: `build_copolymer()`, `pack_amorphous_cell()`, and `equilibrate_until_ok()` now emit cleaner top-level progress summaries.
- Summary robustness: `AnalyzeResult.get_all_prop()` now backfills missing production `volume_nm3` from the cell summary, derives density from topology mass + volume when needed, and falls back to target temperature/pressure only as a last resort with provenance recorded in `fallbacks`.
- Topology parsing: `gmx.topology.parse_itp()` now captures atom masses so density can be reconstructed from topology + box volume when EDR density terms are unavailable.

# YadonPy Log

## 0.7.2 (2026-03-12)
- Analysis UX: `AnalyzeResult` now prints clear step-by-step console progress for `get_all_prop()`, `rdf()`, `msd()`, `sigma()`, `rg()`, and `den_dis()/density_distribution()`, including subtask numbering and elapsed time for long-running GROMACS analyses.
- RDF/CN plots: added all-types summary plots under `06_analysis/rdf/plots/`, including `rdf_all_types.svg`, `cn_all_types.svg`, and `rdf_cn_all_types.svg`.
- MSD plots: added an explicit all-types summary figure `06_analysis/msd/plots/msd_all_types.svg` in addition to the existing linear/log-log overlay plots.
- Conductivity output: console logs now explicitly show the order and completion of NE and EH conductivity calculations, with EH warnings surfaced immediately when a TRR/IONS group is unavailable.

This file tracks notable changes by release.  
(README intentionally does **not** include the changelog.)

## 0.7.0 (2026-03-11)
- New standalone GROMACS export helper: `from yadonpy.io.gmx import write_gmx`.
  - Use `write_gmx(mol=..., out_dir=...)` to generate per-molecule `.gro`, `.itp`, and `.top` files without running the full workflow.
- MOL2 writer rename: public helper is now `write_mol2(...)` under `yadonpy.io.mol2`.
  - `write_mol2_from_rdkit(...)` remains as a compatibility alias.
- Examples: updated relevant scripts to use `write_mol2(...)` and to demonstrate standalone per-species GROMACS export under `90_*_gmx/` folders.

## 0.6.15 (2026-03-09)
- FF assignment UX: `ff_assign()` now prints a formatted per-atom report after successful assignment, including atom type, bonded type, charge, sigma, epsilon, and the force-field type description. Pass `report=False` to suppress it.
- Lightweight charge backends: `qm.assign_charges()` / `core.calc.assign_charges()` now accept `CM1A`, `1.14*CM1A`, `<scale>*CM1A`, `CM5`, `1.2*CM5`, and `<scale>*CM5`.
  - `CM1A` is routed through LigParGen/BOSS.
  - `CM5` is routed through xTB GFN1.
- Public API: expanded `yadonpy.api` and top-level re-exports with `list_forcefields()`, `list_charge_methods()`, `mol_from_smiles()`, `assign_forcefield()`, `assign_charges()`, `conformation_search()`, and `load_from_moldb()`.
- Docs: refreshed `docs/API.md`, `docs/Manual.md`, to cover the new charge-model, reporting, and analysis workflow.
- Analysis outputs: `AnalyzeResult.get_all_prop(save=True)` now writes merged basic analysis outputs under `06_analysis/`, including `summary.json`, `basic_properties.json`, `cell_summary.json`, and per-polymer `Rg`, end-to-end distance, and persistence-length summaries.
- Polymer metrics: polymer species are auto-detected from `system_meta.json`, and metrics are reported separately for each polymer moltype. Box/trajectory summaries now include cell lengths, angles, and volume statistics.

## 0.6.12 (2026-03-06)
- Architecture: added a centralized force-field registry (`yadonpy.ff.registry`) so alias handling and lazy FF construction live in one place instead of being duplicated in the public API.
- Packaging/resource loading: added shared package-resource helpers and switched bundled FF/PDB lookups to them; `pyproject.toml` now ships `core/pdb/*.pdb` so amino-acid residue templates are available from installed packages too.
- API robustness: `parameterize_smiles()` now emits a warning when QM charge assignment fails instead of silently swallowing the error.
- Cleanup: removed bundled `__pycache__` / `.pyc` artifacts from source and examples, and fixed the conductivity analysis docstring escape warning.
- Docs/messages: updated version markers and MolDB guidance to point at the current Examples 07/08 workflow.

## 0.6.11 (2026-03-04)
- Fix (Psi4 compatibility): restored default DFT method keyword to `wb97m-d3bj`.
  - Psi4 v1.10 does **not** provide plain `wb97m` as a method keyword, so previous defaults could crash in OPT/RESP.
  - Added an alias: if you pass `wb97m` (in CSV or scripts), YadonPy will treat it as `wb97m-d3bj`.
- Docs: README/API updated accordingly.

## 0.6.9 (2026-03-04)
- QM defaults: updated the built-in QM level policy.
  - anions (net charge < 0): OPT `wb97m/def2-SVPD`, RESP(ESP) `wb97m/def2-TZVPD`
  - others: OPT `wb97m/def2-SVP`, RESP(ESP) `wb97m/def2-TZVP`
- Auto-inference: total charge and spin multiplicity are now inferred best-effort from SMILES/RDKit (formal charges + radical electrons) across common QM helpers.
- Examples: removed legacy examples 01/06/07 and renumbered the remaining examples accordingly.

## 0.6.8 (2026-03-03)
- MolDB export helper: added `MolDB.mol_gen(mol, work_dir=..., add_to_moldb=False)`.
  - Use it after you already computed a molecule with `qm.conformation_search` + `qm.assign_charges` to generate a **copy-pastable** MolDB snippet under `work_dir/moldb_snippet_*/`.
  - Set `add_to_moldb=True` to directly write into the global MolDB.
- Examples: Example 03 demonstrates generating MolDB snippets for solvent molecules.
- Docs: README/Manual/API updated with `mol_gen` usage.

## 0.6.7 (2026-03-03)
- MolDB safety: `MolDB.autocalculate()` now defaults to **NOT** writing into `~/.yadonpy/moldb`. Instead it creates a fresh MolDB folder under the given `work_dir` (e.g. `moldb_generated_YYYYMMDD_HHMMSS/`).
  - To write into the global MolDB you must pass `add_to_moldb=True`.
- MolDB inspection: added `MolDB.check()` which prints a formatted summary (key/name/smiles(or psmiles)/charge variants) to help spot duplicates/missing variants quickly.
- Docs: README/Manual/API updated to reflect the new default behavior and the new `check()` helper.

## 0.6.6 (2026-03-03)
- Remove legacy cache: deleted the old `basic_top` subsystem (built-in/user pre-baked `.itp/.gro/.top` templates) and all related APIs.
- Data dir simplification: `~/.yadonpy/` now only needs MolDB (`~/.yadonpy/moldb`); no more copying force-field resources into the user home.
- Logging UX: console status tags like `[CMD] [RUN] [DONE] [SKIP] [WARN] [ERROR]` are now color-highlighted (disable via `NO_COLOR=1`).
- Cleanup: removed deprecated files such as `gaff_broken_v0.4.30.py` / `gaff2_broken_v0.4.30.py`.
- Examples: updated all example READMEs/paths; Example 02 now stores PF6- into MolDB; Example 07 is now a MolDB mini-batch sanity check.

## 0.6.5 (2026-03-03)
- Fix: MolDB `autocalculate()` now treats CSV fields `method`/`basis_set` value **Default** (case-insensitive) as *unset*, so Psi4 is never called with `Method=default`.
- Robustness: `sim.qm.assign_charges()` also normalizes `Default/None/Null` placeholders back to YadonPy's built-in QM defaults.

## 0.6.4 (2026-03-02)
- MolDB: `autocalculate()` now accepts friendly aliases: `work_dir=...`, `mem=...`, `psi4_omp=...` (and the common pattern `work_dir=..., omp=<psi4_threads>, mem=<MB>`).
- Bonded override: moved bonded-patch selection to `ff.ff_assign(..., bonded=...)`. Default is *pure force-field bonded terms*; patches are only injected into GROMACS topology when explicitly requested.
- Examples: Example 10 Step 2 now sets `omp_psi4=64`, `mem_mb=20000`, and can (idempotently) run `MolDB.autocalculate(...)` before reuse. New Example 11 shows converting a pasted text table into `template.csv` then building MolDB.

## 0.6.3 (2026-03-02)
- MolDB variants: charges are now stored and selected by a deterministic variant label `(charge_method, basis_set, method)` per canonical SMILES/PSMILES (keeps the legacy `charges.json` for compatibility).
- New handle API: `ff.mol(smiles, basis_set=None, method=None, charge='RESP')` now returns a lightweight `MolSpec` (no geometry/charges). Use `ff.ff_assign(handle)` to resolve it from MolDB and return an RDKit Mol with parameters assigned.
- Compatibility: the previous behavior is available via `GAFF/GAFF2/GAFF2_mod.mol_rdkit(...)`.
- Example 10: `01_build_moldb.py` simplified to `db.read_calc_temp + db.autocalculate(work_root)`; CSV template updated (no total_charge; includes `charge_method,basis_set,method`; no PF6- in the template).
- Conductivity: NE conductivity now ignores *poly-ionic macromolecules* (large natoms + large net charge) by default to prevent unphysical blow-ups; the ignore decision is recorded in `summary.json`/`sigma.json` component metadata.

## 0.6.2 (2026-02-26)
- Data root: default data directory moved to `~/.yadonpy/` (configurable via `YADONPY_HOME` or `YADONPY_DATA_DIR`). Backward-compatible fallback keeps using `~/.local/share/yadonpy/` if already initialized.
- QM naming: `01_qm/..` folders now default to *user script variable names* (best-effort) to avoid opaque auto-names like `C8H13O7-_xxxxxx` when `log_name` is not provided.
- MolDB: default location follows the data root (`~/.yadonpy/moldb/`), still stores **geometry + charges only** (no itp/top/gro).
- basic_top → mol2 shortcut: `GAFF/GAFF2/GAFF2_mod.mol(smiles)` now auto-loads a charge-bearing MOL2 from the built-in/user `basic_top` cache (when present) before falling back to MolDB.
- MOL2 I/O: added `read_mol2_with_charges()` to reliably recover per-atom charges from the MOL2 charge column.
- Examples: consolidated MolDB usage into `examples/10_moldb_precompute_and_reuse/` with a two-step workflow (precompute → reuse).
- Docs: README, Manual, API updated; regenerated `docs/Manual.pdf`.

## 0.5.27 (2026-02-26)
- Examples: merged previous MolDB batch/reuse examples into `examples/10_moldb_precompute_and_reuse/` (two scripts: build DB, then run workflow from DB).
- FF helpers: `GAFF/GAFF2/GAFF2_mod.mol(..., require_ready=...)` and `store_to_db()` added/extended to make MolDB reuse straightforward.

## 0.5.26 (2026-02-26)
- Fix: Example 08 monomer RESP charges could be applied to temporary Mol objects (inside a loop), leaving the original monomer variables uncharged. This could propagate as **all-zero charges** in `02_system/CMC.itp`. Example 08 now updates monomer objects explicitly.
- Fix: `sim.qm.conformation_search` now best-effort applies the optimized geometry back onto the *input* Mol (in-place) to prevent stale-object bugs in list-iteration workflows.
- Fix: `poly.connect_mols` charge conservation at linker deletion sites now checks the actual linker atoms (head/tail + neighbors), instead of atom(0)/atom(-1).

## 0.5.25 (2026-02-26)
- System export: after building `02_system/system.top` + `02_system/system.gro`, YadonPy now best-effort generates a **box-level** `02_system/system.mol2` using ParmEd (top+gro → mol2). This is useful for visualization/interoperability and does not affect MD.

## 0.5.24 (2026-02-26)
- Mol2 export: `write_mol2_from_rdkit()` now prefers the caller's Python variable name (consistent with `02_system` moltype naming). Example outputs like `00_molecules/copoly.mol2` replace opaque auto-names like `C2H6O_8d9587.mol2`.
- Example 08: added `00_molecules/*.mol2` export for key species (CMC/solvents/ions).

## 0.5.23 (2026-02-26)
- MSD: do not add `gmx msd -mol` by default (users requested atom-based MSD for polymers/selections). Improved `-trestart` auto-tuning: target ~5× frame interval (min 10 ps, max 200 ps) and always an integer multiple of the frame interval.
- EH conductivity: strict TRR-based Einstein–Helfand by default (removed the position-only fallback). `gmx current -dsp` is now the only EH path; if TRR/velocities are missing, the EH block reports a clear reason.
- Example 08: run a short production NPT (time=20 ns) after EQ/Additional and perform RDF/MSD/sigma analysis on the production trajectory instead of the equilibration stage.

## 0.5.22 (2026-02-26)
- Fix: `gmx msd` runtime error in GROMACS 2025+ where `-trestart` must be divisible by `-dt`. Auto `trestart` is now forced to an integer multiple of the written frame interval, and we pass `-dt` explicitly when supported.
- Fix/Improve: EH conductivity no longer silently returns null when `gmx current -dsp` fails (e.g., missing velocities). Added a robust fallback that computes an EH-style `-dsp` curve from unwrapped positions (no velocities required), then fits using the same window-selection logic.

## 0.5.21 (2026-02-25)
- RDF/CN: fixed coordination-number reporting in JSON. `rho_target_nm3` now uses the *target particle* number density (atoms of given type per moltype per volume), and `cn_shell` is taken from the raw `gmx rdf -cn` curve when available.
- MSD/Diffusion: MSD analysis now prefers the workflow-produced PBC-fixed trajectory copy `md.xtc.pbc_tmp.xtc` (in the MD stage directory). Improved `-trestart` auto-tuning and more robust diffusion fit window selection (best R² tail window).
- Workflows: PBC post-processing keeps a side-by-side copy (`*.pbc_tmp.*`) while still overwriting the canonical filename for downstream compatibility.

## 0.5.20 (2026-02-25)
- Docs: README updated (cleaner install guide + features) and all example READMEs rewritten for consistency.
- RDF/CN plots: CN axis fixed to 0–6 and first-shell cutoff is annotated (cutoff distance + CN).
- MSD: more robust PBC handling for diffusion analysis (mol → nojump pre-processing) and safer `gmx msd` options.
- Presets: NPT/NVT production trajectories write frames every 0.5 ps by default to reduce coarse/jagged MSD curves.

## 0.5.19 (2026-02-25)
- Documentation refresh: rewritten README and Manual (Markdown), reorganized docs (Install/User Guide/Examples/API).
- No code changes in this documentation-only release.

## 0.5.18
- Fix: RDF group parsing (`ndx_groups`) and step-level timing output.

## 0.5.17
- Fix: Psi4 wrapper dependency checks made lazier and more informative; safer cleanup.

## 0.5.16
- Fix: Prefer user variable names (e.g., `copoly`) when exporting `*.itp` and `system.top`.
