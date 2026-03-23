## 0.8.54 (2026-03-23)

- interface/prep + interface/protocol + interface/__init__ + tests/interface_builder: add `recommend_polymer_diffusion_interface_recipe(...)` plus the new `PolymerDiffusionInterfaceRecipe` data object, extend the diffusion protocol constructors with explicit early-stage control knobs, carry richer geometry metadata through `PolymerAnchoredInterfacePreparation`, and lock the new route-selection behavior down with focused interface-planning regressions;
- examples/10_interface_route_a + 11_interface_route_b + 12_cmcna_interface + tests/release_sanity: rewrite the interface examples into the same linear, script-first style used by Example 02 by removing local helper wrappers, move route/protocol heuristics into the library layer, and enlarge Example 12 into a more conservative `DP=150`, `6`-chain, `1 M` LiPF6 CMC interface workflow that uses a polymer-first probe-and-resize route-B build;
- docs/release hygiene: rewrite `README.md`, `docs/Yadonpy_manul.md`, `docs/Yaonpyd_user_guide.md`, and the versioned API reference for the current architecture, bump the package/docs version to `0.8.54`, and tighten release hygiene so cache directories plus generated `yadonpy.egg-info` artifacts are excluded from release trees.

## 0.8.53 (2026-03-23)

- interface/examples/io/tests: finalize the new interface-building workflow for Examples 10-12, including polymer-first XY-matched electrolyte planning, staged diffusion protocols, release-sanity cleanup of PF6 loading style, and an export fallback that reuses packed-cell cached molecule artifacts before requiring MolDB;
- packaging/release workflow: publish the interface robustness update as `0.8.53`, refresh the package version, README release banner, and versioned API document path to `docs/Yadonpy_API_v0.8.53.md`, and keep release-sanity checks aligned with the new version.

## 0.8.52 (2026-03-23)

- ff/gaff + ff/oplsaa + tests/workdir_and_molspec: stop swallowing MolSpec resolution failures inside `ff_assign(...)`; when `ff.mol(...)` cannot be resolved from MolDB or another downstream loader error occurs, the original exception now propagates instead of falling through to a misleading RDKit type error such as `Kekulize(MolSpec)`;
- moldb/store + examples/01_Li_salt + tests/workdir_and_molspec: make MolDB reload hypervalent MOL2 records such as PF6- via an unsanitized-read fallback with selective sanitization, and teach Example 01 to complete the full build/store/reload/export demonstration even when the optional `psi4` / `resp` stack is missing by falling back to a `gasteiger` charge variant while preserving the normal two-line RESP reuse style in fully provisioned environments;
- package/docs metadata: publish the MolSpec error-propagation fix as `0.8.52`, refresh the package version, README release banner, and versioned API document path to `docs/Yadonpy_API_v0.8.52.md`, and keep release-sanity checks aligned with the new version.

## 0.8.51 (2026-03-23)

- examples/01 + 02 + 05 + 06 + 10 + 11 + 12: remove the PF6 MolDB helper wrappers and inline the intended direct style `ff.mol(...)` followed by `ff.ff_assign(..., bonded="DRIH")`, so the examples express the preferred API usage directly instead of hiding it behind `load_*` or `prepare_*` helpers;
- tests/release_sanity: add a regression that forbids reintroducing PF6 MolDB helper wrappers into the shipped examples, alongside the existing checks that block PF6 on-the-fly rebuilds outside Example 01;
- package/docs metadata: publish the direct-example API cleanup as `0.8.51`, refresh the package version, README release banner, and versioned API document path to `docs/Yadonpy_API_v0.8.51.md`, and keep release-sanity checks aligned with the new version.

## 0.8.50 (2026-03-23)

- examples/02 + 05 + 06 + 10 + 11: replace the remaining on-the-fly PF6 QM/RESP build paths with the same MolDB-backed reuse pattern used by Example 12, so every shipped PF6 workflow except Example 01 now expects the precomputed PF6 RESP + DRIH record and loads it via `ff.mol(...)` plus `ff.ff_assign(..., bonded="DRIH")`;
- examples/07 + tests/release_sanity: update the MolDB reuse snippets and add a release regression that forbids reintroducing the old PF6 on-the-fly build pattern outside Example 01;
- package/docs metadata: publish the PF6 example unification as `0.8.50`, refresh the package version, README release banner, and versioned API document path to `docs/Yadonpy_API_v0.8.50.md`, and keep release-sanity checks aligned with the new version.

## 0.8.49 (2026-03-23)

- examples/12_cmcna_interface: simplify PF6 handling so Example 12 now follows the intended MolDB reuse pattern directly, `PF6 = ff.mol(PF6_smiles)` followed by `PF6 = ff.ff_assign(PF6, bonded="DRIH")`, and fail early with a clear message that Example 01 should be run first when the PF6 MolDB entry is missing or not ready;
- examples: remove the legacy local `src/` bootstrap fallback from all shipped example scripts so release examples consistently exercise the installed package instead of mutating `sys.path` at runtime;
- examples/docs metadata: update the Example 12 and Example 01 guidance to document the PF6 MolDB prerequisite, then publish the example cleanup as `0.8.49` with refreshed package/docs version references.

## 0.8.48 (2026-03-23)

- moldb/store + tests/api: make `MolDB.load_mol(...)` recover from a corrupted `best.mol2` by falling back to other readable `.mol2` files in the same record directory, so export and reuse paths do not fail just because the stable alias file is damaged while the stored geometry copy is still intact;
- examples + tests/release_sanity: remove the local `src/` bootstrap fallback from all shipped example scripts and tighten the release sanity check so published examples always import the installed `yadonpy` package instead of mutating `sys.path` at runtime;
- package/docs metadata: publish the MolDB recovery and example cleanup as `0.8.48`, refresh the package version, README release banner, and versioned API document path to `docs/Yadonpy_API_v0.8.48.md`, and keep release-sanity checks aligned with the new version.

## 0.8.47 (2026-03-23)

- sim/analyzer + tests/runtime: fix polymer equilibrium analysis so Rg checks prefer whole-polymer index groups such as `MOL_<moltype>` before representative-atom groups, which avoids the false `Rg = 0` convergence failure that could trap Example 12 in repeated `additional_eq` rounds;
- sim/analyzer + tests/runtime: relax the density plateau gate for polymer-containing systems relative to simple liquids, so slow polymer/polyelectrolyte density tails are judged with a realistic additional-equilibration threshold instead of an overly strict liquid-like cutoff;
- package/docs metadata: publish the analyzer convergence fix as `0.8.47`, refresh the package version, README release banner, and versioned API document path to `docs/Yadonpy_API_v0.8.47.md`, and keep release-sanity checks aligned with the new version.

## 0.8.46 (2026-03-23)

- io/gromacs_system + tests/workdir_and_molspec: fix mixed-forcefield system export so `export_system_from_cell_meta(...)` resolves each species with its own recorded `ff_name` instead of forcing the caller-wide default onto every species, which restores mixed `GAFF2_mod` plus `MERZ` boxes such as Example 12 probe exports containing `Li+`;
- package/docs metadata: publish the mixed-forcefield export fix as `0.8.46`, refresh the package version, README release banner, and versioned API document path to `docs/Yadonpy_API_v0.8.46.md`, and keep release-sanity checks aligned with the new version.

## 0.8.45 (2026-03-23)

- interface/prep + examples/12_cmcna_interface + tests/interface_builder: stop building the Example 12 probe electrolyte in an explicit cell, switch that probe stage to density-driven packing so `amorphous_cell(...)` retry fallback can reduce density and expand the whole box isotropically, and document/test the new probe-build policy;
- package/docs metadata: publish the Example 12 probe-build robustness update as `0.8.45`, refresh the package version, README release banner, and versioned API document path to `docs/Yadonpy_API_v0.8.45.md`, and keep release-sanity checks aligned with the new version.

## 0.8.44 (2026-03-23)

- interface/prep + examples/12_cmcna_interface + tests/interface_builder: replace the brittle direct final-box electrolyte construction in Example 12 with a restored two-stage probe-and-resize workflow, add library helpers to plan a pack-friendly isotropic probe bulk and then rebuild a resized fixed-XY electrolyte from the equilibrated probe profile, and lock that workflow down with focused planning regressions;
- package/docs metadata: publish the Example 12 probe/resized workflow rewrite as `0.8.44`, refresh the package version, README release banner, and versioned API document path to `docs/Yadonpy_API_v0.8.44.md`, and keep release-sanity checks aligned with the new version.

## 0.8.43 (2026-03-23)

- io/gromacs_system + tests/workdir_and_molspec: accept explicit-cell metadata with `density_g_cm3 = None` during `export_system_from_cell_meta(...)`, so Example 12 and other explicit-box workflows no longer crash in the EQ21 export stage when the cell vectors are authoritative and no target packing density exists;
- package/docs metadata: publish the explicit-cell export fix as `0.8.43`, refresh the package version, README release banner, and versioned API document path to `docs/Yadonpy_API_v0.8.43.md`, and keep release-sanity checks aligned with the new version.

## 0.8.42 (2026-03-23)

- examples/12_cmcna_interface: refactor the CMC-electrolyte interface example into a shorter script that keeps the polymer-anchored interface strategy, folds repeated preparation steps into a few direct helpers, removes debug-heavy charge/planning narration, and leaves a cleaner end-to-end reference workflow for ordinary users;
- package/docs metadata: publish the Example 12 simplification as `0.8.42`, refresh the package version, README release banner, and versioned API document path to `docs/Yadonpy_API_v0.8.42.md`, and keep release-sanity checks aligned with the new version.

## 0.8.41 (2026-03-23)

- workspace instructions: add `.github/copilot-instructions.md` at the workspace root so future work follows the standing release rules to auto-bump the version, archive older version directories and tarballs into `history_version/`, and rebuild a fresh version-matched tar archive for each release;
- package/docs metadata: publish the instruction-file update as `0.8.41`, refresh the package version, README release banner, and versioned API document path to `docs/Yadonpy_API_v0.8.41.md`, and keep release-sanity checks aligned with the new version.

## 0.8.40 (2026-03-23)

- docs/user guide: expand `docs/Yaonpyd_user_guide.md` into a longer practical workflow guide with clearer setup advice, script patterns, example selection, bulk/interface build strategy notes, restart guidance, and troubleshooting for MolDB, packing, and interface preparation;
- docs/release metadata: bump the package version to `0.8.40`, refresh `README.md`, `docs/Yadonpy_manul.md`, and the versioned API reference path to `docs/Yadonpy_API_v0.8.40.md`, and keep release-sanity expectations aligned with the current documentation set.

## 0.8.39 (2026-03-22)

- docs: delete the legacy `docs/API.md`, `docs/Manual.md`, rendered `Manual.pdf`, and the old PDF build helper, then rebuild the release documentation set from scratch as `docs/Yadonpy_API_v0.8.39.md`, `docs/Yadonpy_manul.md`, and `docs/Yaonpyd_user_guide.md` based on the actual package architecture, public API surface, and example workflows;
- release metadata/tests: bump the package version to `0.8.39`, repoint README and release-sanity checks to the new documentation files, and keep the new docs aligned with the current Python 3.11+ baseline and MolDB-first workflow model.

## 0.8.38 (2026-03-22)

- examples/01_Li_salt: rewrite the PF6 example so it first computes PF6 from scratch, exports the built structure/GROMACS files, stores the result into MolDB, and then immediately demonstrates the later MolDB-backed reuse path with the exact script style `PF6 = ff.mol(PF6_smiles)` followed by `PF6 = ff.ff_assign(PF6, bonded="DRIH")`, including comments and README guidance on how to call the database-backed SMILES in later workflows;
- package/docs metadata: publish the Example 01 rewrite as `0.8.38`.

## 0.8.37 (2026-03-22)

- interface/prep + examples/12_cmcna_interface: lift the fixed-XY electrolyte planning and EQ21/Additional/NPT bulk-relaxation flow out of Example 12 into reusable library helpers, so the CMC-electrolyte interface workflow is shorter, easier to read, and reusable by later interface examples instead of keeping the algorithm embedded in one script;
- interface/__init__ + tests/interface_builder: export and lock down the new helper APIs (`plan_fixed_xy_direct_electrolyte_preparation`, `make_orthorhombic_pack_cell`, `equilibrate_bulk_with_eq21`) with focused regressions for bundled electrolyte planning and staged bulk relaxation;
- package/docs metadata: publish the Example 12 workflow/API cleanup as `0.8.37`.

## 0.8.36 (2026-03-22)

- ff/gaff + tests: stop defaulting explicit DRIH/mseminario helper output into `work_dir/bonded_params/...`; bonded overrides now fall back to a hidden cache root instead, so ordinary workflows no longer leave behind an always-created empty `bonded_params` directory when no reusable bonded table is actually needed;
- moldb/store + api + tests: extend MolDB persistence so `update_from_mol(...)` / `mol_gen(...)` also copy and restore QM-derived bonded patch fragments (`_yadonpy_bonded_*`, `_yadonpy_mseminario_*`) for the specific charge variant, allowing high-symmetry inorganic anions such as PF6 to be stored and reloaded with their extra DRIH/mseminario bonded terms intact;
- examples/12_cmcna_interface: refactor the script around reusable molecule-preparation and bulk-equilibration helpers, and make PF6 a MolDB-backed RESP species so repeated Example 12 runs can reuse the stored bonded-aware MolDB entry instead of recalculating the ion every time.

## 0.8.35 (2026-03-22)

- interface/bulk_resize + examples/12_cmcna_interface: keep the Example 12 electrolyte counts planned against the equilibrated CMC box, but derive an XY-locked initial pack box with a pack-friendly lower initial density so the standalone electrolyte build starts with fixed CMC XY and a slightly taller Z instead of trying to force the rounded 1.1 g/cm^3 composition directly into the final box from the first random-placement step;
- core/poly: improve dense explicit-cell packing robustness by ordering amorphous-cell species from larger to smaller packing priority before placement, which reduces the common late-stage failure mode where bulky species like PF6 are inserted only after the box has already been crowded by smaller molecules;
- tests/interface_builder + package/docs metadata: add focused regressions for the XY-locked initial pack-box planner and the new amorphous-cell placement order, then publish the interface-packing robustness update as `0.8.35`.

## 0.8.34 (2026-03-22)

- io/gromacs_system: speed up large-system export by preloading per-species `.gro` templates once, caching packed-fragment coordinates instead of rebuilding them twice, avoiding the old `np.vstack` over all fragment coordinates, and streaming `system.gro` out in chunks instead of accumulating every atom line in one giant Python list before writing;
- core/poly: reduce long-chain random-walk overhead by carrying the current polymer topological distance matrix forward between accepted growth steps instead of rebuilding the same `Chem.GetDistanceMatrix(...)` at the next step handoff, while keeping the existing post-connect validation path intact;
- tests/workdir_and_molspec + package/docs metadata: add focused regressions for preloaded export templates and random-walk distance-matrix reuse, then publish the performance update as `0.8.34`.

## 0.8.33 (2026-03-22)

- io/gromacs_system + io/gromacs_molecule: fix `.gro` writing for large systems by formatting residue and atom serial fields through 5-digit wrap-safe helpers before writing coordinates, so exports beyond atom 99999 keep the fixed-width coordinate columns intact instead of emitting six-digit serials that make `gmx grompp` fail with `Something is wrong in the coordinate formatting of file ... system.gro`;
- tests/workdir_and_molspec + package metadata: add focused regressions that lock the overflow-safe `.gro` line formatting for both system and single-molecule exporters, then publish the fix as `0.8.33`.

## 0.8.32 (2026-03-22)

- gmx/workflows/eq: before each stage `grompp`, parse the current input `.gro` box and automatically cap `rlist`, `rcoulomb`, and `rvdw` to `0.45 * min(box)` when the carried-over box has shrunk below the default `1.2 nm` cutoff budget; this adds a low-level safety net for small or strongly collapsing equilibration boxes so restartable multi-stage workflows fail less often on the next `grompp` with the half-box cut-off error;
- tests/gmx_workflows + package metadata: add a focused regression that locks the new box-aware cutoff clamp into the equilibration workflow and publish the safeguard as `0.8.32`.

## 0.8.31 (2026-03-20)

- sim/preset/eq + examples/12_cmcna_interface: add an EQ21/Additional stage-override path for NPT-like stages so scripts can keep XY fixed during standalone relaxation, then rewrite Example 12 to fill the electrolyte directly into the equilibrated CMC box at an initial `1.1 g/cm^3` estimate and relax that box with fixed XY instead of starting from a sparse probe bulk that can collapse into a GROMACS cut-off violation mid-EQ21;
- tests/interface_builder + package metadata: add focused regressions that verify EQ21 and Additional propagate fixed-XY semiisotropic overrides into their NPT/MD stages, then publish the Example 12 electrolyte-logic update as `0.8.31`.

## 0.8.30 (2026-03-20)

- core/poly: fix the random-walk trial ring-intersection path so compressed trial coordinates created after deleting the consumed head/tail linker atoms are indexed through explicit atom-index maps instead of the original RDKit atom numbering; rigid aromatic double-linker monomers no longer fail with `IndexError` during the lightweight pre-materialization bond/ring screen;
- tests/workdir_and_molspec + package metadata: add a focused regression for a fluorinated aromatic double-linker monomer in the same family as large rigid bulk-build inputs, then publish the random-walk fix as `0.8.30`.

## 0.8.29 (2026-03-20)

- examples/12_cmcna_interface: replace the fragile direct final-box electrolyte packing path with the intended two-stage workflow: first build a probe bulk at a loose target density, equilibrate it, derive a resized electrolyte composition from the equilibrated probe box, then rebuild the interface-aligned electrolyte bulk with fixed CMC XY and a released Z before the downstream interface steps;
- tests/interface_builder + package metadata: add a focused regression that locks the resized-electrolyte plan to the requested interface XY footprint while keeping the probe-profile-driven rescale path intact, then publish the Example 12 workflow rewrite as `0.8.29`.

## 0.8.28 (2026-03-20)

- core/poly: fix `amorphous_cell(...)` retry handling for explicit pack boxes so `density=None` no longer crashes on the old `density *= dec_rate` path, and when a fixed cell is being packed the retry fallback now preserves XY while expanding Z to give tight interface-aligned electrolyte boxes a physically meaningful recovery path instead of repeating the same impossible placement attempt;
- tests/workdir_and_molspec + package metadata: add focused regressions for the new amorphous-cell retry-target behavior, keep the broader polymer/workdir regression file green, and publish the interface-electrolyte packing fix as `0.8.28`.

## 0.8.27 (2026-03-20)

- core/poly: refactor random-walk growth so each retry now prepares a lightweight trial connection geometry first, runs local proximity and ring-intersection checks against the surviving old-chain/new-monomer coordinates, and only materializes the full connected RDKit polymer after the candidate passes, which removes the old eager full-molecule connect/combine/sanitize cost from rejected placements and exposes the same trial geometry for further reuse;
- core/poly: add monomer-side topological-distance pre-caches, random-walk-specific lightweight rollback clones, and a final bonded-term restoration pass keyed off `ff_name`, so the growth loop avoids repeatedly deep-copying full bonded metadata during rollback bookkeeping while still returning a polymer with complete bonded terms;
- tests/workdir_and_molspec + package metadata: add focused regressions for trial cross-distance equivalence, explicit `mon_coord` ring checks, and lightweight bonded-term restoration, keep the broader polymer/workdir regression file green, and publish the optimization as `0.8.27`.

## 0.8.26 (2026-03-20)

- io/gromacs_system + sim/preset/eq: make exported-topology validation recurse through the `#include` chain, require `[ defaults ]` to appear before the first `[ moleculetype ]`, and force `EQ21step` to rebuild cached `02_system` exports whose saved topology tree is now invalid instead of silently reusing them under restart;
- io/gromacs_system: when a scaled export reuses raw molecules via `source_molecules_dir`, also inherit the raw sibling `ff_parameters.itp` block so shared parameter sections stay ahead of molecule includes in the scaled `system.top` that Example 12 bulk EQ uses;
- examples/12_cmcna_interface + tests/workdir_and_molspec + package metadata: rewrite the CMC/electrolyte bulk equilibration path back to explicit old-style export/preflight/EQ script steps, add focused regressions for invalid cached export rebuild and shared-parameter carryover, and publish the fix as `0.8.26`.

## 0.8.25 (2026-03-20)

- core/poly: add a second round of large-chain random-walk performance work by replacing the expensive full Cartesian bond-triangle intersection pairing with a bounding-box prefilter before running the Moller-Trumbore kernel, which reduces wasted ring-intersection checks on far-apart bond/ring pairs during rigid-chain growth;
- core/poly: reduce random-walk topological-distance setup overhead by cloning the temporary RDKit molecule with `Chem.Mol(...)` instead of the heavier YadonPy deep-copy path when building the distance matrix used for clash filtering inside a step;
- tests/workdir_and_molspec + package metadata: add focused regressions for the new ring-intersection prefilter and lightweight RDKit clone helper, then bump the release metadata to `0.8.25`.

## 0.8.24 (2026-03-20)

- interface/builder: add an assembled-interface geometry validator that runs after the final topology-guided whole-molecule canonicalization and records its result in `03_interface/system_meta.json`; it currently checks that no atoms remain outside the primary box and that the assembled slab gap is not strongly negative before MD starts, so broken post-splice geometries fail during build instead of surfacing later in `grompp` or EM;
- tests/release: extend interface builder regressions to assert the new `geometry_validation` metadata stays clean for normal route-A assembly and for the large lateral-shift whole-molecule case, then bump package/docs metadata to `0.8.24`.

## 0.8.23 (2026-03-20)

- interface/builder: harden the post-splice boundary handling by wrapping laterally shifted top-slab molecules back into the target XY box as whole blocks using each molecule COM, instead of leaving the entire shifted block potentially outside the primary box until later tooling happens to fix it; this keeps the lateral phase shift behavior while reducing edge-risk during final interface assembly;
- interface/builder: make slab preparation always re-wrap lateral fragment coordinates after XY scaling, and also when lateral recentring is disabled, so slightly strained or replicated slabs stay inside the intended lateral box before they are written back to `.gro`;
- interface/charge_audit + examples/12_cmcna_interface: promote the Example 12 charge-audit formatting into reusable library helpers and switch the example to use them, so the same packed-cell/export/slab/interface charge checkpoints can be reused in other interface examples without copy-pasting script-local logic.

## 0.8.22 (2026-03-20)

- examples/12_cmcna_interface: switch the default route-A workflow to a more compact starting point by halving the default CMC chain count, raising the initial CMC packing density modestly, reducing the electrolyte probe size and minimum salt-pair floor, and tightening the default slab thickness targets so the example no longer starts from an unnecessarily huge pair of bulk boxes before interface preparation;
- examples/12_cmcna_interface: after the standalone CMC bulk relaxation, run an explicit fixed-XY semiisotropic NPT flattening stage so the later electrolyte resize is anchored to the already-relaxed CMC footprint while Z is allowed to contract slightly, matching the intended "CMC first, then XY-aligned electrolyte" workflow more closely;
- examples/12_cmcna_interface: add charge-audit prints at every major checkpoint (packed cell, equilibrated cell, scaled/raw exports, prepared slabs, assembled interface) and document that the example should rely on InterfaceBuilder's own `-pbc mol` + topology canonicalization path instead of layering on a separate manual pre-splice `-pbc mol` step.

## 0.8.21 (2026-03-20)

- sim/preset/eq: stop the EQ export pair from writing a best-effort full-box `system.mol2` during both the raw and scaled `02_system` exports; large bulk-property jobs already get per-stage MOL2 snapshots later, and skipping this extra ParmEd conversion removes another major `export_system` stall for sparse large boxes without changing the user-facing workflow;
- core/poly: make `_mol_net_charge(...)` recognize the broader set of per-atom charge properties YadonPy already writes (`AtomicCharge_raw`, `RESP`, `RESP_raw`, `ESP`, `MullikenCharge`, `LowdinCharge`, `_GasteigerCharge`) before falling back to formal charge, so packed-cell net-charge warnings and restart metadata do not spuriously treat counter-ions or RESP-only species as neutral;
- interface/builder: topology-canonicalize both representative bulk snapshots and the assembled interface `system.gro` before MD, and bump the interface-build schema so restart does not silently reuse pre-fix interface geometries; this adds one more low-risk whole-molecule safeguard for large Example 12 style systems whose route-A slabs can still arrive at pre-contact MD with awkward wrapped polymers/ions.

## 0.8.20 (2026-03-20)

- gmx/workflows/_util + gmx/workflows/eq: add a topology-guided stage-handoff canonicalization pass for `md.gro` so bonded molecules are rewritten as whole geometries between stages even when the previous step leaves them wrapped across the periodic boundary; this is applied on fresh outputs and on reused restart outputs before the next `grompp` sees them;
- interface/protocol: enable `periodic-molecules = yes` for the staged interface MDP defaults so GROMACS treats interface-spanning molecules consistently during `grompp`, which hardens Example 12 and other large interface systems against `largest distance between excluded atoms is larger than the cut-off distance` failures after pre-contact EM/NVT handoff;
- tests/docs/release: add focused regressions for whole-molecule stage canonicalization and the new interface `periodic-molecules` default, refresh package/docs metadata to `0.8.20`, and publish the fix as a new versioned release.

## 0.8.19 (2026-03-20)

- core/const + core/poly: fix the random-walk progress-bar regression by stopping `tqdm` from being hard-disabled globally; progress bars are now visible again in normal terminal runs and can still be silenced explicitly with `YADONPY_DISABLE_TQDM=1`;
- ff/gaff + ff/merz + ff/oplsaa + ff/dreiding + ff/tip: make `ff_assign(...)` chain-friendly by returning the assigned molecule on success and `False` on failure, so script styles like `mol_P = ff.ff_assign(ff.mol(smiles_P))` and `cation = MERZ().ff_assign(MERZ().mol('[Li+]'))` work without breaking existing `if not result:` / `assert ff.ff_assign(...)` code paths;
- tests/docs/release: add focused regressions for the progress-bar default/override behavior and the requested chained script style, refresh package/docs metadata to `0.8.19`, and publish the fixes as a new versioned release.

## 0.8.18 (2026-03-20)

- core/const + core/poly: fix a regression that had `tqdm` progress bars globally disabled by default, which made random-walk polymer construction look stalled even while `polymerize_rw(...)`, `copolymerize_rw(...)`, and `random_copolymerize_rw(...)` were still running; progress bars are now enabled by default again and can be explicitly suppressed with `YADONPY_DISABLE_TQDM=1`;
- tests/release: add a regression that locks the default progress-bar behavior plus the environment override, refresh package/docs metadata to `0.8.18`, and publish the fix as a new versioned release.

## 0.8.17 (2026-03-20)

- io/gromacs_system + sim/preset/eq: rework the non-interface `export_system` path so EQ21 now builds the raw bulk system once, then derives the charge-scaled production export by reusing the raw molecules, `system.gro`, and `system.ndx` instead of resolving species and rebuilding the same large low-density box a second time; this directly targets the long `export_system` stalls seen after very sparse `amorphous_cell(...)` packing;
- io/gromacs_system: add an internal fast path for reusing previously exported per-species artifacts and box/index files while still regenerating charge-scaled topology files, keeping the external script API unchanged;
- tests/docs/release: add a regression that locks the raw->scaled export reuse path, refresh package/docs metadata to `0.8.17`, and publish the optimization as a new versioned release.

## 0.8.16 (2026-03-20)

- core/poly: fix the remaining random-walk retry-budget regression in `_estimate_rw_rigidity(...)` by removing the stale `Descriptors` dependency and switching the rigidity heuristic to direct molecule APIs plus `rdMolDescriptors`, so newly packaged interface examples do not fail before polymer growth starts;
- release/process: publish this fix as a new versioned release instead of overwriting `0.8.15`, because the previously packaged `0.8.15` archive did not yet contain the random-walk rigidity correction;
- tests/docs/release: rerun focused retry-budget and molecular-weight regressions, refresh release metadata to `0.8.16`, and stage a fresh clean tar archive.

## 0.8.15 (2026-03-20)

- core/molspec + core/calc: add a robust public `molecular_weight(...)` helper, make `as_rdkit_mol(...)` and the legacy mass helpers refresh RDKit property caches defensively, and stop script/runtime mass calculations from depending on `Descriptors.MolWt(...)` succeeding on freshly constructed monoatomic ions or unsanitized hypervalent species;
- ff/merz + examples/io + core/poly: harden monoatomic Merz ion construction, replace direct molecular-weight descriptor calls in Examples 05/10/11/12 plus `io.gromacs_system` and `core.poly`, and stop random-walk rigidity estimation from depending on the removed `Descriptors` import by switching to direct molecule APIs plus `rdMolDescriptors`; electrolyte and interface workflows no longer fail at Li/Na/PF6 mass estimation or early RW budget selection when RDKit cache state is incomplete;
- tests/docs/release: add regressions for robust molecular-weight handling on Merz ions and unsanitized PF6-like species, update script-level API/manual guidance, and bump release markers to `0.8.15`.

## 0.8.14 (2026-03-20)

- interface/bulk_resize: add an electrolyte-specific resize planner that preserves grouped solvent composition and coupled salt-pair concentration instead of independently rounding every species after volume scaling; this produces smaller, polymer-matched electrolyte boxes with less drift in solvent recipe and salt molarity when the target interface footprint is much smaller than the original probe box;
- interface/bulk_resize: extend the electrolyte planner to support grouped solvent families plus multiple coupled salt-pair groups, and add `recommend_electrolyte_alignment(...)` so example defaults derive resized-electrolyte target Z plus fixed-XY semiisotropic NPT duration from the route geometry instead of hard-coded magic numbers;
- examples/interface: propagate the probe-and-resize electrolyte workflow from Example 12 to Examples 10 and 11, so all bundled interface examples now build a probe electrolyte bulk, infer a resized composition from the equilibrated polymer XY footprint, and run a fixed-XY semiisotropic NPT before interface assembly;
- tests/docs/release: add regression coverage for the grouped electrolyte resize planner, mixed-solvent and multi-salt resize edge cases, refresh the interface docs/tutorial text around the shared three-route workflow, and bump release markers to `0.8.14`.

## 0.8.13 (2026-03-20)

- interface/bulk_resize + sim/preset/eq: add a probe-and-resize electrolyte workflow for interface studies. YadonPy can now read the box dimensions of an equilibrated probe bulk, infer its effective density from the known composition, rescale molecule counts to a target XY footprint and target Z span, and optionally run a follow-up semiisotropic NPT stage with fixed XY compressibility so the resized electrolyte stays matched to the polymer box footprint while Z relaxes;
- examples/interface: rewrite Example 12 to build the electrolyte in two passes (probe bulk, then resized bulk matched to the equilibrated CMC XY plane) before assembling the interface, which reduces the total particle count for large interfacial studies while keeping the electrolyte box construction physically tied to an equilibrated reference state;
- tests/docs/release: add regression coverage for bulk-resize planning, fixed-XY semiisotropic NPT overrides, and latest-NPT bulk snapshot discovery, then bump release markers to `0.8.13`.

## 0.8.12 (2026-03-19)

- interface/builder: fix `.gro` read/write handling for large slabs and assembled interfaces whose atom numbering exceeds the fixed 5-column GRO display field; the writer now wraps displayed residue/atom indices safely, and the reader restores atom indices from line order with a fallback coordinate parser for overflowed legacy lines so large prepared slabs can be re-read during final interface assembly instead of failing with `ValueError: could not convert string to float`;
- interface/builder: add optional `numba.jit(nopython=True)` acceleration for the dense-window slab scorer and charge-rebalance candidate picker, while keeping a NumPy/Python fallback when `numba` is unavailable; slab logs now report whether JIT acceleration is active, and the interface build schema was bumped again so cached pre-JIT slab artifacts are rebuilt under the new selection path;
- interface/builder: further accelerate charge-aware slab rebalancing by precomputing scaled fragment charges, COMs, and axial bounds once, then using prefiltered vectorized candidate masks plus lexicographic array scoring instead of rescanning Python lists of remaining fragments on every iteration; this keeps large replicated polymer/electrolyte slab selections responsive even when many counter-fragments are available near the cut plane;
- tests/docs/release: add regression coverage for scaled effective-charge rebalancing plus large-index `.gro` read/write safety, expose an optional `accel` extra for `numba`, and bump release markers to `0.8.12`.

## 0.8.11 (2026-03-19)

- interface/builder: accelerate slab preparation in two places: dense-window slab selection now precomputes fragment COMs, bounds, and masses once and scores candidate cut windows with sorted arrays plus prefix sums instead of repeatedly rescanning every fragment, and the workflow now chooses and charge-balances the slab on the original bulk fragments before applying XY replication so large `3x3` / `4x4` lateral expansions no longer duplicate molecules that will be discarded immediately afterward; slab preparation now also logs per-substep fragment counts, box sizes, slab charge, and elapsed times so long `Prepare bottom slab` stages can be diagnosed from the console;
- core/poly: harden `amorphous_cell(...)` restart reuse by adding a stable workdir-local packed-cell cache alongside the hashed random-walk cache, so restart can still skip cell rebuilding even when the exact hashed cache lookup misses but the latest compatible packed cell is already present;
- core/poly: improve packed-cell console labels with explicit `[RESTART]`, `[SKIP]`, and `[PACK]` messages so continued runs no longer look like they are silently rebuilding an already completed amorphous cell when they are actually restoring or resuming cached work;
- examples/release: stop all bundled example entrypoints from prepending their local `./src` tree ahead of installed packages, so `pip install -e .` and other editable installs are no longer silently overridden; examples now fall back to the local source tree only when `yadonpy` is not importable from the active environment;
- tests/docs/release: add restart-label and stable-cache fallback regressions for `amorphous_cell(...)`, then bump release markers to `0.8.11`.

## 0.8.10 (2026-03-19)

- io/gromacs_system: neutralize tiny exported system-level residual charges directly in the copied `02_system` molecule ITPs after charge scaling, so cached per-molecule artifacts no longer leak a `-0.00x e` box charge into standalone bulk exports;
- interface/builder: stop deriving the assembled interface `[ molecules ]` table from slab metadata, rebuild it from the actual slab topologies instead, and record the resulting interface net charge in `03_interface/system_meta.json` so stale or corrupted slab metadata can no longer inflate the assembled box charge into tens of electrons;
- interface/builder: keep the top slab molecules geometrically whole during lateral phase shifts by removing the per-atom modulo wrapping in the final interface assembly step, which fixes `gmx grompp` failures such as `largest distance between excluded atoms is larger than the cut-off distance` after `01_pre_contact_em`;
- interface/builder: replace the old per-dimension `max(...)` + `ceil(...)` lateral sizing rule with a joint replica/strain search so slabs that are only slightly mismatched can stay at `1x1` with a small, balanced lateral rescale instead of being over-expanded to `2x2`, and unwrap bonded fragments against the source topology before slab preparation so long polymers enter the interface workflow as geometrically consistent whole molecules;
- interface/builder + interface/protocol + gmx/workflows/eq: rebalance charge-sensitive slab cuts by re-adding nearby counter-fragments, invalidate stale pre-fix `03_interface` artifacts with a schema bump, fail interface preflight early if the assembled box still carries a large net charge, and make `01_pre_contact_em` explicitly unconstrained while unconstrained minimization stages skip the `steep_hbonds` bridge entirely;
- tests/docs/release: add focused regressions for exported-system charge neutralization, stale slab metadata, lateral-shift molecule integrity, charge-aware slab rebalancing, optimized lateral sizing, topology-aware fragment unwrapping, and unconstrained interface pre-contact minimization, then bump release markers to `0.8.10`.

## 0.8.9 (2026-03-19)

- core/poly: accelerate single-process random-walk polymer growth by restricting polymer clash checks to old-chain atoms near the newly appended monomer instead of scanning the full chain every step;
- core/poly: add an adaptive retry-budget cap based on monomer rigidity so flexible systems do not waste time on very large retry windows while semi-rigid and rigid monomers keep a more conservative fallback budget;
- core/poly: reduce the public random-walk defaults to a more efficient baseline (`retry=60`, `rollback=3`, `retry_step=80`, `retry_opt_step=4`) across homo-, random-, alternating-, and block-copolymer builders;
- interface/builder: merge duplicated parameter includes from the two slab topologies into one interface-level `molecules/ff_parameters.itp`, de-duplicate identical records, and reject conflicting force-field definitions instead of emitting multiple parameter includes that can trigger `Invalid order for directive atomtypes` in `gmx grompp`;
- interface/builder: normalize mixed molecule ITPs that carry force-field sections before `[ moleculetype ]` by extracting those parameter blocks into the merged interface/slab `ff_parameters.itp`, so embedded `[ atomtypes ]` blocks can no longer slip through after molecule includes during interface MD startup;
- interface/builder: bump the interface-build schema to invalidate stale pre-fix `03_interface/system.top` outputs so restart rebuilds the interface after the merged-parameter logic changes;
- interface/builder: extend interface `system.ndx` with layered aliases such as `BOTTOM1_CMC`, `TOP1_LIPF6`, and `BOTTOM1_TYPE_CMC_C` while keeping the legacy `BOTTOM_*` / `TOP_*` groups for current workflow compatibility;
- interface/builder: add explicit formatted progress prints inside the long-running interface build step so slab preparation and final assembly no longer appear stalled behind one coarse `[RUN] build_interface_*` line;
- interface/protocol: preflight both `system.top` and `system.ndx` before interface MD, fail early on invalid topology include order or missing required groups, and automatically disable pre-contact core freezing when `BOTTOM_CORE` or `TOP_CORE` is empty so `gmx grompp` does not fail on thin or all-surface slabs;
- gmx/workflows/eq + interface/protocol: harden pre-contact interface relaxation by keeping EM helper runs CPU-only, downgrading failed `steep_hbonds` / CG constraint-sensitive minimizations to unconstrained steep minimization, and switching the pre-contact NVT hold to a smaller-step no-constraints stage so rough slab interfaces with temporary voids or bad hydrogen geometry do not abort immediately on LINCS warnings;
- tests/docs/release: add regression coverage for localized clash filtering and adaptive retry budgeting, then bump release markers to `0.8.9`.

## 0.8.7 (2026-03-19)

- interface/builder: expand interface `system.ndx` generation from a small region-only set into a richer inventory that distinguishes region labels, moltypes, atomtypes, per-instance molecules, and representative atoms for post-processing;
- interface/postprocess: add helper functions and an auto-written `system_ndx_groups.json` catalog so downstream scripts can consume the richer interface index groups without reparsing raw `.ndx` text manually;
- interface/builder: bump the interface-build schema signature again so pre-0.8.7 interface artifacts are invalidated and rebuilt automatically when the richer `ndx` layout is requested;
- tests/docs/release: add regression coverage for the new `ndx` groups and bump release markers to `0.8.7`.

## 0.8.6 (2026-03-19)

- interface/protocol: switch the semiisotropic contact and exchange stages to a damped `Berendsen` barostat by default for higher robustness on rough initial interface geometries;
- interface/protocol: keep the final production stage on a configurable production barostat (`C-rescale` by default) so the workflow remains conservative during densification but can still transition to a production-friendly stage;
- core/poly: restore missing atom-level residue metadata when old random-walk polymer caches are reloaded from SDF, and harden `terminate_rw(...)` / `terminate_mols(...)` so older restart artifacts can continue without crashing on `GetPDBResidueInfo()`;
- interface/builder: reorder interface-topology includes so global force-field parameter files such as `ff_parameters.itp` are always written before molecule-type includes, preventing `gmx grompp` failures like `Invalid order for directive atomtypes` during interface MD startup;
- interface/builder: expand the interface-build resume signature with a schema version plus slab/route knobs, so stale pre-fix `03_interface/system.top` outputs are invalidated and rebuilt automatically after builder logic changes;
- tests/docs/release: document the new staged pressure-coupling policy and bump release markers to `0.8.6`.

## 0.8.5 (2026-03-19)

- interface/builder: improve slab construction by selecting the densest target-thickness window along the interface axis instead of always cutting around the box center;
- interface/builder: recenter slab phases laterally before export and apply a configurable lateral phase shift to the top slab during assembly so rough surfaces are less likely to align atom-for-atom;
- interface/tests/docs: record the new builder strategy in metadata, add regression coverage, and bump release markers to `0.8.5`.

## 0.8.4 (2026-03-19)

- examples/interface: simplify Examples 10, 11, and 12 to use a single explicit restart switch with `workdir(..., clean=not restart)` instead of a tri-state local/global mix;
- core/poly: persist and restore `amorphous_cell(...)` packed-cell cache state, including box geometry, so restart can skip completed bulk-box construction before later interface-stage failures;
- interface/protocol: actually run staged interface MD from Examples 10 to 12, pass the interface `.ndx` groups into `grompp`, and freeze the slab core groups during the pre-contact relaxation stages so rough initial interfaces can densify without drifting apart;
- docs/release: bump release markers to `0.8.4`.

## 0.8.3 (2026-03-19)

- analysis: treat systems without polymer moltypes as liquid-like in post-analysis and Rg handling, and use a looser density-only EQ gate for those systems.
- interface: resolve conflicting include basenames during slab assembly by copying colliding files under source-specific names instead of aborting on `ff_parameters.itp` clashes.
- tests/docs: added regression coverage for non-polymer Rg skipping and include-file collision handling, then bumped release markers to `0.8.3`.

## 0.8.2 (2026-03-18)

- core/molspec: restored and exported `as_rdkit_mol()` so resolved `MolSpec` handles can be passed safely into C++-level RDKit descriptor APIs;
- core/naming/examples: routed descriptor-level molecule-weight calls through the new helper where appropriate, including Example 05 and the interface examples;
- tests/docs: added MolSpec-to-RDKit regression coverage and bumped release markers to `0.8.2`.

## 0.8.1 (2026-03-18)

- examples/interface: rewrote Examples 10, 11, and 12 into explicit script-style workflows and removed their dependency on `yadonpy.workflow.steps as wf`;
- examples/interface: Example 12 now builds the CMC bulk and the 1 M LiPF6 EC:DEC:EMC = 3:2:5 electrolyte box through direct molecule preparation and `poly.amorphous_cell(...)` calls before interface assembly;
- docs: updated release markers and interface-workflow text for the v0.8.1 example rewrite.

## 0.8.0 (2026-03-18)

- add a dedicated `yadonpy.interface` layer for interface geometry build and staged interface MD;
- add `InterfaceBuilder`, `InterfaceRouteSpec`, `InterfaceProtocol`, and `InterfaceDynamics`;
- keep interface geometry separate from wall-model settings, with route A and route B APIs;
- add `WorkDir.child(...)` / `workunit(...)` so multi-box studies can live under one root `work_dir`;
- write interface build artifacts directly under `work_dir/<child>/01_snapshots`, `02_slabs`, and `03_interface`;
- add standalone examples for route A, route B, and CMC-Na vs 1M LiPF6 interface studies.

## 0.7.17 (2026-03-18)

- raise the declared Python floor to 3.11 and sync README / API / Manual / release metadata;
- harden `parameterize_smiles()` so requested charge-assignment failures raise by default instead of silently degrading, with an explicit opt-in escape hatch;
- unify `core.poly` restart handling for cell builders so `restart`, legacy `restart_flag`, `WorkDir.restart`, and global runtime defaults resolve through one path;
- refactor repeated cell-builder prep logic inside `core.poly` into shared in-file helpers without splitting the module;
- fix `GromacsRunner._run_capture_tee()` so `verbose=False` no longer streams command output unconditionally;
- add release hygiene via `MANIFEST.in` and remove cached / temporary artifacts from the source tree.

## 0.7.16 (2026-03-18)

- relax the density plateau slope gate in the equilibrium checker from `1e-4 / ps` to `5e-2 / ps` while keeping the `rel_std <= 1%` requirement; this reduces false FAIL reports for slowly settling polymer-electrolyte NPT tails.

## 0.7.15 (2026-03-18)

- QM robustness: force C1 for high-symmetry inorganic polyatomic ions (PF6-, BF4-, ClO4-, AsF6-) in Psi4 tasks.
- QM robustness: if Psi4 optimization still aborts only because the point group changes, keep a C1-symmetrized geometry and continue to the ESP/RESP step instead of failing the workflow.
- QM robustness: synchronize the fallback C1 geometry back into the RDKit conformer before charge fitting and downstream export.

## 0.7.15 (2026-03-18)

- fix MolSpec variable-name inference so `ff.mol(...)` handles keep user-script names such as `solvent_A` instead of collapsing to internal helper aliases like `spec`; this fixes duplicated `spec.itp` includes and incorrect `[ molecules ]` entries in `system.top`.
- add a regression test covering the full `ff.mol(...) -> ff.ff_assign(...) -> amorphous_cell(...) -> export_system_from_cell_meta(...)` path.

## 0.7.13 (2026-03-18)

- add `yadonpy.core.workdir()` and the `WorkDir` path-compatible helper for restart-aware workflow directories;
- keep legacy `ff.mol(...)` handle workflows compatible with polymer-building helpers by caching the resolved RDKit molecule after `ff_assign()`;
- add restart caching to the random-walk polymerization family (`polymerize_rw`, `copolymerize_rw`, `random_copolymerize_rw`, `block_copolymerize_rw`, `terminate_rw`, and the low-level random-walk builder);
- nudge residual amorphous-cell charge below 0.1 e into the first atom of the first species during packing;
- increase the default `grompp -maxwarn` allowance from 1 to 5;
- improve Psi4 optimization robustness for point-group changes by retrying from the latest geometry with symmetry disabled when required;
- update PF6- examples to request `bonded="DRIH"` after RESP charge assignment;
- refresh README, API reference, manual, and example documentation for v0.7.13.
