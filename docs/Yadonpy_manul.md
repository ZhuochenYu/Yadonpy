# YadonPy Manual (v0.8.53)

YadonPy is a SMILES and PSMILES driven molecular-workflow toolkit for building molecules, parameterizing them, packing bulk systems, exporting GROMACS inputs, and running restartable simulation pipelines.

Python requirement: Python 3.11+

Related documents:

- API reference: `docs/Yadonpy_API_v0.8.53.md`
- user guide: `docs/Yaonpyd_user_guide.md`

## 1. Purpose of this manual

This document is the long-form technical manual for the current release. It is not a line-by-line API listing and it is not meant to replace the runnable examples. Instead, it explains how the package is organized, which workflow assumptions the code currently makes, and how to use those assumptions correctly in a real study.

Use this document when you need answers to questions such as:

- what exactly is stored in MolDB and what is regenerated every run;
- which script style is the intended one for production work;
- how YadonPy expects work directories to be structured;
- how bulk and interface workflows move from molecules to equilibrated systems;
- where restartability begins and ends;
- which example is the right starting point for a given study.

If you only need a minimal getting-started path, read the user guide first. If you need callable objects and signatures, use the API reference. If you need the reasoning behind the package behavior, read this manual.

## 2. Design goals

YadonPy is designed around a few stable rules.

### 2.1 Script-first usage

The package favors direct Python scripting over opaque project files. The examples are intended to be edited and reused in real studies. A YadonPy workflow is expected to remain visible in the user script, including molecule definitions, density targets, equilibration presets, and interface route choices.

This has practical consequences:

- the examples are part of the real documentation surface;
- helper APIs are added when several examples need the same logic, not to hide the workflow behind a monolithic launcher;
- most users are expected to keep a study script under version control and treat YadonPy as a library.

### 2.2 MolDB-first persistence

The persistent cache stores expensive molecule-level assets:

- preferred 3D geometry;
- charge variants;
- readiness metadata for those variants;
- bonded patch metadata and bonded sidecar files when a charge variant depends on DRIH or mSeminario bonded terms.

Generated topology files such as `.itp`, `.top`, and `.gro` are treated as build products and are regenerated on demand. This is one of the package's central architectural choices. The long-lived asset is the prepared molecular state, not a previously exported topology tree.

### 2.3 Explicit work directories

Study outputs are written to explicit work directories. Restart behavior is controlled either by explicit keywords or by the runtime defaults. A work directory is meant to answer the question: what happened in this particular study run.

### 2.4 Example-driven workflows

Major capabilities are represented by runnable examples rather than only by abstract API descriptions. For polymer electrolytes and interface systems, the examples are not marketing demos. They are the current reference implementations of intended practice.

### 2.5 Conservative restart philosophy

YadonPy tries to reuse expensive completed steps, but it does not pretend every artifact should always be reused forever. When internal schemas or geometry logic change, the package prefers rebuilding affected outputs rather than silently consuming stale data.

## 3. Conceptual model

To use YadonPy effectively, it helps to think of the package as five linked layers.

### 3.1 Molecular definition layer

This is where molecules begin life as SMILES, PSMILES, or direct RDKit-derived structures. At this stage the object may still be only a handle or an unparameterized molecule.

### 3.2 Molecular preparation layer

This layer gives the molecule the expensive data required for simulation:

- 3D geometry;
- optimized conformation if requested;
- assigned charges;
- atom types;
- bonded terms, including any nontrivial sidecar metadata.

### 3.3 System construction layer

Prepared species are assembled into bulk cells or interface candidates through composition targets, packing rules, explicit box dimensions, density estimates, and polymer builders.

### 3.4 Export and simulation layer

Prepared molecular state is exported to GROMACS inputs. Equilibration or production workflows are then staged under the work directory.

### 3.5 Analysis and reporting layer

After or during simulation, YadonPy writes structured summaries so a study script can reason about its own outputs instead of relying only on free-form terminal logs.

## 4. Molecule creation and representation

This section describes the actual workflow logic that the current release implements.

### 4.1 Common starting points

Inputs usually begin from one of these sources:

- SMILES for small molecules, salts, solvents, and monomers;
- PSMILES for polymer-related workflows;
- MolDB records for precomputed species;
- a molecule object already created by a lower-level RDKit-based preparation flow.

### 4.2 Two main script styles

At the script level there are two common styles.

Direct convenience style:

```python
import yadonpy as yp

mol = yp.mol_from_smiles("CCO", name="ethanol")
```

Explicit force-field handle style:

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
mol = ff.mol("CCO")
```

The second form is important because `ff.mol(...)` can represent a lightweight handle that is resolved during `ff.ff_assign(...)`. For simple scripts, the difference may look cosmetic. In real workflows, the explicit handle style makes it easier to:

- delay work until the exact assignment step;
- pass bonded options such as `bonded="DRIH"` or related force-field-specific controls;
- keep the code close to the example scripts used by the current release.

### 4.3 Recommended naming practice

Assign stable human-readable names to important species early, especially if they are written to `mol2` files or reused in multi-species systems. Consistent naming helps later when reading exported GROMACS files, index groups, or analysis summaries.

### 4.4 Handle versus resolved molecule

In practice, a script may pass through these states:

1. molecular text identity such as SMILES or PSMILES;
2. force-field or MolDB handle;
3. resolved RDKit-backed molecule with coordinates;
4. charge-annotated molecule;
5. fully typed molecule with bonded terms.

It is useful to treat `ff.mol(...)` as a lazy declaration and `ff.ff_assign(...)` as the transition into a prepared molecule suitable for export.

## 5. Geometry generation and charge assignment

### 5.1 Typical charge workflow

YadonPy supports conformer search and charge generation through the QM layer.

Common charge flow:

1. prepare 3D geometry;
2. optionally run conformer search;
3. optionally run optimization;
4. assign charges, often `RESP` for production-quality cases.

The convenience layer exposes this through `assign_charges(...)` and `parameterize_smiles(...)`.

### 5.2 When RESP is worth the cost

RESP is usually the right choice when:

- the species will be reused many times;
- the system contains ions or chemically delicate polar groups;
- you care about reliable electrostatics more than fast preparation;
- you intend to store the result in MolDB and amortize the cost over future studies.

### 5.3 When to store a charged molecule in MolDB

If a molecule required expensive QM work, especially for a nontrivial anion, cation, or charged polymer fragment, it should generally be stored in MolDB once the result is considered good. YadonPy is built around that reuse model.

### 5.4 Diagnostics-first debugging

When charge generation fails, do not begin by guessing which backend is broken. Run the diagnostics report first. That report is intended to catch missing executables, missing Python packages, and backend mismatches early.

## 6. Force-field assignment

### 6.1 Force-field objects

Force-field assignment is performed through force-field objects returned by `get_ff(...)`.

Typical pattern:

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
mol = ff.mol("O=C1OCCO1")
mol = ff.ff_assign(mol)
```

Supported families include GAFF, GAFF2, GAFF2_mod, OPLS-AA, MERZ, and DREIDING.

### 6.2 Choosing a family

Broadly:

- GAFF and GAFF2 families are the common choice for organic molecules and mixed organic electrolytes;
- MERZ is commonly used for monoatomic ions such as Li or Na;
- OPLS-AA examples exist when that family is the desired target;
- DREIDING remains available for workflows that explicitly need it.

The exact chemistry choice is still the user's responsibility. YadonPy standardizes the workflow mechanics more than it enforces one chemical model.

### 6.3 Bonded patch handling

Some species need extra bonded terms that are not treated as generic always-on workflow artifacts.

Current policy:

- ordinary workflows do not depend on an always-created visible `bonded_params` folder;
- special bonded data is attached to the prepared molecule and can be persisted in MolDB for the relevant charge variant;
- high-symmetry inorganic ions such as PF6 can therefore be built once, stored, and later reloaded with their bonded overrides intact.

This policy matters because it keeps the common case clean while still preserving the nonstandard bonded information that certain charged species require.

### 6.4 Assignment reports

After successful force-field assignment, YadonPy prints a formatted atom-by-atom report by default. This should be treated as part of the normal validation loop, not as optional noise. If an unexpected atom type appears, it is usually better to stop at this stage than to discover the issue later in a dense packing or MD run.

## 7. Polymer construction

### 7.1 Polymer workflows are explicit by design

Polymer workflows in YadonPy remain explicit in the script. The package provides builders and helpers, but the script is still expected to define monomer choices, feed ratios, degree of polymerization, tacticity, and termination logic.

### 7.2 Random-walk construction

For random copolymers and related systems, the usual path is:

1. prepare the monomer building blocks;
2. convert feed ratios into probabilities if needed;
3. call the random-walk polymer builder;
4. terminate the resulting chain;
5. assign the final force field;
6. write out a reusable molecule file if the workflow will continue into packing.

### 7.3 Why low initial density often matters

Large polymer chains, especially charged or semi-rigid ones, often fail in direct dense packing. In those cases, a low initial packing density is not a hack. It is the correct first stage of a two-stage strategy:

- first, obtain a physically valid non-overlapping packed cell;
- then let controlled equilibration and pressure coupling reduce the volume.

Example 05 and Example 12 both illustrate this philosophy for CMC-related systems.

## 8. Packing and system generation

### 8.1 Bulk system inputs

Once species are ready, YadonPy builds bulk systems or interfaces by combining:

- molecule specifications;
- composition targets;
- density or explicit cell targets;
- polymer builders or amorphous-cell packers.

Generated GROMACS files are exported from the prepared molecular state rather than loaded from a deprecated topology cache.

### 8.2 Density-driven versus explicit-cell builds

YadonPy supports two broad build styles.

Density-driven build:

- you provide a density target;
- the package derives an initial box estimate;
- the system is packed and then equilibrated.

Explicit-cell build:

- you provide a target cell or target box lengths;
- the package packs directly into that box;
- this is often used when one lateral footprint must match another system.

### 8.3 Why direct final-box packing can fail

If the final target box is already tight, direct packing can spend all of its retry budget before MD even starts. The current release therefore prefers an intermediate strategy for difficult systems:

- choose the chemically meaningful target composition;
- keep the final lateral footprint that must be matched later;
- give the initial pack a looser Z dimension or a lower initial density;
- relax under fixed-XY semiisotropic pressure control until the box converges.

This pattern now appears directly in the reusable interface preparation helpers.

## 9. Export model and generated files

### 9.1 Build products versus persistent assets

One of the easiest mistakes in a YadonPy study is to confuse generated topology trees with persistent molecular state.

Persistent assets:

- molecule identity;
- geometry;
- charges;
- bonded patch metadata.

Build products:

- `.mol2`
- `.itp`
- `.top`
- `.gro`
- exported system directories under a study work tree.

### 9.2 Why export regeneration is intentional

The package regenerates exported files because the authoritative source of truth is the prepared molecule plus the current workflow logic. This avoids silently carrying old topology artifacts into a new code path.

### 9.3 Topology validation

The export layer contains include-order validation logic because malformed include ordering can cause `grompp` failures even when the molecule-level preparation was correct. If topology validation fails, treat it as a workflow error, not a cosmetic warning.

## 10. Equilibration workflows

### 10.1 EQ21 and staged robustness

The package supports staged equilibration workflows for bulk and interface preparation.

One important preset is EQ21, which writes a dedicated staged layout and uses a more conservative GROMACS policy in robust mode. In practice that means:

- one pre-NVT velocity-generation stage by default;
- safer time steps for hot or high-pressure stages;
- more damped intermediate densification stages;
- longer NPT tails by default to reduce brittle collapse behavior.

### 10.2 Additional rounds

The package can continue with additional equilibration rounds if the main check still judges the system insufficiently relaxed. This is useful when a dense or highly charged system requires more than the baseline schedule.

### 10.3 Final NPT tails

A final explicit NPT tail is often the right place to impose a geometry-specific control policy. In interface preparation, this commonly means fixed-XY semiisotropic pressure coupling so the lateral footprint is preserved while Z relaxes.

### 10.4 Workflows are conservative on purpose

The current release deliberately favors stability over minimal wall-clock time in difficult systems. For large polymer electrolytes and interfaces, a slightly longer relaxation is often cheaper than repeated failed rebuilds.

## 11. Interface construction

### 11.1 Route-based assembly

The interface subsystem builds explicit route-based interfacial geometries.

Current route logic:

- Route A is the default fully periodic dual-interface path with no vacuum layer.
- Route B adds vacuum padding and leaves wall forces to the MD protocol layer.

### 11.2 Current interface philosophy

The current release uses a polymer-anchored planning model. In practice, that means:

- the equilibrated polymer bulk XY dimensions define the authoritative interface footprint;
- the top-side electrolyte composition is planned against a compact target slab box rather than the full polymer bulk box;
- the electrolyte is packed into an XY-locked initial box with deliberate Z slack;
- a fixed-XY semiisotropic relaxation is then used to bring the electrolyte side into a better assembly state before slab extraction.

### 11.3 Why this matters for Example 12

CMC-electrolyte interfaces are difficult because both dense packing and electrostatics can punish an over-constrained initial build. The current Example 12 therefore follows a conservative path:

1. build and relax the CMC side first;
2. use the equilibrated CMC XY lengths as the interface reference;
3. derive a compact electrolyte target box for the top slab only;
4. pack the electrolyte in a looser XY-locked box with extra Z room;
5. run fixed-XY semiisotropic relaxation;
6. extract slabs and assemble the route-A interface.

### 11.4 Area mismatch policy

The interface builder includes an area policy that controls which side should dominate if small lateral discrepancies exist. In polymer-anchored workflows the bottom side, usually the polymer bulk, is commonly used as the reference side.

### 11.5 Interface dynamics

After geometry construction, the package can run staged interface dynamics that include pre-contact, contact, exchange, and production phases. Geometry building and interface MD are separate steps by design. This separation lets users inspect or reuse the assembled geometry before committing to the full MD protocol.

## 12. Package structure

### 12.1 Public top level

The package root provides a compact public API for ordinary scripts:

- force-field discovery and creation;
- molecule creation;
- charge assignment;
- one-shot parameterization;
- MolDB loading;
- runtime options;
- interface builder classes and helpers;
- QM shortcut export.

### 12.2 Important subpackages

- `yadonpy.core`: low-level utilities, chemistry helpers, naming, topology helpers, workdir helpers, serialization, and core constants
- `yadonpy.ff`: force-field implementations and registry
- `yadonpy.sim`: QM and simulation presets
- `yadonpy.io`: export layers for molecules and systems
- `yadonpy.gmx`: GROMACS workflow and runner helpers
- `yadonpy.interface`: route-based interface planning, assembly, postprocessing, and dynamics
- `yadonpy.moldb`: persistent molecular database
- `yadonpy.workflow`: higher-level workflow wrappers used by some scripted flows
- `yadonpy.diagnostics`: environment and backend checks

### 12.3 Reading the package effectively

If you need to understand behavior quickly, the most useful order is usually:

1. example script;
2. top-level helper imported by that script;
3. subpackage implementation module;
4. test covering the same helper.

This is usually faster than starting at the package root and reading downward.

## 13. Persistence model

### 13.1 MolDB

MolDB is the persistent object cache. It is intended for expensive molecule-level assets, not for complete simulation systems.

It stores:

- molecule identity by SMILES or PSMILES key;
- preferred geometry;
- charge variants;
- variant readiness metadata;
- bonded patch metadata and sidecar files when needed.

It does not exist to store every `.gro`, `.itp`, or `.top` you may generate during a study.

### 13.2 Work directories

Work directories capture study-specific generated content such as:

- intermediate structures;
- exported topologies;
- equilibration stage folders;
- analysis outputs;
- restartable simulation artifacts.

This separation is important:

- MolDB answers: has this molecule already been prepared.
- work directories answer: what happened in this specific study.

### 13.3 Reuse rules

Good candidates for MolDB reuse:

- salts with expensive RESP preparation;
- solvents used across many systems;
- charged fragments or monomers that appear repeatedly;
- species with special bonded sidecar data.

Poor candidates for MolDB reuse:

- one-off system exports;
- large workdir-level trajectory artifacts;
- arbitrary intermediate `.gro` files from a failed study.

## 14. Runtime and restart behavior

### 14.1 Run options

Runtime defaults are stored in a context-local `RunOptions` object.

Supported global toggles currently include:

- `restart`
- `strict_inputs`

They can be controlled programmatically or through environment variables such as `YADONPY_RESTART`.

The rule is simple:

- explicit keyword arguments in API calls override runtime defaults;
- otherwise the current context-local defaults are used.

### 14.2 What restart should mean in practice

`restart=True` should mean: reuse finished expensive steps when the corresponding outputs are still valid for the current code path.

It should not be interpreted as permission to trust every old artifact blindly. If a workflow schema changes, some generated outputs must be rebuilt.

### 14.3 Recommended script pattern

The common pattern is:

```python
from pathlib import Path
from yadonpy.core import workdir
from yadonpy.runtime import set_run_options

restart = True
set_run_options(restart=restart)
BASE_DIR = Path(__file__).resolve().parent
work_dir = workdir(BASE_DIR / "work_dir", clean=not restart)
```

This keeps restart logic explicit at the script level and gives all downstream steps one stable study root.

## 15. Reporting and outputs

### 15.1 Force-field report

After successful force-field assignment, YadonPy prints a formatted atom-by-atom report by default.

### 15.2 Analysis outputs

Analysis workflows write merged summaries under `work_dir/06_analysis/`. Typical outputs include:

- `thermo_summary.json`
- `basic_properties.json`
- `cell_summary.json`
- `polymer_radius_of_gyration.json`
- `polymer_end_to_end_distance.json`
- `polymer_persistence_length.json`
- `polymer_metrics.json`
- `summary.json`

### 15.3 EQ21 outputs

The EQ21 preset writes a dedicated `03_EQ21/` layout with stage subfolders plus schedule and overview files.

### 15.4 Why structured JSON matters

The JSON summaries are not only for plotting after the fact. They are also meant to support study-level decisions inside your own scripts or notebooks, for example:

- whether a system appears equilibrated enough to proceed;
- whether a polymer metric changed in the expected direction;
- whether conductivity calculations excluded macro-ions as intended.

## 16. Example map and intended learning path

The examples are the clearest guide to intended usage.

- `examples/01_Li_salt`: PF6 build, MolDB store, MolDB-backed reuse, GROMACS export
- `examples/02_polymer_electrolyte`: baseline polymer-electrolyte build flow
- `examples/03_tg_gmx`: glass-transition workflow
- `examples/04_elongation_gmx`: elongation workflow
- `examples/05_cmcna_electrolyte`: charged polymer electrolyte workflow
- `examples/06_polymer_electrolyte_nvt`: NVT-focused polymer electrolyte variant
- `examples/07_moldb_precompute_and_reuse`: precompute MolDB entries and reuse them in a later workflow
- `examples/08_text_to_csv_and_build_moldb`: convert text inputs into structured MolDB-build inputs
- `examples/09_oplsaa_assign`: OPLS-AA assignment examples
- `examples/10_interface_route_a`: full standalone Route A interface workflow
- `examples/11_interface_route_b`: full standalone Route B interface workflow
- `examples/12_cmcna_interface`: fixed-XY CMC versus LiPF6 interface workflow with reusable preparation helpers

Suggested order for a new advanced user:

1. Examples 07 and 08 to understand MolDB preparation and reuse;
2. Example 01 to see a realistic expensive ion preparation cycle;
3. Example 02 to understand the baseline bulk-system flow;
4. Example 05 to understand charged polymer electrolyte handling;
5. Examples 10 and 11 to understand interface route mechanics;
6. Example 12 only after the previous pieces make sense.

## 17. Practical rules for production scripts

- prefer `import yadonpy as yp` for ordinary scripts;
- use the explicit `ff = yp.get_ff(...)` plus `ff.mol(...)` plus `ff.ff_assign(...)` style when you need control over handles or bonded options;
- store expensive species in MolDB once and regenerate exported files on demand;
- treat example scripts as templates for production studies;
- use `yadonpy.diagnostics.doctor(print_report=True)` before debugging backend failures by hand;
- do not force dense final-box placement if a chemically equivalent intermediate packing strategy is available;
- when building interfaces, preserve the authoritative lateral footprint and let Z relax separately if possible;
- inspect typed-molecule reports before blaming later MD stages for chemistry problems introduced earlier.

## 18. Troubleshooting notes

### 18.1 MolDB mismatch problems

If a MolDB lookup succeeds but the later exported topology is not what you expected, verify that the intended charge variant and bonded override were stored for that entry.

### 18.2 QM or RESP failures

If RESP or other QM stages fail, check the diagnostics report first. Then verify backend availability, scratch-space assumptions, and memory settings before changing the chemistry setup.

### 18.3 Dense packing failures

If an interface or dense pack fails late in packing, prefer the newer fixed-XY planning helpers instead of forcing direct final-box placement. Lower initial density and a larger initial Z can be a structurally better answer than raising retry counts alone.

### 18.4 Unexpected force-field warnings

If an angle, bond, or atom-type warning appears in a new workflow, do not assume the newest high-level workflow change caused it. First determine whether the underlying atom typing changed, whether an unsanitized intermediate is being typed, or whether the species should have been loaded from a known-good MolDB entry.

### 18.5 Implausible transport metrics

If a large macro-ion gives implausible Nernst-Einstein conductivity, inspect the analysis summaries because YadonPy records macro-ion exclusion decisions explicitly.

### 18.6 Interface assembly instability

If an interface build succeeds but the early MD stages are unstable, separate the debugging problem into three pieces:

1. was the bulk geometry itself equilibrated enough;
2. was slab extraction physically sensible;
3. did the interface protocol start with overly aggressive pressure coupling.

Debugging all three at once is usually slower than isolating them.

## 19. Final guidance

YadonPy works best when used as a scripted molecular workflow library, not as a black box. The package is intentionally opinionated about a few things:

- expensive molecular preparation should be cached cleanly;
- exported system artifacts are rebuildable products;
- study work must remain explicit and restartable;
- difficult systems should be stabilized by better staging, not by hiding the workflow.

If you keep those principles in mind, the example scripts and helper APIs line up naturally, and the package becomes much easier to extend for new polymer, electrolyte, and interface studies.
