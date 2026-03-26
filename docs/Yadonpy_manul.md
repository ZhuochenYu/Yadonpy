# YadonPy Manual (v0.8.69)

YadonPy is a script-oriented molecular workflow package for polymer, solvent, salt, bulk, and interface studies. This manual explains how the package is organized, what the stable architectural rules are, and how the current release expects real workflows to be staged.

Python requirement: Python 3.11+

Related documents:

- API reference: `docs/Yadonpy_API_v0.8.69.md`
- user guide: `docs/Yaonpyd_user_guide.md`

## 1. Why this manual exists

The user guide explains how to use YadonPy. The API reference explains what can be called. This manual explains why the package behaves the way it does.

Use this manual when you need to reason about:

- what is stored in MolDB and what is deliberately regenerated;
- why work directories are explicit and restart-aware;
- how interface planning is split between bulk preparation, slab build, and staged interface dynamics;
- which behaviors are product behavior versus implementation detail.

## 2. Core design rules

### 2.1 Script first

YadonPy assumes the user wants to keep the real study logic visible in Python.

That means:

- examples are part of the public workflow surface;
- helper APIs exist to reduce duplication, not to hide the workflow entirely;
- major choices such as chemistry, composition, density target, and interface route remain explicit in user scripts.

### 2.2 MolDB first

MolDB is the persistent cache for expensive molecule preparation.

It stores:

- geometry;
- charge variants;
- readiness metadata;
- bonded-patch sidecar information for variants that need it.

It does not exist to preserve old `.gro/.top/.itp` trees as the authoritative source of truth. Those exports are treated as rebuildable products.

### 2.3 Restart-aware work directories

Work directories are not throwaway scratch by default. They are structured records of one workflow execution.

Typical responsibilities:

- hold packed-cell outputs;
- hold exported topologies;
- hold staged equilibration folders;
- hold analysis summaries;
- hold interface assembly artifacts;
- hold resume metadata.

### 2.4 Conservative physical staging

The package favors better staging over blind retry inflation.

For difficult systems this usually means:

- lower initial packing density;
- extra initial `Z` slack;
- fixed-XY or semiisotropic relaxation;
- separate equilibration of components before final interface assembly;
- staged interface release instead of immediate unrestricted contact.
- explicit substrates such as graphite are treated as reusable construction blocks and stacked into the final cell before export, instead of being repaired afterward in coordinate files.

## 3. Conceptual layers

The package is easiest to understand as five layers.

### 3.1 Molecular identity

Input starts as SMILES, PSMILES, an RDKit molecule, or a MolDB-backed handle.

### 3.2 Molecular preparation

This is where geometry, charges, atom types, and bonded terms become usable for simulation.

### 3.3 System construction

Prepared species are turned into bulk or interfacial systems through composition targets, density targets, explicit cells, and packing logic.

### 3.4 Export and simulation

The system is exported to GROMACS inputs and then passed through staged workflows such as EQ21, additional relaxation, NPT production, or staged interface dynamics.

### 3.5 Analysis and reporting

Analysis outputs are written as structured JSON and related artifacts so later code can consume them programmatically.

## 4. Molecules, handles, and resolved structures

### 4.1 Direct molecule creation

At the top level:

```python
import yadonpy as yp

mol = yp.mol_from_smiles("CCO", name="ethanol")
```

### 4.2 Explicit force-field handles

The explicit style is:

```python
import yadonpy as yp

ff = yp.get_ff("gaff2_mod")
mol = ff.mol("CCO")
ok = ff.ff_assign(mol)
```

`ff.mol(...)` can be a lightweight handle. `ff.ff_assign(...)` is the boundary where that handle is resolved into a prepared RDKit-backed molecule.

This matters because it keeps one script style valid for:

- direct preparation from SMILES;
- MolDB-backed reuse;
- explicit bonded overrides such as `bonded="DRIH"`.

## 5. Persistence model

### 5.1 What belongs in MolDB

Good MolDB candidates:

- PF6 or similar ions after expensive RESP or bonded-patch preparation;
- reusable solvents used in many studies;
- monomers or charged fragments that appear across projects.

### 5.2 What belongs in the work directory

Good workdir artifacts:

- system exports;
- packed-cell build folders;
- EQ21 stage folders;
- production trajectories;
- interface slab and assembled-system files;
- per-study analysis outputs.

### 5.3 Why topology export is rebuildable

The package intentionally regenerates `.gro/.top/.itp/.ndx` outputs because the authoritative state is the prepared molecule plus the current export logic. This avoids silently trusting stale topology trees after code changes.

## 6. Force-field and charge workflow

### 6.1 Charge assignment

The common stages are:

1. generate or load geometry;
2. optionally run conformer search;
3. optionally optimize;
4. assign charges;
5. assign the force field.

### 6.2 Assignment reports

After successful assignment, YadonPy prints a per-atom report by default. This is not decorative. It is a first-line chemistry validation tool.

### 6.3 Bonded patch persistence

Some variants, especially certain ions, require bonded-sidecar data. Current MolDB logic preserves that data with the relevant charge variant so later reuse does not silently lose the bonded override.

## 7. Bulk construction model

Bulk construction is usually one of two kinds.

### 7.1 Density-driven packing

Use this when the final cell does not have to match another system later. A density target is enough to derive a starting box.

### 7.2 Explicit-cell or fixed-XY packing

Use this when one system must later match another system laterally, especially in interface workflows.

The current interface helpers use this model for final electrolyte rebuilds:

- keep polymer `XY` fixed;
- give the initial electrolyte box more `Z` room;
- relax under semiisotropic control.

## 8. Equilibration philosophy

### 8.1 EQ21

EQ21 is the robust staged equilibration preset. It writes a dedicated staged layout and deliberately uses a conservative GROMACS policy for difficult systems.

### 8.2 Additional equilibration rounds

YadonPy can keep relaxing if the current equilibrium checks still say the system is not settled enough.

### 8.3 Final NPT tails

The last NPT stage is often the place where geometry-specific control is applied. For example:

- fixed-XY semiisotropic NPT for standalone electrolyte relaxation;
- production NPT after a dense bulk has stabilized.

## 9. Interface architecture

The interface subsystem is intentionally split into three responsibilities.

### 9.1 Bulk preparation

The bulk sides are prepared and equilibrated independently first. This is not optional complexity. It is part of the success strategy.

### 9.2 Slab build and assembly

`InterfaceBuilder` reads equilibrated bulk states, selects slab windows, unwraps bonded fragments, applies lateral sizing and shifting, assembles the interfacial box, and writes:

- `system.gro`
- `system.top`
- `system.ndx`
- `system_meta.json`
- `protocol_manifest.json`

### 9.3 Interface dynamics

`InterfaceDynamics` is separate from geometry assembly. It runs the staged protocol on top of the already assembled interface.

This separation allows:

- geometry inspection before MD;
- route-specific assembly logic without duplicating MD logic;
- preflight validation of topology and index groups before the run.

## 10. Current interface planning model

### 10.1 Polymer-anchored footprint

The equilibrated polymer `XY` lengths define the shared lateral reference.

### 10.2 Electrolyte planning

For difficult systems the package prefers:

1. determine the compact final target volume relevant to the slab;
2. if needed, build a looser isotropic probe electrolyte first;
3. read the equilibrated probe density response;
4. resize the final electrolyte composition to the polymer-matched footprint;
5. rebuild the final electrolyte in a fixed-XY box with extra `Z` slack.

### 10.3 Route and protocol selection

In the current release, `recommend_polymer_diffusion_interface_recipe(...)` centralizes the route and staged-protocol defaults for polymer/electrolyte diffusion studies.

That helper is meant to answer one narrow question:

"Given the already planned polymer-matched interface geometry, which route and staged protocol should this study use by default?"

Its current behavior is:

- neutral systems stay on route A unless vacuum is explicitly requested;
- polyelectrolyte-style systems default to route B with a vacuum buffer;
- stage durations are lengthened for polyelectrolyte-style systems;
- the early stages keep the gap-hold and density-relax logic explicit.

## 11. Example 12 design in the current release

Example 12 is the most demanding shipped workflow and therefore sets the practical standard for interface robustness.

Its intended chain is:

1. build a large CMC bulk;
2. equilibrate CMC first;
3. use the equilibrated CMC `XY` lengths as the interface footprint;
4. build and equilibrate an isotropic electrolyte probe bulk;
5. resize the final electrolyte composition from the equilibrated probe profile;
6. rebuild and relax the final electrolyte in a fixed-XY box;
7. assemble a route-B vacuum-buffered interface;
8. run staged diffusion dynamics with gradual release.

The script remains explicit, but the route/protocol choice is now library-managed instead of being rebuilt by hand inside the example.

The current release also enforces a stricter bulk-equilibration rule for difficult systems: an energy-minimization stage is not considered reusable if GROMACS reports overlapping atoms or non-finite forces. That behavior is intentional. An invalid EM output is a bad packed structure, not a successful restart checkpoint.

## 12. Output contracts that matter

The following files are behaviorally important, not just temporary output:

- `system.top`
- `system.gro`
- `system.ndx`
- `system_meta.json`
- workdir metadata
- resume metadata
- EQ21 schedule files
- interface protocol manifests

When changing code, these must be treated as contracts.

## 13. Validation expectations

When code changes land, the relevant validation order is:

1. syntax and import sanity;
2. focused unit tests for the changed helpers;
3. release-sanity checks for docs and examples;
4. non-GROMACS validation when local GROMACS is unavailable;
5. real MD only in environments that actually provide GROMACS.

## 14. Maintenance constraints

Current maintenance rules for this release:

- keep the workflow script-first;
- do not hide core study logic behind opaque launchers;
- keep MolDB as the reusable chemistry cache rather than reviving persistent topology-cache logic;
- preserve restart/resume semantics as product behavior;
- treat `__pycache__`, `.pytest_cache`, `.yadonpy_cache`, and `src/yadonpy.egg-info` as disposable generated artifacts.

## 15. Final guidance

YadonPy is easiest to maintain correctly when you keep asking:

- what is the real scientific workflow boundary here;
- what is reusable molecular state versus rebuildable exported state;
- which artifact is the source of truth;
- which stage should absorb the complexity so user scripts can stay explicit but not messy.

That is the reasoning behind the current `v0.8.66` layout.
