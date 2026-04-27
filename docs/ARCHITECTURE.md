# YadonPy Architecture

This document explains the design rules behind YadonPy. It is intended for users who
need to understand why the workflow is structured the way it is, and for maintainers who
need to modify the code without breaking the scientific model.

## 1. Script-First by Design

YadonPy is not trying to hide a study protocol behind a large framework.
The intended use is an explicit Python script where the user can still see:

- what molecules are present,
- how charges were assigned,
- which force field was chosen,
- what density targets were used,
- how the bulk or interface was equilibrated,
- which outputs are authoritative.

High-level helpers exist to reduce duplication, not to obscure the workflow.

## 2. The Persistent Layer Is MolDB, Not Exported Topology Files

MolDB is the place for expensive reusable molecular assets:

- canonicalized SMILES or PSMILES,
- prepared 3D geometry,
- charge variants,
- grouped-polyelectrolyte RESP metadata,
- optional bonded patches such as DRIH or modified Seminario outputs.

Exported `.gro`, `.top`, `.itp`, and `.ndx` files are still important artifacts,
but they are generated outputs of a workflow step, not the canonical reusable store.

This matters because:

- topologies can depend on the target force field,
- export options can change,
- charge scaling can be a simulation-time decision,
- interface assembly often needs fresh geometry derived from equilibrated bulk.

## 3. Restart and Resume Are Part of the Product Behavior

Work directories are not throwaway implementation details.

YadonPy uses them to preserve:

- step outputs,
- restart markers,
- manifest files,
- representative structures,
- charge metadata,
- interface assembly records,
- analysis summaries.

When editing or extending the code, avoid:

- silent destructive cleanup,
- hidden reruns,
- accidental reuse of stale state under changed inputs.

The runtime API exists so users can explicitly control restart defaults instead of
each script inventing its own convention.

## 4. Typed Metadata and Schema Boundaries

YadonPy stores provenance in RDKit molecule properties and JSON artifacts. New code should
not hand-roll `HasProp/GetProp/json.loads` or schema stamping. Use the central helpers in
`yadonpy.core.metadata` for:

- RESP/PsiRESP constraint metadata;
- charge-model provenance and QM recipes;
- equilibrium-state summaries;
- benchmark/species-forcefield summaries;
- stable JSON writing with schema versions.

This keeps old MolDB records loadable while making new artifacts auditable and harder to
silently mix across charge recipes, RESP profiles, and benchmark variants.

Example scripts should keep their script-first shape, but shared parsing and runtime
defaults should move into small workflow helpers such as `yadonpy.workflow.EnvReader`.

## 5. QM and RESP Model

The intended RESP path is:

1. Psi4 performs the actual QM calculation.
2. `psiresp-base` handles orientation generation, ESP fitting inputs, and RESP fitting.
3. YadonPy stores the resulting charges and relevant metadata on the molecule and in MolDB.

For charged polymers, grouped-polyelectrolyte RESP is treated as a distinct workflow mode.
That means:

- charged groups are detected explicitly,
- constraints are preserved as metadata,
- later scaling can remain local to the charged groups,
- MolDB can distinguish a grouped-polyelectrolyte variant from an ordinary RESP fit.

## 6. Polymer Construction Rules

Polymer workflows rely on PSMILES, explicit terminal groups, and a random-walk builder.

Important implications:

- terminal groups are part of the chemistry definition;
- the special hydrogen placeholder is not equivalent to an ordinary chemical terminal group;
- retry budgets and rollback behavior directly affect whether a polymerization attempt is robust;
- long or rigid chains should not be forced through the same packing assumptions as short flexible chains.

The current builder logic prefers minimum-correct robustness over brittle short-budget defaults.

## 7. Bulk-First Interface Logic

YadonPy’s current interface philosophy is bulk first, but not bulk-direct-to-stack:

1. equilibrate each phase independently;
2. keep those bulk runs as calibration references for density, composition, and packing difficulty;
3. negotiate the graphite master footprint once in `XY`;
4. rebuild each soft phase directly on that final `XY` footprint with repulsive-only Z walls and explicit vacuum;
5. for graphite-assisted sandwich systems, pre-relax each rebuilt soft phase in a confined `pbc=xy` box before final assembly;
6. assemble the interface or sandwich structure;
7. run staged relaxation.

This avoids a common failure mode where an interface is assembled from unrealistic
unrelaxed packed phases and then expected to fix itself during MD. It also avoids
letting a periodic cut-slab artifact drive the graphite footprint.

The graphite-polymer-electrolyte sandwich builder follows the same rule.
It is meant to be the cleanest high-level expression of this architecture.

## 8. Export Artifacts Are Contractual

Generated files and manifests are part of the behavior:

- `.gro`
- `.top`
- `.itp`
- `.ndx`
- `export_manifest.json`
- `site_map.json`
- `charge_groups.json`
- `resp_constraints.json`
- `residue_map.json`
- interface build manifests

Changing the build logic without checking the exported metadata is incomplete maintenance.

## 9. Force-Field Strategy

YadonPy deliberately supports multiple force-field families because different subproblems
need different defaults:

- `gaff`, `gaff2`, `gaff2_mod` for common organic and electrolyte preparation;
- `merz` for ion-centered models;
- `oplsaa` for OPLS-AA based workflows;
- `dreiding` for broader generic typing coverage.

The force-field registry is designed so the user can select one canonical family name
without needing to know the import path of the implementation class.

## 10. Analysis Strategy

Analysis is not limited to one-size-fits-all whole-molecule metrics.
The package distinguishes between:

- atomic ions,
- molecular centers of mass,
- chain centers of mass,
- residue centers of mass,
- charged-group centers of mass.

That distinction is especially important in polymer electrolytes and charged polymers,
where whole-chain net-charge diffusion can hide the behavior that actually matters physically.

## 11. Refactor Roadmap

The preferred migration path is incremental rather than a big rewrite:

- Phase 1: consolidate metadata accessors, workflow config, MolDB/RESP provenance checks,
  and schema round-trip tests without changing simulation physics.
- Phase 2: split EQ21, LiquidAnneal, recovery, production, gates, and MDP policy into
  smaller internal stage/strategy modules while preserving the public `eq` facade.
- Phase 3: split export and MolDB responsibilities so molecule compatibility, charge
  scaling, system metadata, and topology assembly are independently testable.
- Phase 4: move repeated benchmark-script infrastructure into library helpers so examples
  remain readable scientific protocols instead of hidden product logic.
- Phase 5: add developer-facing checks such as `doctor --full`, MolDB audit, workflow
  dry-run, schema validation, and smoke matrices for common workflows.

## 12. Guidance for Future Changes

When modifying YadonPy, keep these checks in mind:

- Does the change preserve the script-first workflow style?
- Does it keep restart behavior coherent?
- Does it respect MolDB as the persistent preparation layer?
- Does it preserve or intentionally update exported metadata and manifests?
- Does it make the full scientific workflow more reliable, not just one local function?

If a change only passes a local function test but breaks bulk construction, interface
assembly, or metadata recovery, it is not architecturally complete.
