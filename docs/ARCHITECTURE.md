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

## 4. QM and RESP Model

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

## 5. Polymer Construction Rules

Polymer workflows rely on PSMILES, explicit terminal groups, and a random-walk builder.

Important implications:

- terminal groups are part of the chemistry definition;
- the special hydrogen placeholder is not equivalent to an ordinary chemical terminal group;
- retry budgets and rollback behavior directly affect whether a polymerization attempt is robust;
- long or rigid chains should not be forced through the same packing assumptions as short flexible chains.

The current builder logic prefers minimum-correct robustness over brittle short-budget defaults.

## 6. Bulk-First Interface Logic

YadonPy’s current interface philosophy is bulk first:

1. equilibrate each phase independently;
2. extract slabs from equilibrated bulk;
3. align lateral dimensions carefully;
4. for graphite-assisted sandwich systems, pre-relax each soft-phase slab in a confined `pbc=xy` box before final assembly;
5. assemble the interface or sandwich structure;
5. run staged relaxation.

This avoids a common failure mode where an interface is assembled from unrealistic
unrelaxed packed phases and then expected to fix itself during MD.

The graphite-polymer-electrolyte sandwich builder follows the same rule.
It is meant to be the cleanest high-level expression of this architecture.

## 7. Export Artifacts Are Contractual

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

## 8. Force-Field Strategy

YadonPy deliberately supports multiple force-field families because different subproblems
need different defaults:

- `gaff`, `gaff2`, `gaff2_mod` for common organic and electrolyte preparation;
- `merz` for ion-centered models;
- `oplsaa` for OPLS-AA based workflows;
- `dreiding` for broader generic typing coverage.

The force-field registry is designed so the user can select one canonical family name
without needing to know the import path of the implementation class.

## 9. Analysis Strategy

Analysis is not limited to one-size-fits-all whole-molecule metrics.
The package distinguishes between:

- atomic ions,
- molecular centers of mass,
- chain centers of mass,
- residue centers of mass,
- charged-group centers of mass.

That distinction is especially important in polymer electrolytes and charged polymers,
where whole-chain net-charge diffusion can hide the behavior that actually matters physically.

## 10. Guidance for Future Changes

When modifying YadonPy, keep these checks in mind:

- Does the change preserve the script-first workflow style?
- Does it keep restart behavior coherent?
- Does it respect MolDB as the persistent preparation layer?
- Does it preserve or intentionally update exported metadata and manifests?
- Does it make the full scientific workflow more reliable, not just one local function?

If a change only passes a local function test but breaks bulk construction, interface
assembly, or metadata recovery, it is not architecturally complete.
