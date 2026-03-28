# YadonPy Manual (v0.8.75)

## 1. Purpose

This manual describes the architectural rules of YadonPy. It is intended for users and developers who need to understand why the workflow is structured the way it is, what data are persistent, and where physical or software-level assumptions enter the pipeline.

Related documents:

- API reference: `docs/Yadonpy_API_v0.8.75.md`
- user guide: `docs/Yaonpyd_user_guide.md`

## 2. Architectural principles

### 2.1 Script-first workflow

YadonPy is designed so that the scientific protocol remains legible in the user script. Helper APIs reduce duplication, but they do not replace the explicit statement of:

- molecular identities;
- composition;
- target density or explicit cell dimensions;
- force field;
- equilibration strategy;
- interface route and protocol.

### 2.2 MolDB-first persistence

MolDB is the persistent cache for expensive molecular preparation. Its role is to preserve:

- geometry;
- charge variants;
- charge/basis/method metadata;
- bonded-patch sidecars when required.

From `v0.8.72`, charge variants may also preserve grouped-polyelectrolyte metadata:

- `polyelectrolyte_mode`
- `polyelectrolyte_detection`
- `constraint_signature`
- grouped charge definitions
- grouped RESP constraint payloads

MolDB variant lookup now uses those fields as part of compatibility selection. This prevents an ordinary RESP variant from being mistaken for a grouped-polyelectrolyte RESP variant for the same monomer or repeat unit.

From `v0.8.75`, the release no longer ships a bundled MolDB archive. Reference species are rebuilt explicitly from plain CSV inputs under `examples/07_moldb_precompute_and_reuse/`.

### 2.2.1 Polyelectrolyte variant discipline

For charged polymer monomers or repeat units, the persistent record must preserve three independent layers:

- chemistry identity (`canonical`, `kind`, `connectors`)
- QM provenance (`charge`, `basis_set`, `method`)
- grouped-charge semantics (`polyelectrolyte_mode`, `polyelectrolyte_detection`, `constraint_signature`, `charge_groups`, `resp_constraints`)

The grouped-charge semantics are treated as first-class cache identity, not as optional annotations.

System-level GROMACS exports are treated as reproducible products, not as the primary persistent source.

### 2.3 Restart-aware work directories

Work directories are structured execution records. Restart logic is expected to reuse valid outputs, but only when their inputs and schema remain compatible with the current run definition.

### 2.4 Conservative physical staging

For difficult systems, YadonPy prefers better staging over brute-force retries. This includes:

- lower initial packing density;
- additional free volume along the least constrained axis;
- separated component equilibration;
- staged contact and release for interfaces;
- explicit substrate blocks instead of post hoc coordinate editing.

## 3. Layer model

YadonPy can be understood as five layers.

### 3.1 Molecular identity

Input begins as SMILES, PSMILES, RDKit molecules, or MolDB-backed handles.

### 3.2 Molecular preparation

This layer assigns:

- geometry;
- charges;
- atom types;
- bonded terms;
- reusable metadata.

### 3.3 System construction

Prepared species are converted into bulk or interface systems through:

- composition targets;
- density or explicit box constraints;
- packing and stacking logic;
- species-level metadata propagation.

### 3.4 Export and simulation

System state is exported into GROMACS artifacts and passed into staged workflows such as EQ21, additional relaxation, interface diffusion, elongation, or glass-transition protocols.

### 3.5 Analysis and reporting

Analysis is written as files that are both human-readable and machine-readable, so later code can consume the outputs programmatically. From `v0.8.73`, the analysis layer is also expected to preserve physical semantics explicitly rather than inferring them from loose moltype selections.

From `v0.8.74`, build/export correctness is tightened as well:

- strict input checking is the default resume behavior for build/export/interface stages;
- implicit ion registry injection is removed; ions must be passed explicitly into `amorphous_cell(...)`;
- exported systems now include `site_map.json` and `export_manifest.json`;
- large-system packing writes `.yadonpy/amorphous_cell_pack_diagnostics.json` when a work directory is available.

## 4. Charge workflow model

### 4.1 Separation of raw charge template and simulation scaling

YadonPy now enforces a strict separation between:

- **raw molecular charges**, produced by QM fitting and eligible for MolDB persistence;
- **simulation-level scaled charges**, applied only during export or system construction.

The raw RESP template is the authoritative molecular charge state. Scaling is a system-level modeling choice.

### 4.2 PsiRESP as the RESP/ESP backend

From `v0.8.71`, RESP and ESP are implemented through **PsiRESP** rather than the old Psi4 `resp` plugin path.

The reason is architectural rather than cosmetic. The workflow requires:

- explicit charge-sum constraints;
- explicit equivalence constraints;
- auditable grouped constraints for charged polymer motifs.

PsiRESP provides these facilities in a stable and inspectable form.

### 4.3 Polyelectrolyte mode

`polyelectrolyte_mode=True` activates grouped charge detection and grouped RESP constraints.

The workflow is:

1. detect charged groups by built-in template;
2. if template matching fails, try graph-based local charged-subgraph detection;
3. create one charge-sum constraint per charged group;
4. constrain the neutral remainder as one charge-sum region;
5. add conservative equivalence constraints on the neutral remainder only;
6. preserve the resulting metadata on the molecule and in exports.

### 4.4 Failure policy

If grouped charged-region identification fails:

- RESP is still allowed to finish;
- the fallback is recorded explicitly;
- downstream charge scaling falls back to whole-molecule scaling;
- the failure is not silent.

## 5. Polyelectrolyte metadata model

The current implementation preserves four categories of metadata:

- `charge_groups.json`
- `resp_constraints.json`
- `residue_map.json`
- `charge_scaling_report.json`

These files are written during export and are intended for:

- tracing atom-index groups back to chemistry;
- validating grouped charge constraints;
- validating local scaling behavior;
- external analysis by residue or charged-group.

At the MolDB layer, the same grouped information is now preserved per charge variant when it exists on the source molecule. This is necessary because two RESP fits with the same `charge/basis/method` but different grouped-constraint logic should not collapse onto one indistinguishable database variant.

## 6. Residue model for polymers

### 6.1 Residues are monomer-level export units

For polymeric species, YadonPy now preserves residue information through export:

- each repeat unit is written as a residue;
- terminal groups remain separate residues when available;
- small molecules and ions remain single-residue species.

### 6.2 Why this matters

This change is required for rigorous local scaling and later analysis, because charged-group indices must remain auditable after system export.

## 7. Bulk construction model

### 7.1 Density-driven packing

This mode is appropriate when no later interface requires an externally fixed lateral footprint.

### 7.2 Explicit-cell or fixed-lateral-dimension packing

This mode is required when a bulk phase will later be assembled into an interface or stacked against a substrate or another phase.

## 8. Interface model

The interface subsystem is deliberately split into three responsibilities:

1. **bulk preparation**
2. **geometric assembly**
3. **interface dynamics**

This separation allows geometry inspection, cached reuse, and route-specific assembly logic without entangling all MD logic into one monolithic path.

### 8.1 Route A

Route A is the fully periodic interface workflow.

### 8.2 Route B

Route B is the vacuum-buffered and wall-ready interface workflow for `pbc = xy` use cases.

## 9. Substrate model

Graphite and other explicit substrates are treated as reusable construction blocks. They are built as proper molecular/cell objects and stacked into the final cell before export rather than being repaired afterward in coordinate files.

## 10. Export model

`export_system_from_cell_meta(...)` is the main system-export bridge for modern workflows.

Its responsibilities include:

- resolving per-species artifacts;
- preserving molecule type names;
- preserving polymer residue identity when present;
- applying simulation-level charge scaling;
- generating topology, coordinate, and index artifacts;
- writing machine-readable metadata about charge scaling and residues.

## 11. Simulation presets

### 11.1 EQ21

EQ21 remains the main conservative equilibration preset. It is intended for robust equilibration rather than minimal wall-clock cost.

### 11.2 NPT reporting

NPT-capable workflows now generate a convergence plot that overlays:

- density;
- volume;
- three box lengths.

The plot is written in normalized form so the series can be interpreted together despite differing units.

### 11.3 Structured post-processing

From `v0.8.73`, three rules are fixed.

1. **MSD is metric-specific**

- single-atom ions default to `ion_atomic_msd`;
- small molecules default to `molecule_com_msd`;
- polymers default to `chain_com_msd`;
- polymers may additionally expose `residue_com_msd` and `charged_group_com_msd`.

2. **RDF/CN are site-first**

- site-level coordination analysis is the default;
- atomtype-wide RDF is retained only as an explicit auxiliary mode;
- center-species resolution is fail-closed, not heuristic.

3. **Charged-polymer conductivity is group-based**

- Nernst-Einstein conductivity uses charged-group diffusion coefficients when charged-group metadata are available;
- the charge carried into the equation is the group formal charge, not the whole-chain net charge;
- if a charged polymer lacks charged-group MSD metadata, its conductivity contribution is omitted and recorded as such.

## 12. Packaging and release hygiene

Release trees should not retain generated clutter such as:

- `__pycache__`
- `.pytest_cache`
- `.yadonpy_cache`
- `yadonpy.egg-info`

The current release process also requires:

- archive of the previous numbered release into `history_version`;
- only the current version directory and current `.tar` in the root release tree;
- GitHub synchronization after each update.

## 13. Environment baseline

The recommended baseline for QM-enabled workflows is:

- Python `3.11`
- `psi4`
- `psiresp`
- `dftd3-python`
- `rdkit`

Because Psi4 and PsiRESP are most stable through conda, the project documentation treats conda as the primary installation path for the QM stack.

## 14. Current limitations

The current polyelectrolyte implementation is deliberately conservative.

Known limits include:

- charged-group detection is template-first and therefore not exhaustive;
- graph fallback is structural, not chemically complete for all exotic motifs;
- local charge scaling currently preserves grouped totals but does not attempt a second constrained redistribution on the neutral remainder after scaling;
- site classification is conservative and therefore not exhaustive for every unusual heteroatom environment;
- charged-polymer conductivity requires charged-group metadata and will be omitted rather than guessed if that metadata are absent.

These limits are explicit design tradeoffs in favor of auditable behavior over aggressive automatic inference.
