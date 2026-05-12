# YadonPy Technical Notes

This document collects technical provenance notes that are useful for auditing packaged
data and force-field behavior but do not belong in the main user-facing guide.

## 1. OPLS-AA Import Provenance

YadonPy’s packaged OPLS-AA data was refreshed from the upstream `moltemplate` project.

### Upstream source

- Repository: `https://github.com/jewettaij/moltemplate`
- Primary upstream inputs:
  - `moltemplate/force_fields/oplsaa2024.lt`
  - `moltemplate/force_fields/oplsaa2024_original_format/Jorgensen_et_al-2024-The_Journal_of_Physical_Chemistry_B.sup-2.par`

### Local importer

- Import script: `tools/oplsaa/import_moltemplate_oplsaa2024.py`

### What the importer does

1. Rebuilds the OPLS-AA particle table from the upstream parameter source.
2. Merges bonded coefficient tables from `oplsaa2024.lt` into YadonPy’s packaged JSON.
3. Preserves YadonPy’s local SMARTS typing layer, which is necessary because the upstream
   source does not provide a ready-to-use RDKit rule table for all supported chemistries.

### Unit conversions

- Lennard-Jones `sigma`: Angstrom -> nm via `0.1`
- Lennard-Jones `epsilon`: kcal/mol -> kJ/mol via `4.184`
- Bond `k`: converted to GROMACS harmonic function-1 form via `k * 4.184 * 200`
- Bond `r0`: Angstrom -> nm via `0.1`
- Angle `k`: converted to GROMACS harmonic function-1 form via `k * 4.184 * 2`
- Dihedral Fourier terms: `abs(Vn) * 4.184 / 2`, with phase determined from the OPLS sign convention

The `200` and `2` factors are intentional. LAMMPS/moltemplate harmonic bond and
angle coefficients use an energy form without the leading `0.5`; GROMACS
function 1 uses `0.5 * k * dx^2`. Using `400`/`4` would make the exported
GROMACS bond and angle terms exactly twice as stiff. The helper
`audit_bundled_oplsaa_parameter_sanity()` checks sentinel terms such as
`CT-HC` and `HC-CT-HC` to catch this regression before launching MD.

### OPLS-AA polyelectrolyte EQ21 start-up

For CMC-Na / carbonate / LiPF6 test systems, the unstable part of the workflow
was not the final `2 fs + h-bonds` production setting.  The failure occurred
during the first unconstrained EQ21 NVT thermalization: multi-chain CMC-Na cells
could develop local X-H / ionized-side-group instabilities at `1 fs`, while the
same EM output survived a short `0.5 fs` pre-NVT and then continued at `1 fs`.

YadonPy therefore applies a narrow automatic safeguard:

- if the exported system metadata contains an OPLS-AA species with
  `polyelectrolyte_mode=True` or localized charge groups,
- and the user did not explicitly set `eq21_pre_nvt_dt_ps`,
- EQ21 resolves `02_preNVT` to `min(eq21_dt_ps, 0.0005 ps)`.

Later EQ21 stages still use the normal robust timestep policy, and production
can still use the default `h-bonds + 2 fs` setting after the constrained-settle
bridge.  This is a start-up stabilization policy, not a change to the intended
production ensemble.

## 2. Si-H Parameter Provenance

YadonPy ships explicit Si-H bonded terms in `gaff2_mod.json`.

### Calculation source

- Host: `zcyu@192.9.207.150`
- Engine: `yadonpy.sim.qm.bond_angle_params_mseminario`
- QM backend: `Psi4`
- Workflow: geometry optimization + Hessian + modified Seminario projection
- Level of theory: `wB97M-D3BJ / def2-SVP`

### Probe set

- `[SiH4]`
- `C[SiH3]`
- `O[SiH3]`
- `[SiH3]O[SiH3]`

These probes were used to populate the currently packaged explicit terms:

- `si,hi`
- `hi,si,hi`
- `ci,si,hi`
- `oi,si,hi`
- `oss,si,hi`

### Adopted packaged values

- `si,hi`: `r0 = 0.148585 nm`, `k = 166875.184 kJ/mol/nm^2`
- `hi,si,hi`: `theta0 = 109.122 deg`, `k = 388.318 kJ/mol/rad^2`
- `ci,si,hi`: `theta0 = 110.638 deg`, `k = 435.224 kJ/mol/rad^2`
- `oi,si,hi`: `theta0 = 110.000 deg`, `k = 555.292 kJ/mol/rad^2`
- `oss,si,hi`: `theta0 = 109.659 deg`, `k = 464.084 kJ/mol/rad^2`

### Intentional limitation

The common `hi,si,oss,si` torsion is still a conservative surrogate cloned from
`X,c3,os,X`. That is intentional: the current modified Seminario workflow derives
bond and angle terms, not torsions.

## 3. Why These Notes Stay Separate

These records are important for:

- reproducing packaged force-field data,
- auditing where specific parameters came from,
- understanding why certain terms are explicit while others remain surrogate values.

They are not part of the ordinary workflow entry points, so they live here instead of
inflating the user guide or README.
