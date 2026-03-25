# Si-H QM Source Note (2026-03-25)

This note records the origin of the explicit Si-H bonded parameters shipped in `gaff2_mod.json`.

## Calculation source

- host: `zcyu@192.9.207.150`
- path: `/home/zcyu/si_h_qm_probe_20260325`
- code: `yadonpy_v0.8.62/src`
- engine: `yadonpy.sim.qm.bond_angle_params_mseminario`
- QM backend: `Psi4`
- level of theory: `wB97M-D3BJ / def2-SVP`
- workflow: geometry optimization + Hessian + modified Seminario bond/angle projection

## Probe set

- `si,hi` and `hi,si,hi`: `[SiH4]`, `C[SiH3]`, `O[SiH3]`, `[SiH3]O[SiH3]`
- `ci,si,hi`: `C[SiH3]`
- `oi,si,hi`: `O[SiH3]`
- `oss,si,hi`: `[SiH3]O[SiH3]`

## Adopted explicit terms

- `si,hi`: `r0 = 0.148585 nm`, `k = 166875.184 kJ/mol/nm^2`
- `hi,si,hi`: `theta0 = 109.122 deg`, `k = 388.318 kJ/mol/rad^2`
- `ci,si,hi`: `theta0 = 110.638 deg`, `k = 435.224 kJ/mol/rad^2`
- `oi,si,hi`: `theta0 = 110.000 deg`, `k = 555.292 kJ/mol/rad^2`
- `oss,si,hi`: `theta0 = 109.659 deg`, `k = 464.084 kJ/mol/rad^2`

## Non-QM term retained intentionally

- `hi,si,oss,si` is still a surrogate compatibility torsion cloned from `X,c3,os,X`.
- Reason: modified Seminario in YadonPy derives bond and angle terms only; it does not derive torsions.

## Raw summaries

- `docs/si_h_qm_probe_20260325_summary.json`
- `docs/si_h_qm_probe_20260325_typed_summary.json`
