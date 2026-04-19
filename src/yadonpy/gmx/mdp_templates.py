"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


MINIM_STEEP_MDP = """; yadonpy: energy minimization (steep, no constraints)
; NOTE:
;   This stage is designed to be robust right after packing.
;   We intentionally disable constraints to avoid LINCS failures.
integrator               = steep
emtol                    = {emtol}
emstep                   = {emstep}
nsteps                   = {nsteps}

; nonbonded settings (yzc-gmx-gen style)
cutoff-scheme            = {cutoff_scheme}
nstlist                  = {nstlist}
verlet-buffer-tolerance  = {verlet_buffer_tolerance}
rlist                    = {rlist}

vdwtype                  = {vdwtype}
rvdw                     = {rvdw}
coulombtype              = {coulombtype}
rcoulomb                 = {rcoulomb}

fourierspacing           = {fourierspacing}
pme-order                = {pme_order}
ewald-rtol               = {ewald_rtol}
DispCorr                 = {dispcorr}

pbc                      = {pbc}
periodic-molecules       = {periodic_molecules}

{wall_mdp}

constraints              = none

{extra_mdp}
"""


MINIM_CG_MDP = """; yadonpy: energy minimization (cg)
integrator               = cg
emtol                    = {emtol}
emstep                   = {emstep}
nsteps                   = {nsteps}

; nonbonded settings (yzc-gmx-gen style)
cutoff-scheme            = {cutoff_scheme}
nstlist                  = {nstlist}
verlet-buffer-tolerance  = {verlet_buffer_tolerance}
rlist                    = {rlist}

vdwtype                  = {vdwtype}
rvdw                     = {rvdw}
coulombtype              = {coulombtype}
rcoulomb                 = {rcoulomb}

fourierspacing           = {fourierspacing}
pme-order                = {pme_order}
ewald-rtol               = {ewald_rtol}
DispCorr                 = {dispcorr}

pbc                      = {pbc}
periodic-molecules       = {periodic_molecules}

{wall_mdp}

constraints              = {constraints}
constraint_algorithm     = {constraint_algorithm}
lincs_iter               = {lincs_iter}
lincs_order              = {lincs_order}

{extra_mdp}
"""


MINIM_STEEP_HBONDS_MDP = """; yadonpy: energy minimization (steep, constraints h-bonds)
; NOTE:
;   This stage is a bridge between steep (no constraints) and CG.
;   Steep can tolerate initial constraint failures that CG cannot handle.
integrator               = steep
emtol                    = {emtol}
emstep                   = {emstep}
nsteps                   = {nsteps}

; nonbonded settings (yzc-gmx-gen style)
cutoff-scheme            = {cutoff_scheme}
nstlist                  = {nstlist}
verlet-buffer-tolerance  = {verlet_buffer_tolerance}
rlist                    = {rlist}

vdwtype                  = {vdwtype}
rvdw                     = {rvdw}
coulombtype              = {coulombtype}
rcoulomb                 = {rcoulomb}

fourierspacing           = {fourierspacing}
pme-order                = {pme_order}
ewald-rtol               = {ewald_rtol}
DispCorr                 = {dispcorr}

pbc                      = {pbc}
periodic-molecules       = {periodic_molecules}

{wall_mdp}

constraints              = {constraints}
constraint_algorithm     = {constraint_algorithm}
lincs_iter               = {lincs_iter}
lincs_order              = {lincs_order}

{extra_mdp}
"""


# Backward-compatible alias: previous releases had MINIM_MDP as a single-stage
# steepest-descent minimization. We now follow the yzc-gmx-gen approach
# (steep/none -> cg/constraints). Any caller importing MINIM_MDP gets the
# CG template (the final minimization stage).
MINIM_MDP = MINIM_CG_MDP


NVT_MDP = """; yadonpy: NVT equilibration
integrator               = md
dt                       = {dt}
nsteps                   = {nsteps}

; nonbonded settings (yzc-gmx-gen style)
cutoff-scheme            = {cutoff_scheme}
nstlist                  = {nstlist}
verlet-buffer-tolerance  = {verlet_buffer_tolerance}
rlist                    = {rlist}

vdwtype                  = {vdwtype}
rvdw                     = {rvdw}
coulombtype              = {coulombtype}
rcoulomb                 = {rcoulomb}

fourierspacing           = {fourierspacing}
pme-order                = {pme_order}
ewald-rtol               = {ewald_rtol}
DispCorr                 = {dispcorr}

pbc                      = {pbc}
periodic-molecules       = {periodic_molecules}

{wall_mdp}

constraints              = {constraints}
constraint_algorithm     = {constraint_algorithm}
lincs_iter               = {lincs_iter}
lincs_order              = {lincs_order}

; temperature coupling
tcoupl                    = V-rescale
tc-grps                   = {tc_grps}
tau_t                     = {tau_t}
ref_t                     = {ref_t}

; velocity generation
gen_vel                   = {gen_vel}
gen_temp                  = {gen_temp}
gen_seed                  = {gen_seed}
; output control
nstxout                  = {nstxout_trr}
; compressed coordinates (xtc)
nstxout-compressed       = {nstxout}
nstvout                  = {nstvout}
nstenergy                = {nstenergy}
nstlog                   = {nstlog}

{extra_mdp}
"""


NPT_MDP = """; yadonpy: NPT equilibration
integrator               = md
dt                       = {dt}
nsteps                   = {nsteps}

; nonbonded settings (yzc-gmx-gen style)
cutoff-scheme            = {cutoff_scheme}
nstlist                  = {nstlist}
verlet-buffer-tolerance  = {verlet_buffer_tolerance}
rlist                    = {rlist}

vdwtype                  = {vdwtype}
rvdw                     = {rvdw}
coulombtype              = {coulombtype}
rcoulomb                 = {rcoulomb}

fourierspacing           = {fourierspacing}
pme-order                = {pme_order}
ewald-rtol               = {ewald_rtol}
DispCorr                 = {dispcorr}

pbc                      = {pbc}
periodic-molecules       = {periodic_molecules}

{wall_mdp}

constraints              = {constraints}
constraint_algorithm     = {constraint_algorithm}
lincs_iter               = {lincs_iter}
lincs_order              = {lincs_order}

; temperature coupling
tcoupl                    = V-rescale
tc-grps                   = {tc_grps}
tau_t                     = {tau_t}
ref_t                     = {ref_t}

; pressure coupling
pcoupl                    = {pcoupl}
pcoupltype                = {pcoupltype}
tau_p                     = {tau_p}
ref_p                     = {ref_p}
compressibility           = {compressibility}

; velocity generation
gen_vel                   = {gen_vel}
gen_temp                  = {gen_temp}
gen_seed                  = {gen_seed}
; output control
nstxout                  = {nstxout_trr}
; compressed coordinates (xtc)
nstxout-compressed       = {nstxout}
nstvout                  = {nstvout}
nstenergy                = {nstenergy}
nstlog                   = {nstlog}

{extra_mdp}
"""


NVT_NO_CONSTRAINTS_MDP = """; yadonpy: NVT equilibration (no constraints)
integrator               = md
dt                       = {dt}
nsteps                   = {nsteps}

; nonbonded settings (yzc-gmx-gen style)
cutoff-scheme            = {cutoff_scheme}
nstlist                  = {nstlist}
rlist                    = {rlist}

vdwtype                  = {vdwtype}
rvdw                     = {rvdw}
coulombtype              = {coulombtype}
rcoulomb                 = {rcoulomb}

fourierspacing           = {fourierspacing}
pme-order                = {pme_order}
ewald-rtol               = {ewald_rtol}
DispCorr                 = {dispcorr}

pbc                      = {pbc}
periodic-molecules       = {periodic_molecules}

{wall_mdp}

constraints              = none

; temperature coupling
tcoupl                    = V-rescale
tc-grps                   = {tc_grps}
tau_t                     = {tau_t}
ref_t                     = {ref_t}

; velocity generation
gen_vel                   = {gen_vel}
gen_temp                  = {gen_temp}
gen_seed                  = {gen_seed}
; output control
nstxout                  = {nstxout_trr}
; compressed coordinates (xtc)
nstxout-compressed       = {nstxout}
nstvout                  = {nstvout}
nstenergy                = {nstenergy}
nstlog                   = {nstlog}

{extra_mdp}
"""


NPT_NO_CONSTRAINTS_MDP = """; yadonpy: NPT equilibration (no constraints)
integrator               = md
dt                       = {dt}
nsteps                   = {nsteps}

; nonbonded settings (yzc-gmx-gen style)
cutoff-scheme            = {cutoff_scheme}
nstlist                  = {nstlist}
rlist                    = {rlist}

vdwtype                  = {vdwtype}
rvdw                     = {rvdw}
coulombtype              = {coulombtype}
rcoulomb                 = {rcoulomb}

fourierspacing           = {fourierspacing}
pme-order                = {pme_order}
ewald-rtol               = {ewald_rtol}
DispCorr                 = {dispcorr}

pbc                      = {pbc}
periodic-molecules       = {periodic_molecules}

{wall_mdp}

constraints              = none

; temperature coupling
tcoupl                    = V-rescale
tc-grps                   = {tc_grps}
tau_t                     = {tau_t}
ref_t                     = {ref_t}

; pressure coupling
pcoupl                    = {pcoupl}
pcoupltype                = {pcoupltype}
tau_p                     = {tau_p}
ref_p                     = {ref_p}
compressibility           = {compressibility}

; velocity generation
gen_vel                   = {gen_vel}
gen_temp                  = {gen_temp}
gen_seed                  = {gen_seed}
; output control
nstxout                  = {nstxout_trr}
; compressed coordinates (xtc)
nstxout-compressed       = {nstxout}
nstvout                  = {nstvout}
nstenergy                = {nstenergy}
nstlog                   = {nstlog}

{extra_mdp}
"""


DEFORM_NVT_MDP = """; yadonpy: uniaxial deformation (NVT + deform)
integrator               = md
dt                       = {dt}
nsteps                   = {nsteps}

cutoff-scheme            = Verlet
nstlist                  = 20
rlist                    = {rlist}
rcoulomb                 = {rcoulomb}
rvdw                      = {rvdw}

coulombtype               = PME
vdwtype                   = Cut-off
pbc                       = {pbc}
periodic-molecules        = {periodic_molecules}

{wall_mdp}

constraints               = h-bonds
constraint_algorithm      = lincs

; temperature coupling
tcoupl                    = V-rescale
tc-grps                   = System
tau_t                     = {tau_t}
ref_t                     = {ref_t}

; no pressure coupling when deforming
pcoupl                    = no

; deformation (nm/ps) for box vectors (x y z xy xz yz)
deform                   = {deform_x} {deform_y} {deform_z} 0 0 0
refcoord_scaling         = com

; output control
nstxout                  = {nstxout_trr}
; compressed coordinates (xtc)
nstxout-compressed       = {nstxout}
nstvout                  = {nstvout}
nstenergy                = {nstenergy}
nstlog                   = {nstlog}
"""


@dataclass(frozen=True)
class MdpSpec:
    template: str
    params: Dict[str, object]

    def render(self) -> str:
        # ------------------------------------------------------------------
        # Safety/robustness policy
        # ------------------------------------------------------------------
        # Historically, some workflows used the Parrinello–Rahman barostat
        # early in equilibration. For low-density packed cells this can lead
        # to pathological volume/step behavior and downstream performance
        # problems. YadonPy defaults to the modern stochastic cell-rescale
        # barostat (C-rescale; Bernetti & Bussi 2020) and *forbids* emitting
        # Parrinello–Rahman in generated mdp files.
        #
        # We enforce this here (close to the final mdp text) so that even
        # user-provided overrides cannot accidentally re-enable PR.
        params = dict(self.params)
        if "periodic-molecules" in params and "periodic_molecules" not in params:
            params["periodic_molecules"] = params["periodic-molecules"]
        pc = params.get("pcoupl")
        if isinstance(pc, str):
            s = pc.strip().lower().replace("_", "-")
            if s in {
                "parrinello-rahman",
                "parrinello rahman",
                "parrinello",
                "p-r",
                "pr",
            }:
                params["pcoupl"] = "C-rescale"

        return self.template.format(**params)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
        return path


def default_mdp_params() -> Dict[str, object]:
    return {
        "emtol": 1000.0,
        "emstep": 0.01,
        "nsteps": 5000,
        "dt": 0.002,
        # yzc-gmx-gen style nonbond defaults
        "cutoff_scheme": "Verlet",
        "nstlist": 20,
        "verlet_buffer_tolerance": 0.005,
        "rlist": 1.2,
        "rcoulomb": 1.2,
        "rvdw": 1.2,
        "vdwtype": "Cut-off",
        "coulombtype": "PME",
        "pme_order": 4,
        "fourierspacing": 0.12,
        "ewald_rtol": 1.0e-5,
        "dispcorr": "EnerPres",
        # constraints (robust default for polymer/ionic systems)
        "constraints": "h-bonds",
        "constraint_algorithm": "lincs",
        "lincs_iter": 2,
        "lincs_order": 8,
        # thermostat
        "tc_grps": "System",
        "tau_t": 0.5,
        "ref_t": 298.15,
        # barostat (default: C-rescale; PR is forbidden in MdpSpec.render())
        "pcoupl": "C-rescale",
        "pcoupltype": "isotropic",
        "tau_p": 2.0,
        "ref_p": 1.0,
        "compressibility": 4.5e-5,
        "deform_x": 0.0,
        "deform_y": 0.0,
        "deform_z": 0.0,
        # velocity generation seed (for NVT when gen_vel=yes)
        "gen_seed": -1,
        "gen_vel": "no",
        "gen_temp": 298.15,
        # output control (yzc-gmx-gen style: lean trajectory by default)
        # - trr/velocity output disabled unless user opts in
        # - xtc written every 10k steps (20 ps at dt=2 fs)
        "nstxout": 10000,
        "nstxout_trr": 10000,
        "nstvout": 10000,
        "nstenergy": 10000,
        "nstlog": 10000,
        "extra_mdp": "",
        "pbc": "xyz",
        "periodic_molecules": "no",
        "wall_mdp": "",
    }
