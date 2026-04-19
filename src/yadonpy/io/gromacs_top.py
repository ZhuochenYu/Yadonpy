"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AmberGaffScaling:
    """Default GAFF/AMBER 1-4 scaling for GROMACS topologies."""

    fudge_lj: float = 0.5
    fudge_qq: float = 0.8333333333


@dataclass(frozen=True)
class GromacsDefaults:
    """A complete GROMACS ``[ defaults ]`` specification."""

    comb_rule: int = 2
    gen_pairs: str = "yes"
    fudge_lj: float = 0.5
    fudge_qq: float = 0.8333333333


def defaults_for_ff_name(ff_name: str | None) -> GromacsDefaults:
    """Return conservative GROMACS ``[ defaults ]`` for a force-field family."""
    name = str(ff_name or "").strip().lower()
    if name == "oplsaa":
        # OPLS-AA / Jorgensen defaults: geometric LJ combination and 0.5/0.5 1-4 scaling.
        return GromacsDefaults(comb_rule=3, gen_pairs="yes", fudge_lj=0.5, fudge_qq=0.5)
    if name in {"dreiding", "dreiding_ut"}:
        return GromacsDefaults(comb_rule=2, gen_pairs="yes", fudge_lj=0.5, fudge_qq=0.8333333333)
    return GromacsDefaults()


def defaults_block(
    *,
    comb_rule: int = 2,
    gen_pairs: str = "yes",
    scaling: AmberGaffScaling = AmberGaffScaling(),
) -> str:
    """Return a GROMACS ``[ defaults ]`` block."""
    defaults = GromacsDefaults(
        comb_rule=int(comb_rule),
        gen_pairs=str(gen_pairs),
        fudge_lj=float(scaling.fudge_lj),
        fudge_qq=float(scaling.fudge_qq),
    )
    return defaults_block_from_spec(defaults)


def defaults_block_from_spec(defaults: GromacsDefaults) -> str:
    """Render a GROMACS ``[ defaults ]`` block from a resolved spec."""
    return (
        "[ defaults ]\n"
        "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ\n"
        f"  1  {int(defaults.comb_rule)}  {str(defaults.gen_pairs)}  "
        f"{float(defaults.fudge_lj):.6f}  {float(defaults.fudge_qq):.10f}\n\n"
    )
