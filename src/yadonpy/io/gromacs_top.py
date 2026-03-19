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


def defaults_block(*, comb_rule: int = 2, gen_pairs: str = "yes", scaling: AmberGaffScaling = AmberGaffScaling()) -> str:
    """Return a GROMACS [ defaults ] block with GAFF scaling."""
    # nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ
    return (
        "[ defaults ]\n"
        "; nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ\n"
        f"  1  {comb_rule}  {gen_pairs}  {scaling.fudge_lj:.6f}  {scaling.fudge_qq:.10f}\n\n"
    )
