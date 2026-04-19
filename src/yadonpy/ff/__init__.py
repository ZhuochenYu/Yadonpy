"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from .gaff import GAFF
from .gaff2 import GAFF2
from .gaff2_mod import GAFF2_mod
from .dreiding import Dreiding

# Added: OPLS-AA (SMARTS-based typing + ffoplsaa-style parameter DB)
from .oplsaa import OPLSAA
from .oplsaa_reference import audit_oplsaa_reference

# Added: Merz ion FF (OPC/OPC3 ion LJ parameters)
from .merz import MERZ
from .registry import available_forcefields, canonical_forcefield_name, create_forcefield

 # (basic_top/library subsystem removed in v0.6.6; MolDB is the only cache.)


def gaff2(*, variant: str = "mod", **kwargs):
    """Factory for GAFF2-family force fields.

    By default, YadonPy uses the more robust GAFF2_mod variant.

    Args:
        variant: "mod" (default) -> GAFF2_mod, "classic" -> GAFF2
        **kwargs: forwarded to the underlying class constructor
    """
    v = (variant or "mod").lower().strip()
    if v in ("classic", "gaff2", "orig", "original"):
        return GAFF2(**kwargs)
    return GAFF2_mod(**kwargs)

__all__ = [
    "GAFF",
    "GAFF2",
    "GAFF2_mod",
    "Dreiding",
    "OPLSAA",
    "audit_oplsaa_reference",
    "MERZ",
    "gaff2",
    "available_forcefields",
    "canonical_forcefield_name",
    "create_forcefield",
]
