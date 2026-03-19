"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

from __future__ import annotations

from . import eq

__all__ = ["eq", "PresetNotPortedError", "not_ported"]


class PresetNotPortedError(NotImplementedError):
    """Raised when a legacy RadonPy preset is requested but not ported."""


def not_ported(name: str) -> None:
    raise PresetNotPortedError(
        f"Preset '{name}' is not ported to GROMACS yet. "
        "Use `yadonpy.gmx.workflows` for supported presets (eq/tg/elong/quick)."
    )
