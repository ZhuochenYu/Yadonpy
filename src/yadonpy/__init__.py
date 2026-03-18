"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Design inspiration: RadonPy and yuzc's yzc-gmx-gen toolkit.
"""

from __future__ import annotations

__version__ = "0.4.35"

# Convenience re-exports for examples / user scripts
from .sim import qm  # noqa: F401

__all__ = ["__version__", "qm"]
