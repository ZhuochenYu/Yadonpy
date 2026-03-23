"""YadonPy is an automated, SMILES-driven workflow developed by yuzc for property calculations of
polymer/solvent/salt blend systems using GROMACS-based molecular dynamics.

Its software design is inspired by RadonPy (an automated workflow for polymer bulk-property simulations
developed by a Japanese research group) and by yuzc's in-house yzc-gmx-gen toolkit. To the best of our
knowledge, this project does not raise copyright issues.
"""

# ******************************************************************************
# YadonPy core.__init__
# ******************************************************************************

from .molspec import MolSpec, as_rdkit_mol, molecular_weight  # noqa: F401
from .workdir import WorkDir, workdir, workunit  # noqa: F401
from ..interface import build_interface, build_interface_from_workdirs  # noqa: F401

__all__ = ["MolSpec", "as_rdkit_mol", "molecular_weight", "WorkDir", "workdir", "workunit", "build_interface", "build_interface_from_workdirs"]

